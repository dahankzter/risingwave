// Copyright 2026 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use iceberg::transaction::{ApplyTransactionAction, Transaction};
use itertools::Itertools;
use risingwave_connector::connector_common::IcebergCommittedSnapshot;
use risingwave_connector::sink::SinkError;
use risingwave_connector::sink::catalog::SinkId;
use thiserror_ext::AsReport;
use tokio::sync::oneshot::Sender;
use tokio::task::JoinHandle;

use super::*;

#[derive(Debug, Clone)]
pub(super) enum SnapshotExpirationProtection {
    NoProtection,
    Skip,
    Watermark(IcebergCommittedSnapshot),
}

impl SnapshotExpirationProtection {
    fn protect_with(&mut self, snapshot: Option<&IcebergCommittedSnapshot>) {
        let Some(snapshot) = snapshot else {
            *self = Self::Skip;
            return;
        };

        if matches!(self, Self::Skip) {
            return;
        }

        match self {
            Self::NoProtection => {
                *self = Self::Watermark(snapshot.clone());
            }
            Self::Watermark(existing) => {
                if snapshot.timestamp_ms < existing.timestamp_ms {
                    *existing = snapshot.clone();
                }
            }
            Self::Skip => {}
        }
    }
}

impl IcebergCompactionManagerInner {
    pub(super) fn begin_snapshot_expiration(
        &mut self,
        sink_id: SinkId,
    ) -> Option<SnapshotExpirationProtection> {
        if !self
            .snapshot_expiration_in_progress_sink_ids
            .insert(sink_id)
        {
            return None;
        }

        Some(self.snapshot_expiration_protection(sink_id))
    }

    pub(super) fn finish_snapshot_expiration(&mut self, sink_id: SinkId) {
        self.snapshot_expiration_in_progress_sink_ids
            .remove(&sink_id);
    }

    pub(super) fn snapshot_expiration_protection(
        &self,
        sink_id: SinkId,
    ) -> SnapshotExpirationProtection {
        let mut protection = SnapshotExpirationProtection::NoProtection;

        if let Some(track) = self.sink_schedules.get(&sink_id)
            && let Some(gc_watermark_snapshot) = track.processing_gc_watermark_snapshot()
        {
            protection.protect_with(gc_watermark_snapshot);
        }

        protection
    }
}

impl IcebergCompactionManager {
    pub fn gc_loop(manager: Arc<Self>, interval_sec: u64) -> (JoinHandle<()>, Sender<()>) {
        assert!(
            interval_sec > 0,
            "Iceberg GC interval must be greater than 0"
        );
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let join_handle = tokio::spawn(async move {
            tracing::info!(
                interval_sec = interval_sec,
                "Starting Iceberg GC loop with configurable interval"
            );
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_sec));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = manager.perform_gc_operations().await {
                            tracing::error!(error = ?e.as_report(), "GC operations failed");
                        }
                    },
                    _ = &mut shutdown_rx => {
                        tracing::info!("Iceberg GC loop is stopped");
                        return;
                    }
                }
            }
        });

        (join_handle, shutdown_tx)
    }

    async fn perform_gc_operations(&self) -> MetaResult<()> {
        let sink_ids = {
            let guard = self.inner.read();
            guard
                .snapshot_expiration_sink_ids
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        };

        tracing::info!("Starting GC operations for {} tables", sink_ids.len());

        for sink_id in sink_ids {
            if let Err(e) = self.check_and_expire_snapshots(sink_id).await {
                tracing::error!(error = ?e.as_report(), "Failed to perform GC for sink {}", sink_id);
            }
        }

        tracing::info!("GC operations completed");
        Ok(())
    }

    pub async fn check_and_expire_snapshots(&self, sink_id: SinkId) -> MetaResult<()> {
        const MAX_SNAPSHOT_AGE_MS_DEFAULT: i64 = 24 * 60 * 60 * 1000;
        let now = chrono::Utc::now().timestamp_millis();

        let iceberg_config = self.load_iceberg_config(sink_id).await?;
        if !iceberg_config.enable_snapshot_expiration {
            let mut guard = self.inner.write();
            guard.snapshot_expiration_sink_ids.remove(&sink_id);
            return Ok(());
        }

        let default_snapshot_expiration_timestamp_ms = now - MAX_SNAPSHOT_AGE_MS_DEFAULT;

        let mut snapshot_expiration_timestamp_ms =
            match iceberg_config.snapshot_expiration_timestamp_ms(now) {
                Some(timestamp) => timestamp,
                None => default_snapshot_expiration_timestamp_ms,
            };

        let Some(snapshot_expiration_protection) =
            self.inner.write().begin_snapshot_expiration(sink_id)
        else {
            tracing::info!(
                catalog_name = iceberg_config.catalog_name(),
                table_name = iceberg_config.full_table_name()?.to_string(),
                %sink_id,
                "Skip snapshots expiration because another expiration is in progress",
            );
            return Ok(());
        };
        let _snapshot_expiration_guard = scopeguard::guard(self.inner.clone(), move |inner| {
            inner.write().finish_snapshot_expiration(sink_id);
        });

        match snapshot_expiration_protection {
            SnapshotExpirationProtection::NoProtection => {}
            SnapshotExpirationProtection::Skip => {
                tracing::info!(
                    catalog_name = iceberg_config.catalog_name(),
                    table_name = iceberg_config.full_table_name()?.to_string(),
                    %sink_id,
                    "Skip snapshots expiration because an iceberg compaction task has no observed GC watermark",
                );
                return Ok(());
            }
            SnapshotExpirationProtection::Watermark(snapshot) => {
                let original_snapshot_expiration_timestamp_ms = snapshot_expiration_timestamp_ms;
                snapshot_expiration_timestamp_ms =
                    snapshot_expiration_timestamp_ms.min(snapshot.timestamp_ms);
                tracing::info!(
                    catalog_name = iceberg_config.catalog_name(),
                    table_name = iceberg_config.full_table_name()?.to_string(),
                    %sink_id,
                    gc_watermark_branch = %snapshot.branch,
                    gc_watermark_snapshot_id = snapshot.snapshot_id,
                    gc_watermark_timestamp_ms = snapshot.timestamp_ms,
                    original_snapshot_expiration_timestamp_ms,
                    protected_snapshot_expiration_timestamp_ms = snapshot_expiration_timestamp_ms,
                    "Protect snapshots expiration with iceberg compaction GC watermark",
                );
            }
        }

        let catalog = iceberg_config.create_catalog().await?;
        let mut table = catalog
            .load_table(&iceberg_config.full_table_name()?)
            .await
            .map_err(|e| SinkError::Iceberg(e.into()))?;

        let metadata = table.metadata();
        let snapshots = metadata.snapshots().collect_vec();

        if !snapshots
            .iter()
            .any(|snapshot| snapshot.timestamp_ms() < snapshot_expiration_timestamp_ms)
        {
            return Ok(());
        }

        tracing::info!(
            catalog_name = iceberg_config.catalog_name(),
            table_name = iceberg_config.full_table_name()?.to_string(),
            %sink_id,
            snapshots_len = snapshots.len(),
            snapshot_expiration_timestamp_ms = snapshot_expiration_timestamp_ms,
            snapshot_expiration_retain_last = ?iceberg_config.snapshot_expiration_retain_last,
            clear_expired_files = ?iceberg_config.snapshot_expiration_clear_expired_files,
            clear_expired_meta_data = ?iceberg_config.snapshot_expiration_clear_expired_meta_data,
            "try trigger snapshots expiration",
        );

        let txn = Transaction::new(&table);

        let mut expired_snapshots = txn
            .expire_snapshot()
            .expire_older_than(snapshot_expiration_timestamp_ms)
            .clear_expire_files(iceberg_config.snapshot_expiration_clear_expired_files)
            .clear_expired_meta_data(iceberg_config.snapshot_expiration_clear_expired_meta_data);

        if let Some(retain_last) = iceberg_config.snapshot_expiration_retain_last {
            expired_snapshots = expired_snapshots.retain_last(retain_last);
        }

        let before_metadata = table.metadata_ref();
        let tx = expired_snapshots
            .apply(txn)
            .map_err(|e| SinkError::Iceberg(e.into()))?;
        table = tx
            .commit(catalog.as_ref())
            .await
            .map_err(|e| SinkError::Iceberg(e.into()))?;

        if iceberg_config.snapshot_expiration_clear_expired_files {
            table
                .cleanup_expired_files(&before_metadata)
                .await
                .map_err(|e| SinkError::Iceberg(e.into()))?;
        }

        tracing::info!(
            catalog_name = iceberg_config.catalog_name(),
            table_name = iceberg_config.full_table_name()?.to_string(),
            %sink_id,
            "Expired snapshots for iceberg table",
        );

        Ok(())
    }
}
