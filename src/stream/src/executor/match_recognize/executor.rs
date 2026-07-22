// Copyright 2025 RisingWave Labs
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

//! Streaming `MATCH_RECOGNIZE` executor (v1, event-time).
//!
//! Scope: append-only input, `ONE ROW PER MATCH`,
//! `AFTER MATCH SKIP {PAST LAST ROW | TO NEXT ROW | TO {FIRST | LAST} <var>}`,
//! `MEASURES` with per-variable navigation (`FIRST`/`LAST`/bare `var.col`) and `CLASSIFIER()`.
//!
//! Event-time model: rows are buffered per partition (in any arrival order). Matching is driven by
//! the watermark on the leading `ORDER BY` column: when the watermark advances to `w`, every row
//! with `order_key < w` is *safe* (its content is final), so the buffer is sorted by order key and
//! the `< w` prefix is matched. A match becomes final once it is followed by another decided row (so
//! the greedy match is known maximal); consumed rows are evicted. This handles out-of-order arrival
//! within the watermark's lateness and bounds state. How a final match reaches the output depends on
//! the emit mode (see "Emission modes" below).
//!
//! Physical `PREV` needs no retention machinery: the binder only admits `PREV(.., k)` on variables
//! at least `k` rows from the match start, so those reads stay inside the match span, whose rows
//! are always retained while the match is live. Physical `NEXT` in `DEFINE` is rejected at bind
//! time: a row's verdict would depend on rows after it, and doing that correctly needs
//! per-candidate decidability (an out-of-range read must be a wait for exactly that candidate —
//! not a NULL verdict, and not a global horizon that can starve an idle partition whose match is
//! already decidable). Until that lands, the safe prefix is the only boundary.
//!
//! The boundary is **strict**. A RisingWave watermark `w` promises only that no future row will have
//! `order_key < w`; a row with `order_key == w` may still arrive, and `watermark_filter` forwards it
//! (it keeps `event_time >= watermark`). So a row at exactly `w` is not final, and neither is a
//! `WITHIN` deadline at exactly `w` — a completing row at `w` can still land inside the bound. Every
//! finality decision here is therefore `< w`, and the wakeup frontier's complement is `>= w` so that
//! a partition is always revisited for its rows sitting at `w`. Emit and eviction share the same
//! predicate on purpose: were the emit side stricter than the eviction side, eviction would delete
//! the rows of a match the emit side is still holding, dropping it silently.
//!
//! `AFTER MATCH SKIP TO FIRST|LAST <var>` has no valid resume row when the target is bound to no row
//! of a match, or resolves to the match's own first row. Both are data-dependent, so instead of
//! failing the actor (which would crash-loop a committed materialized view) the resume position
//! degrades to a weaker skip strategy and the degradation is *reported* — see [`SkipMode::next_pos`]
//! and [`report_skip_degradation_once`].
//!
//! Measures are evaluated at match time, not at arrival: a measure references specific matched rows
//! (e.g. `FIRST(a.ts)`, `LAST(b.v)`), which are only known once the match and its per-row pattern
//! variable labels are found. Each measure is an expression over a synthetic row whose columns are
//! produced by its [`MeasureSlot`]s from the matched rows.
//!
//! State: only the raw buffered rows are persisted to a state table (row layout `[seq, input
//! columns...]`; satisfied pattern variables are re-derived by matching on scan, never stored) —
//! written through on arrival, deleted on consumption — and restored on recovery.
//!
//! Emission modes (the `emit_on_update` flag, set by the planner): under EMIT ON WINDOW CLOSE
//! (`emit_on_update = false`) emission is *final-only* — the completed match above is emitted as an
//! append-only `Insert` at the watermark, and nothing is ever retracted. Under the plain form
//! (`emit_on_update = true`) emission is a retract changelog: at each barrier every partition touched
//! since the last barrier is re-matched over its *whole* buffer ("as if input ended now", not just the
//! safe prefix), its provisional set is diffed against what was last emitted, and the `Delete`/`Insert`
//! ops are yielded (keyed by `_match_id`, the match's start-row seq). In this mode the watermark path
//! emits nothing; it only evicts, after an emit-before-finalize diff closes the same-epoch
//! never-emitted-match hole (see the `Message::Watermark` arm). Finalization never *creates* a
//! retraction — a match over the whole buffer cannot be invalidated by being declared final — so the
//! diff base (`last_emitted`) is rebuilt from the buffer on recovery/rescale without re-emitting.
//!
//! Matching is driven by a per-partition [`IncrementalMatcher`] (kept in an in-memory `matchers`
//! cache): each visit feeds the newly-safe rows and reads `provisional()`, which by construction
//! equals a from-scratch `find_matches_dynamic` over the safe prefix (see the incremental module's
//! differential oracle), so emission and eviction are byte-identical to the previous full-rescan
//! path. The cache is a pure derivation of state-table content — dropped on recovery and on any
//! vnode-bitmap change, and rebuilt lazily per partition by feeding its recovered buffer. It carries
//! the live-window seqs + labels (never row data) — the identities the emit-on-update diff retracts
//! against. Eviction and empty-partition removal keep the buffer bounded to the live (unfinalized)
//! window, so the work per watermark is bounded by that window rather than the partition's history.
//!
//! Incremental cost, per mode: the matcher freezes a leading prefix of matches whose scan region is
//! dead at the current boundary and rescans only the mutable suffix, so its payoff is the frozen
//! prefix it skips across visits. Under EMIT ON WINDOW CLOSE that payoff is nil — eviction advances the
//! retained front to the live frontier every watermark, so the frozen prefix is dropped as fast as it
//! forms and `next_pos` rebases back toward 0 each visit; eviction alone bounds the work, and the
//! incrementality here is plumbing the plain form needs. Under emit-on-update it does pay off: the
//! whole-buffer barrier diffs run between the watermark evictions, so for append-mostly input the
//! frozen prefix persists across barriers and each diff rescans only the newly-appended suffix.

use std::collections::{HashMap, HashSet, hash_map};
use std::ops::Bound;

use futures::{StreamExt, pin_mut};
use risingwave_common::array::{Op, StreamChunk};
use risingwave_common::hash::VnodeBitmapExt;
use risingwave_common::row::{OwnedRow, Row, RowExt, once};
use risingwave_common::types::{DataType, Datum, DefaultOrd, ScalarImpl, ToOwnedDatum};
use risingwave_common::util::iter_util::ZipEqFast;
use risingwave_common::util::row_id::RowIdGenerator;
use risingwave_expr::ExprError;
use risingwave_expr::aggregate::{AggCall, BoxedAggregateFunction, build_append_only};
use risingwave_expr::expr::{EvalErrorReport, NonStrictExpression, build_non_strict_from_prost};
use risingwave_pb::stream_plan::{
    MatchRecognizeDefine as PbMatchRecognizeDefine,
    MatchRecognizeMeasure as PbMatchRecognizeMeasure,
};
use risingwave_storage::StateStore;
use risingwave_storage::store::PrefetchOptions;

use super::incremental::{
    Finalized, IncrementalMatcher, Seq, SeqMatch, diff_provisional, plan_provisional_rows,
};
use super::nfa::{CandidateMatcher, LabeledMatch, Nfa, SkipDegradation, SkipMode};
use crate::common::table::state_table::StateTable;
use crate::executor::prelude::*;
use crate::task::ActorEvalErrorReport;

/// Report an `AFTER MATCH SKIP` degradation ([`SkipDegradation`]) — unless the same degradation was
/// already reported in this watermark pass, in which case it is dropped.
///
/// The condition is data-dependent and deliberately not fatal (see [`SkipMode::next_pos`] for why an
/// error would turn a committed materialized view into a crash loop), so the only thing left is to
/// make it visible. It goes to the actor's [`EvalErrorReport`], which is the surface every expression
/// evaluation error in this operator already uses: the rate-limited `stream_expr_error` log and the
/// `user_compute_error` metric, labelled `["ExprError", executor_name, fragment_id]`.
///
/// **The carrier is new, the surface is not.** Nothing else in the tree hands `EvalErrorReport` a
/// *synthesized* error — every other reporter passes on an error produced by an actual expression
/// evaluation. [`ExprError`] is nonetheless the only type the trait accepts, and
/// `ExprError::InvalidParam` is the honest fit: the query's `AFTER MATCH SKIP` parameter cannot be
/// honored. (`ExprError::Custom` was rejected — it is the UDF error channel and slated for removal;
/// `Internal`/`InvalidState` would misreport a user-query problem as an engine fault.) Two
/// consequences to expect when reading the output:
///
///  * the log line carries the surface's fixed prefix `failed to evaluate expression`, hardcoded in
///    `ActorContext::on_compute_error`, even though no expression was evaluated here. The actionable
///    content is the `error=` field, which is self-contained;
///  * the metric labels separate this operator from others, but not this operator's own
///    `DEFINE`/`MEASURES`/`WITHIN` evaluation errors, which report through the same labels. So the
///    metric reads as "this `MATCH_RECOGNIZE` query is unhealthy" and the log line is what says why.
///
/// **Volume policy.** The cause is a property of the query, not of one row: a skip target that no
/// match can ever bind degrades on every match, forever, and even a target that only *sometimes*
/// fails to bind (`PATTERN (a? b)` with `SKIP TO FIRST b`, degrading on the matches where `a` did not
/// bind) repeats without bound. The diagnostic names the skip clause, its target variable and the
/// applied fallback — and nothing row-, match- or partition-specific — so every repetition within one
/// watermark pass is a byte-identical duplicate carrying no new information, at the cost of a
/// `format!` on the emit path. `already_reported` therefore holds the kinds already reported in this
/// pass (at most two exist, and `Vec::new()` allocates only if one actually fires); it is reset per
/// pass, so a persisting condition keeps producing one report per kind per watermark — a steady
/// signal, bounded by watermark frequency rather than by match or partition count. The trade-off is
/// deliberate: the metric counts *passes* that degraded, not degradations.
fn report_skip_degradation_once(
    report: &impl EvalErrorReport,
    skip: &SkipMode,
    degradation: SkipDegradation,
    already_reported: &mut Vec<SkipDegradation>,
) {
    if already_reported.contains(&degradation) {
        return;
    }
    already_reported.push(degradation);
    // The mode is named once, by `clause_name`; `describe` names the target variable and the fallback.
    report.report(ExprError::InvalidParam {
        name: skip.clause_name(),
        reason: degradation.describe(skip).into(),
    });
}

/// How a [`MeasureSlot`] resolves against the rows of a match (mirrors the planner's slot kinds).
#[derive(Clone, Copy, PartialEq, Eq)]
enum MeasureSlotKind {
    /// Column value of the first row labeled `var` within the match.
    First,
    /// Column value of the last row labeled `var` (also a bare `var.col` under FINAL semantics).
    Last,
    /// The pattern variable bound to the match's last row.
    Classifier,
    /// Number of rows in the whole match (`COUNT(*)`).
    CountStar,
    /// Number of rows labeled `var` with a non-null `col` (`COUNT(var.col)`).
    Count,
    /// Minimum `col` over rows labeled `var` (`MIN(var.col)`).
    Min,
    /// Maximum `col` over rows labeled `var` (`MAX(var.col)`).
    Max,
    /// `SUM(var.col)`, evaluated by the slot's [`AggSlot`] aggregate kernel. `AVG` is lowered to a
    /// `Sum` slot plus a `Count` slot and a division expression, so it has no kind of its own.
    Sum,
}

/// A `SUM`/`AVG` aggregate kernel for a slot, plus the input column type used to feed it.
struct AggSlot {
    func: BoxedAggregateFunction,
    col_type: DataType,
}

/// One navigation input that a measure expression reads. The executor materializes one value per
/// slot from a match's rows and labels, forming the synthetic row the measure is evaluated over.
struct MeasureSlot {
    kind: MeasureSlotKind,
    /// Pattern variables this slot navigates over (several for a `SUBSET`). A row matches if its
    /// label is any of these. Empty for [`MeasureSlotKind::Classifier`].
    vars: Vec<String>,
    /// Input column index to read. Unused for [`MeasureSlotKind::Classifier`].
    col_idx: usize,
    /// The aggregate kernel for [`MeasureSlotKind::Sum`] (`AVG` is lowered to `Sum` plus `Count`).
    agg: Option<AggSlot>,
}

/// A `MEASURES` item compiled for execution.
pub struct CompiledMeasure {
    /// Expression over the synthetic per-match row: `InputRef(i)` reads `slots[i]`.
    expr: NonStrictExpression,
    slots: Vec<MeasureSlot>,
}

impl CompiledMeasure {
    /// Builds a compiled measure from its protobuf, building any aggregate kernels its slots need.
    pub fn from_protobuf(
        pb: &PbMatchRecognizeMeasure,
        error_report: impl EvalErrorReport + 'static,
    ) -> StreamExecutorResult<Self> {
        let expr = build_non_strict_from_prost(
            pb.expr
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MATCH_RECOGNIZE measure missing expression"))?,
            error_report,
        )?;
        let slots = pb
            .slots
            .iter()
            .map(|s| {
                let kind = match s.kind {
                    0 => MeasureSlotKind::Last,
                    1 => MeasureSlotKind::First,
                    2 => MeasureSlotKind::Classifier,
                    3 => MeasureSlotKind::CountStar,
                    4 => MeasureSlotKind::Count,
                    5 => MeasureSlotKind::Min,
                    6 => MeasureSlotKind::Max,
                    7 => MeasureSlotKind::Sum,
                    // Fail fast on an unknown kind rather than silently treating it as LAST, which
                    // would change measure semantics under a corrupt plan or version skew.
                    other => {
                        return Err(anyhow::anyhow!(
                            "invalid MATCH_RECOGNIZE measure slot kind: {other}"
                        )
                        .into());
                    }
                };
                let agg = match kind {
                    MeasureSlotKind::Sum => {
                        let call =
                            AggCall::from_protobuf(s.agg_call.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "MATCH_RECOGNIZE SUM/AVG measure slot missing agg_call"
                                )
                            })?)?;
                        let col_type = call.args.arg_types()[0].clone();
                        let func = build_append_only(&call)?;
                        Some(AggSlot { func, col_type })
                    }
                    _ => None,
                };
                Ok(MeasureSlot {
                    kind,
                    vars: s.vars.clone(),
                    col_idx: s.col_idx as usize,
                    agg,
                })
            })
            .collect::<StreamExecutorResult<Vec<_>>>()?;
        Ok(CompiledMeasure { expr, slots })
    }
}

impl MeasureSlot {
    /// Resolves this slot against a match: `rows[start..]` are the matched rows and `labels[i]` is
    /// the pattern variable bound to `rows[start + i]`.
    async fn resolve(
        &self,
        rows: &[BufferedRow],
        start: usize,
        labels: &[String],
    ) -> StreamExecutorResult<Datum> {
        // The column value of the row at match-relative index `j`.
        let col_at = |j: usize| rows[start + j].row.datum_at(self.col_idx).to_owned_datum();
        // Whether a row's label is one this slot navigates over (a plain var, or any SUBSET member).
        let matches = |l: &String| self.vars.iter().any(|v| v == l);
        Ok(match self.kind {
            MeasureSlotKind::Classifier => {
                labels.last().map(|s| ScalarImpl::Utf8(s.as_str().into()))
            }
            MeasureSlotKind::First => labels.iter().position(&matches).and_then(col_at),
            MeasureSlotKind::Last => labels.iter().rposition(&matches).and_then(col_at),
            MeasureSlotKind::CountStar => Some(ScalarImpl::Int64(labels.len() as i64)),
            MeasureSlotKind::Count => {
                let n = labels
                    .iter()
                    .enumerate()
                    .filter(|(j, l)| matches(l) && col_at(*j).is_some())
                    .count();
                Some(ScalarImpl::Int64(n as i64))
            }
            MeasureSlotKind::Min => labels
                .iter()
                .enumerate()
                .filter(|(_, l)| matches(l))
                .filter_map(|(j, _)| col_at(j))
                .min_by(|a, b| a.default_cmp(b)),
            MeasureSlotKind::Max => labels
                .iter()
                .enumerate()
                .filter(|(_, l)| matches(l))
                .filter_map(|(j, _)| col_at(j))
                .max_by(|a, b| a.default_cmp(b)),
            MeasureSlotKind::Sum => {
                let agg = self.agg.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("MATCH_RECOGNIZE SUM measure slot has no kernel")
                })?;
                // Feed the kernel a single-column chunk of the col values over the matching rows.
                let input: Vec<(Op, OwnedRow)> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| matches(l))
                    .map(|(j, _)| (Op::Insert, OwnedRow::new(vec![col_at(j)])))
                    .collect();
                if input.is_empty() {
                    None
                } else {
                    let chunk = StreamChunk::from_rows(&input, std::slice::from_ref(&agg.col_type));
                    let mut state = agg.func.create_state()?;
                    agg.func.update(&mut state, &chunk).await?;
                    agg.func.get_result(&state).await?
                }
            }
        })
    }
}

/// How a [`DefineSlot`] resolves against the candidate row (mirrors the planner's slot kinds).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefineSlotKind {
    /// The candidate row's own column.
    SelfCol,
    /// `PREV(col, offset)`: a row `offset` positions earlier in the ordered partition.
    Prev,
    /// `NEXT(col, offset)`: `offset` positions later.
    Next,
    /// `FIRST(var.col)`: the first row labeled `vars` in the in-progress match. The candidate row
    /// counts as tentatively labeled when `vars` contains the variable being defined (see
    /// [`DefineMatcher::slot_value`]).
    RunningFirst,
    /// `LAST(var.col)` / bare other-variable reference: the last such row (running), which is the
    /// candidate row itself when `vars` contains the variable being defined.
    RunningLast,
}

/// One input a `DEFINE` predicate reads (mirrors the planner's [`DefineSlot`]).
struct DefineSlot {
    kind: DefineSlotKind,
    vars: Vec<String>,
    col_idx: usize,
    offset: usize,
}

/// A `DEFINE` predicate compiled for execution: a boolean condition over a synthetic slot row.
pub struct CompiledDefine {
    symbol: String,
    condition: NonStrictExpression,
    slots: Vec<DefineSlot>,
}

impl CompiledDefine {
    pub fn from_protobuf(
        pb: &PbMatchRecognizeDefine,
        error_report: impl EvalErrorReport + 'static,
    ) -> StreamExecutorResult<Self> {
        let condition = build_non_strict_from_prost(
            pb.condition
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MATCH_RECOGNIZE define missing condition"))?,
            error_report,
        )?;
        let slots = pb
            .slots
            .iter()
            .map(|s| {
                let kind = match s.kind {
                    0 => DefineSlotKind::SelfCol,
                    1 => DefineSlotKind::Prev,
                    2 => DefineSlotKind::Next,
                    3 => DefineSlotKind::RunningFirst,
                    4 => DefineSlotKind::RunningLast,
                    // Fail fast on an unknown kind rather than silently treating it as a self-column
                    // reference, which would change the DEFINE predicate's meaning under a corrupt
                    // plan or version skew.
                    other => {
                        return Err(StreamExecutorError::from(anyhow::anyhow!(
                            "invalid MATCH_RECOGNIZE define slot kind: {other}"
                        )));
                    }
                };
                Ok(DefineSlot {
                    kind,
                    vars: s.vars.clone(),
                    col_idx: s.col_idx as usize,
                    offset: s.offset as usize,
                })
            })
            .collect::<StreamExecutorResult<Vec<_>>>()?;
        Ok(CompiledDefine {
            symbol: pb.symbol.clone(),
            condition,
            slots,
        })
    }
}

/// Evaluates `DEFINE` predicates against the in-progress match, driving the NFA. Holds the sorted
/// safe-prefix rows of one partition and the compiled `DEFINE`s; a variable with no `DEFINE` is
/// universally true.
struct DefineMatcher<'a> {
    rows: &'a [BufferedRow],
    safe_len: usize,
    defines: &'a HashMap<String, CompiledDefine>,
    /// `WITHIN` span predicate over `[last_order_key, first_order_key]`. Applied as a candidate is
    /// bound so the NFA prunes any extension that would push the match's span past the bound,
    /// yielding the longest match that fits the window rather than rejecting an overshooting greedy
    /// match after the fact.
    within: Option<&'a NonStrictExpression>,
}

impl DefineMatcher<'_> {
    /// The value a slot reads for a candidate at `pos` being tested for pattern variable `var`, where
    /// `match_start` is the match's first row and `labels[k]` is the variable bound to
    /// `rows[match_start + k]`.
    ///
    /// `labels` covers only the rows *already* bound, so for running navigation the candidate is the
    /// implicit trailing label: while its membership is still tentative, the running set a `DEFINE`
    /// predicate sees is `labels ++ [var]`. It therefore participates in `RunningFirst`/`RunningLast`
    /// whenever the slot's variable set contains `var` — including via a `SUBSET` that has `var` as a
    /// member. This is what makes `DEFINE a AS LAST(a.v) = a.v` a tautology, as SQL:2016 requires: a
    /// pattern-variable-qualified column reference *is* `RUNNING LAST` of that column, and the binder
    /// already lowers the bare `a.v` inside `a`'s own `DEFINE` to the candidate row.
    fn slot_value(
        &self,
        slot: &DefineSlot,
        var: &str,
        pos: usize,
        match_start: usize,
        labels: &[String],
    ) -> Datum {
        let col_at = |i: usize| self.rows[i].row.datum_at(slot.col_idx).to_owned_datum();
        let in_var = |l: &str| slot.vars.iter().any(|v| v == l);
        // Whether the candidate row itself belongs to the set this slot navigates over.
        let candidate_in_var = in_var(var);
        match slot.kind {
            DefineSlotKind::SelfCol => col_at(pos),
            DefineSlotKind::Prev => pos.checked_sub(slot.offset).and_then(col_at),
            DefineSlotKind::Next => {
                // Unreachable from SQL: the binder rejects physical NEXT in DEFINE (a verdict must
                // not depend on rows after the candidate). Kept for proto compatibility; a stale
                // plan reaching it gets a conservative NULL.
                let i = pos + slot.offset;
                if i < self.safe_len { col_at(i) } else { None }
            }
            // The candidate is the running first only when no earlier row of the match is in the set.
            DefineSlotKind::RunningFirst => labels
                .iter()
                .position(|l| in_var(l))
                .map(|k| match_start + k)
                .or_else(|| candidate_in_var.then_some(pos))
                .and_then(col_at),
            // The candidate is the newest row, so it is the running last whenever it is in the set.
            DefineSlotKind::RunningLast => candidate_in_var
                .then_some(pos)
                .or_else(|| {
                    labels
                        .iter()
                        .rposition(|l| in_var(l))
                        .map(|k| match_start + k)
                })
                .and_then(col_at),
        }
    }
}

impl CandidateMatcher for DefineMatcher<'_> {
    async fn matches(
        &self,
        var: &str,
        pos: usize,
        labels: &[String],
    ) -> StreamExecutorResult<bool> {
        let match_start = pos - labels.len();
        // A pattern variable with no DEFINE matches every row; one with a DEFINE must satisfy it.
        if let Some(def) = self.defines.get(var) {
            let synthetic: Vec<Datum> = def
                .slots
                .iter()
                .map(|slot| self.slot_value(slot, var, pos, match_start, labels))
                .collect();
            let value = def
                .condition
                .eval_row_infallible(&OwnedRow::new(synthetic))
                .await;
            if !value.is_some_and(|s| s.into_bool()) {
                return Ok(false);
            }
        }
        // WITHIN: binding `pos` extends the match to span `[match_start, pos]`. Reject the candidate
        // if that span exceeds the bound, so the NFA backtracks to a shorter match that fits.
        if let Some(within) = self.within {
            let first_key = self.rows[match_start].order_key.clone();
            let last_key = self.rows[pos].order_key.clone();
            let span_row = OwnedRow::new(vec![last_key, first_key]);
            let value = within.eval_row_infallible(&span_row).await;
            if !value.is_some_and(|s| s.into_bool()) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Align a partition's [`IncrementalMatcher`] with the freshly-scanned safe prefix `rows[0..safe_len]`
/// (PK-ordered, so the fed rows should be a leading run of it) before its `provisional()` is read.
/// On return, `provisional()` equals a from-scratch `find_matches_dynamic(safe_len, matcher, skip)`
/// over `rows[0..safe_len]` — in every case below, not just the append path.
///
/// The common case is a plain append: the safe rows already fed are a prefix of `rows`, so the tail
/// `rows[fed_len..safe_len]` is fed via [`IncrementalMatcher::advance`]. If a safe row arrived
/// out-of-order — its sorted position precedes an already-fed row, so the first divergence sits at a
/// position `< fed.len()` — the matcher is rolled back with [`IncrementalMatcher::truncate_from_seq`]
/// at the first displaced fed row's seq (the resync point), then the corrected sorted suffix from the
/// divergence is re-fed. `matcher` is this visit's `DefineMatcher`; it resolves absolute positions
/// into `rows`, so the matcher's positions stay aligned with the buffer.
///
/// The third case is an over-feed rollback: more rows were fed than this visit's window (under
/// emit-on-update the barrier/emit-before-finalize paths feed the *whole* buffer, and the watermark
/// eviction pass then narrows back to the safe prefix). The truncation rolls the over-fed tail back,
/// but it also drops the provisional matches over the retained fed suffix `[next_pos, safe_len)` —
/// and with nothing left to re-feed (`advance` would be a no-op), those matches must be re-derived
/// in place via [`IncrementalMatcher::rescan`], or `provisional()` would silently under-report the
/// mutable suffix until the next feed.
async fn refresh_matcher(
    inc: &mut IncrementalMatcher,
    rows: &[BufferedRow],
    safe_len: usize,
    matcher: &(impl CandidateMatcher + Sync),
) -> StreamExecutorResult<()> {
    // First position where the freshly-read safe buffer diverges from the already-fed seqs.
    let (diverge_at, resync) = {
        let fed = inc.fed_seqs();
        let mut i = 0;
        while i < safe_len && i < fed.len() && rows[i].seq == fed[i] {
            i += 1;
        }
        // A divergence with fed rows still remaining means a safe row sorts before an already-fed
        // one (or, at `i == safe_len`, that the matcher was over-fed past this window): `fed[i]` is
        // the first fed row whose sorted position shifted (the resync seq).
        let resync = (i < fed.len()).then(|| fed[i]);
        (i, resync)
    };
    if let Some(resync) = resync {
        inc.truncate_from_seq(resync, matcher).await?;
    }
    // Feed the (corrected) sorted suffix from the divergence point. Empty in the steady state where
    // nothing new became safe (a no-op `advance` then) — and in the over-feed rollback, where the
    // retained rows are still fed and only their truncation-dropped provisional matches need
    // re-deriving: rescan in place instead.
    let new_seqs: Vec<Seq> = rows[diverge_at..safe_len].iter().map(|r| r.seq).collect();
    if new_seqs.is_empty() && resync.is_some() {
        inc.rescan(matcher).await?;
    } else {
        inc.advance(&new_seqs, matcher).await?;
    }
    Ok(())
}

/// Build the output row for one match: the `PARTITION BY` columns, the `MEASURES` values evaluated
/// over the match's rows, and the trailing `_match_id` (the match's start-row `seq`). `start` is the
/// match's first buffer position and `labels[i]` binds `rows[start + i]`.
///
/// Shared by the watermark (EMIT ON WINDOW CLOSE) emit path and the barrier (emit-on-update) diff
/// path so both produce byte-identical output rows — extracting it out of the former watermark emit
/// block is a pure refactor, preserving its measure evaluation order and row layout. `WITHIN` is
/// enforced inside the matcher before a match reaches here, so there is no span check to repeat.
async fn build_match_row(
    measures: &[CompiledMeasure],
    rows: &[BufferedRow],
    partition_key: &OwnedRow,
    start: usize,
    labels: &[String],
) -> StreamExecutorResult<OwnedRow> {
    // Evaluate each measure over the synthetic row its slots produce from the matched rows + labels.
    let mut measure_datums: Vec<Datum> = Vec::with_capacity(measures.len());
    for measure in measures {
        let mut synthetic = Vec::with_capacity(measure.slots.len());
        for slot in &measure.slots {
            synthetic.push(slot.resolve(rows, start, labels).await?);
        }
        let synthetic = OwnedRow::new(synthetic);
        let value = measure.expr.eval_row_infallible(&synthetic).await;
        measure_datums.push(value);
    }
    // The match's identity is its start row's `seq`: deterministic across recovery replay (re-emission
    // after a rollback reproduces byte-identical output, unlike a freshly minted id), and unique
    // forever because an emitted match's start row is always evicted eventually (the same invariant
    // that prevents cross-watermark double emits under EMIT ON WINDOW CLOSE), so no later match can
    // ever share the start. This stable, replay-deterministic identity is what the emit-on-update
    // changelog diff retracts against.
    let match_id = rows[start].seq;
    let measures_row = OwnedRow::new(measure_datums);
    Ok(partition_key
        .chain(&measures_row)
        // Unwrap the seq newtype to its raw `i64` here, where the output `_match_id` datum is built.
        .chain(once(Some(ScalarImpl::Int64(match_id.0))))
        .into_owned_row())
}

/// Refresh a partition's cached matcher over the *whole* buffer — in emit-on-update the match window
/// is all data-so-far ("as if input ended now"), not the watermark-safe prefix — and return the
/// seq→buffer-position index used to map each seq-anchored provisional match back to its rows for
/// measure evaluation. Shared by the two emit-on-update derivations so they scan identically:
/// [`emit_partition_diff`] (the barrier diff) and [`compute_partition_emitted`] (the recovery reseed).
/// `rows` is the partition's buffer, already read by the caller, and must be non-empty; `inc` is the
/// partition's cached matcher (`matchers.entry(..)`), left refreshed over the whole buffer.
async fn refresh_partition_matcher(
    inc: &mut IncrementalMatcher,
    rows: &[BufferedRow],
    defines: &HashMap<String, CompiledDefine>,
    within: Option<&NonStrictExpression>,
) -> StreamExecutorResult<HashMap<Seq, usize>> {
    let safe_len = rows.len();
    let matcher = DefineMatcher {
        rows,
        safe_len,
        defines,
        within,
    };
    refresh_matcher(inc, rows, safe_len, &matcher).await?;
    let mut seq_to_pos: HashMap<Seq, usize> = HashMap::with_capacity(safe_len);
    for (p, r) in rows.iter().enumerate() {
        seq_to_pos.insert(r.seq, p);
    }
    Ok(seq_to_pos)
}

/// Compute one partition's current emit-on-update provisional set as `(match, output row)` pairs,
/// WITHOUT diffing or touching `last_emitted`, building EVERY match's output row fresh. Used by
/// [`rebuild_last_emitted`] to seed the base on recovery/rescale — there is no prior base to reuse
/// rows from, so all rows are built — emitting nothing.
///
/// Deterministic in the buffer content: the same buffer yields the same matcher feed, the same
/// `provisional()`, and the same `build_match_row` output rows. That determinism is what lets the
/// recovery/rescale rebuild seed the base without re-emitting.
async fn compute_partition_emitted(
    inc: &mut IncrementalMatcher,
    partition_key: &OwnedRow,
    rows: &[BufferedRow],
    defines: &HashMap<String, CompiledDefine>,
    within: Option<&NonStrictExpression>,
    measures: &[CompiledMeasure],
) -> StreamExecutorResult<Vec<(SeqMatch, OwnedRow)>> {
    let seq_to_pos = refresh_partition_matcher(inc, rows, defines, within).await?;
    let mut new_emitted: Vec<(SeqMatch, OwnedRow)> = Vec::with_capacity(inc.provisional().len());
    for m in inc.provisional() {
        let start = seq_to_pos[&m.start_seq];
        let out_row = build_match_row(measures, rows, partition_key, start, &m.labels).await?;
        new_emitted.push((m.clone(), out_row));
    }
    Ok(new_emitted)
}

/// Compute one partition's emit-on-update changelog: refresh the matcher over the whole buffer, diff
/// the new provisional set against the last-emitted base, and advance the base to it. Returns the
/// `Delete`/`Insert` ops for the caller to stream — a plain async fn cannot `yield` — while updating
/// `last_emitted` and the cached matcher in place. `rows` is the partition's buffer, already read by
/// the caller, and must be non-empty (the caller handles the drained-partition case).
///
/// Output rows are built lazily: [`plan_provisional_rows`] classifies each new match against the base
/// first, and the (expensive) `build_match_row` runs ONLY for a brand-new or content-changed
/// identity; an unchanged match reuses the base's stored row (byte-identical by construction). In the
/// steady state a partition's provisional set is unchanged barrier-to-barrier, so this rebuilds
/// nothing.
///
/// Shared by two callers so they produce byte-identical output and stay in lockstep:
/// * the barrier arm — the normal emit-on-update path, diffing every dirty partition before commit;
/// * the watermark arm's emit-before-finalize — which runs this first, before eviction removes any
///   rows, to close the never-emitted-match hole (see that call site).
///
/// The diff is deterministic in the buffer content. A partition handled by the watermark arm's
/// emit-before-finalize is cleared from `dirty_partitions` there, so the barrier no longer re-diffs
/// it — that re-diff was a guaranteed no-op (see the call site). A partition NOT woken this epoch is
/// still diffed here at the barrier.
async fn emit_partition_diff(
    inc: &mut IncrementalMatcher,
    last_emitted: &mut HashMap<OwnedRow, Vec<(SeqMatch, OwnedRow)>>,
    partition_key: &OwnedRow,
    rows: &[BufferedRow],
    defines: &HashMap<String, CompiledDefine>,
    within: Option<&NonStrictExpression>,
    measures: &[CompiledMeasure],
) -> StreamExecutorResult<Vec<(Op, OwnedRow)>> {
    let seq_to_pos = refresh_partition_matcher(inc, rows, defines, within).await?;
    let provisional = inc.provisional();

    // Classify each new match against the base BEFORE building any row: reuse the base's stored
    // output row for an unchanged identity, (re)build only for a new or content-changed one.
    let prev = last_emitted
        .get(partition_key)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let plan = plan_provisional_rows(prev, provisional);
    let mut new_emitted: Vec<(SeqMatch, OwnedRow)> = Vec::with_capacity(provisional.len());
    // `plan` is positional over `provisional` (one entry per provisional match, by construction of
    // `plan_provisional_rows`), so the equal-length zip asserts that invariant instead of silently
    // truncating if it were ever broken.
    for (m, reuse) in provisional.iter().zip_eq_fast(plan) {
        let out_row = match reuse {
            Some(i) => prev[i].1.clone(),
            None => {
                let start = seq_to_pos[&m.start_seq];
                build_match_row(measures, rows, partition_key, start, &m.labels).await?
            }
        };
        new_emitted.push((m.clone(), out_row));
    }

    // Ops in `diff_provisional`'s order (start-seq ascending, Delete-before-Insert per revision); the
    // base then advances to the new set.
    let ops = diff_provisional(prev, &new_emitted);
    // Advance the base. An empty set diffs identically against an absent entry and an empty one (both
    // are `&[]`), so drop the key rather than keep an empty vec (mirrors `rebuild_last_emitted`).
    // When the key already exists (the steady state), overwrite in place to avoid cloning it.
    if new_emitted.is_empty() {
        last_emitted.remove(partition_key);
    } else if let Some(slot) = last_emitted.get_mut(partition_key) {
        *slot = new_emitted;
    } else {
        last_emitted.insert(partition_key.clone(), new_emitted);
    }
    Ok(ops)
}

/// Rebuild the emit-on-update diff base (`last_emitted`) after recovery or a vnode-bitmap change,
/// WITHOUT emitting anything. For every partition with a buffer in the currently-owned vnodes,
/// recompute its provisional set over the whole buffer and seed `last_emitted` with it (populating
/// `matchers` as a side effect). By determinism the recomputed set is byte-identical to what the
/// pre-crash / pre-rescale actor last emitted for the same committed buffer, so it already equals
/// the rows downstream holds at the recovered epoch — re-emitting them would double-Insert. Hence
/// this diffs nothing and yields nothing; the next barrier diffs genuine new input against this base
/// and emits only the delta. Partitions rebuilt here are deliberately left OUT of `dirty_partitions`
/// (nothing changed), so a later chunk marks them dirty as usual and the next barrier diffs against
/// this rebuilt base.
///
/// Emit-on-update only: EMIT ON WINDOW CLOSE has no `last_emitted` and must not pay this cost.
///
/// Cost: one full matcher pass over each owned partition's buffer at startup — the recovery-latency
/// trade the design (spec §6) chose over persisting a second changelog state table (the base is a
/// pure derivation of the buffer, so recomputing it is cheaper overall than maintaining it on disk).
///
/// Partition enumeration scans the BUFFER table itself per owned vnode: its PK is `(partition
/// columns, ORDER BY columns, seq)` and `iter_with_vnode` yields memcomparable PK order, so a
/// partition's rows are contiguous within its vnode — one pass groups consecutive rows by partition
/// key, holding a single partition's rows resident at a time (the same memory bound as the
/// watermark/barrier scans), and is complete by construction: a partition has rows iff the scan
/// sees them.
///
/// Enumerating from the buffer table itself — not the wakeup-frontier index — is deliberate: the
/// buffer is the authoritative record of which partitions hold rows (complete by construction,
/// above), while the frontier index is only a wakeup *schedule*. "Every buffered partition has a
/// frontier entry" is a separate invariant this rebuild does not depend on — it currently holds
/// (emit-on-update requires `WITHIN` at plan time, and a surviving safe row always schedules a
/// within-deadline wakeup), but historically the no-`WITHIN` shape kept buffered rows with no
/// frontier entry, and a frontier-driven rebuild would have skipped exactly those partitions and
/// duplicate-Inserted their already-emitted matches at the next barrier. The buffer scan stays the
/// correct-by-construction enumerator regardless of that invariant.
#[expect(clippy::too_many_arguments)]
async fn rebuild_last_emitted<S: StateStore>(
    state_table: &StateTable<S>,
    matchers: &mut HashMap<OwnedRow, IncrementalMatcher>,
    last_emitted: &mut HashMap<OwnedRow, Vec<(SeqMatch, OwnedRow)>>,
    nfa: &Nfa,
    skip: &SkipMode,
    defines: &HashMap<String, CompiledDefine>,
    within: Option<&NonStrictExpression>,
    measures: &[CompiledMeasure],
    input_arity: usize,
    time_col: usize,
    partition_key_indices: &[usize],
) -> StreamExecutorResult<()> {
    let vnodes: Vec<_> = state_table.vnodes().iter_vnodes().collect();
    for vnode in vnodes {
        let sub_range: (Bound<OwnedRow>, Bound<OwnedRow>) = (Bound::Unbounded, Bound::Unbounded);
        let iter = state_table
            .iter_with_vnode(vnode, &sub_range, PrefetchOptions::default())
            .await?;
        pin_mut!(iter);
        // Current group: one partition's key and its buffered rows (in PK order, i.e. ORDER BY
        // order), seeded into `last_emitted` when the key changes or the vnode's scan ends. The
        // loop runs one extra iteration with no next row, so the end-of-scan flush shares the
        // partition-boundary flush below.
        let mut cur: Option<(OwnedRow, Vec<BufferedRow>)> = None;
        loop {
            // Stored row layout: `[ seq, <input cols..> ]` (same parse as the watermark/barrier
            // scans); the partition key is projected from the input columns, exactly as at ingest.
            let next = match iter.next().await.transpose()? {
                Some(row) => {
                    let row = row.into_owned_row();
                    let seq = Seq(row.datum_at(0).expect("seq not null").into_int64());
                    let input_row = OwnedRow::new(
                        (1..1 + input_arity)
                            .map(|i| row.datum_at(i).to_owned_datum())
                            .collect(),
                    );
                    let partition_key =
                        (&input_row).project(partition_key_indices).into_owned_row();
                    let order_key = input_row.datum_at(time_col).to_owned_datum();
                    let brow = BufferedRow {
                        seq,
                        order_key,
                        row: input_row,
                    };
                    Some((partition_key, brow))
                }
                None => None,
            };
            match (&mut cur, next) {
                (Some((k, rows)), Some((pk, brow))) if *k == pk => rows.push(brow),
                (slot, next) => {
                    // Partition boundary (or end of the vnode's scan): seed the finished group,
                    // then start the new one (or stop).
                    if let Some((k, rows)) = slot.take() {
                        let inc = matchers
                            .entry(k.clone())
                            .or_insert_with(|| IncrementalMatcher::new(nfa, skip.clone()));
                        let new_emitted =
                            compute_partition_emitted(inc, &k, &rows, defines, within, measures)
                                .await?;
                        // A partition with no provisional matches diffs identically against an
                        // absent base entry and an empty one, so skip the empty entry and keep the
                        // base populated only for partitions that actually emitted something.
                        if !new_emitted.is_empty() {
                            last_emitted.insert(k, new_emitted);
                        }
                    }
                    match next {
                        Some((pk, brow)) => *slot = Some((pk, vec![brow])),
                        None => break,
                    }
                }
            }
        }
    }
    Ok(())
}

pub struct MatchRecognizeExecutorArgs<S: StateStore> {
    pub ctx: ActorContextRef,
    pub input: Executor,
    /// Output schema: the `PARTITION BY` columns followed by the `MEASURES` columns.
    pub schema: Schema,
    pub chunk_size: usize,
    pub partition_key_indices: Vec<usize>,
    pub order_key_indices: Vec<usize>,
    pub measures: Vec<CompiledMeasure>,
    pub defines: Vec<CompiledDefine>,
    /// `WITHIN` span check over `[last_order_key, first_order_key]`; rejects matches that exceed it.
    pub within: Option<NonStrictExpression>,
    /// `WITHIN` deadline `first_order_key + interval` over a synthetic `[first_order_key]` row; the
    /// watermark at which a partial starting at that row expires. Used to wake idle partitions to
    /// evict timed-out partials. `None` when there is no `WITHIN`.
    pub within_deadline: Option<NonStrictExpression>,
    pub nfa: Nfa,
    pub skip: SkipMode,
    /// Where the actor's compute-error reports go. The compiled `DEFINE`/`MEASURES`/`WITHIN`
    /// expressions already report evaluation errors through it; the executor itself uses it for
    /// `AFTER MATCH SKIP` degradations (see [`report_skip_degradation_once`]).
    pub eval_error_report: ActorEvalErrorReport,
    /// Number of input columns; the buffered raw input row stored per row in the state table.
    pub input_arity: usize,
    pub state_table: StateTable<S>,
    /// Wakeup frontier: `pk (partition...) -> next_wakeup_order_key`. Point-looked-up by partition.
    pub frontier_meta_table: StateTable<S>,
    /// Wakeup frontier: `pk (next_wakeup_order_key, partition...)`, distributed by partition.
    pub frontier_index_table: StateTable<S>,
    /// Emit-On-Update mode (the plain form, without `EMIT ON WINDOW CLOSE`): the planner has
    /// already committed to this node emitting a retract stream, and the executor delivers it —
    /// a provisional-match changelog diffed at each barrier over the whole buffer, with the
    /// watermark doing finalization only. See the executor field of the same name for details.
    pub emit_on_update: bool,
}

pub struct MatchRecognizeExecutor<S: StateStore> {
    ctx: ActorContextRef,
    input: Executor,
    schema: Schema,
    chunk_size: usize,
    partition_key_indices: Vec<usize>,
    /// Input column index of the leading ORDER BY column (the watermark column). The full ORDER BY
    /// is encoded in the state-table key, so the buffer scans back already ordered; the executor
    /// only needs the leading column here, to find the safe prefix against the watermark.
    time_col: usize,
    measures: Vec<CompiledMeasure>,
    /// Compiled `DEFINE` predicates keyed by their pattern variable.
    defines: HashMap<String, CompiledDefine>,
    within: Option<NonStrictExpression>,
    /// `WITHIN` deadline expr (see [`MatchRecognizeExecutorArgs`]); folded into `next_wakeup` so an
    /// idle partition is woken to evict a partial that has timed out.
    within_deadline: Option<NonStrictExpression>,
    nfa: Nfa,
    skip: SkipMode,
    /// Where `AFTER MATCH SKIP` degradations are reported (see [`MatchRecognizeExecutorArgs`]).
    eval_error_report: ActorEvalErrorReport,
    input_arity: usize,
    state_table: StateTable<S>,
    /// Wakeup frontier (see [`MatchRecognizeExecutorArgs`]). Maintained on insert so a watermark can
    /// visit only the partitions that need attention.
    frontier_meta_table: StateTable<S>,
    frontier_index_table: StateTable<S>,
    /// Emit-On-Update mode (see [`MatchRecognizeExecutorArgs`]). When set, matches are emitted as a
    /// retract stream: at each barrier the provisional match set of every touched partition is
    /// diffed against what was last emitted and the resulting `Delete`/`Insert` ops are yielded (see
    /// the `Message::Barrier` arm). The watermark path then only evicts and does not emit. When
    /// unset (EMIT ON WINDOW CLOSE), the watermark path emits completed matches as an append stream.
    emit_on_update: bool,
}

/// A buffered input row, materialized from the state table while processing one partition.
struct BufferedRow {
    /// Per-actor monotonic id; the state-table key tiebreaker (keeps rows with equal ORDER BY keys
    /// distinct and stably ordered). The raw `i64` is read from / written to the state table at the
    /// storage boundary and unwrapped (`.0`) only where the `_match_id` output datum is built.
    seq: Seq,
    /// Leading ORDER BY value (a copy of `row[time_col]`), compared against the watermark to find
    /// the safe prefix. The buffer arrives pre-sorted by the full ORDER BY key (state-table PK).
    order_key: Datum,
    /// The raw input row, read by DEFINE and MEASURES navigation slots at match time.
    row: OwnedRow,
}

impl<S: StateStore> MatchRecognizeExecutor<S> {
    pub fn new(args: MatchRecognizeExecutorArgs<S>) -> Self {
        let time_col = args.order_key_indices[0];
        let defines = args
            .defines
            .into_iter()
            .map(|d| (d.symbol.clone(), d))
            .collect();
        Self {
            ctx: args.ctx,
            input: args.input,
            schema: args.schema,
            chunk_size: args.chunk_size,
            partition_key_indices: args.partition_key_indices,
            time_col,
            measures: args.measures,
            defines,
            within: args.within,
            within_deadline: args.within_deadline,
            nfa: args.nfa,
            skip: args.skip,
            eval_error_report: args.eval_error_report,
            input_arity: args.input_arity,
            state_table: args.state_table,
            frontier_meta_table: args.frontier_meta_table,
            frontier_index_table: args.frontier_index_table,
            emit_on_update: args.emit_on_update,
        }
    }

    /// Memoized WITHIN-deadline lookup for one partition visit. `deadline(rows[i]) = order_key +
    /// interval` is consulted by the boundary-emit guard, the eviction scan, and the wakeup
    /// recompute; the expression is evaluated at most once per row (`memo[i]`) instead of once per
    /// consulting site.
    async fn deadline_at(
        deadline_expr: &NonStrictExpression,
        rows: &[BufferedRow],
        memo: &mut [Option<Datum>],
        i: usize,
    ) -> Datum {
        if memo[i].is_none() {
            let synthetic = OwnedRow::new(vec![rows[i].order_key.clone()]);
            memo[i] = Some(deadline_expr.eval_row_infallible(&synthetic).await);
        }
        memo[i].as_ref().unwrap().clone()
    }

    /// Fold a wakeup key into a running minimum (`None` = nothing folded yet).
    fn fold_min(acc: &mut Datum, k: &ScalarImpl) {
        let lower = match acc {
            Some(m) => k.default_cmp(m).is_lt(),
            None => true,
        };
        if lower {
            *acc = Some(k.clone());
        }
    }

    /// Maintain the wakeup frontier for one partition on insert. If the partition has no frontier
    /// entry, or the chunk's earliest new order key precedes its recorded wakeup, (re)point the
    /// frontier at `new_min`. On insert `next_wakeup` only ever moves *earlier*; a watermark pass
    /// recomputes it precisely (including any WITHIN-driven expiry) after processing the partition.
    ///
    /// `frontier_meta`: `pk (partition...) -> next_wakeup` — updated in place (pk unchanged).
    /// `frontier_index`: `pk (next_wakeup, partition...)` — `next_wakeup` is part of the key, so a
    /// change is a delete of the old entry plus an insert of the new one.
    async fn update_frontier_on_insert(
        meta: &mut StateTable<S>,
        index: &mut StateTable<S>,
        partition_key: &OwnedRow,
        new_min: Datum,
    ) -> StreamExecutorResult<()> {
        // This point-reads the meta table per touched partition. With `forbid_preload_all_rows` a
        // cold partition round-trips to the state store, so it adds a read to the ingest path. For
        // high-cardinality workloads a bounded in-memory frontier cache (mirroring
        // `append_only_dedup`'s `ManagedLruCache`) would absorb hot partitions; deferred until the
        // watermark path consumes the frontier and profiling justifies the added complexity.
        let p = partition_key.len();
        let old = meta.get_row(partition_key).await?;
        let old_wakeup: Option<Datum> = old.as_ref().map(|r| r.datum_at(p).to_owned_datum());
        let should_update = match &old_wakeup {
            None => true,
            Some(ow) => new_min
                .as_ref()
                .unwrap()
                .default_cmp(ow.as_ref().unwrap())
                .is_lt(),
        };
        if !should_update {
            return Ok(());
        }
        // Build the written rows by chaining *references* (the row params are `impl Row`), so neither
        // `partition_key` nor the old meta row is cloned for the meta/index writes — only the small
        // single-datum wakeup rows are allocated.
        let wakeup_row = OwnedRow::new(vec![new_min.clone()]);
        // meta: pk (partition...) -> next_wakeup. pk is unchanged, so update in place.
        match &old {
            Some(old_row) => meta.update(old_row, partition_key.chain(&wakeup_row)),
            None => meta.insert(partition_key.chain(&wakeup_row)),
        }
        // index: pk (next_wakeup, partition...). next_wakeup is part of the key, so a change is a
        // delete of the old entry plus an insert of the new one.
        if let Some(ow) = &old_wakeup {
            let old_wakeup_row = OwnedRow::new(vec![ow.clone()]);
            index.delete(old_wakeup_row.chain(partition_key));
        }
        let new_index_key = OwnedRow::new(vec![new_min]);
        index.insert(new_index_key.chain(partition_key));
        Ok(())
    }

    /// Remove a partition's wakeup-frontier entry from both tables. Used when a processed partition
    /// is empty or has no future row to wake for. `old_wakeup` is the partition's current frontier
    /// value (already in hand from the index scan), so neither table needs a point read — it doubles
    /// as the meta old-value, valid because meta and index are kept in lockstep (see the
    /// candidate-processing comment in the watermark arm).
    fn remove_frontier(
        meta: &mut StateTable<S>,
        index: &mut StateTable<S>,
        partition_key: &OwnedRow,
        old_wakeup: &Datum,
    ) {
        let ow = OwnedRow::new(vec![old_wakeup.clone()]);
        index.delete((&ow).chain(partition_key));
        meta.delete(partition_key.chain(&ow));
    }

    /// Re-point a partition's wakeup frontier from `old_wakeup` to `new_wakeup`. The index PK leads
    /// with the wakeup, so its entry is delete+insert; the meta PK is the partition, so it updates
    /// in place. Rows are chained by reference to avoid cloning the partition key. `old_wakeup` (from
    /// the index scan) doubles as the meta old-value; this is valid only while meta and index stay in
    /// lockstep (see the candidate-processing comment in the watermark arm).
    fn move_frontier(
        meta: &mut StateTable<S>,
        index: &mut StateTable<S>,
        partition_key: &OwnedRow,
        old_wakeup: &Datum,
        new_wakeup: Datum,
    ) {
        let ow = OwnedRow::new(vec![old_wakeup.clone()]);
        let nw = OwnedRow::new(vec![new_wakeup]);
        index.delete((&ow).chain(partition_key));
        index.insert((&nw).chain(partition_key));
        meta.update(partition_key.chain(&ow), partition_key.chain(&nw));
    }
}

impl<S: StateStore> Execute for MatchRecognizeExecutor<S> {
    fn execute(self: Box<Self>) -> BoxedMessageStream {
        self.execute_inner().boxed()
    }
}

impl<S: StateStore> MatchRecognizeExecutor<S> {
    #[try_stream(ok = Message, error = StreamExecutorError)]
    async fn execute_inner(self: Box<Self>) {
        let Self {
            ctx,
            input,
            schema,
            chunk_size,
            partition_key_indices,
            time_col,
            measures,
            defines,
            within,
            within_deadline,
            nfa,
            skip,
            eval_error_report,
            input_arity,
            mut state_table,
            mut frontier_meta_table,
            mut frontier_index_table,
            emit_on_update,
        } = *self;

        let mut input = input.execute();
        let barrier = expect_first_barrier(&mut input).await?;
        let first_epoch = barrier.epoch;
        yield Message::Barrier(barrier);
        state_table.init_epoch(first_epoch).await?;
        frontier_meta_table.init_epoch(first_epoch).await?;
        frontier_index_table.init_epoch(first_epoch).await?;

        // Generator for the buffer-table PK tiebreaker `seq`, assigned to every input row. A
        // snowflake-style id (timestamp + owned vnode + sequence) is unique across this actor's
        // lifetime and across rescaling, so distinct rows never collide on `(partition, seq)`.
        // The output `_match_id` is NOT minted here — it is the match's start-row `seq` (see the
        // emit path), making emitted output deterministic across recovery replay. Rebuilt on a
        // vnode-bitmap change below.
        let mut row_id_gen = RowIdGenerator::new(
            state_table.vnodes().iter_vnodes(),
            state_table.vnodes().len(),
        );

        // In-memory lower bound on the minimum `next_wakeup` across this actor's frontier entries.
        // `None` = unknown (startup, or after a vnode-bitmap change) — the next watermark must scan
        // and re-establish it; `Some(None)` = the frontier is known empty; `Some(Some(k))` = no
        // entry is earlier than `k`. Purely an optimization, never persisted: it lets a watermark
        // with nothing due skip the per-vnode index scans with a single comparison.
        let mut min_wakeup: Option<Datum> = None;

        // Per-partition incremental matchers, keyed by partition key. A pure in-memory derivation of
        // state-table content: created empty here (so it is empty after recovery — the state table is
        // authoritative and each partition's matcher rebuilds lazily from its scanned buffer) and
        // cleared on any vnode-bitmap change below (the rescale-desync precedent — a stale matcher for
        // a re-homed partition must never survive). Each holds only the live window's seqs + labels.
        let mut matchers: HashMap<OwnedRow, IncrementalMatcher> = HashMap::new();

        // Emit-on-update bookkeeping (unused under EMIT ON WINDOW CLOSE). `dirty_partitions`
        // accumulates the partition keys that received rows since the last barrier — the barrier arm
        // re-diffs exactly those. `last_emitted` is the diff base: per partition, the `(match,
        // output row)` set we last told downstream is live, so a Delete can re-emit the exact prior
        // row. Both are in-memory derivations. `last_emitted` is rebuilt silently from the buffer on
        // recovery (just below) and on a vnode-bitmap change (see `rebuild_last_emitted`) so neither
        // re-emits rows downstream already holds; `matchers` (and `dirty_partitions`) are cleared on
        // a vnode-bitmap change and rebuilt lazily.
        let mut dirty_partitions: HashSet<OwnedRow> = HashSet::new();
        let mut last_emitted: HashMap<OwnedRow, Vec<(SeqMatch, OwnedRow)>> = HashMap::new();

        // No in-memory partition buffer: the state table is the single source of truth. Inserts are
        // written through immediately; each watermark scans the buffered rows back in PK order
        // (partition, order key, seq), processing one partition at a time and holding only that
        // partition's rows resident — so memory is bounded by the largest single partition's live
        // buffer, not by a whole vnode or the number of distinct keys. The buffer itself needs no
        // in-memory rebuild on recovery/rescale (the state table is authoritative); the one derived
        // structure that cannot be reconstructed without re-emitting is the emit-on-update diff base,
        // rebuilt silently next.

        // Recovery: the state tables were just restored to the recovered epoch, but `last_emitted`
        // (the emit-on-update diff base) is in-memory and starts empty. If the first barrier diffed
        // each partition's provisional set against an empty base it would re-Insert every match —
        // duplicates, since those rows are already in the MV from before the crash. Instead rebuild
        // the base silently now, BEFORE processing any input: recompute each owned partition's
        // provisional set and seed `last_emitted`, emitting nothing. By determinism the recomputation
        // equals what downstream already holds at the recovered epoch, so the next barrier emits only
        // genuine new deltas (and a partition rebuilt here is not marked dirty). On a cold start the
        // tables are empty, so this is a no-op. EMIT ON WINDOW CLOSE has no diff base — skip it.
        if emit_on_update {
            rebuild_last_emitted(
                &state_table,
                &mut matchers,
                &mut last_emitted,
                &nfa,
                &skip,
                &defines,
                within.as_ref(),
                &measures,
                input_arity,
                time_col,
                &partition_key_indices,
            )
            .await?;
        }

        #[for_await]
        for msg in input {
            match msg? {
                Message::Chunk(chunk) => {
                    // Append-only input: write each row through to the state table. DEFINE and
                    // MEASURES are evaluated later, at watermark time, against the buffered rows.
                    let chunk = chunk.compact_vis();
                    // Earliest new order key per partition touched by this chunk, so the wakeup
                    // frontier is updated once per distinct partition (O(#partitions in chunk)) rather
                    // than once per row.
                    let mut chunk_min: HashMap<OwnedRow, Datum> = HashMap::new();
                    // Fold the min for a *run* of consecutive same-partition rows in `run` before
                    // touching `chunk_min`, so a partition-clustered chunk (the common shape) projects
                    // and owns one partition key per run instead of one per row — the map's owned-key
                    // `entry` probe is the reason the old code materialized a key for every row. The
                    // in-run test compares the projected key by reference (no allocation); a partition
                    // that recurs in a later run merges back into the same entry on flush, so an
                    // unclustered chunk still owns at most one key per row, never more. `run` holds the
                    // current `(partition_key, min_order_key)`.
                    let mut run: Option<(OwnedRow, Datum)> = None;
                    // Merge a finished run's min into `chunk_min`, keeping the smaller order key.
                    let flush = |chunk_min: &mut HashMap<OwnedRow, Datum>, key: OwnedRow, min: Datum| {
                        match chunk_min.entry(key) {
                            hash_map::Entry::Occupied(mut e) => {
                                if min.as_ref().unwrap().default_cmp(e.get().as_ref().unwrap()).is_lt()
                                {
                                    *e.get_mut() = min;
                                }
                            }
                            hash_map::Entry::Vacant(e) => {
                                e.insert(min);
                            }
                        }
                    };
                    for (op, row_ref) in chunk.rows() {
                        // The input is required to be append-only (enforced at planning time), so only
                        // Insert is expected. Fail loud on anything else rather than silently
                        // producing wrong matches if that contract is ever violated upstream.
                        if !matches!(op, Op::Insert) {
                            return Err(anyhow::anyhow!(
                                "MATCH_RECOGNIZE requires append-only input but received a {:?} record",
                                op
                            )
                            .into());
                        }
                        let order_key = row_ref.datum_at(time_col).to_owned_datum();
                        // A row with a NULL order key has no event time: it can never fall under the
                        // watermark, so it would never be finalized or evicted (it would linger
                        // forever) and cannot be meaningfully ordered against other rows. Drop it, as
                        // event-time processing does with NULL-rowtime rows.
                        if order_key.is_none() {
                            continue;
                        }
                        // Extend the current run when this row's partition is unchanged (a by-reference
                        // comparison over the projected partition columns — no owned row built);
                        // otherwise flush the run and open a new one, owning the key exactly here.
                        let projected = row_ref.project(&partition_key_indices);
                        match &mut run {
                            Some((key, min)) if key.iter().eq(projected.iter()) => {
                                if order_key.as_ref().unwrap().default_cmp(min.as_ref().unwrap()).is_lt()
                                {
                                    *min = order_key;
                                }
                            }
                            _ => {
                                if let Some((key, min)) = run.take() {
                                    flush(&mut chunk_min, key, min);
                                }
                                run = Some((projected.to_owned_row(), order_key));
                            }
                        }
                        // State-table row layout: `[ seq, <input cols..> ]` — the partition columns
                        // and order key are columns of the stored input row. Written through by
                        // reference: `insert` takes `impl Row`, so the seq datum is chained onto the
                        // borrowed chunk row with no intermediate owned materialization. The minted
                        // id passes through `Seq` and is unwrapped (`.0`) at the datum, mirroring the
                        // `Seq(..)` wrap at every read — the storage boundary is grep-auditable in
                        // both directions.
                        let seq = Seq(row_id_gen.next());
                        state_table
                            .insert(once(Some(ScalarImpl::Int64(seq.0))).chain(row_ref));
                    }
                    // Flush the final run into the map before the per-partition frontier update below.
                    if let Some((key, min)) = run.take() {
                        flush(&mut chunk_min, key, min);
                    }
                    // One frontier update per distinct partition. On insert `next_wakeup` only ever
                    // moves earlier (a new row can only make a partition need attention sooner); the
                    // watermark path (the `Message::Watermark` arm below) reads the frontier to decide
                    // which partitions to visit, then recomputes each visited partition's entry
                    // precisely. Insert and watermark both mutate `frontier_meta` and `frontier_index`
                    // in lockstep — the invariant the watermark path relies on when it reuses an index
                    // datum as the meta old-value.
                    for (partition_key, new_min) in chunk_min {
                        // Keep the in-memory lower bound a lower bound: an insert only ever moves a
                        // partition's wakeup earlier. (Unknown stays unknown until a scan.)
                        if let (Some(acc), Some(k)) = (&mut min_wakeup, &new_min) {
                            Self::fold_min(acc, k);
                        }
                        Self::update_frontier_on_insert(
                            &mut frontier_meta_table,
                            &mut frontier_index_table,
                            &partition_key,
                            new_min,
                        )
                        .await?;
                        // Under emit-on-update, remember this partition so the next barrier re-diffs
                        // its provisional set. `partition_key` is owned here and unused afterwards, so
                        // move it in (no clone); under EMIT ON WINDOW CLOSE it is simply dropped.
                        if emit_on_update {
                            dirty_partitions.insert(partition_key);
                        }
                    }
                }
                Message::Watermark(watermark) => {
                    // Only the leading ORDER BY column drives matching.
                    if watermark.col_idx != time_col {
                        continue;
                    }
                    let w = watermark.val;

                    // Idle fast path: if the in-memory lower bound on the frontier's minimum
                    // `next_wakeup` is known and past `w`, no partition can be due — skip the
                    // per-vnode index scans entirely. Unknown (`None`) falls through to a scan,
                    // which re-establishes the exact value below.
                    //
                    // Deliberately `> w` (and the candidate check below `<= w`), i.e. one step *less*
                    // strict than the finality predicates: a wakeup at exactly `w` is admitted, so the
                    // partition is visited and simply finds nothing newly final. Waking a partition
                    // early only costs a scan; the wakeup tests must never be stricter than the
                    // finality tests, or a partition that is genuinely due is skipped.
                    match &min_wakeup {
                        Some(None) => continue,
                        Some(Some(k)) if k.default_cmp(&w).is_gt() => continue,
                        _ => {}
                    }

                    // Frontier-driven: visit only the partitions whose `next_wakeup <= w`, instead
                    // of sweeping every live partition. Per owned vnode, scan the index (ordered by
                    // next_wakeup) and stop at the first entry past the watermark; then, for each
                    // candidate partition, read just that partition from the buffer table, match /
                    // emit / evict, and recompute its frontier entry. Work is therefore proportional
                    // to the partitions that actually need attention, not to the number of live
                    // partitions.
                    let mut builder = StreamChunkBuilder::new(chunk_size, schema.data_types());
                    // Recomputed exactly during the scan: the first past-`w` index entry per vnode
                    // plus every re-pointed candidate wakeup — every remaining frontier entry is one
                    // or the other.
                    let mut next_min: Datum = None;
                    // `AFTER MATCH SKIP` degradations already reported in this pass; see
                    // `report_skip_degradation_once` for the policy. Empty (and unallocated) unless a
                    // skip target actually fails to resolve.
                    let mut reported_degradations: Vec<SkipDegradation> = Vec::new();
                    let vnodes: Vec<_> = state_table.vnodes().iter_vnodes().collect();
                    for vnode in vnodes {
                        // 1. Collect candidate `(old_wakeup, partition)` from the index, then drop the
                        //    iterator — a state-table delete cannot interleave with an open iterator,
                        //    and we mutate the index below. The index is PK-ordered by
                        //    `(next_wakeup, partition)`, so candidates (`next_wakeup <= w`) sort first
                        //    and we stop at the first row past `w`.
                        let mut candidates: Vec<(Datum, OwnedRow)> = Vec::new();
                        {
                            let sub_range: (Bound<OwnedRow>, Bound<OwnedRow>) =
                                (Bound::Unbounded, Bound::Unbounded);
                            let iter = frontier_index_table
                                .iter_with_vnode(vnode, &sub_range, PrefetchOptions::default())
                                .await?;
                            pin_mut!(iter);
                            while let Some(item) = iter.next().await {
                                // index row = [next_wakeup, partition...]
                                let row = item?.into_owned_row();
                                let next_wakeup = row.datum_at(0).to_owned_datum();
                                // Stop at the first entry past the watermark. Sound because the index
                                // PK leads with `next_wakeup` ascending and `iter_with_vnode` yields
                                // rows in memcomparable PK order, which agrees with `default_cmp` on
                                // the order-key type (negatives sort before positives; a tie on
                                // `next_wakeup` falls through to the partition suffix, so every
                                // equal-wakeup candidate is seen before any greater one). `next_wakeup`
                                // is never NULL: NULL order keys are dropped at ingest, and a WITHIN
                                // deadline is only scheduled when it evaluates non-null. The first
                                // past-`w` entry also seeds the recomputed in-memory lower bound
                                // (every entry before it is a candidate being reprocessed below).
                                match &next_wakeup {
                                    Some(k) if k.default_cmp(&w).is_le() => {}
                                    Some(k) => {
                                        Self::fold_min(&mut next_min, k);
                                        break;
                                    }
                                    None => break,
                                }
                                let partition_key = OwnedRow::new(
                                    (1..row.len())
                                        .map(|i| row.datum_at(i).to_owned_datum())
                                        .collect(),
                                );
                                candidates.push((next_wakeup, partition_key));
                            }
                        }

                        // 2. Process each candidate partition. `old_wakeup` is this partition's
                        //    frontier value as stored in the *index*; `move_frontier`/`remove_frontier`
                        //    below reuse it as the *meta* old-value instead of point-reading meta. That
                        //    is sound only because meta and index are always mutated together (by
                        //    `update_frontier_on_insert`, `move_frontier`, `remove_frontier`), each
                        //    writing the same wakeup to both — so they never disagree on a partition's
                        //    `next_wakeup`. The consistent-old-value check does not backstop this: a
                        //    stale old-value on an untouched, already-committed row is accepted, not
                        //    rejected, so the lockstep must hold by construction.
                        for (old_wakeup, partition_key) in candidates {
                            // Read this partition's rows from the buffer table in PK order
                            // (partition, order key, seq) — already ORDER BY ordered, no sort — then
                            // drop the iterator so the evicting deletes below can run in place.
                            let mut rows: Vec<BufferedRow> = Vec::new();
                            {
                                let sub_range: (Bound<OwnedRow>, Bound<OwnedRow>) =
                                    (Bound::Unbounded, Bound::Unbounded);
                                let iter = state_table
                                    .iter_with_prefix(
                                        &partition_key,
                                        &sub_range,
                                        PrefetchOptions::default(),
                                    )
                                    .await?;
                                pin_mut!(iter);
                                while let Some(item) = iter.next().await {
                                    let row = item?.into_owned_row();
                                    let seq = Seq(row.datum_at(0).expect("seq not null").into_int64());
                                    let input_row = OwnedRow::new(
                                        (1..1 + input_arity)
                                            .map(|i| row.datum_at(i).to_owned_datum())
                                            .collect(),
                                    );
                                    let order_key = input_row.datum_at(time_col).to_owned_datum();
                                    rows.push(BufferedRow {
                                        seq,
                                        order_key,
                                        row: input_row,
                                    });
                                }
                            }

                            if rows.is_empty() {
                                // Stale frontier entry for an empty partition: drop it, and any
                                // matcher (its buffer is gone, so its cache must not linger).
                                Self::remove_frontier(
                                    &mut frontier_meta_table,
                                    &mut frontier_index_table,
                                    &partition_key,
                                    &old_wakeup,
                                );
                                matchers.remove(&partition_key);
                                continue;
                            }

                            // Emit-on-update: close the never-emitted-match hole before evicting.
                            // Within an epoch the message order is Chunk -> Watermark -> Barrier. A
                            // complete match whose rows arrived *this* epoch and become
                            // watermark-safe now is about to be evicted below — but no barrier has
                            // diffed this partition yet, so its `Insert` was never emitted, and the
                            // barrier that follows sees the rows already gone (nothing to diff).
                            // Downstream would permanently miss a real match. So run the same
                            // whole-buffer diff the barrier arm runs FIRST, guaranteeing every match
                            // finalization removes from the diffable set has been emitted before its
                            // rows leave. EMIT ON WINDOW CLOSE is immune (it emits at the watermark),
                            // hence the `emit_on_update` gate.
                            //
                            // Gated on the partition being dirty: only a partition that received
                            // rows since the last barrier can hold an un-emitted match (anything from
                            // an earlier epoch was emitted at that epoch's barrier, so it is already
                            // in `last_emitted` and the eviction prune below drops it without a
                            // retraction). For an already-emitted match this diff is a no-op anyway,
                            // so the gate only skips wasted work. `dirty_partitions` is drained at the
                            // barrier, so at watermark time it names exactly the partitions with
                            // un-emitted rows this epoch.
                            //
                            // This runs before the eviction's own matcher refresh (over the safe
                            // prefix, below), which rolls the matcher back to that prefix; the diff
                            // is deterministic, so re-diffing at the barrier (if the partition stays
                            // dirty) is a no-op — a pure reordering of the same ops into the
                            // pre-barrier part of the epoch, never different content.
                            if emit_on_update && dirty_partitions.contains(&partition_key) {
                                let inc = matchers
                                    .entry(partition_key.clone())
                                    .or_insert_with(|| IncrementalMatcher::new(&nfa, skip.clone()));
                                let ops = emit_partition_diff(
                                    inc,
                                    &mut last_emitted,
                                    &partition_key,
                                    &rows,
                                    &defines,
                                    within.as_ref(),
                                    &measures,
                                )
                                .await?;
                                for (op, out_row) in ops {
                                    if let Some(c) = builder.append_row(op, out_row) {
                                        yield Message::Chunk(c);
                                    }
                                }
                                // This partition's provisional changelog is now emitted and
                                // `last_emitted` advanced to its whole-buffer set. The eviction below
                                // only prunes finalized matches from that base WITHOUT a retract, so
                                // the base stays consistent with the post-eviction buffer. Within an
                                // epoch the message order is Chunk -> Watermark -> Barrier, so no chunk
                                // can re-dirty this partition before the barrier — meaning the
                                // barrier's re-read + re-diff of it would be a guaranteed no-op (same
                                // surviving buffer, same pruned base, byte-identical rows). Clear it so
                                // the barrier skips that wasted pass; a partition NOT woken this
                                // watermark stays dirty and is diffed at the barrier as usual.
                                dirty_partitions.remove(&partition_key);
                            }

                            // The prefix with leading `order_key < w` is final. Strictly below `w`:
                            // the watermark contract only promises that no future row has
                            // `order_key < w`, so a row *at* `w` may still arrive (`watermark_filter`
                            // admits `event_time >= watermark`) and could sort before an already
                            // buffered row at `w` under a multi-column ORDER BY.
                            let safe_len = rows
                                .iter()
                                .take_while(|r| {
                                    matches!(&r.order_key, Some(k) if k.default_cmp(&w).is_lt())
                                })
                                .count();
                            // WITHIN-deadline memo for the safe prefix (see `deadline_at`); only
                            // sized when a WITHIN bound exists.
                            let mut deadline_memo: Vec<Option<Datum>> = if within_deadline.is_some()
                            {
                                vec![None; safe_len]
                            } else {
                                Vec::new()
                            };
                            // Evaluate DEFINE predicates against the in-progress match; the matcher
                            // borrows `rows`.
                            let matcher = DefineMatcher {
                                rows: &rows,
                                safe_len,
                                defines: &defines,
                                within: within.as_ref(),
                            };
                            // Drive matching through this partition's incremental matcher instead of
                            // rescanning the whole safe prefix. Feed the newly-safe rows (resyncing on
                            // out-of-order arrival, and rolling back + rescanning after an
                            // emit-on-update whole-buffer feed) and read `provisional()`, which
                            // `refresh_matcher` guarantees equals `find_matches_dynamic(safe_len,
                            // &matcher, &skip)` — so every downstream emit/evict decision below is
                            // unchanged.
                            let inc = matchers
                                .entry(partition_key.clone())
                                .or_insert_with(|| IncrementalMatcher::new(&nfa, skip.clone()));
                            refresh_matcher(inc, &rows, safe_len, &matcher).await?;
                            // Map seq-anchored provisional matches back to buffer positions. A match
                            // spans contiguous positions, so `end = start + labels.len()`, exactly as
                            // `find_matches_dynamic` reports it.
                            let mut seq_to_pos: HashMap<Seq, usize> =
                                HashMap::with_capacity(safe_len);
                            for (p, r) in rows[..safe_len].iter().enumerate() {
                                seq_to_pos.insert(r.seq, p);
                            }
                            let found: Vec<LabeledMatch> = inc
                                .provisional()
                                .iter()
                                .map(|m| {
                                    let start = seq_to_pos[&m.start_seq];
                                    LabeledMatch {
                                        start,
                                        end: start + m.labels.len(),
                                        labels: m.labels.clone(),
                                    }
                                })
                                .collect();
                            let mut cursor = 0usize;
                            for m in found {
                                if m.start < cursor {
                                    continue;
                                }
                                // At the safe boundary we normally wait for a trailing safe row to
                                // confirm the greedy match is maximal (it might still extend). But
                                // under WITHIN, once the watermark is strictly past this match's
                                // deadline (its first row's order_key + interval), no future row can
                                // legally extend it — any extension would fall inside the
                                // now-fully-safe window — so it is final and must be emitted now,
                                // before the WITHIN eviction below drops its rows. `deadline < w`,
                                // not `<=`: at `deadline == w` a row at `order_key == w` may still
                                // arrive and still fall within the bound, so the match can grow.
                                // This test must stay in exact lockstep with the WITHIN eviction
                                // predicate below — a stricter emit than eviction deletes a match's
                                // rows while it is still being held, silently dropping it.
                                // Without WITHIN, keep waiting.
                                if m.end >= safe_len {
                                    let within_final = if let Some(dl) = &within_deadline {
                                        let deadline = Self::deadline_at(
                                            dl,
                                            &rows,
                                            &mut deadline_memo,
                                            m.start,
                                        )
                                        .await;
                                        matches!(&deadline, Some(d) if d.default_cmp(&w).is_lt())
                                    } else {
                                        false
                                    };
                                    // A boundary match whose accepting path is TERMINAL — no
                                    // continuation of the automaton could consume another row, for
                                    // any future data (`may_extend` is false) — is final now: the
                                    // wait above is for a proof of maximality that nothing could
                                    // ever supply. Holding it would starve an idle partition
                                    // forever: without WITHIN the frontier recompute below finds
                                    // neither a future row nor a deadline and drops the partition,
                                    // so the last complete match would never be emitted unless an
                                    // unrelated row happened to arrive.
                                    let terminal = !within_final
                                        && !nfa.may_extend(m.start, m.end, &matcher).await?;
                                    // `break`, not `continue`, when the match is held. `found` is
                                    // ordered by start; the WITHIN bound is a constant interval and
                                    // rows are PK-sorted by order key, so a match's deadline (its
                                    // first order key + interval) is monotone non-decreasing in its
                                    // start. Once one boundary match is not yet within-final, no
                                    // later one can be either, so stopping is correct. `continue`
                                    // would be actively wrong: it could emit a later, shorter
                                    // match, advance `cursor` past this held boundary match, and
                                    // then let the eviction below delete this match's start row —
                                    // losing it. With `break` the held match's rows are retained (a
                                    // complete match at the boundary is `reaches_boundary_alive`),
                                    // and any shorter overlapping match is re-found on a later
                                    // watermark, never lost. A terminal match, by contrast, is
                                    // emitted and the loop continues: consuming it cannot mask a
                                    // later match (the scan resumes at its skip position exactly as
                                    // for any emitted match).
                                    if !within_final && !terminal {
                                        break;
                                    }
                                }
                                // EMIT ON WINDOW CLOSE: emit the completed match now. Under
                                // emit-on-update the changelog is emitted at barriers via provisional
                                // diffs (see the `Message::Barrier` arm), so the watermark path only
                                // advances the eviction cursor here and never emits — otherwise a
                                // match would be delivered twice (once here, once in the diff). The
                                // break/continue/cursor logic runs in both modes, so eviction is
                                // unchanged.
                                if !emit_on_update {
                                    let out_row = build_match_row(
                                        &measures,
                                        &rows,
                                        &partition_key,
                                        m.start,
                                        &m.labels,
                                    )
                                    .await?;
                                    // Stream the match into the chunk builder, flushing a full chunk
                                    // the moment it fills — output memory stays bounded by one chunk.
                                    if let Some(c) = builder.append_row(Op::Insert, out_row) {
                                        yield Message::Chunk(c);
                                    }
                                }
                                // Where the scan resumes after this match. A variable-targeted skip
                                // whose target row does not exist in this match degrades to a weaker
                                // strategy instead of failing the actor; that is reported, not
                                // silent (see `report_skip_degradation_once`).
                                let (next_cursor, degradation) =
                                    skip.next_pos(m.start, m.end, &m.labels);
                                cursor = next_cursor;
                                if let Some(degradation) = degradation {
                                    report_skip_degradation_once(
                                        &eval_error_report,
                                        &skip,
                                        degradation,
                                        &mut reported_degradations,
                                    );
                                }
                            }

                            // Evict finalized rows that can no longer be part of any match. Retain
                            // from the earliest row that is still a live match start; drop everything
                            // before it. A start at position `p` is live only if BOTH hold:
                            //
                            //  * it is structurally alive at the safe boundary — a match from `p` can
                            //    still reach the boundary given the safe rows so far. (It is not enough
                            //    that `p` can merely *begin* the pattern: for `(a b)`, a buffer
                            //    `[a, x, x]` whose `a` is followed only by non-matching safe rows can
                            //    never complete, though the `a` can still begin it.)
                            //
                            //  * its WITHIN window is still open — `order_key + interval >= w`. Once
                            //    the watermark is strictly past that deadline, every row within the
                            //    bound is final, so if no match from `p` has completed by now none ever
                            //    can, and `p` is dead even though the NFA is structurally still
                            //    expecting more input. This is what bounds idle-partition state (see
                            //    the doc's state bound): reaches_boundary_alive alone would keep a lone
                            //    `[a]` forever. The window is still open at `deadline == w`, because a
                            //    row at `order_key == w` may still arrive and still fall inside the
                            //    bound — this predicate must stay in exact lockstep with the WITHIN
                            //    boundary-emit test above, or a match held there has its rows deleted
                            //    here and is lost.
                            //
                            // The partition iterator is already dropped, so delete in place.
                            let mut retain_from = safe_len;
                            for (p, _) in rows.iter().enumerate().take(safe_len).skip(cursor) {
                                if let Some(deadline_expr) = &within_deadline {
                                    let deadline = Self::deadline_at(
                                        deadline_expr,
                                        &rows,
                                        &mut deadline_memo,
                                        p,
                                    )
                                    .await;
                                    // Window closed (deadline < w): `p` is dead, skip it. A null
                                    // deadline (eval-error only) fails this test, so the position is
                                    // conservatively retained — degenerate, deliberate.
                                    if matches!(&deadline, Some(d) if d.default_cmp(&w).is_lt()) {
                                        continue;
                                    }
                                }
                                if nfa.reaches_boundary_alive(p, safe_len, &matcher).await? {
                                    retain_from = p;
                                    break;
                                }
                            }
                            // Delete by reference: the seq datum chained onto the borrowed buffered
                            // row (`delete` takes `impl Row`), so no owned copy is built per evictee.
                            for c in &rows[0..retain_from] {
                                state_table
                                    .delete(once(Some(ScalarImpl::Int64(c.seq.0))).chain(&c.row));
                            }

                            // Keep the matcher aligned with the post-eviction buffer. The evicted rows
                            // are `rows[0..retain_from]`; the surviving front becomes `rows[retain_from]`.
                            // The matcher owns the finalize-vs-rebuild decision: given the first
                            // surviving row's seq as the eviction boundary, `finalize_evicted_prefix`
                            // rebases in place when it soundly can (PAST LAST ROW, boundary fed and
                            // within the frozen prefix, no straddle) and otherwise returns
                            // `MustRebuild` — so no gate here can trip a debug assertion or underflow.
                            // The executor only handles the whole-buffer-drained case (`retain_from ==
                            // rows.len()`), where there is no surviving boundary row to rebase onto.
                            // Every drop path lets the next visit rebuild lazily from the scanned
                            // buffer, keeping `provisional()` equal to a from-scratch scan.
                            if retain_from > 0 {
                                let must_rebuild = if retain_from < rows.len() {
                                    let inc = matchers
                                        .get_mut(&partition_key)
                                        .expect("matcher inserted above");
                                    matches!(
                                        inc.finalize_evicted_prefix(rows[retain_from].seq),
                                        Finalized::MustRebuild
                                    )
                                } else {
                                    true
                                };
                                if must_rebuild {
                                    matchers.remove(&partition_key);
                                }
                            }

                            // Under emit-on-update, keep `last_emitted` consistent with eviction:
                            // every match whose start row was just evicted is finalized — a permanent
                            // result already delivered as an Insert (the emit-before-finalize block
                            // above guarantees it was emitted this epoch if it had not been already)
                            // — so drop it from the diff base *without* emitting a retract. Matches
                            // whose start survives stay in the base and are re-diffed at the next
                            // barrier. Keying on the surviving seqs covers both the
                            // `finalize_evicted_prefix` and matcher-drop paths above uniformly (a
                            // live/emitted match's start is never evicted, since
                            // `reaches_boundary_alive` keeps it).
                            //
                            // WITHIN-expiry semantics (why a retract is never correct here):
                            //  * A *partial* (incomplete) match dying at its WITHIN deadline was
                            //    never in the provisional set — only *complete* matches are — so it
                            //    was never emitted; its rows just evict, invisibly to downstream.
                            //    Nothing to retract, and it is not in `last_emitted` to prune.
                            //  * A *complete* provisional match that WITHIN-expiry finalizes (its
                            //    deadline passed, so it can extend no further — see the boundary-emit
                            //    guard above) is *final*: it was emitted (above), stays live
                            //    downstream, and is only dropped from the diffable base here — no
                            //    retraction. Finality cannot invalidate a match computed over the same
                            //    buffer (spec §5), so the watermark path never *creates* a retract; a
                            //    retract arises only through the normal barrier diff, when later input
                            //    supersedes a still-provisional match.
                            if emit_on_update
                                && retain_from > 0
                                && last_emitted.contains_key(&partition_key)
                            {
                                let surviving: HashSet<Seq> =
                                    rows[retain_from..].iter().map(|r| r.seq).collect();
                                let prev = last_emitted
                                    .get_mut(&partition_key)
                                    .expect("checked contains_key");
                                prev.retain(|(m, _)| surviving.contains(&m.start_seq));
                                let now_empty = prev.is_empty();
                                if now_empty {
                                    last_emitted.remove(&partition_key);
                                }
                            }

                            // Recompute the wakeup frontier as the earlier of two events:
                            //
                            //  (a) the next row to become safe — the earliest surviving row that is not
                            //      yet final, i.e. whose `order_key >= w`. This is the exact complement
                            //      of the safe prefix (`order_key < w`), so every surviving unprocessed
                            //      row is covered: a row sitting exactly at `w` still needs a later
                            //      watermark to become safe, and scheduling it at `w` is what guarantees
                            //      the partition is revisited for it. Survivors `rows[retain_from..]`
                            //      are PK-ordered, so it is the first such row.
                            //
                            //  (b) the earliest WITHIN expiry of a retained live partial — so an idle
                            //      partition (no future row) is still woken to evict a partial that
                            //      times out. The earliest retained safe partial is `rows[retain_from]`
                            //      (if it is `< w`); its deadline is `first_order_key + interval`,
                            //      evaluated from `within_deadline`. After that watermark the existing
                            //      eviction predicate (which already honours WITHIN) drops it.
                            //
                            // If neither exists, drop the frontier entry — a later insert re-schedules.
                            let row_wakeup: Option<Datum> = rows[retain_from..]
                                .iter()
                                .find(|r| {
                                    matches!(&r.order_key, Some(k) if k.default_cmp(&w).is_ge())
                                })
                                .map(|r| r.order_key.clone());
                            let within_wakeup: Option<Datum> = match &within_deadline {
                                Some(deadline_expr)
                                    if retain_from < rows.len()
                                        && matches!(
                                            &rows[retain_from].order_key,
                                            Some(k) if k.default_cmp(&w).is_lt()
                                        ) =>
                                {
                                    let dl = Self::deadline_at(
                                        deadline_expr,
                                        &rows,
                                        &mut deadline_memo,
                                        retain_from,
                                    )
                                    .await;
                                    // A null deadline (only on eval error) carries no schedule.
                                    dl.is_some().then_some(dl)
                                }
                                _ => None,
                            };
                            let new_wakeup: Option<Datum> = match (row_wakeup, within_wakeup) {
                                (Some(a), Some(b)) => Some(
                                    if a.as_ref().unwrap().default_cmp(b.as_ref().unwrap()).is_le()
                                    {
                                        a
                                    } else {
                                        b
                                    },
                                ),
                                (a, b) => a.or(b),
                            };
                            match new_wakeup {
                                Some(nw) => {
                                    // The re-pointed wakeup is >= w (a row or deadline sitting exactly
                                    // at `w` is not yet final, so `w` itself is a legitimate next
                                    // wakeup), so it belongs in the recomputed lower bound. `nw == w`
                                    // is harmless: watermarks advance strictly, so the next one picks
                                    // the entry up, and a repeated `w` merely re-runs an idempotent
                                    // pass. It can also make `nw == old_wakeup`, which `move_frontier`
                                    // handles — a delete plus an insert of the same key collapses to an
                                    // update in the mem table.
                                    if let Some(k) = &nw {
                                        Self::fold_min(&mut next_min, k);
                                    }
                                    Self::move_frontier(
                                        &mut frontier_meta_table,
                                        &mut frontier_index_table,
                                        &partition_key,
                                        &old_wakeup,
                                        nw,
                                    )
                                }
                                None => Self::remove_frontier(
                                    &mut frontier_meta_table,
                                    &mut frontier_index_table,
                                    &partition_key,
                                    &old_wakeup,
                                ),
                            }
                        }
                    }
                    // The scan visited every owned vnode, so `next_min` is now the exact minimum
                    // remaining `next_wakeup` — known until the next insert lowers it.
                    min_wakeup = Some(next_min);
                    if let Some(c) = builder.take() {
                        yield Message::Chunk(c);
                    }
                }
                Message::Barrier(barrier) => {
                    // Emit-on-update: before committing this epoch, emit the provisional changelog for
                    // every partition that received rows since the last barrier. For each dirty
                    // partition we re-scan its whole buffer (in emit-on-update the match window is the
                    // *entire* buffer — "as if input ended now" — not the watermark-safe prefix the
                    // watermark path matches over), refresh its matcher over that whole buffer,
                    // rebuild each provisional match's output row, and diff against what we last
                    // emitted; the diff's Delete/Insert ops are the retract stream. Reads see this
                    // epoch's uncommitted inserts/deletes (read-your-writes), so the buffer is exactly
                    // the current content; emitting here — before commit and before the barrier is
                    // yielded — keeps these chunks in the pre-barrier epoch. `dirty_partitions` is
                    // only populated in this mode, so the block is a no-op under EMIT ON WINDOW CLOSE.
                    if emit_on_update && !dirty_partitions.is_empty() {
                        let mut builder = StreamChunkBuilder::new(chunk_size, schema.data_types());
                        for partition_key in dirty_partitions.drain() {
                            // Read this partition's whole buffer in PK order (partition, order key,
                            // seq) — already ORDER BY ordered — then drop the iterator before the
                            // commit below mutates the table.
                            let mut rows: Vec<BufferedRow> = Vec::new();
                            {
                                let sub_range: (Bound<OwnedRow>, Bound<OwnedRow>) =
                                    (Bound::Unbounded, Bound::Unbounded);
                                let iter = state_table
                                    .iter_with_prefix(
                                        &partition_key,
                                        &sub_range,
                                        PrefetchOptions::default(),
                                    )
                                    .await?;
                                pin_mut!(iter);
                                while let Some(item) = iter.next().await {
                                    let row = item?.into_owned_row();
                                    let seq = Seq(row.datum_at(0).expect("seq not null").into_int64());
                                    let input_row = OwnedRow::new(
                                        (1..1 + input_arity)
                                            .map(|i| row.datum_at(i).to_owned_datum())
                                            .collect(),
                                    );
                                    let order_key = input_row.datum_at(time_col).to_owned_datum();
                                    rows.push(BufferedRow {
                                        seq,
                                        order_key,
                                        row: input_row,
                                    });
                                }
                            }

                            // Whole buffer drained (all rows evicted): every finalized match was
                            // already dropped from `last_emitted` by the eviction path (without a
                            // retract — a finalized match is a permanent result), so there is nothing
                            // left to diff. Forget the now-dead partition's cache entries.
                            if rows.is_empty() {
                                matchers.remove(&partition_key);
                                last_emitted.remove(&partition_key);
                                continue;
                            }

                            // Diff this partition's whole-buffer provisional set against the diff
                            // base and stream the ops; the helper advances the base to what we just
                            // emitted (same code the watermark arm's emit-before-finalize runs).
                            let inc = matchers
                                .entry(partition_key.clone())
                                .or_insert_with(|| IncrementalMatcher::new(&nfa, skip.clone()));
                            let ops = emit_partition_diff(
                                inc,
                                &mut last_emitted,
                                &partition_key,
                                &rows,
                                &defines,
                                within.as_ref(),
                                &measures,
                            )
                            .await?;
                            for (op, out_row) in ops {
                                if let Some(c) = builder.append_row(op, out_row) {
                                    yield Message::Chunk(c);
                                }
                            }
                        }
                        if let Some(c) = builder.take() {
                            yield Message::Chunk(c);
                        }
                    }

                    // Commit all three state tables at this epoch. The frontier tables are
                    // distributed by partition like the buffer table, so they re-shard together on a
                    // vnode-bitmap change.
                    let post_commit = state_table.commit(barrier.epoch).await?;
                    let meta_post_commit = frontier_meta_table.commit(barrier.epoch).await?;
                    let index_post_commit = frontier_index_table.commit(barrier.epoch).await?;
                    let update_vnode_bitmap = barrier.as_update_vnode_bitmap(ctx.id);
                    yield Message::Barrier(barrier);
                    meta_post_commit
                        .post_yield_barrier(update_vnode_bitmap.clone())
                        .await?;
                    index_post_commit
                        .post_yield_barrier(update_vnode_bitmap.clone())
                        .await?;
                    // On a vnode-bitmap change (rescaling) the set of partitions this actor owns
                    // shifts. There is no in-memory buffer to reload — the state table is
                    // authoritative and the next watermark scans whatever vnodes are now owned. Rebuild
                    // the id generator only when the owned set may have *grown* (`cache_may_stale`), so
                    // new ids fall in the now-owned range. This is deliberately narrower than
                    // `RowIdGenExecutor`, which rebuilds on any bitmap change: there the generated id
                    // *is* the row-id distribution key, so a stale vnode would misplace rows. Here `seq`
                    // is only a within-partition PK tiebreaker (and, via a match's start row, the
                    // output `_match_id`) — not a distribution key, so a lost vnode never misplaces
                    // state, and the snowflake timestamp keeps ids unique regardless. Rebuilding on
                    // growth alone suffices.
                    if let Some((_, cache_may_stale)) =
                        post_commit.post_yield_barrier(update_vnode_bitmap).await?
                    {
                        // The owned-vnode set changed: the in-memory frontier lower bound no longer
                        // describes it. Unknown forces the next watermark to scan and re-establish.
                        min_wakeup = None;
                        // Drop every cached matcher: a re-homed partition's matcher would be a stale
                        // derivation of a buffer this actor no longer owns (the rescale-desync
                        // precedent). Each surviving partition rebuilds lazily from its scanned buffer.
                        matchers.clear();
                        // Drop the emit-on-update diff base for the same reason, then rebuild it
                        // silently from the now-owned buffers. Clearing alone would make the next
                        // barrier re-Insert every re-homed partition's provisional matches, but
                        // downstream still holds those rows from the pre-rescale actor's emissions;
                        // by determinism the recomputation over the same committed buffer matches
                        // them, so seeding the base without emitting keeps downstream correct. The
                        // owned set is authoritative here: `post_yield_barrier` has applied the new
                        // bitmap to all three tables, so the rebuild scans exactly the partitions
                        // this actor now owns. This is emit-on-update only.
                        last_emitted.clear();
                        // Partitions rebuilt below must not be left dirty (nothing changed). The
                        // barrier emit above already drained `dirty_partitions`; clear defensively so
                        // a rebuilt partition is only re-diffed once a later chunk marks it dirty.
                        dirty_partitions.clear();
                        if emit_on_update {
                            rebuild_last_emitted(
                                &state_table,
                                &mut matchers,
                                &mut last_emitted,
                                &nfa,
                                &skip,
                                &defines,
                                within.as_ref(),
                                &measures,
                                input_arity,
                                time_col,
                                &partition_key_indices,
                            )
                            .await?;
                        }
                        if cache_may_stale {
                            row_id_gen = RowIdGenerator::new(
                                state_table.vnodes().iter_vnodes(),
                                state_table.vnodes().len(),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use risingwave_expr::expr::LogReport;
    use risingwave_pb::expr::expr_node::{RexNode, Type as PbExprType};
    use risingwave_pb::expr::{ExprNode, FunctionCall as PbFunctionCall};
    use risingwave_pb::stream_plan::MatchRecognizeDefineSlot as PbDefineSlot;

    use super::*;
    use crate::executor::match_recognize::nfa::{LabeledMatch, Pattern, Quantifier};

    /// Slot kinds as the planner encodes them (see `MatchRecognizeDefineSlot.kind`).
    const KIND_SELF: u32 = 0;
    const KIND_PREV: u32 = 1;
    const KIND_RUNNING_FIRST: u32 = 3;
    const KIND_RUNNING_LAST: u32 = 4;

    /// One `int` column named `v` at index 0; `order_key` mirrors the physical position (unused
    /// without `WITHIN`, but kept consistent).
    fn buffered(vals: &[i32]) -> Vec<BufferedRow> {
        vals.iter()
            .enumerate()
            .map(|(i, v)| BufferedRow {
                seq: i as i64,
                order_key: Some(ScalarImpl::Int32(i as i32)),
                row: OwnedRow::new(vec![Some(ScalarImpl::Int32(*v))]),
            })
            .collect()
    }

    fn input_ref(idx: u32) -> ExprNode {
        ExprNode {
            function_type: PbExprType::Unspecified as i32,
            return_type: Some(DataType::Int32.to_protobuf()),
            rex_node: Some(RexNode::InputRef(idx)),
        }
    }

    /// `slots[0] = slots[1]`, i.e. the navigation slot compared against the candidate's own column.
    fn nav_eq_self_condition() -> ExprNode {
        ExprNode {
            function_type: PbExprType::Equal as i32,
            return_type: Some(DataType::Boolean.to_protobuf()),
            rex_node: Some(RexNode::FuncCall(PbFunctionCall {
                children: vec![input_ref(0), input_ref(1)],
            })),
        }
    }

    /// A navigation slot over column `v`.
    fn nav_slot(kind: u32, vars: &[&str], offset: u32) -> PbDefineSlot {
        PbDefineSlot {
            kind,
            vars: vars.iter().map(|v| (*v).to_owned()).collect(),
            col_idx: 0,
            offset,
        }
    }

    /// `DEFINE <symbol> AS <nav> = <symbol>.v`, compiled through the real proto lowering so the slot
    /// kinds are the planner's.
    fn nav_eq_self(symbol: &str, nav: PbDefineSlot) -> (String, CompiledDefine) {
        let pb = PbMatchRecognizeDefine {
            symbol: symbol.to_owned(),
            condition: Some(nav_eq_self_condition()),
            slots: vec![nav, nav_slot(KIND_SELF, &[], 0)],
        };
        (
            symbol.to_owned(),
            CompiledDefine::from_protobuf(&pb, LogReport).unwrap(),
        )
    }

    fn plus(var: &str) -> Pattern {
        Pattern::Quantified(
            Box::new(Pattern::Var(var.to_owned())),
            Quantifier::Plus,
            false,
        )
    }

    fn labels(s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    /// All matches over `vals`, with the whole buffer safe (no watermark boundary in play).
    async fn find_all(
        nfa: &Nfa,
        defines: &HashMap<String, CompiledDefine>,
        vals: &[i32],
    ) -> Vec<LabeledMatch> {
        let rows = buffered(vals);
        let matcher = DefineMatcher {
            rows: &rows,
            safe_len: rows.len(),
            defines,
            within: None,
        };
        nfa.find_matches_dynamic(rows.len(), &matcher, &SkipMode::PastLastRow)
            .await
            .unwrap()
    }

    /// `DEFINE a AS LAST(a.v) = a.v` is a tautology: SQL:2016 defines a pattern-variable-qualified
    /// column reference as `RUNNING LAST` of that column, and the binder already resolves the bare
    /// `a.v` inside `a`'s own DEFINE to the candidate row. So the running navigation must see the
    /// candidate too — including on the match's first row, where no earlier `a` exists.
    #[tokio::test]
    async fn define_running_last_of_self_sees_candidate() {
        let defines = HashMap::from([nav_eq_self("a", nav_slot(KIND_RUNNING_LAST, &["a"], 0))]);
        assert_eq!(
            find_all(&Nfa::compile(&plus("a")), &defines, &[1, 2, 3]).await,
            vec![LabeledMatch {
                start: 0,
                end: 3,
                labels: labels("aaa"),
            }]
        );
    }

    /// The eviction walker shares the finder's satisfies-source, so it must reach the same verdict:
    /// a lone row that satisfies `a AS LAST(a.v) = a.v` is a live partial match of `(a b)` and must
    /// be retained. Were the two to disagree, eviction would delete rows the matcher still needs.
    #[tokio::test]
    async fn define_running_last_of_self_keeps_start_alive() {
        let rows = buffered(&[1]);
        let defines = HashMap::from([nav_eq_self("a", nav_slot(KIND_RUNNING_LAST, &["a"], 0))]);
        let matcher = DefineMatcher {
            rows: &rows,
            safe_len: rows.len(),
            defines: &defines,
            within: None,
        };
        let nfa = Nfa::compile(&Pattern::Concat(vec![
            Pattern::Var("a".to_owned()),
            Pattern::Var("b".to_owned()),
        ]));
        assert!(
            nfa.reaches_boundary_alive(0, rows.len(), &matcher)
                .await
                .unwrap()
        );
    }

    /// `DEFINE a AS FIRST(a.v) = a.v`: the candidate is the *first* `a` only while no earlier `a` is
    /// bound, so this holds for the match's first row and then pins later rows to that value.
    #[tokio::test]
    async fn define_running_first_of_self_sees_candidate() {
        let defines = HashMap::from([nav_eq_self("a", nav_slot(KIND_RUNNING_FIRST, &["a"], 0))]);
        assert_eq!(
            find_all(&Nfa::compile(&plus("a")), &defines, &[5, 5, 7]).await,
            vec![
                // 5, 5 share the first value; 7 breaks it and starts its own match.
                LabeledMatch {
                    start: 0,
                    end: 2,
                    labels: labels("aa"),
                },
                LabeledMatch {
                    start: 2,
                    end: 3,
                    labels: labels("a"),
                },
            ]
        );
    }

    /// Running navigation that falls back on `labels` still indexes from the match's start. Here
    /// `x AS PREV(x.v) = x.v` cannot hold at position 0 (there is no previous row), so the match
    /// starts at 1, and `a+` binds two rows, so the third row's `FIRST(a.v)` resolves through
    /// `labels[1]` — pinning the `match_start + k` arithmetic where neither term is 0.
    #[tokio::test]
    async fn define_running_first_indexes_from_match_start() {
        let defines = HashMap::from([
            nav_eq_self("x", nav_slot(KIND_PREV, &[], 1)),
            nav_eq_self("a", nav_slot(KIND_RUNNING_FIRST, &["a"], 0)),
        ]);
        let nfa = Nfa::compile(&Pattern::Concat(vec![
            Pattern::Var("x".to_owned()),
            plus("a"),
        ]));
        assert_eq!(
            // `x` = the second 9 (its physical predecessor is the first 9); the run of 7s is `a+`,
            // whose `FIRST` is `rows[2]`, so the trailing 5 ends the match.
            find_all(&nfa, &defines, &[9, 9, 7, 7, 5]).await,
            vec![LabeledMatch {
                start: 1,
                end: 4,
                labels: labels("xaa"),
            }]
        );
    }

    /// `SUBSET u = (a, b)` + `DEFINE a AS LAST(u.v) = a.v`: the candidate is tentatively an `a`, and
    /// `a ∈ u`, so it counts as the running last of `u`. Both spellings lower to this same slot —
    /// `LAST(u.v)` via the navigation path and the bare `u.v` via the input-ref rewriter (whose
    /// self-reference exemption is name-exact and therefore misses the subset).
    #[tokio::test]
    async fn define_running_last_of_subset_containing_self_sees_candidate() {
        // The slot's `vars` is `members_of(u)`, which preserves the SUBSET's declaration order, so
        // both orders must behave identically: membership is a set test, not a look at `vars[0]`.
        for members in [["a", "b"], ["b", "a"]] {
            let defines =
                HashMap::from([nav_eq_self("a", nav_slot(KIND_RUNNING_LAST, &members, 0))]);
            assert_eq!(
                find_all(&Nfa::compile(&plus("a")), &defines, &[1, 2, 3]).await,
                vec![LabeledMatch {
                    start: 0,
                    end: 3,
                    labels: labels("aaa"),
                }],
                "SUBSET u = ({}, {})",
                members[0],
                members[1]
            );
        }
    }

    /// Navigation over a variable set that does *not* contain the candidate's own variable keeps
    /// resolving to the earlier row: `DEFINE b AS LAST(a.v) = b.v` compares against the `a`, never
    /// against the candidate `b`. This is the shape every existing DEFINE test uses.
    #[tokio::test]
    async fn define_running_last_of_other_var_excludes_candidate() {
        let defines = HashMap::from([nav_eq_self("b", nav_slot(KIND_RUNNING_LAST, &["a"], 0))]);
        let nfa = Nfa::compile(&Pattern::Concat(vec![
            Pattern::Var("a".to_owned()),
            Pattern::Var("b".to_owned()),
        ]));
        // Equal values: the `b` row equals the running `a`, so `(a b)` matches.
        assert_eq!(
            find_all(&nfa, &defines, &[7, 7]).await,
            vec![LabeledMatch {
                start: 0,
                end: 2,
                labels: labels("ab"),
            }]
        );
        // Different values: had the candidate been treated as the running last of `a`, this would
        // become a tautology and match.
        assert_eq!(find_all(&nfa, &defines, &[7, 8]).await, vec![]);
    }

    /// Collects what the executor reports, so the `AFTER MATCH SKIP` diagnostic can be asserted
    /// without an actor (in production the report goes to `ActorEvalErrorReport`).
    #[derive(Clone, Default)]
    struct CollectReport(Arc<Mutex<Vec<String>>>);

    impl EvalErrorReport for CollectReport {
        fn report(&self, error: ExprError) {
            self.0.lock().unwrap().push(error.to_string());
        }
    }

    impl CollectReport {
        fn messages(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    /// The reported error must be actionable on its own — it is what lands in the `error=` field of
    /// the `stream_expr_error` log line. Pinned verbatim: it names the skip mode (once), the target
    /// variable, and the strategy the resume position degraded to.
    #[test]
    fn skip_degradation_report_names_clause_and_fallback() {
        let report = CollectReport::default();
        let mut reported = Vec::new();
        report_skip_degradation_once(
            &report,
            &SkipMode::ToLast("c".to_owned()),
            SkipDegradation::TargetAbsent,
            &mut reported,
        );
        report_skip_degradation_once(
            &report,
            &SkipMode::ToFirst("a".to_owned()),
            SkipDegradation::TargetAtMatchStart,
            &mut reported,
        );
        assert_eq!(
            report.messages(),
            vec![
                "Invalid parameter AFTER MATCH SKIP TO LAST: target variable `c` is bound to no row \
                 of the match, so there is no row to resume at; the scan resumed past the match's \
                 last row instead (degraded to SKIP PAST LAST ROW)",
                "Invalid parameter AFTER MATCH SKIP TO FIRST: target variable `a` resolves to the \
                 match's own first row, so resuming there would re-find the same match forever; the \
                 scan resumed at the row after the match's first row instead (degraded to SKIP TO \
                 NEXT ROW)",
            ]
        );
    }

    /// Volume policy: the degradation repeats without bound (on every match, when no match can bind
    /// the target), and the message has no row, match or partition identity, so a per-match report
    /// would be byte-identical chatter. One report per kind per watermark pass.
    #[test]
    fn skip_degradation_report_is_deduplicated_per_pass() {
        let report = CollectReport::default();
        let skip = SkipMode::ToLast("x".to_owned());
        let mut reported = Vec::new();
        for _ in 0..5 {
            report_skip_degradation_once(
                &report,
                &skip,
                SkipDegradation::TargetAbsent,
                &mut reported,
            );
        }
        assert_eq!(report.messages().len(), 1, "{:?}", report.messages());
        // A different degradation is a different diagnostic, so it is reported once too.
        report_skip_degradation_once(
            &report,
            &skip,
            SkipDegradation::TargetAtMatchStart,
            &mut reported,
        );
        assert_eq!(report.messages().len(), 2, "{:?}", report.messages());
        // The next pass starts with a fresh set, so a persisting condition keeps being visible.
        let mut next_pass = Vec::new();
        report_skip_degradation_once(
            &report,
            &skip,
            SkipDegradation::TargetAbsent,
            &mut next_pass,
        );
        assert_eq!(report.messages().len(), 3, "{:?}", report.messages());
    }
}
