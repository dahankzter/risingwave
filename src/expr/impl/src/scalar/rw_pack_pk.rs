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

use risingwave_common::row::Row;
use risingwave_common::util::memcmp_encoding;
use risingwave_common::util::sort_util::OrderType;
use risingwave_expr::{ExprError, Result, function};

/// Format version byte prepended to every `_rw_pack_pk` output so that we can
/// evolve the encoding scheme in the future without colliding with old payloads.
const FORMAT_VERSION: u8 = 0x01;

/// Internal scalar function `_rw_pack_pk(args...) -> bytea`.
///
/// For each input row, the output is the concatenation of
/// `[FORMAT_VERSION, memcmp_encode(arg0), memcmp_encode(arg1), ...]`
/// where each argument is encoded with `OrderType::ascending()`.
///
/// The result is deterministic for a given input row. The Iceberg v3 sink planner
/// injects a call to this function over the upstream stream-key columns that are
/// not written out to iceberg (i.e. columns present in the upstream plan but not
/// in the sink output schema — `is_hidden=true` in `sink_desc.columns`, filtered
/// out by `build_sink_param`). The resulting bytes are stored as a synthetic
/// `_rw_extra_pk` column so the extended iceberg primary key uniquely identifies
/// every upstream row.
#[function("_rw_pack_pk(...) -> bytea")]
fn _rw_pack_pk(row: impl Row, writer: &mut impl std::io::Write) -> Result<()> {
    writer.write_all(&[FORMAT_VERSION]).unwrap();
    for datum in row.iter() {
        let encoded = memcmp_encoding::encode_value(datum, OrderType::ascending())
            .map_err(|e| ExprError::Internal(e.into()))?;
        writer.write_all(&encoded).unwrap();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use risingwave_common::row::OwnedRow;
    use risingwave_common::types::ScalarImpl;
    use risingwave_expr::expr::{Expression, build_from_pretty};

    #[tokio::test]
    async fn test_rw_pack_pk_deterministic_and_distinct() {
        let expr = build_from_pretty("(rw_pack_pk:bytea $0:int4 $1:varchar)");
        let r1 = OwnedRow::new(vec![
            Some(ScalarImpl::Int32(1)),
            Some(ScalarImpl::Utf8("a".into())),
        ]);
        let r2 = OwnedRow::new(vec![
            Some(ScalarImpl::Int32(1)),
            Some(ScalarImpl::Utf8("a".into())),
        ]);
        let r3 = OwnedRow::new(vec![
            Some(ScalarImpl::Int32(1)),
            Some(ScalarImpl::Utf8("b".into())),
        ]);
        let v1 = expr.eval_row(&r1).await.unwrap().unwrap();
        let v2 = expr.eval_row(&r2).await.unwrap().unwrap();
        let v3 = expr.eval_row(&r3).await.unwrap().unwrap();
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        if let ScalarImpl::Bytea(b) = v1 {
            assert_eq!(b[0], 0x01);
        } else {
            panic!("expected bytea");
        }
    }

    #[tokio::test]
    async fn test_rw_pack_pk_null_distinguished() {
        let expr = build_from_pretty("(rw_pack_pk:bytea $0:int4 $1:varchar)");
        let r_null_a = OwnedRow::new(vec![None, Some(ScalarImpl::Utf8("a".into()))]);
        let r_null_b = OwnedRow::new(vec![
            Some(ScalarImpl::Int32(0)),
            Some(ScalarImpl::Utf8("a".into())),
        ]);
        let r_null_both = OwnedRow::new(vec![None, None]);
        let v_a = expr.eval_row(&r_null_a).await.unwrap().unwrap();
        let v_b = expr.eval_row(&r_null_b).await.unwrap().unwrap();
        let v_both = expr.eval_row(&r_null_both).await.unwrap().unwrap();
        assert_ne!(
            v_a, v_b,
            "NULL must encode distinctly from any concrete value"
        );
        assert_ne!(v_a, v_both);
        assert_ne!(v_b, v_both);
    }

    #[tokio::test]
    async fn test_rw_pack_pk_cross_type_distinct() {
        // Same logical value 1 with different integer widths must encode differently
        // because their memcmp-encoded widths differ.
        let expr_int4 = build_from_pretty("(rw_pack_pk:bytea $0:int4)");
        let expr_int8 = build_from_pretty("(rw_pack_pk:bytea $0:int8)");
        let r4 = OwnedRow::new(vec![Some(ScalarImpl::Int32(1))]);
        let r8 = OwnedRow::new(vec![Some(ScalarImpl::Int64(1))]);
        let v4 = expr_int4.eval_row(&r4).await.unwrap().unwrap();
        let v8 = expr_int8.eval_row(&r8).await.unwrap().unwrap();
        assert_ne!(v4, v8, "different integer widths produce different bytes");
    }

    #[tokio::test]
    async fn test_rw_pack_pk_supports_various_types() {
        use std::str::FromStr;

        use risingwave_common::types::{Date, Decimal, Time, Timestamp};

        let expr = build_from_pretty(
            "(rw_pack_pk:bytea $0:decimal $1:timestamp $2:date $3:time $4:bytea)",
        );
        let dec = Decimal::from_str("3.14").unwrap();
        let ts = Timestamp::from_str("2026-05-28 12:00:00").unwrap();
        let d = Date::from_ymd_uncheck(2026, 5, 28);
        let t = Time::from_hms_uncheck(12, 0, 0);
        let b: Box<[u8]> = Box::new([1u8, 2, 3]);
        let row = OwnedRow::new(vec![
            Some(ScalarImpl::Decimal(dec)),
            Some(ScalarImpl::Timestamp(ts)),
            Some(ScalarImpl::Date(d)),
            Some(ScalarImpl::Time(t)),
            Some(ScalarImpl::Bytea(b)),
        ]);
        let v1 = expr.eval_row(&row).await.unwrap().unwrap();
        let v2 = expr.eval_row(&row).await.unwrap().unwrap();
        assert_eq!(v1, v2);
        if let ScalarImpl::Bytea(b) = v1 {
            assert_eq!(b[0], 0x01, "format version byte still present");
            assert!(
                b.len() > 1,
                "encoding should produce more than just the version byte"
            );
        } else {
            panic!("expected bytea");
        }
    }
}
