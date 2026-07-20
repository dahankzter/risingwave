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

//! Incremental driver over the row-pattern [`Nfa`].
//!
//! The batch matcher [`Nfa::find_matches_dynamic`] rescans the whole buffer from position 0 on every
//! call. Under append-only input (rows arriving in `ORDER BY` order) most of that work is redundant:
//! `AFTER MATCH SKIP` makes the matches before a committed skip-resume point immutable, because no
//! row appended *after* them can change a match that already terminated *before* them. This wrapper
//! keeps that skip-resume point as a scan cursor and, on each [`IncrementalMatcher::advance`], reruns
//! `find_matches_dynamic` **only over the suffix that can still change** — never reimplementing the
//! NFA traversal (and so never bypassing its greedy/reluctant preference, fresh-visited-scope, or
//! `WITHIN` invariants).
//!
//! Matches are anchored by row *seq* rather than buffer *position* so they stay stable when earlier
//! rows are evicted (a later task); positions are an internal detail of the current buffer.
//!
//! v1 freezing rule (see [`IncrementalMatcher::advance`]): a match is frozen once its `end` lies
//! strictly before the buffer boundary; a match ending *at* the boundary is still open and is
//! re-attempted on the next advance, since an appended row may extend it.

use crate::executor::error::StreamExecutorResult;
use crate::executor::match_recognize::nfa::{CandidateMatcher, LabeledMatch, Nfa, SkipMode};

/// A match anchored by row seqs (stable across eviction), not buffer positions. `start_seq` is the
/// seq of the match's first row; `end_seq` is one past the seq of its last row (so `end_seq -
/// start_seq` equals the row count only while seqs are contiguous). `labels[i]` is the pattern
/// variable bound to the match's `i`-th row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqMatch {
    pub start_seq: i64,
    pub end_seq: i64,
    pub labels: Vec<String>,
}

/// Incremental wrapper around [`Nfa::find_matches_dynamic`] for append-only input.
///
/// Rows are fed in `ORDER BY` order via [`IncrementalMatcher::advance`]; their buffer position is
/// implied by feed order and mapped back to a stable seq through `seq_index`. Everything before
/// `next_pos` (the skip-resume point after the last frozen match) is immutable and never rescanned.
pub struct IncrementalMatcher {
    /// Owned clone of the compiled pattern. `Nfa` is `Clone` and small, so we hold it by value rather
    /// than borrow, keeping this struct free of a lifetime the executor would have to thread through.
    nfa: Nfa,
    /// `AFTER MATCH SKIP` strategy, shared with the batch path.
    skip: SkipMode,
    /// All matches over the rows fed so far. `matched[..frozen_count]` are frozen (immutable under
    /// future appends); the rest is the provisional tail recomputed on every advance.
    matched: Vec<SeqMatch>,
    /// Number of leading entries of `matched` that are frozen.
    frozen_count: usize,
    /// Buffer position where the next rescan begins: the skip-resume point after the last frozen
    /// match (0 while nothing is frozen). The suffix `[next_pos, n_rows)` is the only mutable region.
    next_pos: usize,
    /// `seq_index[pos]` is the seq of the row fed at buffer position `pos`. Its length is the number
    /// of rows fed so far (the batch `n_rows`).
    seq_index: Vec<i64>,
}

/// Adapts a [`CandidateMatcher`] so that a scan over the suffix `[offset, ..)` sees suffix-relative
/// positions `0, 1, ...` while the underlying matcher still resolves absolute buffer positions. This
/// lets us drive [`Nfa::find_matches_dynamic`] over just the mutable suffix using the *same* matcher
/// the batch path uses, without adding a start-offset parameter to the NFA.
struct OffsetMatcher<'a, M> {
    inner: &'a M,
    offset: usize,
}

impl<M: CandidateMatcher + Sync> CandidateMatcher for OffsetMatcher<'_, M> {
    fn matches(
        &self,
        var: &str,
        pos: usize,
        labels: &[String],
    ) -> impl std::future::Future<Output = StreamExecutorResult<bool>> + Send {
        self.inner.matches(var, pos + self.offset, labels)
    }
}

impl IncrementalMatcher {
    pub fn new(nfa: &Nfa, skip: SkipMode) -> Self {
        Self {
            nfa: nfa.clone(),
            skip,
            matched: Vec::new(),
            frozen_count: 0,
            next_pos: 0,
            seq_index: Vec::new(),
        }
    }

    /// Feed rows appended in `ORDER BY` order. `new_row_seqs` are the seqs of the newly appended rows;
    /// their buffer positions are the next positions after the rows fed so far. An empty call is a
    /// no-op.
    ///
    /// Rescans only the mutable suffix `[next_pos, n_rows)` via [`Nfa::find_matches_dynamic`] (through
    /// an [`OffsetMatcher`]), replacing the provisional tail of `matched`. It then freezes the leading
    /// suffix matches whose `end` is strictly before the new boundary `n_rows` and advances `next_pos`
    /// to the last frozen match's skip-resume position; a match ending at the boundary stays open and
    /// will be re-attempted next time, since an appended row may extend it.
    pub async fn advance(
        &mut self,
        new_row_seqs: &[i64],
        matcher: &(impl CandidateMatcher + Sync),
    ) -> StreamExecutorResult<()> {
        if new_row_seqs.is_empty() {
            return Ok(());
        }
        self.seq_index.extend_from_slice(new_row_seqs);
        let n_rows = self.seq_index.len();

        // Rescan the mutable suffix only. The offset matcher maps suffix-relative positions produced
        // by the scan back onto absolute buffer positions the real matcher understands.
        let offset = self.next_pos;
        let offset_matcher = OffsetMatcher {
            inner: matcher,
            offset,
        };
        let tail = self
            .nfa
            .find_matches_dynamic(n_rows - offset, &offset_matcher, &self.skip)
            .await?;

        // Lift suffix-relative spans back to absolute buffer positions.
        let tail_abs: Vec<LabeledMatch> = tail
            .into_iter()
            .map(|m| LabeledMatch {
                start: m.start + offset,
                end: m.end + offset,
                labels: m.labels,
            })
            .collect();

        // Freeze the leading run of matches that end strictly before the boundary; stop at the first
        // match ending at the boundary (still open — an appended row may extend it).
        let newly_frozen = tail_abs.iter().take_while(|m| m.end < n_rows).count();
        if newly_frozen > 0 {
            let last = &tail_abs[newly_frozen - 1];
            self.next_pos = self.skip.next_pos(last.start, last.end, &last.labels);
        }

        // Drop the previous provisional tail and reattach the freshly scanned suffix.
        let new_tail: Vec<SeqMatch> = tail_abs.iter().map(|m| self.to_seq_match(m)).collect();
        self.matched.truncate(self.frozen_count);
        self.matched.extend(new_tail);
        self.frozen_count += newly_frozen;

        Ok(())
    }

    /// Current provisional matches over everything fed so far, as if input ended now.
    pub fn provisional(&self) -> &[SeqMatch] {
        &self.matched
    }

    /// Convert an absolute-position [`LabeledMatch`] into a seq-anchored [`SeqMatch`]. A match always
    /// spans at least one row (`end > start`), so `end - 1` is a valid position.
    fn to_seq_match(&self, m: &LabeledMatch) -> SeqMatch {
        SeqMatch {
            start_seq: self.seq_index[m.start],
            end_seq: self.seq_index[m.end - 1] + 1,
            labels: m.labels.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::executor::match_recognize::incremental::IncrementalMatcher;
    use crate::executor::match_recognize::nfa::{Nfa, Pattern, Quantifier, SetMatcher, SkipMode};

    /// Oracle: feeding rows incrementally (in any split) must equal one batch
    /// `find_matches_dynamic` over the same rows.
    async fn assert_equiv(
        nfa: &Nfa,
        skip: SkipMode,
        rows: &[BTreeSet<String>],
        split_at: &[usize],
    ) {
        let matcher = SetMatcher::new(rows.to_vec());
        let batch = nfa
            .find_matches_dynamic(rows.len(), &matcher, &skip)
            .await
            .unwrap();
        let mut inc = IncrementalMatcher::new(nfa, skip.clone());
        let mut fed = 0usize;
        for &cut in split_at.iter().chain(std::iter::once(&rows.len())) {
            let seqs: Vec<i64> = (fed..cut).map(|i| i as i64).collect();
            inc.advance(&seqs, &matcher).await.unwrap();
            fed = cut;
        }
        let inc_matches: Vec<(usize, usize)> = inc
            .provisional()
            .iter()
            .map(|m| (m.start_seq as usize, m.end_seq as usize))
            .collect();
        let batch_matches: Vec<(usize, usize)> = batch.iter().map(|m| (m.start, m.end)).collect();
        assert_eq!(inc_matches, batch_matches);
    }

    fn sets(labels: &[&str]) -> BTreeSet<String> {
        labels.iter().map(|s| s.to_string()).collect()
    }

    /// One row per non-whitespace char, each satisfying the single variable named by that char.
    fn from_str(s: &str) -> Vec<BTreeSet<String>> {
        s.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| BTreeSet::from([c.to_string()]))
            .collect()
    }

    /// `n` rows that each satisfy both `a` and `b`, so quantifier preference (not the predicate)
    /// decides the split — mirrors `nfa`'s own `ab_rows` helper.
    fn ab_rows(n: usize) -> Vec<BTreeSet<String>> {
        vec![BTreeSet::from(["a".to_string(), "b".to_string()]); n]
    }

    fn quant(inner: Pattern, q: Quantifier, reluctant: bool) -> Pattern {
        Pattern::Quantified(Box::new(inner), q, reluctant)
    }

    #[tokio::test]
    async fn incremental_equals_batch_in_order() {
        // pattern (a b+) — greedy trailing quantifier exercises the "still-open trailing match must
        // re-scan" rule.
        let pat = Pattern::Concat(vec![
            Pattern::Var("a".into()),
            Pattern::Quantified(Box::new(Pattern::Var("b".into())), Quantifier::Plus, false),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = vec![
            sets(&["a"]),
            sets(&["b"]),
            sets(&["b"]),
            sets(&["a"]),
            sets(&["b"]),
        ];
        // every split point, including feeding one row at a time
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[1]).await;
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[1, 2, 3, 4]).await;
        assert_equiv(&nfa, SkipMode::ToNextRow, &rows, &[2]).await;
    }

    #[tokio::test]
    async fn alternation_incremental_equals_batch() {
        // (a | b) c — the standard alternation shape from nfa.rs's own tests. The trailing `c`
        // makes a match ending at the buffer boundary re-attempt until the `c` row arrives.
        let pat = Pattern::Concat(vec![
            Pattern::Alt(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]),
            Pattern::Var("c".into()),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("acbc");
        for split in [&[1][..], &[2][..], &[1, 2, 3][..], &[3][..]] {
            assert_equiv(&nfa, SkipMode::PastLastRow, &rows, split).await;
        }
    }

    #[tokio::test]
    async fn range_quantifier_incremental_equals_batch() {
        // a{2,3} — a bounded range that greedily takes up to three, with the optional third copy
        // ending at the boundary (open) until the next row disambiguates it.
        let pat = quant(
            Pattern::Var("a".into()),
            Quantifier::Range {
                min: 2,
                max: Some(3),
            },
            false,
        );
        let nfa = Nfa::compile(&pat);
        let rows = from_str("aaxaa");
        for split in [&[1][..], &[2][..], &[1, 2, 3, 4][..], &[3][..]] {
            assert_equiv(&nfa, SkipMode::PastLastRow, &rows, split).await;
        }
    }

    #[tokio::test]
    async fn reluctant_quantifier_incremental_equals_batch() {
        // a+? b over rows that each satisfy both a and b: the reluctant `a+?` takes the fewest `a`,
        // so each match is exactly "ab" and the split between them is preference-, not predicate-,
        // driven.
        let pat = Pattern::Concat(vec![
            quant(Pattern::Var("a".into()), Quantifier::Plus, true),
            Pattern::Var("b".into()),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = ab_rows(4);
        for split in [&[1][..], &[2][..], &[1, 2, 3][..]] {
            assert_equiv(&nfa, SkipMode::PastLastRow, &rows, split).await;
        }
    }

    #[tokio::test]
    async fn to_next_row_overlap_incremental_equals_batch() {
        // a+ with SKIP TO NEXT ROW: overlapping matches (0,3),(1,3),(2,3) over "aaa" — the exact
        // overlap case from nfa.rs's `find_matches_skip_to_next_row_overlaps`. Every open match
        // ends at the boundary, so none freezes until a non-`a` row (or nothing) follows.
        let pat = quant(Pattern::Var("a".into()), Quantifier::Plus, false);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("aaa");
        for split in [&[1][..], &[2][..], &[1, 2][..]] {
            assert_equiv(&nfa, SkipMode::ToNextRow, &rows, split).await;
        }
        // and with a trailing non-`a` row that closes all three matches strictly before the boundary
        let rows = from_str("aaab");
        assert_equiv(&nfa, SkipMode::ToNextRow, &rows, &[1, 2, 3]).await;
    }

    #[tokio::test]
    async fn empty_advances_are_noops() {
        // A repeated split point feeds an empty `advance(&[])`; a leading `0` feeds one before any
        // real row. Both must be no-ops, so the final matches still equal the batch answer.
        let pat = Pattern::Concat(vec![
            Pattern::Var("a".into()),
            quant(Pattern::Var("b".into()), Quantifier::Plus, false),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("abbab");
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[2, 2, 3]).await; // empty in the middle
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[0, 1, 3]).await; // empty at the very start
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[1, 3, 3, 3]).await; // empties at the end
    }

    #[tokio::test]
    async fn single_row_feeds_over_twelve_rows() {
        // Feed 12-row inputs one row at a time (split = [1..=11]) for several patterns.
        let one_at_a_time: Vec<usize> = (1..12).collect();

        let ab_plus = Nfa::compile(&Pattern::Concat(vec![
            Pattern::Var("a".into()),
            quant(Pattern::Var("b".into()), Quantifier::Plus, false),
        ]));
        assert_equiv(
            &ab_plus,
            SkipMode::PastLastRow,
            &from_str("abbabbaabbab"),
            &one_at_a_time,
        )
        .await;

        let alt_c = Nfa::compile(&Pattern::Concat(vec![
            Pattern::Alt(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]),
            Pattern::Var("c".into()),
        ]));
        assert_equiv(
            &alt_c,
            SkipMode::PastLastRow,
            &from_str("acbcacxbcacb"),
            &one_at_a_time,
        )
        .await;
        assert_equiv(
            &alt_c,
            SkipMode::PastLastRow,
            &from_str("acbcbcacacbc"),
            &one_at_a_time,
        )
        .await;

        // overlapping matches under ToNextRow, single-fed
        let a_plus = Nfa::compile(&quant(Pattern::Var("a".into()), Quantifier::Plus, false));
        assert_equiv(
            &a_plus,
            SkipMode::ToNextRow,
            &from_str("aaxaaaxaaaax"),
            &one_at_a_time,
        )
        .await;
    }

    /// Known v1 limitation, kept as an executable spec for a later task (do not delete). The v1
    /// freezing rule — "a match is frozen once its `end` is strictly before the buffer boundary" —
    /// is unsound for an alternation whose *shorter* alternative accepts strictly before the
    /// boundary while a *longer*, higher-preference alternative is still alive *at* the boundary.
    ///
    /// Here `(a b c) | a` over `[a, b]` returns the fallback `a` match `(0,1)` (the `a b c` branch
    /// is alive but not yet accepting at the boundary). `end = 1 < 2`, so v1 freezes `(0,1)`. When
    /// `c` is then appended, the batch answer is `(0,3)` ("abc") — so the frozen incremental result
    /// diverges. The sound fix (deferred to a later task) gates freezing on
    /// [`crate::executor::match_recognize::nfa::Nfa::reaches_boundary_alive`], the same liveness
    /// predicate eviction already uses; it is intentionally *not* applied here to keep v1's freezing
    /// rule exactly as specified.
    #[ignore = "documents a known v1 freezing-rule soundness gap; addressed in a later task"]
    #[tokio::test]
    async fn known_limitation_alternation_alive_at_boundary_diverges() {
        let pat = Pattern::Alt(vec![
            Pattern::Concat(vec![
                Pattern::Var("a".into()),
                Pattern::Var("b".into()),
                Pattern::Var("c".into()),
            ]),
            Pattern::Var("a".into()),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("abc");
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[2]).await;
    }
}
