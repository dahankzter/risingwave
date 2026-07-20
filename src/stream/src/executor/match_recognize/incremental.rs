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
//! Freezing rule (see [`IncrementalMatcher::advance`]): a match — and the scan region behind it up
//! to its skip-resume position — freezes only once *every* position in that region is dead at the
//! current boundary per [`Nfa::reaches_boundary_alive`] (the same liveness predicate row eviction
//! uses). A dead position's scan outcome can never change under appended rows, because no path from
//! it can consume past the old boundary; so the whole region's scan behavior — matches found, gaps
//! skipped, and the resume point — is final. Any live position (a still-open trailing match, or a
//! gap where a longer, higher-preference alternative is still in flight) keeps the region
//! provisional and re-attempted on the next advance.
//!
//! A late (out-of-order) row that sorts *before* rows already fed is handled by
//! [`IncrementalMatcher::truncate_from_seq`]: it rolls state back to a scan-resume point at or before
//! the insertion, re-verifying the freezing gate against the truncation boundary (freezing is only
//! sound against the boundary it was checked at), after which the caller re-feeds the corrected
//! sorted suffix through `advance`.

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
    /// run of suffix matches whose entire scan region `[cursor, skip-resume)` is dead at the boundary
    /// per [`Nfa::reaches_boundary_alive`], advancing `next_pos` to the last frozen match's resume
    /// position. Checking the *whole region* — not just the match's start — matters: a gap position
    /// before the match can be alive (a longer, higher-preference alternative still in flight) and a
    /// future row could then produce a match there that consumes past this one, so nothing behind
    /// that gap may freeze. Liveness is checked with the raw `matcher` at absolute positions (only
    /// the finder needs the offset adapter, because it always scans from 0).
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

        // Freeze the leading run of matches whose scan region `[cursor, resume)` is entirely dead at
        // the boundary. A dead position's scan outcome is final — no path from it can consume past
        // `n_rows - 1`, so appended rows can never be reached from it and the greedy attempt there
        // returns the same result over any future buffer. Matches freeze strictly in order (there is
        // a single cursor), so stop at the first region containing a live position. A match ending
        // at the boundary is covered without a special case: its own accepting path reaches the
        // boundary, so its start is alive and the region check fails.
        let mut newly_frozen = 0usize;
        let mut cursor = self.next_pos;
        'freeze: for m in &tail_abs {
            let resume = self.skip.next_pos(m.start, m.end, &m.labels);
            for p in cursor..resume {
                if self.nfa.reaches_boundary_alive(p, n_rows, matcher).await? {
                    break 'freeze;
                }
            }
            cursor = resume;
            newly_frozen += 1;
        }
        self.next_pos = cursor;

        // Drop the previous provisional tail and reattach the freshly scanned suffix.
        let new_tail: Vec<SeqMatch> = tail_abs.iter().map(|m| self.to_seq_match(m)).collect();
        self.matched.truncate(self.frozen_count);
        self.matched.extend(new_tail);
        self.frozen_count += newly_frozen;

        Ok(())
    }

    /// Invalidate everything at and after the fed row identified by `seq`, so the caller can re-feed
    /// a corrected sorted suffix (an out-of-order row landing before rows already fed). `seq` is the
    /// stable identity of the first buffered row whose sorted position changes; the executor computes
    /// it (the first buffered order key `>=` the late row's), and here we only map it back to a fed
    /// position via `seq_index`.
    ///
    /// A seq never fed (e.g. an order key beyond everything buffered) is a no-op; truncating at the
    /// first fed row (position 0) is a full reset. After this call `provisional()` never returns a
    /// match overlapping the truncated region.
    ///
    /// Why this needs the `matcher` (and so mirrors [`IncrementalMatcher::advance`]'s freezing gate
    /// rather than a purely positional rule): a match froze against a *later* boundary, and freezing
    /// only requires every region position to be dead at *that* boundary — a position may still hold
    /// a path that stays alive *through* the rows now being truncated (a longer, higher-preference
    /// alternative that only died past the truncation point). Such a frozen match is not final once
    /// those rows change, even when its own span ends before the truncation point. So we recompute
    /// the surviving frozen prefix with the exact gate `advance` uses — region-wide
    /// [`Nfa::reaches_boundary_alive`] — but against the truncation boundary. Only a region entirely
    /// dead at that boundary is independent of the truncated/re-fed rows and may be kept; the rest
    /// (and every provisional match) is dropped and re-derived by the following `advance`, which
    /// rescans from the rewound `next_pos`.
    pub async fn truncate_from_seq(
        &mut self,
        seq: i64,
        matcher: &(impl CandidateMatcher + Sync),
    ) -> StreamExecutorResult<()> {
        // Seqs are stable row identities, not sort keys, so `seq_index` is not ordered by value; find
        // the exact entry. A missing seq means nothing buffered at/after it changed — leave state as
        // is.
        let Some(trunc_pos) = self.seq_index.iter().position(|&s| s == seq) else {
            return Ok(());
        };

        let mut kept = 0usize;
        let mut cursor = 0usize;
        'keep: for m in &self.matched[..self.frozen_count] {
            // Frozen matches are stored in scan order, so each start is at or after the cursor; search
            // forward from there to recover its buffer position (seqs are not positions).
            let start_pos = cursor
                + self.seq_index[cursor..]
                    .iter()
                    .position(|&s| s == m.start_seq)
                    .expect("frozen match start seq must still be fed at truncation");
            let end_pos = start_pos + m.labels.len();
            // A match reaching into (or across) the truncated region cannot survive: the re-fed rows
            // may change its greedy extent or its skip-resume point.
            if end_pos > trunc_pos {
                break;
            }
            // `resume <= end_pos <= trunc_pos`, so every checked position is in the retained region.
            let resume = self.skip.next_pos(start_pos, end_pos, &m.labels);
            for p in cursor..resume {
                if self.nfa.reaches_boundary_alive(p, trunc_pos, matcher).await? {
                    break 'keep;
                }
            }
            cursor = resume;
            kept += 1;
        }

        self.next_pos = cursor;
        self.frozen_count = kept;
        self.matched.truncate(kept);
        self.seq_index.truncate(trunc_pos);
        Ok(())
    }

    /// Finalize (semantically remove) every match lying wholly within the evicted prefix and return
    /// them in scan order. The caller is evicting those rows from its buffer, so the matches leave
    /// the diffable set and their fed-row bookkeeping is dropped here; after this call
    /// [`IncrementalMatcher::provisional`] returns only still-revisable matches.
    ///
    /// `seq` is the first row seq that is *not* evicted (an exclusive upper bound), mirroring
    /// [`IncrementalMatcher::truncate_from_seq`]'s position convention: everything at sorted
    /// positions before that row's position is gone. A match is finalized when its last row lies
    /// before that position; a match starting at or after it is kept. A seq that was never fed is a
    /// no-op (the caller passes the first *surviving* buffered row's seq, which is still fed).
    ///
    /// Two contract invariants are `debug_assert`ed — the executor guarantees both by construction
    /// (finalization happens only at watermark boundaries where the frozen prefix is provably final),
    /// so they document the contract rather than guard runtime input:
    /// - the boundary lies within the frozen prefix (`final_pos <= next_pos`): finalization must
    ///   never reach into the open, still-revisable trailing region;
    /// - no match straddles the boundary: each match either ends at/before it (finalized) or starts
    ///   at/after it (kept).
    ///
    /// Positions are then **rebased**: the evicted rows physically leave the front of the logical
    /// buffer, so `seq_index` drains its prefix and `next_pos`/`frozen_count` shift down. Only rows
    /// at positions `>= final_pos` survive, and rebasing is a uniform downward shift of those rows;
    /// paths *forward* from a surviving position consume only surviving (unchanged) rows, so the
    /// freezing-soundness argument (a frozen region is dead at its boundary) is preserved unchanged.
    /// Match spans are anchored by seq, so both the retained and the returned [`SeqMatch`]es keep
    /// their identities without adjustment.
    pub fn finalize_before_seq(&mut self, seq: i64) -> Vec<SeqMatch> {
        // Seqs are stable identities, not sort keys; find the exact entry. A missing seq means the
        // boundary row is not (or no longer) buffered here — nothing to finalize.
        let Some(final_pos) = self.seq_index.iter().position(|&s| s == seq) else {
            return Vec::new();
        };
        debug_assert!(
            final_pos <= self.next_pos,
            "finalization boundary at position {final_pos} reaches past the frozen prefix \
             (next_pos {})",
            self.next_pos
        );

        // Finalized matches are a leading run of the frozen prefix: only frozen matches can end
        // within `[0, next_pos)` (a provisional match starts at `>= next_pos`), and matches are
        // stored in scan order. Walk them, recovering each start position from `seq_index` (seqs are
        // identities, not positions), and stop at the first match that ends past the boundary.
        let mut finalized = 0usize;
        let mut cursor = 0usize;
        for m in &self.matched[..self.frozen_count] {
            let start_pos = cursor
                + self.seq_index[cursor..]
                    .iter()
                    .position(|&s| s == m.start_seq)
                    .expect("finalized match start seq must still be fed");
            let end_pos = start_pos + m.labels.len();
            if end_pos <= final_pos {
                finalized += 1;
                // Later matches start strictly after this one (`resume > start`), so search forward.
                cursor = start_pos + 1;
            } else {
                debug_assert!(
                    start_pos >= final_pos,
                    "match [{start_pos}, {end_pos}) straddles finalization boundary {final_pos}"
                );
                break;
            }
        }

        // Detach the finalized matches (scan order, seqs intact) and rebase the buffer.
        let removed: Vec<SeqMatch> = self.matched.drain(..finalized).collect();
        self.frozen_count -= finalized;
        self.next_pos -= final_pos;
        self.seq_index.drain(..final_pos);
        removed
    }

    /// Current provisional matches over everything fed so far, as if input ended now.
    pub fn provisional(&self) -> &[SeqMatch] {
        &self.matched
    }

    /// Seqs of the rows fed so far, in feed (buffer-position) order — i.e. `seq_index`. The executor
    /// reads this to align the matcher with the freshly-scanned state-table buffer each visit and to
    /// detect an out-of-order safe row (one whose sorted position precedes an already-fed row).
    pub fn fed_seqs(&self) -> &[i64] {
        &self.seq_index
    }

    /// Number of leading buffer positions that are frozen (immutable under future appends) — i.e.
    /// `next_pos`, the scan-resume point. [`IncrementalMatcher::finalize_before_seq`] may only evict a
    /// prefix that lies within this frozen region, so the executor compares its eviction boundary
    /// against this before finalizing (and drops-and-rebuilds the matcher when eviction reaches past
    /// it). Distinct from [`IncrementalMatcher::frozen`], which counts frozen *matches*, not positions.
    pub fn frozen_prefix_len(&self) -> usize {
        self.next_pos
    }

    /// Number of leading `provisional()` entries that are frozen (final under future appends).
    /// Test-only observability for asserting freezing behavior directly.
    #[cfg(test)]
    fn frozen(&self) -> usize {
        self.frozen_count
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

    use super::{IncrementalMatcher, SeqMatch};
    use crate::executor::error::StreamExecutorResult;
    use crate::executor::match_recognize::nfa::{
        CandidateMatcher, Nfa, Pattern, Quantifier, SetMatcher, SkipMode,
    };

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

    fn labels(ls: &[&str]) -> Vec<String> {
        ls.iter().map(|s| s.to_string()).collect()
    }

    /// Provisional matches as `(start_pos, end_pos, labels)`. In the truncation tests seqs are
    /// assigned equal to final sorted position, so a seq is directly its position and these triples
    /// line up with the batch oracle's position-anchored spans.
    fn provisional_triples(inc: &IncrementalMatcher) -> Vec<(usize, usize, Vec<String>)> {
        inc.provisional()
            .iter()
            .map(|m| (m.start_seq as usize, m.end_seq as usize, m.labels.clone()))
            .collect()
    }

    /// Batch oracle over `rows` as `(start, end, labels)` triples (labels included so a *steal* —
    /// a row rebinding to a different variable — is caught, not just span changes).
    async fn batch_triples(
        nfa: &Nfa,
        skip: &SkipMode,
        rows: &[BTreeSet<String>],
    ) -> Vec<(usize, usize, Vec<String>)> {
        let matcher = SetMatcher::new(rows.to_vec());
        nfa.find_matches_dynamic(rows.len(), &matcher, skip)
            .await
            .unwrap()
            .iter()
            .map(|m| (m.start, m.end, m.labels.clone()))
            .collect()
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

    /// The freezing gate must consult boundary liveness, not just "match ended before the boundary".
    /// `(a b c) | a` over `[a, b]` returns the fallback `a` match `(0,1)` while the longer,
    /// higher-preference `a b c` branch is still alive *at* the boundary (waiting for `c`). The
    /// naive `end < n_rows` rule would freeze `(0,1)`; the liveness gate sees position 0 alive and
    /// defers, so when `c` arrives the rescan finds the batch answer `(0,3)` ("abc").
    #[tokio::test]
    async fn alternation_alive_at_boundary_defers_freezing() {
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
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[1, 2]).await;
    }

    /// A *gap* position (no match there yet) can be the live one: `(a n n n) | n` over `[a, n, n]`
    /// finds only `n` matches, but position 0's `a n n n` branch is alive at the boundary — one more
    /// `n` turns the batch answer into the single match `(0,4)`. Freezing the early `n` matches
    /// (whose own starts are dead) would lose it, so the gate must check every position in the
    /// would-be-frozen region, not just match starts.
    #[tokio::test]
    async fn live_gap_position_defers_freezing() {
        let pat = Pattern::Alt(vec![
            Pattern::Concat(vec![
                Pattern::Var("a".into()),
                Pattern::Var("n".into()),
                Pattern::Var("n".into()),
                Pattern::Var("n".into()),
            ]),
            Pattern::Var("n".into()),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("annn");
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[3]).await;
        assert_equiv(&nfa, SkipMode::PastLastRow, &rows, &[1, 2, 3]).await;
    }

    /// A greedy trailing quantifier keeps the last match alive at the buffer end forever: `(a b+)`
    /// fed one row at a time never freezes its trailing match, and the match keeps extending as
    /// each `b` arrives.
    #[tokio::test]
    async fn trailing_quantified_match_extends_without_freezing() {
        let pat = Pattern::Concat(vec![
            Pattern::Var("a".into()),
            quant(Pattern::Var("b".into()), Quantifier::Plus, false),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("abbb");
        let matcher = SetMatcher::new(rows.clone());
        let mut inc = IncrementalMatcher::new(&nfa, SkipMode::PastLastRow);

        // [a]: `a` alone doesn't satisfy `a b+`, but it is alive (a `b` may arrive) — no match yet,
        // nothing frozen.
        inc.advance(&[0], &matcher).await.unwrap();
        assert_eq!(inc.provisional(), &[]);
        assert_eq!(inc.frozen(), 0);

        // Each appended `b` extends the same match by one row; it always ends at the buffer end, so
        // it stays alive and never freezes.
        for (seq, expected_end) in [(1i64, 2i64), (2, 3), (3, 4)] {
            inc.advance(&[seq], &matcher).await.unwrap();
            assert_eq!(
                inc.provisional(),
                &[SeqMatch {
                    start_seq: 0,
                    end_seq: expected_end,
                    labels: std::iter::once("a".to_string())
                        .chain(std::iter::repeat_n(
                            "b".to_string(),
                            expected_end as usize - 1
                        ))
                        .collect(),
                }]
            );
            assert_eq!(inc.frozen(), 0);
        }
    }

    /// Step 1 out-of-order reinsert with a *steal*. Rows arrive `[r0, r1, r3, r4]`; a late `r2`
    /// lands between `r1` and `r3`. Pattern `a+ b` over the pre-insert rows `[{a}, {a,b}, {x}, {x}]`
    /// freezes the match `a b` = `(0,2)` with `r1` bound as the closing `b`. The late `r2 = {b}`
    /// inserted at position 2 lets the greedy `a+` swallow `r1` as an extra `a` and bind `r2` as the
    /// `b`, so the batch answer over `[{a}, {a,b}, {b}, {x}, {x}]` is the longer `(0,3)` = `a a b`
    /// (r1 stolen from `b` to `a`). Truncating at `r3`'s seq must invalidate the frozen `(0,2)` — its
    /// end reaches the truncation point — and rewind so the re-feed re-derives `(0,3)`.
    #[tokio::test]
    async fn out_of_order_reinsert_equals_batch() {
        let pat = Pattern::Concat(vec![
            quant(Pattern::Var("a".into()), Quantifier::Plus, false),
            Pattern::Var("b".into()),
        ]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        // Buffer before the late arrival (sorted positions 0..4).
        let pre_rows = vec![sets(&["a"]), sets(&["a", "b"]), sets(&["x"]), sets(&["x"])];
        let pre_matcher = SetMatcher::new(pre_rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2, 3], &pre_matcher).await.unwrap();
        // `a b` = (0,2) freezes with r1 bound `b`.
        assert_eq!(provisional_triples(&inc), vec![(0, 2, labels(&["a", "b"]))]);
        assert_eq!(inc.frozen(), 1);

        // r2 = {b} sorts between r1 (pos 1) and r3 (pos 2). r3 is the first buffered row whose sorted
        // position changes, so the caller truncates at r3's seq (2).
        inc.truncate_from_seq(2, &pre_matcher).await.unwrap();
        // The frozen match reached the truncation point, so nothing survives.
        assert_eq!(provisional_triples(&inc), vec![]);
        assert_eq!(inc.frozen(), 0);

        // Re-feed the sorted suffix [r2, r3, r4] with their final positions as seqs.
        let final_rows = vec![
            sets(&["a"]),
            sets(&["a", "b"]),
            sets(&["b"]),
            sets(&["x"]),
            sets(&["x"]),
        ];
        let final_matcher = SetMatcher::new(final_rows.clone());
        inc.advance(&[2, 3, 4], &final_matcher).await.unwrap();

        assert_eq!(
            provisional_triples(&inc),
            batch_triples(&nfa, &skip, &final_rows).await
        );
        assert_eq!(
            provisional_triples(&inc),
            vec![(0, 3, labels(&["a", "a", "b"]))]
        );
    }

    /// Truncation landing in the *middle* of the frozen region: an earlier frozen match survives
    /// while a later one is dropped, so `next_pos` rewinds to the survivor's resume point (not 0).
    /// Pattern `a b` over `[{a},{b},{x},{a},{b},{x}]` freezes both `(0,2)` and `(3,5)`. A late `{a}`
    /// sorts at position 3 (before the second match): truncating at that row's seq keeps `(0,2)` and
    /// invalidates `(3,5)`, and the re-feed re-derives the shifted second match `(4,6)`.
    #[tokio::test]
    async fn truncate_inside_frozen_region_keeps_earlier_matches() {
        let pat = Pattern::Concat(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        let pre_rows = vec![
            sets(&["a"]),
            sets(&["b"]),
            sets(&["x"]),
            sets(&["a"]),
            sets(&["b"]),
            sets(&["x"]),
        ];
        let pre_matcher = SetMatcher::new(pre_rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2, 3, 4, 5], &pre_matcher).await.unwrap();
        assert_eq!(
            provisional_triples(&inc),
            vec![(0, 2, labels(&["a", "b"])), (3, 5, labels(&["a", "b"]))]
        );
        assert_eq!(inc.frozen(), 2);

        // A late {a} sorts at position 3; the old row at position 3 is the first whose sorted position
        // changes, so the caller truncates at its seq (3).
        inc.truncate_from_seq(3, &pre_matcher).await.unwrap();
        // (0,2) survives (its region is dead at boundary 3); (3,5) reaches past it and is dropped.
        assert_eq!(provisional_triples(&inc), vec![(0, 2, labels(&["a", "b"]))]);
        assert_eq!(inc.frozen(), 1);

        // Re-feed the sorted suffix [late {a}, old rows] from position 3, with final positions as seqs.
        let final_rows = vec![
            sets(&["a"]),
            sets(&["b"]),
            sets(&["x"]),
            sets(&["a"]),
            sets(&["a"]),
            sets(&["b"]),
            sets(&["x"]),
        ];
        let final_matcher = SetMatcher::new(final_rows.clone());
        inc.advance(&[3, 4, 5, 6], &final_matcher).await.unwrap();

        assert_eq!(
            provisional_triples(&inc),
            batch_triples(&nfa, &skip, &final_rows).await
        );
        assert_eq!(
            provisional_triples(&inc),
            vec![(0, 2, labels(&["a", "b"])), (4, 6, labels(&["a", "b"]))]
        );
    }

    /// Truncating at the first fed row is a full reset. A late `{a}` sorting before everything shifts
    /// all positions, so the caller truncates at seq 0; state must clear entirely, and re-feeding the
    /// whole corrected sequence must equal the batch answer.
    #[tokio::test]
    async fn truncate_to_zero_resets_and_refeeds() {
        let pat = Pattern::Concat(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        let pre_rows = vec![sets(&["a"]), sets(&["b"]), sets(&["x"])];
        let pre_matcher = SetMatcher::new(pre_rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2], &pre_matcher).await.unwrap();
        assert_eq!(provisional_triples(&inc), vec![(0, 2, labels(&["a", "b"]))]);
        assert_eq!(inc.frozen(), 1);

        // A late {a} sorts before r0, so r0 (seq 0) is the first row whose position changes.
        inc.truncate_from_seq(0, &pre_matcher).await.unwrap();
        assert_eq!(provisional_triples(&inc), vec![]);
        assert_eq!(inc.frozen(), 0);

        let final_rows = vec![sets(&["a"]), sets(&["a"]), sets(&["b"]), sets(&["x"])];
        let final_matcher = SetMatcher::new(final_rows.clone());
        inc.advance(&[0, 1, 2, 3], &final_matcher).await.unwrap();

        assert_eq!(
            provisional_triples(&inc),
            batch_triples(&nfa, &skip, &final_rows).await
        );
        assert_eq!(provisional_triples(&inc), vec![(1, 3, labels(&["a", "b"]))]);
    }

    /// THE case a positional (matcher-free) truncation rule gets wrong — do not simplify
    /// `truncate_from_seq` back to "drop frozen matches whose end position >= trunc_pos".
    ///
    /// Pattern `(a b c d) | (a b)` (long branch preferred) over `[{a},{b},{c},{x}]`: the short
    /// branch matches `(0,2)`, and it freezes only once the `x` at position 3 kills the long branch
    /// (at boundary 3 the long branch is still alive — `a b c` reaches the boundary inside the
    /// automaton — so no freeze happens there). A late `{d}` then sorts at position 3, displacing
    /// the `x`. Truncating at the x-row's seq re-checks the frozen region against boundary 3, where
    /// position 0 is alive again, so `(0,2)` must be dropped even though its end (2) lies strictly
    /// before the truncation position (3); the re-feed then derives the long match `(0,4)`. The
    /// positional rule keeps `(0,2)` and rewinds to its resume point 2 — no match can start at
    /// `{c}`/`{d}`/`{x}`, so it would wrongly answer `(0,2)` forever.
    #[tokio::test]
    async fn truncation_recheck_drops_frozen_match_alive_at_new_boundary() {
        let pat = Pattern::Alt(vec![
            Pattern::Concat(vec![
                Pattern::Var("a".into()),
                Pattern::Var("b".into()),
                Pattern::Var("c".into()),
                Pattern::Var("d".into()),
            ]),
            Pattern::Concat(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]),
        ]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        let pre_rows = vec![sets(&["a"]), sets(&["b"]), sets(&["c"]), sets(&["x"])];
        let pre_matcher = SetMatcher::new(pre_rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2, 3], &pre_matcher).await.unwrap();
        // The `x` kills the long branch at position 3, so the short `(0,2)` freezes.
        assert_eq!(provisional_triples(&inc), vec![(0, 2, labels(&["a", "b"]))]);
        assert_eq!(inc.frozen(), 1);

        // The late {d} sorts at position 3; the old {x} row (seq 3) is the first buffered row whose
        // sorted position changes, so the caller truncates at its seq.
        inc.truncate_from_seq(3, &pre_matcher).await.unwrap();
        // Discriminator: the frozen (0,2) ends *before* the truncation position, yet position 0 is
        // alive at the new boundary — the liveness re-check must drop it. The positional rule keeps
        // it here, and these two assertions (and the batch check below) fail under that rule.
        assert_eq!(provisional_triples(&inc), vec![]);
        assert_eq!(inc.frozen(), 0);

        // Re-feed the sorted suffix [late {d}, old {x}] with final positions as seqs.
        let final_rows = vec![
            sets(&["a"]),
            sets(&["b"]),
            sets(&["c"]),
            sets(&["d"]),
            sets(&["x"]),
        ];
        let final_matcher = SetMatcher::new(final_rows.clone());
        inc.advance(&[3, 4], &final_matcher).await.unwrap();

        assert_eq!(
            provisional_triples(&inc),
            batch_triples(&nfa, &skip, &final_rows).await
        );
        assert_eq!(
            provisional_triples(&inc),
            vec![(0, 4, labels(&["a", "b", "c", "d"]))]
        );
    }

    /// Truncating at a seq that was never fed is a no-op: neither a seq past everything buffered nor
    /// one exactly one-past-the-end may touch state, and later appends must still equal the batch.
    #[tokio::test]
    async fn truncate_unknown_seq_is_noop() {
        let pat = Pattern::Concat(vec![
            Pattern::Var("a".into()),
            quant(Pattern::Var("b".into()), Quantifier::Plus, false),
        ]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        let rows = from_str("abbc");
        let matcher = SetMatcher::new(rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2], &matcher).await.unwrap();
        let before = inc.provisional().to_vec();
        let before_frozen = inc.frozen();

        // A seq far past everything buffered, and the seq exactly one past the last fed row: both are
        // absent from `seq_index`, so both leave state untouched.
        inc.truncate_from_seq(99, &matcher).await.unwrap();
        inc.truncate_from_seq(3, &matcher).await.unwrap();
        assert_eq!(inc.provisional(), before.as_slice());
        assert_eq!(inc.frozen(), before_frozen);

        // Appending really does append (nothing corrupted): the final answer equals the batch.
        inc.advance(&[3], &matcher).await.unwrap();
        assert_eq!(
            provisional_triples(&inc),
            batch_triples(&nfa, &skip, &rows).await
        );
    }

    /// `SeqMatch`es (e.g. the finalized-and-returned ones) as `(start, end, labels)` triples, so
    /// they line up with the position-anchored batch oracle (seqs equal final sorted positions in
    /// these tests).
    fn seq_triples(ms: &[SeqMatch]) -> Vec<(usize, usize, Vec<String>)> {
        ms.iter()
            .map(|m| (m.start_seq as usize, m.end_seq as usize, m.labels.clone()))
            .collect()
    }

    /// Generic oracle: feeding rows incrementally (in any split) through `matcher` must equal one
    /// batch `find_matches_dynamic` with the *same* `matcher`. Unlike [`assert_equiv`] this takes an
    /// arbitrary [`CandidateMatcher`] (not just [`SetMatcher`]), so a matcher that applies its own
    /// pruning — e.g. the `WITHIN` span prune — can be driven through the incremental path.
    async fn assert_equiv_with<M: CandidateMatcher + Sync>(
        nfa: &Nfa,
        skip: SkipMode,
        n_rows: usize,
        matcher: &M,
        split_at: &[usize],
    ) {
        let batch: Vec<(usize, usize, Vec<String>)> = nfa
            .find_matches_dynamic(n_rows, matcher, &skip)
            .await
            .unwrap()
            .iter()
            .map(|m| (m.start, m.end, m.labels.clone()))
            .collect();
        let mut inc = IncrementalMatcher::new(nfa, skip.clone());
        let mut fed = 0usize;
        for &cut in split_at.iter().chain(std::iter::once(&n_rows)) {
            let seqs: Vec<i64> = (fed..cut).map(|i| i as i64).collect();
            inc.advance(&seqs, matcher).await.unwrap();
            fed = cut;
        }
        assert_eq!(provisional_triples(&inc), batch);
    }

    /// A [`CandidateMatcher`] that models the `WITHIN` span prune the executor applies inside
    /// `DefineMatcher::matches` (see `executor.rs`): binding a candidate at `pos` extends the match
    /// to span `[match_start, pos]`, and the executor rejects the candidate when that span exceeds
    /// the bound, so the NFA backtracks to the longest match that fits the window. `WITHIN` lives
    /// entirely inside the `CandidateMatcher`; this module has no `WITHIN` logic of its own — it
    /// hands the matcher straight to `find_matches_dynamic` and `reaches_boundary_alive` — so
    /// driving a span-pruning matcher through the incremental path and checking equality with the
    /// batch path proves the pass-through. (`nfa.rs`'s `SetMatcher` has no `WITHIN`, and the real
    /// `DefineMatcher` needs the executor's expression/row machinery, so we model the prune here.)
    struct WithinSetMatcher {
        rows: Vec<BTreeSet<String>>,
        /// Max span in order-key units. Seqs equal positions here, so the span of a candidate at
        /// `pos` is `pos - match_start == labels.len()`.
        max_span: usize,
    }

    impl CandidateMatcher for WithinSetMatcher {
        async fn matches(
            &self,
            var: &str,
            pos: usize,
            labels: &[String],
        ) -> StreamExecutorResult<bool> {
            if !self.rows[pos].contains(var) {
                return Ok(false);
            }
            let match_start = pos - labels.len();
            Ok(pos - match_start <= self.max_span)
        }
    }

    /// (a) Finalize mid-stream, then keep feeding: the finalized prefix is removed from and returned
    /// out of the diffable set, `provisional()` keeps only the still-revisable matches, and the
    /// union `provisional() ∪ returned` equals the batch oracle over all rows — with the rebased
    /// bookkeeping proven by advancing further after the finalization and still matching the oracle.
    #[tokio::test]
    async fn finalize_removes_prefix_and_rebases_bookkeeping() {
        let pat = Pattern::Concat(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        // 0:a 1:b 2:x 3:a 4:b 5:x 6:a 7:b 8:x  -> batch matches (0,2),(3,5),(6,8).
        let full = from_str("abxabxabx");
        let m_full = SetMatcher::new(full.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2, 3, 4, 5], &m_full).await.unwrap();
        assert_eq!(
            provisional_triples(&inc),
            vec![(0, 2, labels(&["a", "b"])), (3, 5, labels(&["a", "b"]))]
        );
        assert_eq!(inc.frozen(), 2);

        // Finalize everything before seq 3 (evict sorted positions [0,3)): removes the wholly-inside
        // match (0,2); (3,5) starts at the boundary and is kept.
        let removed = inc.finalize_before_seq(3);
        assert_eq!(seq_triples(&removed), vec![(0, 2, labels(&["a", "b"]))]);
        assert_eq!(provisional_triples(&inc), vec![(3, 5, labels(&["a", "b"]))]);
        assert_eq!(inc.frozen(), 1);

        // Keep feeding rows 6,7,8. Their buffer positions are now rebased (row 3 sits at position 0),
        // so the matcher indexes the surviving buffer `full[3..]`.
        let m_tail = SetMatcher::new(full[3..].to_vec());
        inc.advance(&[6, 7, 8], &m_tail).await.unwrap();
        assert_eq!(
            provisional_triples(&inc),
            vec![(3, 5, labels(&["a", "b"])), (6, 8, labels(&["a", "b"]))]
        );

        // Union of the returned finalized match and the surviving provisional set equals the batch
        // oracle over the whole run.
        let mut union = seq_triples(&removed);
        union.extend(provisional_triples(&inc));
        assert_eq!(union, batch_triples(&nfa, &skip, &full).await);
    }

    /// (a) Bookkeeping consistency across *all three* operations after a finalization: finalize a
    /// prefix, advance to derive more matches, then take a late (out-of-order) row that reinserts
    /// into the already-rebased tail — `truncate_from_seq` + re-feed — and the union still equals the
    /// batch oracle over the corrected full sequence. Exercises seq→position mapping against the
    /// rebased `seq_index` in both `advance` and `truncate_from_seq`.
    #[tokio::test]
    async fn finalize_then_truncate_and_advance_equals_batch() {
        let pat = Pattern::Concat(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        // 0:a 1:b 2:x 3:a 4:b 5:x 6:a 7:b 8:x
        let pre = from_str("abxabxabx");
        let m_pre = SetMatcher::new(pre.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2, 3, 4, 5, 6, 7, 8], &m_pre).await.unwrap();
        assert_eq!(
            provisional_triples(&inc),
            vec![
                (0, 2, labels(&["a", "b"])),
                (3, 5, labels(&["a", "b"])),
                (6, 8, labels(&["a", "b"])),
            ]
        );
        assert_eq!(inc.frozen(), 3);

        // Finalize before seq 3 (evict [0,3)); returns (0,2), rebases so row 3 is now position 0.
        let removed = inc.finalize_before_seq(3);
        assert_eq!(seq_triples(&removed), vec![(0, 2, labels(&["a", "b"]))]);
        assert_eq!(
            provisional_triples(&inc),
            vec![(3, 5, labels(&["a", "b"])), (6, 8, labels(&["a", "b"]))]
        );

        // A late {a} sorts at global position 6 (before old row 6): old row 6 (seq 6) is the first
        // buffered row whose sorted position changes, so the caller truncates at seq 6. The matcher
        // indexes the rebased surviving buffer `pre[3..]`.
        let m_pre_tail = SetMatcher::new(pre[3..].to_vec());
        inc.truncate_from_seq(6, &m_pre_tail).await.unwrap();
        // (3,5) survives (its region is dead at the truncation boundary); (6,8) reaches past it and
        // is dropped, to be re-derived by the re-feed.
        assert_eq!(provisional_triples(&inc), vec![(3, 5, labels(&["a", "b"]))]);

        // Corrected full sequence with the late {a} inserted at position 6:
        // 0:a 1:b 2:x 3:a 4:b 5:x 6:a 7:a 8:b 9:x  -> batch (0,2),(3,5),(7,9).
        let corrected = from_str("abxabxaabx");
        // Re-feed the sorted suffix from global position 6 (seqs 6..=9), matcher over the rebased
        // surviving buffer `corrected[3..]`.
        let m_corr_tail = SetMatcher::new(corrected[3..].to_vec());
        inc.advance(&[6, 7, 8, 9], &m_corr_tail).await.unwrap();

        let mut union = seq_triples(&removed);
        union.extend(provisional_triples(&inc));
        assert_eq!(union, batch_triples(&nfa, &skip, &corrected).await);
    }

    /// (a) Robustness: a seq that was never fed is a no-op, and finalizing before the very first row
    /// (`final_pos == 0`) evicts nothing. Neither may touch state, and later appends still equal the
    /// batch.
    #[tokio::test]
    async fn finalize_unknown_or_zero_seq_is_noop() {
        let pat = Pattern::Concat(vec![Pattern::Var("a".into()), Pattern::Var("b".into())]);
        let nfa = Nfa::compile(&pat);
        let skip = SkipMode::PastLastRow;

        let rows = from_str("abxab");
        let matcher = SetMatcher::new(rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, skip.clone());
        inc.advance(&[0, 1, 2, 3], &matcher).await.unwrap();
        let before = inc.provisional().to_vec();
        let before_frozen = inc.frozen();

        assert_eq!(inc.finalize_before_seq(99), vec![]); // never fed
        assert_eq!(inc.finalize_before_seq(0), vec![]); // final_pos == 0, evict nothing
        assert_eq!(inc.provisional(), before.as_slice());
        assert_eq!(inc.frozen(), before_frozen);

        inc.advance(&[4], &matcher).await.unwrap();
        assert_eq!(
            provisional_triples(&inc),
            batch_triples(&nfa, &skip, &rows).await
        );
    }

    /// (b) `WITHIN` parity: a matcher that applies the `WITHIN` span prune drives identically through
    /// the incremental path and the batch path. `a b+` with a max span of one row (so a match may
    /// span at most two rows, `a b`) over `abbabb`: without `WITHIN` the greedy `b+` swallows both
    /// `b`s per match; `WITHIN` caps each match at `ab`. The incremental path must track that through
    /// both matching and the freeze-gate liveness check — proving pass-through, since this module has
    /// no `WITHIN` logic of its own.
    #[tokio::test]
    async fn within_span_prune_incremental_equals_batch() {
        let pat = Pattern::Concat(vec![
            Pattern::Var("a".into()),
            quant(Pattern::Var("b".into()), Quantifier::Plus, false),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("abbabb");
        let matcher = WithinSetMatcher {
            rows: rows.clone(),
            max_span: 1,
        };
        for split in [
            &[1][..],
            &[2][..],
            &[3][..],
            &[1, 2, 3, 4, 5][..],
            &[2, 4][..],
        ] {
            assert_equiv_with(&nfa, SkipMode::PastLastRow, rows.len(), &matcher, split).await;
        }
    }

    /// (c) Finalization must never reach into the open (non-frozen) trailing region: the boundary
    /// has to lie within the frozen prefix. `a b+` fed one row at a time keeps its trailing greedy
    /// match alive at the buffer end forever, so nothing freezes (`next_pos == 0`). Finalizing before
    /// seq 2 (a position past the frozen prefix) must trip the debug assertion.
    #[tokio::test]
    #[should_panic(expected = "frozen prefix")]
    async fn finalize_into_open_region_panics() {
        let pat = Pattern::Concat(vec![
            Pattern::Var("a".into()),
            quant(Pattern::Var("b".into()), Quantifier::Plus, false),
        ]);
        let nfa = Nfa::compile(&pat);
        let rows = from_str("abb");
        let matcher = SetMatcher::new(rows.clone());

        let mut inc = IncrementalMatcher::new(&nfa, SkipMode::PastLastRow);
        inc.advance(&[0, 1, 2], &matcher).await.unwrap();
        // Trailing greedy match (0,3) stays alive at the boundary: nothing frozen.
        assert_eq!(inc.frozen(), 0);
        // seq 2 sits at position 2, past the frozen prefix (next_pos == 0): must panic.
        inc.finalize_before_seq(2);
    }
}
