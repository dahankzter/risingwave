# Row Pattern Recognition (`MATCH_RECOGNIZE`)

`MATCH_RECOGNIZE` (SQL:2016 row pattern recognition) finds matches of a regular-expression-like
pattern over the rows of a partition, ordered by a time column, and emits one row per match. It is
the streaming-SQL form of complex event processing (CEP): chains like "a login, then three failed
actions, then a withdrawal within five minutes".

This document covers the streaming implementation. The supported v1 subset is:

- **append-only input only** — a retraction mid-partial-match would invalidate an in-progress or
  completed match, so the semantics over a changelog are ill-defined. The binder/planner rejects
  non-append-only input. (Flink restricts `MATCH_RECOGNIZE` to append-only for the same reason.)
- **`ONE ROW PER MATCH`** — `ALL ROWS PER MATCH` is not yet supported.
- `PARTITION BY` (required, plain columns) and `ORDER BY` (required, leading column must carry a
  watermark).
- `PATTERN`: concatenation, alternation (`|`), grouping, quantifiers (`*`, `+`, `?`, `{n,m}` and
  their reluctant `*?` forms), and `PERMUTE`.
- `DEFINE` predicates with full running navigation (`PREV`/`NEXT`/`FIRST`/`LAST` and bare `A.col`).
- `MEASURES` with `FIRST`/`LAST`/bare `A.col`, `CLASSIFIER()`, `SUBSET`, and the aggregates
  `COUNT(*)`/`COUNT`/`MIN`/`MAX`/`SUM`/`AVG`.
- `AFTER MATCH SKIP PAST LAST ROW` / `TO NEXT ROW` / `TO FIRST|LAST <var>`.
- `WITHIN <interval>` (a streaming time bound on the match span).

## Feature support

The clause is modeled on the two reference implementations RisingWave users come from: Apache Flink
SQL (streaming) and Google BigQuery (batch). The table summarizes RisingWave's v1 support against
them. Flink and BigQuery columns reflect their public documentation as of June 2026 (see Sources);
✅ supported, ❌ not supported, ➖ not applicable.

| Feature | Flink SQL | BigQuery | RisingWave v1 |
| --- | :---: | :---: | :---: |
| Streaming | ✅ | ❌ | ✅ |
| Batch | ✅ | ✅ | ❌ |
| `ONE ROW PER MATCH` | ✅ | ✅ ² | ✅ |
| `ALL ROWS PER MATCH` | ✅ | ❌ | ❌ |
| Concatenation, `*` `+` `?` `{n,m}` | ✅ | ✅ | ✅ |
| Reluctant quantifiers (`*?`) | ✅ ¹ | ✅ | ✅ |
| Alternation (`A \| B`) | ❌ | ✅ | ✅ |
| Grouping + quantifier (`(A B)+`) | ❌ | ✅ | ✅ |
| `PERMUTE` | ❌ | ❌ | ✅ |
| Anchors (`^` `$`) | ❌ | ✅ | ❌ |
| Exclusion (`{- … -}`) | ❌ | ❌ | ❌ |
| Running nav in `DEFINE` (`A.col`, `FIRST`/`LAST`) | ✅ | ✅ | ✅ |
| Physical `PREV` in `DEFINE` | ❌ ³ | ✅ | ✅ |
| Physical `NEXT` in `DEFINE` | ❌ ³ | ✅ | ❌ ⁴ |
| `MEASURES` `FIRST`/`LAST` | ✅ | ✅ | ✅ |
| Aggregates in `MEASURES` (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`) | ✅ | ✅ | ✅ |
| `CLASSIFIER()` | ❌ | ✅ | ✅ |
| `MATCH_NUMBER()` | ❌ | ✅ | ❌ |
| `SUBSET` | ❌ | ❌ | ✅ |
| `AFTER MATCH SKIP PAST LAST ROW` / `TO NEXT ROW` | ✅ | ✅ | ✅ |
| `AFTER MATCH SKIP TO FIRST`/`LAST <var>` | ✅ | ❌ | ✅ |
| `WITHIN` (time bound) | ✅ | ❌ | ✅ |
| Checkpoint / recovery / rescaling | ✅ | ➖ | ✅ |

¹ Flink supports reluctant `+?` / `*?` but not the reluctant optional `??`.
² BigQuery has no `ROWS PER MATCH` keyword; it emits one row per match and requires aggregation in
`MEASURES` (use `ARRAY_AGG` for all-rows-style output).
³ Flink expresses physical offsets through `LAST(expr, n)` rather than `PREV`/`NEXT`.
⁴ Rejected at bind time: a row's verdict would depend on rows after it, which needs per-candidate
decidability (an out-of-range read as a wait for exactly that candidate) — future work.

Sources: [Apache Flink — Pattern Recognition](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/sql/queries/match_recognize/),
[BigQuery — `MATCH_RECOGNIZE` clause](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#match_recognize_clause).

## Planning pipeline

The clause flows through the usual layers; each is a thin, conventional addition:

- **Parser** (`src/sqlparser`): `TableFactor::MatchRecognize` plus the `Measure`, `RowsPerMatch`,
  `AfterMatchSkip`, `MatchRecognizePattern`, `RepetitionQuantifier`, and `SubsetDefinition` AST
  nodes.
- **Binder** (`src/frontend/src/binder/relation/match_recognize.rs`): produces `BoundMatchRecognize`
  and registers the output columns (`PARTITION BY` columns, then the measures). The interesting work
  is *lowering* `MEASURES` and `DEFINE` (see below).
- **Logical plan** (`logical_match_recognize.rs`): `LogicalMatchRecognize` with the standard trait
  set. `PredicatePushdown` is a barrier (a predicate over computed output columns must not push below
  the operator); `ColPrunable` prunes the input to the columns the clause's expressions actually
  read. `to_stream` enforces the v1 restrictions and shards the input by the `PARTITION BY` key.
- **Stream plan** (`stream_match_recognize.rs` + `generic/match_recognize.rs`): `StreamMatchRecognize`
  is hash-sharded on the partition columns and declares three internal state tables (see
  [State and fault tolerance](#state-and-fault-tolerance)). Its output stream kind follows the emit
  mode: append-only under `EMIT ON WINDOW CLOSE`, retract under the plain emit-on-update form (see
  [Emission modes](#emission-modes)).

### Lowering `MEASURES` and `DEFINE`

Pattern-variable references (`A.price`, `FIRST(B.ts)`, `PREV(price)`) have no direct analog in the
expression framework, so the binder lowers each measure and define to **an ordinary expression over
a synthetic row**, plus a list of *slots* that describe how to build that synthetic row from a
match:

- A `MeasureSlot` / `DefineSlot` records a navigation kind (`First`, `Last`, `Classifier`, `Prev`,
  `Next`, `RunningFirst`, `RunningLast`, the aggregates, …), the pattern variables it ranges over
  (several, for a `SUBSET`), and the input column it reads.
- The lowered expression is a normal `ExprImpl` whose `InputRef(i)` reads `slots[i]`.

`DEFINE` navigation is **`RUNNING`**, and the candidate row counts as tentatively labeled: while
`A`'s predicate is being tested, `RunningFirst`/`RunningLast` over any variable set containing `A`
(directly or through a `SUBSET`) sees the candidate as the newest such row. That is what makes
`DEFINE A AS LAST(A.v) = A.v` a tautology, matching SQL:2016, where a pattern-variable-qualified
column reference *is* `RUNNING LAST` of that column — the binder already lowers the bare `A.v` inside
`A`'s own `DEFINE` straight to the candidate row, so the two spellings must agree. `MEASURES` is
unaffected: it runs over completed matches whose labels already include the last row.

The corollary is that `DEFINE A AS A.v > LAST(A.v)` is **unsatisfiable** — it compares the candidate
against itself (`x > x`), so `A` never matches and the view stays empty. It does not mean "greater
than the previous `A`". The standard spells that with a logical offset, `LAST(A.v, 1)`, which is not
implemented yet (rejected at bind time). Use the physical `PREV(A.v)` instead, as
`e2e_test/streaming/match_recognize_define_nav.slt` does for its V-shape:
`define down as down.price < prev(down.price)`.

This keeps all type checking, coercion, and constant folding in the existing expression machinery:
the executor materializes the synthetic row per match (or per candidate, for `DEFINE`) and evaluates
the expression over it. `DEFINE` navigation functions are pulled out of the predicate by an AST
pre-walk into a synthetic placeholder relation so the remaining predicate binds normally.

### The hidden match id

A partition can contain many matches, and two matches may produce byte-identical `PARTITION BY` +
`MEASURES` output, so those columns are not a unique key. The output therefore carries a **hidden
`_match_id` column** (the same mechanism sources use for `_row_id`); the stream key is the partition
columns plus `_match_id`. It is hidden, so `SELECT *` returns only the user columns.

The executor fills `_match_id` with the **match's start row's `seq`** (the buffered row's hidden
snowflake PK tiebreaker, assigned once at ingest). This is unique forever — an emitted match's start
row is always evicted, so no later match can share it (the same invariant that prevents
cross-watermark double emits) — and, unlike an id minted at emission time, it is **deterministic
across recovery replay**: re-emitting a match after a rollback reproduces byte-identical output. A
stable, replay-deterministic match identity is also the key the plain form's emit-on-update changelog
retracts against, and what lets recovery rebuild that changelog's diff base from the buffer alone
without persisting it (see [Emission modes](#emission-modes)).

### Emission modes

The emit mode is chosen per query by the presence of `EMIT ON WINDOW CLOSE`, and the plan node
carries it (`emit_on_update`, proto field 16). Both modes share the buffer, the NFA, and the
matching/eviction machinery; they differ only in *when* and *how* a match reaches the output.

**`EMIT ON WINDOW CLOSE` — final-only.** The operator emits **only final matches, at the watermark**:
a match is output — as an `Insert`, so the output is append-only — once a later safe row confirms the
greedy match is maximal and no late row can still change it. This is RisingWave's Emit-On-Window-Close
behavior; the plan node declares it (`emit_on_window_close = true`) so a query naming
`EMIT ON WINDOW CLOSE` gets exactly this.

**Plain form — emit-on-update.** Without `EMIT ON WINDOW CLOSE`, the operator emits under RisingWave's
default semantics: a **retract changelog**, corrected as the picture fills in. The model is
whole-buffer, not safe-prefix. At each barrier, every partition that received rows since the last
barrier is re-matched over its *entire* buffer ("as if input ended now"), yielding a *provisional*
match set; that set is diffed against the set the operator last emitted for the partition, and the
difference is emitted. The changelog is keyed by `_match_id` (the match's start-row seq), so the diff
is by identity:

- a match present now but not before (a new start) → `Insert`;
- a match present before but not now (its start no longer begins any match) → `Delete`;
- a match present in both whose extent, labels, or output row changed → `Delete(old)` then
  `Insert(new)` under the same key;
- an unchanged match → nothing.

The watermark's role shrinks to **finalization**: it evicts rows that can no longer belong to any
match, which is what lets a provisional match settle and stop being re-diffed. It never emits in this
mode.

**Emit-before-finalize (watermark ordering).** Within an epoch the message order is
Chunk → Watermark → Barrier. A match whose rows all arrive in *this* epoch and become watermark-safe
now would be evicted by the watermark before any barrier had diffed the partition — its `Insert` would
never be emitted, and the following barrier would see the rows already gone (nothing to diff),
permanently losing a real match. So in emit-on-update mode the watermark path runs the same
whole-buffer diff **before** it evicts; every match that finalization removes from the diffable set has
thus already been emitted before its rows leave. The diff is deterministic in the buffer content, so
running it at the watermark and again at the barrier is not a double emit — the second run sees no
change and is a no-op, it only pulls the same ops earlier within the epoch.

**Finalization never creates a retraction.** A match computed over the whole buffer stays valid when
the watermark later declares it final: finality removes future *revisability*, it cannot invalidate a
match already computed over the rows that exist. So the watermark path only ever *drops* a finalized
match from the diff base (a permanent result — no op emitted); a `Delete` arises only from the barrier
diff, when genuinely later input supersedes a still-provisional match. Every retraction therefore
corrects a match the operator itself had emitted.

**Recovery without an emission state table.** The operator persists no changelog and no last-emitted
state — only the raw buffer (see [State and fault tolerance](#state-and-fault-tolerance)). The
emit-on-update diff base is in-memory, and after a crash or a rescale it is rebuilt by recomputing each
owned partition's provisional set from the restored buffer and seeding the base with it, emitting
nothing. This is sound precisely because both the match identity (`_match_id` = start-row seq) and each
match's output row are **deterministic functions of the buffer**: the recomputed set is byte-identical
to what the pre-crash actor last emitted for the same committed buffer, so it already equals the rows
downstream holds at the recovered epoch. Seeding the base to that set means the next barrier emits only
genuine new deltas — no duplicate `Insert`s, no spurious `Delete`s. A persisted emission state table
would be redundant: it could only store what the buffer already determines, at the cost of a second
table to write every barrier. The trade is recovery-time recomputation (one matcher pass per owned
partition at startup) for zero steady-state write amplification.

One composition consequence carries over from Emit-On-Window-Close: that mode emits no downstream
watermark, so a stateful operator that needs one (e.g. an aggregation) cannot sit above
`MATCH_RECOGNIZE` **inside the same** `EMIT ON WINDOW CLOSE` view; compose across views instead (the
view's output is append-only, and a downstream default-emit view can aggregate it). The plain
emit-on-update form outputs an ordinary retract stream and composes like any other default-emit
materialized view. Stateless operators (projections, filters) compose freely within either.

## The NFA

`src/stream/src/executor/match_recognize/nfa.rs` is a self-contained, pure module (unit-tested
without a cluster). A `Pattern` (variable / concat / alternation / quantified / permute) is compiled
by Thompson construction into an `Nfa` whose labelled transitions are pattern variables.

Matching is **predicate-driven**: rather than precomputing which variables each row satisfies, the
matcher consults a `CandidateMatcher` as it walks the NFA, so `DEFINE` predicates that depend on the
running match (e.g. `B AS B.price > A.price`) can be evaluated against the rows matched so far. The
matcher returns the *first accepting path in transition order*; greedy quantifiers order the loop
edge first (longest match), reluctant quantifiers order the exit edge first (shortest), and
alternation prefers its first branch. `PERMUTE` expands to the alternation of all orderings (capped
to keep the factorial bounded).

## The executor

`MatchRecognizeExecutor` follows the standard append-only, watermark-driven executor shape (compare
`eowc_over_window`):

- **Buffering.** Each input row is written through to the state table; nothing is held in memory
  between watermarks. Rows may arrive out of order.
- **Matching on watermark.** When the watermark on the leading `ORDER BY` column advances to `w`,
  the executor visits only the partitions that need attention, found via the **wakeup frontier** (see
  [The wakeup frontier](#the-wakeup-frontier)): per owned vnode it range-scans `frontier_index_table`
  for entries with `next_wakeup <= w` (the index is PK-ordered by `next_wakeup`, so it stops at the
  first entry past `w`), then for each candidate partition reads just that partition from the buffer
  table with `iter_with_prefix`, in PK order — `(partition, order_key, seq)`, already `ORDER BY`
  ordered, so no in-memory sort. Every row with `order_key < w` is final; the matcher runs over the
  safe prefix. A match becomes final once a later safe row follows it (so the greedy match is known
  maximal), **or immediately if the finder's preferred result can no longer change** — no path the
  finder prefers over the current accepting one could become accepting with more rows, as for a
  fixed `(a b)`, an ordered alternation whose first-listed branch already accepted, or a reluctant
  quantifier's short result. That wait would otherwise starve an idle partition forever. The check
  follows the matcher's preference order exactly (`Nfa::may_extend`): a lower-priority path — a
  later-listed alternation branch, a greedy loop the finder already exited on real data — never
  holds a match. `AFTER MATCH SKIP` decides where the scan resumes. Under `EMIT ON WINDOW CLOSE` the
  final match is emitted here, streaming straight into a `StreamChunkBuilder` flushed a chunk at a
  time; under the plain form the watermark path emits nothing and only evicts, emission having
  happened at barriers (see [Emission modes](#emission-modes)). After processing, the partition's
  frontier entry is
  recomputed (the earlier of its next future row and the earliest WITHIN expiry of a retained
  partial) or dropped. Work per watermark is therefore proportional to the
  partitions that need attention, not to the number of live partitions; the working set is the largest
  single candidate partition's live rows plus one output chunk.

  Physical `PREV` needs no retention machinery: the binder admits `PREV(.., k)` only on variables at
  least `k` rows from the match start (an exact minimum-distance walk over the pattern), so every
  `PREV` read lands inside the match span, whose rows are retained while the match is live. Shapes
  that could read before the match — where rows are no longer retained, so the same row would flip
  its verdict depending on whether eviction has run — are rejected at bind time until lookbehind
  retention is designed as its own change. Physical `NEXT` in `DEFINE` is rejected at bind time
  entirely: a verdict depending on rows after the candidate needs per-candidate decidability, and a
  global deferral (hold everything by the max offset) can starve a partition whose match is already
  decidable.
- **Measures at match time.** Measures reference specific matched rows (`FIRST(a.ts)`, `LAST(b.v)`),
  known only once the match and its per-row variable labels are found, so each measure's synthetic
  row is built from the matched rows and the expression evaluated then. `WITHIN` is enforced during
  matching, pruning candidates that would push the span past the bound.
- **Eviction.** Rows before the earliest position that could still *begin* a match are evicted. Each
  candidate partition is read with its own `iter_with_prefix` scan, which is dropped before the
  evicting deletes run (a state-table delete cannot interleave with an open iterator over the same
  table), so deletes apply in place per partition. Together with the watermark this bounds state to
  the live (unfinalized) window (see [State bound and `WITHIN`](#state-bound-and-within)).

  Eviction fires for every partition the frontier wakes. A partition gaining a newly-safe row is
  woken by the row term of `next_wakeup`; an *idle* partition holding a retained partial bounded by
  `WITHIN` is woken by the deadline term (see [The wakeup frontier](#the-wakeup-frontier)) when that
  partial times out, so its dead rows are released even with no further input.

Matching is **incremental**. Each partition keeps a cached `IncrementalMatcher`
(`src/stream/src/executor/match_recognize/incremental.rs`) that feeds newly appended rows and rescans
only the mutable suffix behind the last frozen match, rather than re-running the whole buffer. A
leading run of matches *freezes* once its entire scan region is dead at the current boundary — per the
same `reaches_boundary_alive` liveness predicate eviction uses — so a frozen match can never change
under later appends and is never rescanned; work is proportional to the newly-safe rows, not the live
window. The cache is a pure derivation of the buffer (live-window seqs and per-row labels only, never
row data): dropped on recovery and on any vnode-bitmap change, rebuilt lazily per partition by
re-feeding the restored buffer, and its `provisional()` set is by construction byte-identical to a
from-scratch scan over the same rows. How much the incrementality saves is mode-dependent — under
`EMIT ON WINDOW CLOSE` eviction trims the frozen prefix as fast as it forms, so there is no cross-visit
saving; under emit-on-update the whole-buffer barrier diffs run between evictions, so for append-mostly
input the frozen prefix persists across barriers and each diff rescans only the newly-appended suffix
(see [Emission modes](#emission-modes)).

### The watermark boundary is strict

A RisingWave watermark `w` promises only that **no future row will have `order_key < w`** — a row
with `order_key == w` may still arrive, and `WatermarkFilterExecutor` forwards it (it keeps
`event_time >= watermark`). Every finality decision in this operator is therefore expressed with a
strict `< w`:

- the safe prefix is `order_key < w`;
- a `WITHIN` window is closed only when `deadline < w` (at `deadline == w` a completing row at
  `order_key == w` still falls inside the bound);
- a match held at the safe boundary becomes `WITHIN`-final under that same `deadline < w`.

The last two must stay in lockstep. If the emit test were stricter than the eviction test, eviction
would delete the rows of a match the emit side is still holding, and the match would be lost with no
trace.

The wakeup frontier is the **complement**: a partition's next row-driven wakeup is its earliest
surviving row with `order_key >= w`, so a row sitting exactly at `w` still schedules a revisit. The
frontier's own lookups (the idle fast path and the candidate range scan, both `next_wakeup <= w`)
stay one step *less* strict on purpose — waking a partition that turns out to have nothing final
costs only a scan, whereas a wakeup test stricter than the finality tests would skip a partition that
is genuinely due and park its match forever.

### Invalid `AFTER MATCH SKIP` targets degrade, and say so

`AFTER MATCH SKIP TO FIRST|LAST <var>` has no valid resume row in two data-dependent cases — the same
query hits them or not depending on which rows arrive:

| case | example | resume position | degrades to |
|---|---|---|---|
| the target is bound to no row of the match | `PATTERN (a b?)` matching only `a`, `SKIP TO LAST b` | the match end | `SKIP PAST LAST ROW` |
| the target is the match's own first row | `PATTERN (a b)`, `SKIP TO FIRST a` | `match start + 1` | `SKIP TO NEXT ROW` |

The SQL standard prescribes a runtime error for both (Oracle raises ORA-62511 / ORA-62512; Flink
likewise). This implementation keeps the degradation and **reports** it instead of raising it. The
condition is data-dependent, and the materialized view is already committed by the time any row
arrives: an error would abort the actor, recovery would replay the same rows, and the actor would die
again — a recoverable query turned into a crash loop. No RisingWave streaming operator fails an actor
for a data-dependent condition; every hard error in this operator is a contract or plan violation
(non-append-only input, an unknown slot kind) that recovery cannot fix either way.

So the degradation is made visible rather than fatal. `SkipMode::next_pos` returns the resume position
plus an optional diagnostic (keeping `nfa.rs` pure — it holds no error reporter), and the executor's
emit path routes it to the actor's `EvalErrorReport`: the same *surface* expression evaluation errors in
this operator already use, i.e. the rate-limited `stream_expr_error` log and the `user_compute_error`
metric. The reported error names the skip mode, its target variable and the strategy actually applied:

```
Invalid parameter AFTER MATCH SKIP TO LAST: target variable `c` is bound to no row of the match,
so there is no row to resume at; the scan resumed past the match's last row instead (degraded to
SKIP PAST LAST ROW)
```

Two things about that surface are worth knowing when reading the output. First, while the surface is
precedented, the *carrier* is not: no other `EvalErrorReport` user synthesizes an error — they all pass
on one produced by an actual expression evaluation. `ExprError` is simply the only type the trait
accepts, and `InvalidParam` is the honest fit (the query's `AFTER MATCH SKIP` parameter cannot be
honored); `ExprError::Custom` was rejected as the UDF error channel and slated for removal. A
consequence is that the log line carries the surface's fixed prefix `failed to evaluate expression`,
hardcoded in `ActorContext::on_compute_error`, even though nothing was evaluated — the actionable
content is the self-contained `error=` field, not the head of the line. Second, the metric labels
(`["ExprError", executor_name, fragment_id]`) separate this operator from others but not from this
operator's own `DEFINE`/`MEASURES`/`WITHIN` evaluation errors, so the metric reads as "this
`MATCH_RECOGNIZE` query is unhealthy" and the log line is the artifact that says why.

Reporting is deduplicated **per kind per watermark pass**. The cause is a property of the query, not of
one row: a target no match can bind degrades on every match forever, and one that only sometimes fails
to bind (`PATTERN (a? b)` with `SKIP TO FIRST b`) still repeats without bound. Every repetition within a
pass would be a byte-identical duplicate, so one report per kind per pass keeps the signal steady and
bounded by watermark frequency rather than by match or partition count; the metric therefore counts
passes that degraded, not individual degradations.

Note the related bind-time check: a `SKIP TO FIRST|LAST <var>` target that is not a *pattern* variable
at all (e.g. a `DEFINE`-only symbol) is rejected when the query is bound, so it never reaches the
executor.

## State and fault tolerance

The operator declares three internal state tables: the **buffer table** plus the two **wakeup
frontier** tables (below). The buffer table has layout `[ seq (i64), <input columns…> ]`, keyed by
`(partition columns, ORDER BY columns, seq)` and distributed by the partition key. Keying by the
order columns keeps the buffer physically sorted by `(partition, order key)`, so a partition can be
read back in key order and processed without an in-memory sort. `seq` is a per-actor monotonic id
that breaks ties between rows with equal `ORDER BY` keys. Only the raw buffered rows are persisted —
the NFA is recompiled from the pattern at startup and `DEFINE`/`MEASURES` are evaluated at match
time, so neither is stored. (This is less state than Flink's CEP, which persists the partial-match
SharedBuffer.)

- **Recovery.** The state table is authoritative, so there is no in-memory buffer to rebuild: after
  recovery the next watermark simply scans the (restored) state table per owned vnode (an empty-prefix
  scan cannot compute a vnode on a distributed table). In emit-on-update mode the one derived structure
  that would otherwise re-emit — the last-emitted diff base — is rebuilt from the restored buffer before
  any input is processed; by determinism this reproduces exactly what downstream already holds, so
  recovery re-emits nothing (see [Emission modes](#emission-modes)).
- **Rescaling.** On a vnode-bitmap change the set of partitions an actor owns shifts; the state table
  migrates the affected vnodes, and the next watermark scans whatever the actor now owns. There is no
  in-memory cache to reload or drop.
- **Parallelism.** Matching is independent per partition, so the input is hash-sharded by the
  `PARTITION BY` key and each actor owns its partitions' state.

### The wakeup frontier

A watermark only needs to touch partitions that have a newly-safe row (or, eventually, a retained
partial expiring by `WITHIN`). Without help, finding them means scanning every live partition each
watermark — `O(#live partitions)`, which dominates for high-cardinality `PARTITION BY` (per-player,
per-session, …). The frontier makes the work proportional to the partitions that actually need
attention. It is two internal tables, for two access patterns:

- `frontier_meta_table` — pk `(partition…)` → `next_wakeup_order_key`. Point-looked-up by partition
  on the insert path, so a chunk can re-point a partition's wakeup in one lookup.
- `frontier_index_table` — pk `(next_wakeup_order_key, partition…)`, **distributed by partition**.
  Range-scanned per owned vnode for `next_wakeup <= watermark`. The PK leads with the wakeup so the
  scan is a key-prefix scan that stops at the first entry past the watermark, while distributing by
  the partition keeps a partition's index entry on the same vnode as its buffered rows (so they
  re-shard together on rescale).

`next_wakeup_order_key` is the earliest `order_key` at which the partition next needs attention,
which is the earlier of two events:

- **the next row to become safe** — its earliest unprocessed `order_key`. On insert this is moved
  earlier when a chunk brings an earlier row (aggregated once per partition per chunk, not per row).
- **the earliest `WITHIN` expiry of a retained partial** — `first_order_key + interval`, computed
  from the `within_deadline` expression. Without it, an idle partition (no future row) holding a
  partial bounded by `WITHIN` would drop its frontier entry and never be revisited, leaking the
  partial until — if ever — a new row arrived. With it, the partition is woken exactly when the
  partial times out, and the eviction predicate (which already honours `WITHIN`) releases it.

On watermark the value is recomputed to the minimum of those two after processing, or the entry is
dropped when neither applies (a later insert re-schedules it). Both tables are committed with the
buffer table at each barrier, so the frontier and the buffer stay consistent across recovery and
rescale.

### State bound and `WITHIN`

State is bounded to the **live (unfinalized) window** — the rows that could still begin or extend a
match. What bounds that window depends on whether the pattern carries a `WITHIN` clause:

- **With `WITHIN <interval>`** the span of any match is capped, so once the watermark is strictly past
  a row's `order_key + interval` that row can no longer begin or extend a match and is evicted. State per
  partition is bounded by the `WITHIN` window, and total state by that window times the number of
  live partitions.
- **Without `WITHIN`** a buffered prefix can be completed by an arbitrarily distant future row — e.g.
  `PATTERN (A B)` retains an `A` until some later `B` arrives, however long that takes — so an
  unmatched partial is kept indefinitely. This is correct SQL semantics (a streaming join without a
  time bound retains its build side the same way), but it means state is bounded only by the number
  of distinct `PARTITION BY` keys, not by time. For an unbounded key space (per-session, per-device,
  …) it grows without limit.

What that growth costs differs by emit mode:

- Under **`EMIT ON WINDOW CLOSE`** resident memory is bounded regardless — the executor streams
  partitions from the state table and holds nothing between watermarks — so the unbounded quantity
  without `WITHIN` is only the *persisted* state on the storage engine, not process memory. The
  binder emits a `NOTICE` when such a query has no `WITHIN`, as a reminder.
- Under **emit-on-update** the operator additionally keeps two in-memory derivations of the buffer
  across barriers: the changelog diff base (`last_emitted`) and the per-partition matcher cache. A
  `WITHIN` bound is what expires a match and lets its rows evict, draining those structures; without
  it they grow with `PARTITION BY` key cardinality — unbounded *process* memory. Emit-on-update
  therefore **requires a `WITHIN` clause at plan time** (a `NotSupported` error otherwise, with the
  hint to add `WITHIN` or declare `EMIT ON WINDOW CLOSE`), rather than warn at bind time and OOM at
  runtime.

An opt-in state TTL (dropping partials older than a configurable age, trading completeness for a
hard bound) is possible future work.

## Limitations and future work

- `ALL ROWS PER MATCH`, batch execution, and non-append-only input are not supported.
- The incremental matcher's cross-visit CPU saving is realized only in emit-on-update mode; under
  `EMIT ON WINDOW CLOSE`, eviction trims the frozen prefix each watermark, so it saves nothing there
  (see [The executor](#the-executor)).
- Without a `WITHIN` clause, unmatched partials are retained indefinitely, so state is bounded only
  by `PARTITION BY` key cardinality (see [State bound and `WITHIN`](#state-bound-and-within)).
  Emit-on-update rejects that shape at plan time (the in-memory diff base and matcher cache would be
  unbounded); `EMIT ON WINDOW CLOSE` allows it with a binder `NOTICE`, since only persisted state
  grows there.
- Anchors (`^`, `$`) and pattern exclusions (`{- … -}`) are parsed but rejected at planning time.
