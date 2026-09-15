# Program Analysis and Verification — Final Project

Two static analyses, implemented in Python 3 (stdlib only):

- `parity/` — Relational parity analysis for integer programs (Section 1)
- `shape/`  — Relational shape analysis for acyclic/unshared linked lists (Section 2)
- `common/` — Shared CFG parser + generic worklist fixpoint engine
- `tests/`  — Unit tests plus the property-based verification tools described below

See `report.pdf` for the formal domains, transfer functions, and the
soundness/termination arguments.

Neither analysis uses widening: both domains have finite height for any
fixed program, so plain worklist iteration terminates on its own.

## Requirements

Python 3.8+, no third-party packages.

## Running

From the repository root:

```
# Parity analysis: run on one or more test programs
python3 -m parity.analysis parity/interesting-tests/1-given-example.txt
python3 -m parity.analysis parity/interesting-tests/*.txt      # run all 5

# Shape analysis: same pattern
python3 -m shape.analysis shape/interesting-tests/1-given-example.txt
python3 -m shape.analysis shape/interesting-tests/*.txt        # run all 5
```

Each run prints, per `assert` edge in the program: `VERIFIED` (the
analysis proved the assertion holds on every execution) or `POSSIBLY
VIOLATED` (the analysis could not prove it — either because the
assertion is genuinely false on some path, or because of a real
precision limit). Parity test 3 and shape test 4 each contain an
assertion that is genuinely false and is correctly reported as
`POSSIBLY VIOLATED`.

Shape analysis additionally reports `VIOLATION` lines for memory-safety,
cycle, and sharing problems, checked against the converged fixpoint at
each mutating command. Thanks to the bounded disjunctive extension
(report.pdf, "Loop-carried aliasing"), these are reported as *certain*
violations rather than merely possible ones.

## Input format

Identical to the project PDF's examples:
```
var1 var2 ... varN

Lsrc  <command>   Ldst
Lsrc  <command>   Ldst
...
```

## Test suites

Each analysis has exactly 5 "interesting" test programs in its
`interesting-tests/` directory.

### Parity (`parity/interesting-tests/`)
1. `1-given-example.txt` — the project's own example; all asserts verify.
2. `2-decrement-precision-limit.txt` — **deliberately unverifiable**:
   `j := 4; i := j - 1; assert (ODD i)`. `i` is provably 3 (odd) by hand,
   but `assign_decr` only refines `i` when `j` is *provably odd*
   (`parity/analysis.py`) — odd values are never 0, so the "flip parity"
   rule is sound; otherwise (as here, where `j`'s tracked parity is EVEN)
   truncation at 0 could make the flip unsound, so `i` is forgotten
   instead. The domain only tracks `j`'s parity, not its exact value, so
   it cannot tell this EVEN `j` is actually the nonzero value 4 rather
   than 0 — a genuine, deliberate precision trade-off (see
   `TestDecrementIsConservative` / `TestDecrementRefinesWhenJIsOdd` in
   `tests/test_parity_analysis.py` for the odd-`j` case where the
   refinement *does* fire), not a bug. Reported "possibly violated"
   rather than a false "verified".
3. `3-negative-unconditional-even.txt` — **deliberately unverifiable**
   (`EVEN i` unconditionally isn't actually always true here); shows the
   tool correctly reports "possibly violated" rather than a false
   "verified".
4. `4-multihop-chain.txt` — a 3-variable renaming chain that exposed a
   completeness gap in naive per-disjunct entailment checking (fixed by
   exact coset enumeration; see `report.pdf` §2.4 / `parity/analysis.py`'s
   `check_assert` docstring).
5. `5-double-increment-nonrelational-ok.txt` — a case a plain
   *non-relational* parity domain would already get right; included as a
   contrast/control to show where the relational machinery is/isn't
   actually earning its keep.

Tests 1, 4, and 5 verify with no issues; tests 2 and 3 each contain a
deliberately unverifiable assertion (one because it's genuinely false,
one because of a documented precision limit) and are correctly reported
as "possibly violated".

### Shape (`shape/interesting-tests/`)
1. `1-given-example.txt` — the project's own example. All 9 assertions
   verify, with zero safety warnings.

   **Typo correction:** the project PDF's own listing of this example is
   internally inconsistent: `L31 t := yy.n L32` targets node `L32`, but the
   two edges that should originate there are instead labeled `L34 assume
   (t = NULL) L40` / `L34 assume (t != NULL) L35` — `L32` and `L34` never
   otherwise appear, so as written the CFG has a dangling node `L32` with no
   outgoing edges and an unreachable node `L34`. We use `L32` consistently
   for both (i.e. the two `assume` edges leave from `L32`, matching `L31`'s
   target) — the only reading that makes this part of the CFG well-formed
   and symmetric with the `xx`-traversal loop just above it.
2. `2-disjunct-cap-precision-limit.txt` — **deliberately unverifiable**:
   two pointers `p`, `q` are moved together (both reset to `NULL`, or
   both left alone) across 3 independent nondeterministic branches, so on
   all 8 concrete paths their nullity stays correlated — a genuine
   invariant, true by construction. `DisjunctiveShapeState` caps live
   per-branch states at `max_disjuncts=4` (`shape/domain.py`); the 3rd
   branch point forces 8 live states to be pairwise-joined back down to
   4, and whichever pair mixes a both-`NULL` path with a both-`NONNULL`
   path collapses `p`/`q`'s nullity independently to unknown, losing the
   correlation. A genuine, structural precision limit of the bounded
   disjunctive domain (with only 2 such branches — 4 disjuncts, within
   the cap — the same assertion correctly verifies), not a soundness bug.
3. `3-cycle-violation.txt` — PDF "Additional examples" #2: correctly
   reports a *certain* `CYCLE` violation.
4. `4-merge-aliasing.txt` — PDF "Additional examples" #3: safety holds
   and every assertion verifies except `t = yy.n`, which is genuinely
   false here (`yy.n` has been reset to `x`, a non-null node, while `t`
   is `NULL`) and is correctly reported as `POSSIBLY VIOLATED`. The added
   `assert (LS y xx)` correctly **verifies**: `yy.n := x` splices `x`'s
   list onto `y`'s tail, so `y` provably reaches `xx`.
5. `5-null-deref-negative.txt` — our own negative test: a pointer that's
   only *sometimes* re-nulled before a dereference; correctly reported as
   a `MEMORY SAFETY` violation.

## Verification tools (beyond the required test suites)

- `tests/` — `python -m unittest discover -s tests -v` runs the full unit
  test suite (126 tests as of this revision).
- `tests/monotonicity_check_parity.py` — exhaustive/randomized check that
  every parity transfer function is monotone.
- `tests/exhaustive_monotonicity_check.py` and
  `tests/monotonicity_check_liveness_filter.py` — the same property for
  shape's transfer functions, the latter specifically exercising the
  liveness filter used by the sharing check.
- `tests/differential_fuzz_shape.py` — ground-truth differential fuzzing:
  runs random programs through both a concrete heap simulator and the
  real abstract transfer functions, and checks every fact the abstract
  state claims against the concrete execution.
- `tests/loop_differential_check.py` — unrolls small loop programs
  against ground truth for trip counts 0–6, to validate the bounded
  disjunctive extension used for loop-carried precision.

### Running the verification tools

From the repository root:

```
python3 -m unittest discover -s tests -v            # 126 unit tests
python3 -m tests.monotonicity_check_parity          # parity, ~500k pairs (seconds)
python3 -m tests.exhaustive_monotonicity_check      # shape, ~1.5M pairs (several minutes)
python3 -m tests.monotonicity_check_liveness_filter # shape sharing-check liveness filter
python3 -m tests.differential_fuzz_shape            # shape, ground-truth (several minutes)
python3 tests/loop_differential_check.py            # shape, loop/disjunctive
```

## AI disclosure

AI assistance (Claude) was used during this project as a second opinion on
design decisions and implementation choices, and to generate additional
tests and help find edge cases in the analyses.
