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
2. `2-independent-tautology.txt` — unrelated variables; checks the
   assert-checker doesn't over- or under-claim on trivial tautologies.
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

All five verify with no issues on the current code.

### Shape (`shape/interesting-tests/`)
1. `1-given-example.txt` — the project's own example. All 9 assertions
   verify, with zero safety warnings.
2. `2-sharing-violation.txt` — PDF "Additional examples" #1: correctly
   reports a *certain* `SHARING` violation; the assertions downstream of
   the now-invalid write correctly drop to unverified.
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
