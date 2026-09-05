# Program Analysis and Verification — Final Project

Two static analyses, implemented in Python 3 (stdlib only):

- `impl/parity/` — Parity analysis for integer programs (Section 1)
- `impl/shape/`  — Shape analysis for acyclic/unshared linked lists (Section 2)
- `impl/common/` — Shared CFG parser + generic worklist fixpoint engine

See `report.pdf` for the formal domains, transfer functions, and
soundness/termination arguments. See `00-plan-and-decisions.md` for the
design-decision log written before/during implementation (kept as-is,
including a couple of "found via testing" corrections — see below).

## Requirements

Python 3.8+, no third-party packages.

## Running

From the `impl/` directory:

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
VIOLATED` (the analysis could not prove it — either because the assertion
is actually false on some path, or because of a precision limit of the
abstraction). Parity test 3 and shape test 4 each contain an assertion
that is genuinely false and is correctly reported as `POSSIBLY VIOLATED`.
Shape analysis additionally reports `VIOLATION` lines for memory-safety /
cycle / sharing problems, found by a post-hoc check against the converged
fixpoint at each mutating command.

## Input format

Identical to the project PDF's examples:
```
var1 var2 ... varN

Lsrc  <command>   Ldst
Lsrc  <command>   Ldst
...
```
One quirk found in the provided `example-list.txt`: the edge
`L31 t := yy.n L32` is followed by two `assume` edges written as sourced
from `L34` rather than `L32` — a typo, given the identical pattern is
written correctly (`L21`/`L22`) for the `x`-loop just above it. Our copy in
`shape/interesting-tests/1-given-example.txt` unifies both to `L32`.

## Test suites

Each analysis has exactly 5 "interesting" test programs in its
`interesting-tests/` directory (see the analysis-specific notes below).

### Parity (`impl/parity/interesting-tests/`)
1. `1-given-example.txt` — the project's own example; all asserts verify.
2. `2-independent-tautology.txt` — unrelated variables; checks the
   assert-checker doesn't over- or under-claim on trivial tautologies.
3. `3-negative-unconditional-even.txt` — **deliberately unverifiable**
   (`EVEN i` unconditionally isn't actually always true here); shows the
   tool correctly reports "possibly violated" rather than a false
   "verified".
4. `4-multihop-chain.txt` — a 3-variable renaming chain. This test is what
   *exposed* the need to replace naive per-disjunct entailment checking
   with exact coset enumeration (see `parity/analysis.py`'s
   `check_assert` docstring) — an earlier, more naive implementation
   wrongly reported this as unverifiable.
5. `5-double-increment-nonrelational-ok.txt` — a case a plain
   *non-relational* parity domain would already get right; included as a
   contrast/control to show where the relational machinery is/isn't
   actually earning its keep.

### Shape (`impl/shape/interesting-tests/`)
1. `1-given-example.txt` — the project's own example (typo-fixed, see
   above).
2. `2-sharing-violation.txt` — PDF "Additional examples" #1: correctly
   reports a SHARING violation.
3. `3-cycle-violation.txt` — PDF "Additional examples" #2: correctly
   reports a CYCLE violation (definite — the analysis proves `y` reaches
   `yy`, so `yy.n := y` must close a cycle).
4. `4-merge-aliasing.txt` — PDF "Additional examples" #3: safety holds and
   every assertion verifies except `t = yy.n`, which is genuinely false
   here (`yy.n` has been reset to `x`, a non-null node, while `t` is NULL)
   and is correctly reported as `POSSIBLY VIOLATED`. The added
   `assert (LS y xx)` — expected to exceed the abstraction's precision in
   the original write-up — now verifies: after `yy.n := x` splices `x`'s
   list onto `y`'s tail, `y` provably reaches `xx`.
5. `5-null-deref-negative.txt` — our own negative test: a pointer that's
   only *sometimes* re-nulled before a dereference; correctly reported as
   a MEMORY SAFETY violation.

## Shape analysis — resolved `y`-side precision gap

An earlier version verified the `x`-side assertions on the main example
(`LS x p`, `(ODD x xx)(EVEN x xx)`, `(ODD x xx)(ODD x z)`) but *not* the
structurally symmetric `y`-side ones (`LS y yy`, `t = yy.n`), even though
both are true — the `x`/`y` asymmetry discussed in the write-up and
slides. It is now fixed; all nine assertions on `1-given-example.txt`
verify.

Root cause: on the `y`-traversal, `succ[y]` briefly reads as `"NULL"`
early in the fixpoint (before `y`'s list is built), so `t := yy.n` takes
the "`y.n` is NULL" branch and writes `reach[y][t] = NO` into the merged
state at the loop head. A later join with the real value
(`reach[y][t] = shift(reach[y][yy])`) gives `TOP`, which monotone
iteration never recovers, and `yy := t` feeds it back to pin
`reach[y][yy] = TOP`. The `x`-side never hits this because `succ[x]` is
always a known non-NULL slot, so `t := xx.n` never takes the NULL branch.

Fix (`shape/analysis.py`): `do_assign_deref` now records the exact
`y.n == x` relationship on *every* branch, so it is representation-
identical on all paths and survives their CFG join instead of collapsing
`succ[y]` to `None`; a small `_resolve_succ` helper lets the succ-consuming
sites treat a successor that points at a proven-NULL slot identically to
`"NULL"`. The latter also removes some false-positive sharing conflicts
that had been suppressing precise list-building updates — which is why
shape test 3's cycle is now reported as definite and test 4's
`assert (LS y xx)` now verifies (both sound; see the per-test notes
above). Shape tests 2 and 5 are unchanged.

## AI-assisted development disclosure
