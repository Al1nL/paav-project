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
VIOLATED` (the analysis could not prove it — either because it's a real
bug, or a real precision limit of the abstraction; test 3 in each suite is
a deliberate example of the former, and the shape suite's known limitation
below is an example of the latter). Shape analysis additionally reports
`VIOLATION` lines for memory-safety / cycle / sharing problems, found by a
post-hoc check against the converged fixpoint at each mutating command.

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
   reports a (possible) CYCLE violation.
4. `4-merge-aliasing.txt` — PDF "Additional examples" #3: the added
   `assert (LS y xx)` is correctly reported as unverifiable.
5. `5-null-deref-negative.txt` — our own negative test: a pointer that's
   only *sometimes* re-nulled before a dereference; correctly reported as
   a MEMORY SAFETY violation.

## Known precision limitation (shape analysis)

On the main example, the `x`-side assertions (`LS x p`,
`(ODD x xx)(EVEN x xx)`, `(ODD x xx)(ODD x z)`) all verify. The
structurally symmetric `y`-side assertions (`LS y yy`, `t = yy.n`) do not
verify, even though they are true. This was tracked down (see inline
comments in `shape/analysis.py` and the design-decisions log) to a
self-consistent but overly-coarse fixed point our domain reaches for the
`y`-traversal loop specifically — confirmed to be a genuine stable fixed
point (not an under-iterated computation) rather than an outright
soundness bug. We were not able to fully resolve the `x`/`y` asymmetry
within the project's time budget; it's flagged here rather than hidden,
and discussed as a concrete example of a real precision limit in the
write-up and slides.

## AI-assisted development disclosure
