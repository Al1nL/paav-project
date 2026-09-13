"""
Dedicated monotonicity check for the liveness filter added to
find_sharing_conflict (see analysis.py's docstring / report.pdf for the
rationale). The existing exhaustive_monotonicity_check.py calls
do_field_assign(s, x, y) WITHOUT n_vars, which defaults to None and
disables the new filter entirely (falls back to the pre-existing
behavior) -- so it does not exercise this code path at all. This script
specifically passes n_vars < n so some coordinates count as "aux" for
the liveness computation, exactly as the real analyze() does.

Reuses random_state / single_field_weakenings / random_multi_field_weakening
/ check_function from exhaustive_monotonicity_check.py unchanged.

Usage: python tests/monotonicity_check_liveness_filter.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape import domain as D
from shape.analysis import do_field_assign, find_sharing_conflict
from tests.exhaustive_monotonicity_check import check_function


def run():
    n = 6          # 2 "named" variables (0,1) + 4 aux-like coordinates (2..5)
    n_vars = 2
    num_bases = 4000

    checks = [
        ("do_field_assign(x=0,y=1,n_vars=2)",
         lambda s: do_field_assign(s, 0, 1, n_vars)),
        ("do_field_assign(x=0,y=NULL,n_vars=2)",
         lambda s: do_field_assign(s, 0, None, n_vars)),
    ]

    any_violation = False
    for idx, (name, fn) in enumerate(checks):
        checked, violation = check_function(name, fn, n, num_bases, seed=5000 + idx)
        if violation:
            any_violation = True
            print(f"{name}: VIOLATION after {checked} pair checks")
            a, b, ra, rb = violation
            print("  a:", a)
            print("  b:", b)
            print("  f(a):", ra)
            print("  f(b):", rb)
        else:
            print(f"{name}: clean (checked {checked} pairs)")

    # Direct sanity check of the liveness reasoning itself: a "dangling
    # chain" scenario -- z2 has succ pointing at z3, z3 is not aliased to
    # anything and nothing points at IT -- so a real second in-edge
    # arriving only via z3 should still be flagged as at least MAYBE
    # (z2 -> z3 makes z3 live, one hop further than the simple 1-hop
    # heuristic would catch if it only checked "does anything point at
    # z3 directly" without transitive closure).
    print()
    print("Direct scenario check (2-hop dangling chain z0 -> z2 -> z3):")
    s = D.top(6)
    s = D.set_claimed(s, 0, True)
    s = D.set_claimed(s, 2, True)
    s = D.set_claimed(s, 3, True)
    s = D.set_nullity(s, 0, D.NONNULL)
    s = D.set_nullity(s, 2, D.NONNULL)
    s = D.set_nullity(s, 3, D.NONNULL)
    s = D.set_succ(s, 0, 2)   # var 0 (live, named) -> z2
    s = D.set_succ(s, 2, 3)   # z2 -> z3  (z3 only reachable via z2, which is live)
    s = D.close_eq(s)
    conflict = find_sharing_conflict(s, x=1, y=3, n_vars=2)
    print("  find_sharing_conflict(x=1,y=3) with a live 2-hop path to z3:", conflict)
    ok1 = conflict != None  # noqa: E711 -- must be "MAYBE" or a definite index, not None
    print("  PASS" if ok1 else "  FAIL (unsoundly ignored a live 2-hop candidate)")

    print()
    print("Direct scenario check (truly dangling z4, nothing points at it):")
    s2 = D.top(6)
    s2 = D.set_claimed(s2, 0, True)
    s2 = D.set_claimed(s2, 4, True)
    s2 = D.set_nullity(s2, 0, D.NONNULL)
    s2 = D.set_nullity(s2, 4, D.NONNULL)
    s2 = D.set_succ(s2, 4, None)  # z4's successor unresolved, but NOTHING live points at z4
    s2 = D.close_eq(s2)
    conflict2 = find_sharing_conflict(s2, x=1, y=3, n_vars=2)
    print("  find_sharing_conflict(x=1,y=3) with z4 truly dangling:", conflict2)
    ok2 = conflict2 is None
    print("  PASS (correctly ignored)" if ok2 else "  (conservatively still flagged MAYBE -- sound, just less precise here)")

    print()
    print("ALL CLEAN" if not any_violation and ok1 else "ISSUES FOUND")
    return 0 if (not any_violation and ok1) else 1


if __name__ == "__main__":
    sys.exit(run())
