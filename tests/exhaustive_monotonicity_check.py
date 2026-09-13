"""
Monotonicity checker for shape analysis state transformers.
Validates transfer(a) <= transfer(b) for random base states a <= b.
"""
import itertools
import os
import random
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape import domain as D
from shape.analysis import (
    do_field_assign,
    do_assign_deref,
    copy_row,
    reset_as_null,
    reset_as_fresh_new,
    free_var_slot,
    make_transfer,
)

EQ_VALS = [D.TRUE, D.FALSE, D.UNKNOWN]
REACH_VALS = [D.NO, D.ODD, D.EVEN, D.EITHER, D.TOP_REACH]


def random_state(n, fresh=(), max_tries=20):
    """`fresh` indices are forced to their guaranteed-unused starting
    shape (UNKNOWN eq to everything including the NULL coordinate, None
    succ, TOP_REACH both ways, claimed=False) -- this is what a real
    edge's dedicated `aux` slot always looks like before that edge's own
    transfer function ever touches it (see free_var_slot's docstring).
    Without this, an earlier version of this script let `aux` collide
    with a "real", independently-random variable's data, which produced
    several false-positive "violations" that were really just testing a
    state shape the real analyzer can never produce.

    The eq table is (n+1)x(n+1): index n is the reserved NULL coordinate
    (nullity[i] is just eq[i][n] now -- see domain.py). The raw random
    table is then closed via D.close_eq, since only closed tables are
    valid domain elements (see module docstring); if that collapses to
    BOTTOM, retry with a fresh random draw."""
    m = n + 1
    for _ in range(max_tries):
        eq = [
            [
                D.TRUE if i == j else (D.UNKNOWN if (i in fresh or j in fresh) else random.choice(EQ_VALS))
                for j in range(m)
            ]
            for i in range(m)
        ]
        for i in range(m):
            for j in range(i + 1, m):
                eq[j][i] = eq[i][j]
        succ_choices = list(range(n)) + ["NULL", None]
        succ = tuple(None if i in fresh else random.choice(succ_choices) for i in range(n))
        reach = [
            [
                D.TOP_REACH if (i in fresh or j in fresh) else random.choice(REACH_VALS)
                for j in range(n)
            ]
            for i in range(n)
        ]
        claimed = tuple(False if i in fresh else random.choice([True, False]) for i in range(n))
        s = D.State(n=n, eq=tuple(tuple(r) for r in eq),
                    succ=succ, reach=tuple(tuple(r) for r in reach), claimed=claimed)
        s = D.close_eq(s)
        if not D.is_bottom(s):
            return s
    return None  # gave up; caller skips this base


def single_field_weakenings(s, n):
    """Every state obtained by weakening exactly one field-cell of s.
    Weakening an eq cell (including the NULL coordinate, index n) can
    only ever produce a state that's still closed under substitution --
    UNKNOWN never propagates further -- so no need to re-close here."""
    m = n + 1
    for i in range(m):
        for j in range(m):
            if i != j and s.eq[i][j] != D.UNKNOWN:
                yield s.with_(eq=D._set2(s.eq, i, j, D.UNKNOWN))
    for i in range(n):
        if s.succ[i] is not None:
            yield s.with_(succ=D._set1(s.succ, i, None))
    for i in range(n):
        for j in range(n):
            if s.reach[i][j] != D.TOP_REACH:
                for other in REACH_VALS:
                    joined = D.reach_join(s.reach[i][j], other)
                    if joined != s.reach[i][j]:
                        yield s.with_(reach=D._set2(s.reach, i, j, joined))
    for i in range(n):
        if s.claimed[i] is False:
            yield s.with_(claimed=D._set1(s.claimed, i, True))


def random_multi_field_weakening(s, n, p=0.3):
    m = n + 1
    def we(v): return D.UNKNOWN if random.random() < p else v
    def ws(v): return None if random.random() < p else v
    def wr(v): return D.reach_join(v, random.choice(REACH_VALS)) if random.random() < p else v
    def wc(v): return True if (v is False and random.random() < p) else v
    eq = [[we(v) if i != j else D.TRUE for j, v in enumerate(row)] for i, row in enumerate(s.eq)]
    for i in range(m):
        for j in range(i + 1, m):
            eq[j][i] = eq[i][j]
    succ = tuple(ws(v) for v in s.succ)
    reach = [[wr(v) for v in row] for row in s.reach]
    claimed = tuple(wc(v) for v in s.claimed)
    return D.State(n=n, eq=tuple(tuple(r) for r in eq),
                    succ=succ, reach=tuple(tuple(r) for r in reach), claimed=claimed)


def report_violation(a, b, ra, rb, n):
    print("    --- a  ---", a)
    print("    --- b  ---", b)
    print("    --- f(a) ---", ra)
    print("    --- f(b) ---", rb)
    if D.is_bottom(rb):
        print("    f(b) is BOTTOM while f(a) is not -- a itself must not be BOTTOM here")
        return
    m = n + 1
    for i in range(m):
        for j in range(m):
            if rb.eq[i][j] != D.UNKNOWN and ra.eq[i][j] != rb.eq[i][j]:
                label = "NULLITY" if j == n and i != n else "EQ"
                print(f"    {label}[{i}][{j}] mismatch: {ra.eq[i][j]} vs {rb.eq[i][j]}")
    for i in range(n):
        for j in range(n):
            if not D.reach_leq(ra.reach[i][j], rb.reach[i][j]):
                print(f"    REACH[{i}][{j}] mismatch: {ra.reach[i][j]} vs {rb.reach[i][j]}")
        if rb.succ[i] is not None and ra.succ[i] != rb.succ[i]:
            print(f"    SUCC[{i}] mismatch: {ra.succ[i]} vs {rb.succ[i]}")
        if not rb.claimed[i] and ra.claimed[i]:
            print(f"    CLAIMED[{i}] mismatch: {ra.claimed[i]} vs {rb.claimed[i]}")


def check_function(name, fn, n, num_bases, seed, multi_per_base=8, fresh=()):
    random.seed(seed)
    checked = 0
    for _ in range(num_bases):
        a = random_state(n, fresh=fresh)
        if a is None:
            continue
        try:
            ra = fn(a)
        except Exception:
            continue
        candidates = list(single_field_weakenings(a, n))
        for _ in range(multi_per_base):
            candidates.append(random_multi_field_weakening(a, n))
        for b in candidates:
            if D.is_bottom(D.close_eq(b)) or not D.leq(a, b):
                continue
            try:
                rb = fn(b)
            except Exception:
                continue
            checked += 1
            if not D.leq(ra, rb):
                return checked, (a, b, ra, rb)
    return checked, None


def run():
    # x=0, y=1, z=2, w=3 (third-party witness variables) -- AUX MUST be a
    # 5th, dedicated index (4), never overlapping any of those, exactly
    # as the real analyze() guarantees (aux indices always start beyond
    # n_vars). Reusing index 3 as both "w" and "aux" (an earlier version
    # of this script's bug) tests a state shape that can never actually
    # arise, and produces false-positive "violations" that are really
    # just the aux slot colliding with an unrelated witness variable.
    n = 5
    AUX = 4
    num_bases = 4000

    transfer = make_transfer({"x": 0, "y": 1, "z": 2}, edge_aux={})

    def edge(cmd):
        return types.SimpleNamespace(cmd=cmd)

    # (name, fn, needs_fresh_aux)
    checks = [
        ("do_field_assign(x=0,y=1)", lambda s: do_field_assign(s, 0, 1), False),
        ("do_field_assign(x=0,y=NULL)", lambda s: do_field_assign(s, 0, None), False),
        (f"do_assign_deref(x=0,y=1,aux={AUX})", lambda s: do_assign_deref(s, 0, 1, AUX), True),
        (f"copy_row(dst=0,src=1,aux={AUX})", lambda s: copy_row(s, 0, 1, AUX), True),
        (f"reset_as_null(x=0,aux={AUX})", lambda s: reset_as_null(s, 0, AUX), True),
        (f"reset_as_fresh_new(x=0,aux={AUX})", lambda s: reset_as_fresh_new(s, 0, AUX), True),
        (f"free_var_slot(x=0,aux={AUX})", lambda s: free_var_slot(s, 0, AUX), True),
        ("transfer assume_eq(x,y)", lambda s: transfer(edge(("assume_eq", "x", "y")), s), False),
        ("transfer assume_neq_null(x)", lambda s: transfer(edge(("assume_neq_null", "x")), s), False),
        ("transfer assume_eq_null(x)", lambda s: transfer(edge(("assume_eq_null", "x")), s), False),
    ]

    any_violation = False
    for idx, (name, fn, needs_aux) in enumerate(checks):
        fresh = {AUX} if needs_aux else ()
        checked, violation = check_function(name, fn, n, num_bases, seed=1000 + idx, fresh=fresh)
        if violation:
            any_violation = True
            print(f"  {name}: VIOLATION  (found after {checked} pair checks)")
            report_violation(*violation, n)
        else:
            print(f"  {name}: clean  (checked {checked} pairs)")

    print()
    print("ALL CLEAN" if not any_violation else "VIOLATIONS FOUND")
    return 0 if not any_violation else 1


if __name__ == "__main__":
    sys.exit(run())
