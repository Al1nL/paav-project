"""
Property-based monotonicity check for the parity domain/analysis
(parity/domain.py, parity/analysis.py), in the same spirit as
tests/exhaustive_monotonicity_check.py for shape -- which is what caught
a real soundness bug in shape/analysis.py's assume_eq handling. No
equivalent existed for parity before this file; given that unit tests
alone missed that shape bug, parity's soundness claim was resting on
hand-argument + unit tests only, which is a materially weaker basis.

For every transfer function f and many random states a, and many random
WEAKENINGS b of a (a <= b by construction: b's solution set is a superset
of a's, built by adding extra basis vectors -- i.e. b admits every
solution a does, plus possibly more), checks f(a) <= f(b).

Usage (from repo root):
    python -m tests.monotonicity_check_parity
or:
    python tests/monotonicity_check_parity.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parity import domain as D
from parity.analysis import make_transfer


def random_state(n_dims, seed):
    """A uniformly-random affine subspace of GF(2)^n_dims: random point p,
    random rank-k basis (k itself random, 0..n_dims) of random vectors."""
    rnd = random.Random(seed)
    p = rnd.getrandbits(n_dims)
    k = rnd.randint(0, n_dims)
    raw = [rnd.getrandbits(n_dims) for _ in range(k)]
    basis = D.reduce_basis(raw)
    return D.State(p=p, basis=basis)


def single_weakenings(s, n_dims):
    """Every state obtained by adding exactly one extra, not-yet-spanned
    basis vector (every possible one) -- a minimal weakening."""
    if D.is_bottom(s):
        return
    for v in range(1, 1 << n_dims):
        if not D.in_span(v, s.basis):
            yield D.State(p=s.p, basis=D.reduce_basis(list(s.basis) + [v]))


def random_multi_weakening(s, n_dims, seed, max_extra=3):
    if D.is_bottom(s):
        return s
    rnd = random.Random(seed)
    extra = [rnd.getrandbits(n_dims) for _ in range(rnd.randint(1, max_extra))]
    return D.State(p=s.p, basis=D.reduce_basis(list(s.basis) + extra))


def le(a, b):
    return D.leq(a, b)


def check_function(name, fn, n_dims, num_bases, seed):
    checked = 0
    for i in range(num_bases):
        a = random_state(n_dims, seed * 100000 + i)
        try:
            ra = fn(a)
        except Exception:
            continue
        candidates = list(single_weakenings(a, n_dims))
        for j in range(6):
            candidates.append(random_multi_weakening(a, n_dims, seed * 1009 + i * 17 + j))
        for b in candidates:
            if not le(a, b):
                continue
            try:
                rb = fn(b)
            except Exception:
                continue
            checked += 1
            if not le(ra, rb):
                return checked, (a, b, ra, rb)
    return checked, None


def dump(label, s):
    print(f"  --- {label} ---")
    if D.is_bottom(s):
        print("    BOTTOM")
        return
    print(f"    p={s.p:#b} basis={[f'{v:#b}' for v in s.basis]}")


def run():
    n_vars = 3
    n_dims = n_vars + 1  # +1 reserved aux coordinate, matching analyze()'s convention
    var_index = {"i": 0, "j": 1, "k": 2}
    aux_idx = 3
    transfer = make_transfer(var_index, aux_idx)

    def edge(cmd):
        class E:
            pass
        e = E()
        e.cmd = cmd
        return e

    checks = [
        ("assign_var(i:=j)", lambda s: transfer(edge(("assign_var", "i", "j")), s)),
        ("assign_var(i:=i) [self-alias]", lambda s: transfer(edge(("assign_var", "i", "i")), s)),
        ("assign_incr(i:=j+1)", lambda s: transfer(edge(("assign_incr", "i", "j")), s)),
        ("assign_incr(i:=i+1) [self-alias]", lambda s: transfer(edge(("assign_incr", "i", "i")), s)),
        ("assign_decr(i:=j-1)", lambda s: transfer(edge(("assign_decr", "i", "j")), s)),
        ("assign_const(i:=4)", lambda s: transfer(edge(("assign_const", "i", 4)), s)),
        ("assign_const(i:=5)", lambda s: transfer(edge(("assign_const", "i", 5)), s)),
        ("assign_nondet(i:=?)", lambda s: transfer(edge(("assign_nondet", "i")), s)),
        ("assume_eq(i=j)", lambda s: transfer(edge(("assume_eq", "i", "j")), s)),
        ("assume_eq_const(i=4)", lambda s: transfer(edge(("assume_eq_const", "i", 4)), s)),
        ("assume_neq(i!=j)", lambda s: transfer(edge(("assume_neq", "i", "j")), s)),
        ("assume_true", lambda s: transfer(edge(("assume_true",)), s)),
        ("assume_false", lambda s: transfer(edge(("assume_false",)), s)),
        ("D.rename_coord(0->3) direct", lambda s: D.rename_coord(s, 0, 3)),
        ("D.forget(0) direct", lambda s: D.forget(s, 0)),
        ("D.join(s, top) direct", lambda s: D.join(s, D.top(n_dims))),
    ]

    any_violation = False
    for idx, (name, fn) in enumerate(checks):
        checked, violation = check_function(name, fn, n_dims, num_bases=1500, seed=3000 + idx)
        if violation:
            any_violation = True
            a, b, ra, rb = violation
            print(f"{name}: VIOLATION (after {checked} pair checks)")
            dump("a", a)
            dump("b", b)
            dump("f(a)", ra)
            dump("f(b)", rb)
        else:
            print(f"{name}: clean (checked {checked} pairs)")

    print()
    print("ALL CLEAN" if not any_violation else "VIOLATIONS FOUND")
    return 0 if not any_violation else 1


if __name__ == "__main__":
    sys.exit(run())
