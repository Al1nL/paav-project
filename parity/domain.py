"""
Relational parity domain: an affine subspace of GF(2)^n, n = #variables.

Representation ("generator form"): a state is either BOTTOM, or a pair
    (p: int bitmask, basis: tuple[int, ...])
meaning the solution set { p XOR (xor-combination of a subset of basis) }.
Bit i of an int corresponds to variable index i (parity of that variable:
0=Even, 1=Odd).

This form makes JOIN (affine hull of a union of two flats) cheap:
    join((p1,B1), (p2,B2)) = (p1, reduce(B1 + B2 + [p1^p2]))

Implication/entailment of a linear equation c.x = d, and the "meet with a
hyperplane" operation used by assignments/assume, are implemented via
Gaussian elimination over GF(2) on the basis (bit-parallel: vectors are
plain Python ints, XOR = addition mod 2, popcount&1 = dot product mod 2).
"""

from dataclasses import dataclass
from typing import Tuple


BOTTOM = "BOTTOM"


def popcount_parity(x: int) -> int:
    return bin(x).count("1") & 1


def dot(a: int, b: int) -> int:
    return popcount_parity(a & b)


@dataclass(frozen=True)
class State:
    p: int
    basis: Tuple[int, ...]   # always kept reduced: linearly independent,
                             # canonical (sorted by highest set bit desc.)


def _highest_bit(x: int) -> int:
    return x.bit_length() - 1


def reduce_basis(vectors):
    """Gaussian-eliminate a list of vectors (ints) into a canonical,
    linearly-independent, reduced (RREF-like) tuple, dropping zeros."""
    vecs = [v for v in vectors if v != 0]
    basis_by_pivot = {}  # pivot bit -> vector
    for v in vecs:
        cur = v
        while cur != 0:
            piv = _highest_bit(cur)
            if piv in basis_by_pivot:
                cur ^= basis_by_pivot[piv]
            else:
                basis_by_pivot[piv] = cur
                break
    # full reduction: eliminate each pivot's bit from all other rows (RREF)
    changed = True
    while changed:
        changed = False
        for piv, vec in list(basis_by_pivot.items()):
            for other_piv, other_vec in list(basis_by_pivot.items()):
                if other_piv == piv:
                    continue
                if (other_vec >> piv) & 1:
                    basis_by_pivot[other_piv] = other_vec ^ vec
                    changed = True
    result = tuple(sorted(basis_by_pivot.values(), key=lambda v: -_highest_bit(v)))
    return result


def in_span(vec: int, basis: Tuple[int, ...]) -> bool:
    cur = vec
    for v in basis:
        piv = _highest_bit(v)
        if (cur >> piv) & 1:
            cur ^= v
    return cur == 0


def top(n_dims: int) -> State:
    """n_dims = number of program variables PLUS the one reserved aux/scratch
    coordinate used internally by assign_linear for aliased assignments."""
    basis = tuple(1 << i for i in range(n_dims))
    return State(p=0, basis=reduce_basis(basis))


def is_bottom(s) -> bool:
    return s == BOTTOM


def meet_equation(s, c: int, d: int):
    """Intersect state s with the hyperplane {x : dot(c,x) = d}.
    Sound 'assume'/assignment-refinement primitive."""
    if is_bottom(s):
        return BOTTOM
    # find a basis vector v* with dot(c, v*) == 1
    pivot = None
    for v in s.basis:
        if dot(c, v) == 1:
            pivot = v
            break
    if pivot is None:
        # constraint doesn't touch the free directions: just check base point
        if dot(c, s.p) != d:
            return BOTTOM
        return s
    new_basis = []
    for v in s.basis:
        if v == pivot:
            continue
        if dot(c, v) == 1:
            v = v ^ pivot
        new_basis.append(v)
    new_p = s.p
    if dot(c, new_p) != d:
        new_p = new_p ^ pivot
    return State(p=new_p, basis=reduce_basis(new_basis))


def forget(s, var_idx: int):
    """Existentially quantify out variable var_idx (e.g. before an
    assignment to it): the var becomes fully free again."""
    if is_bottom(s):
        return BOTTOM
    e = 1 << var_idx
    return State(p=s.p, basis=reduce_basis(list(s.basis) + [e]))


def rename_coord(s, old_idx: int, new_idx: int):
    """Relabel coordinate old_idx -> new_idx (new_idx must currently be
    unused/free in s). This is a lossless isomorphism -- unlike `forget`,
    it does not erase any relation; it's the tool used to correctly
    handle aliased/self-referential assignments like `i := i + 1` (see
    assign_linear) without accidentally erasing relations to other
    variables that "forget" alone would destroy."""
    if is_bottom(s):
        return BOTTOM
    old_bit = 1 << old_idx
    new_bit = 1 << new_idx

    def move(x):
        if x & old_bit:
            x = (x & ~old_bit) | new_bit
        return x

    new_p = move(s.p)
    new_basis = list(move(v) for v in s.basis) + [old_bit]  # old slot now fully free
    return State(p=new_p, basis=reduce_basis(new_basis))


def assign_linear(s, var_idx: int, other_idx, flip: bool, aux_idx: int = None):
    """i := j (flip=False) or i := j+1 / j-1 (flip=True), other_idx=index of j.
    If other_idx is None, this is i := K (flip used as the K-parity bit).

    aux_idx: a scratch coordinate not used by any program variable, needed
    to correctly handle the aliased case i==j (e.g. `i := i + 1`) -- see
    rename_coord's docstring for why a naive forget-based approach is
    unsound-by-omission here (it silently entails the WRONG thing: it
    would make new-i's parity fully unconstrained instead of "old-i
    flipped").
    """
    if other_idx is None:
        s = forget(s, var_idx)
        if is_bottom(s):
            return BOTTOM
        e_i = 1 << var_idx
        return meet_equation(s, e_i, 1 if flip else 0)

    assert aux_idx is not None, "aux_idx required for variable-to-variable assignment"
    s = rename_coord(s, var_idx, aux_idx)  # snapshot old i (if aliased) into aux; i now free
    if is_bottom(s):
        return BOTTOM
    src_idx = aux_idx if other_idx == var_idx else other_idx
    e_i = 1 << var_idx
    e_src = 1 << src_idx
    c = e_i ^ e_src
    d = 1 if flip else 0
    s = meet_equation(s, c, d)
    s = forget(s, aux_idx)  # drop the scratch slot; correctly marginalizes it out
    return s


def join(a, b) -> State:
    if is_bottom(a):
        return b
    if is_bottom(b):
        return a
    combined = list(a.basis) + list(b.basis) + [a.p ^ b.p]
    return State(p=a.p, basis=reduce_basis(combined))


def leq(a, b) -> bool:
    """a <= b  iff  solutions(a) subseteq solutions(b)."""
    if is_bottom(a):
        return True
    if is_bottom(b):
        return False
    # a's base point must be in b's flat, and a's directions subseteq span(b)
    if not in_span(a.p ^ b.p, b.basis):
        return False
    for v in a.basis:
        if not in_span(v, b.basis):
            return False
    return True


def entails(s, c: int, d: int) -> bool:
    """Does state s guarantee the linear fact dot(c,x) = d for every
    concrete state it represents?"""
    if is_bottom(s):
        return True  # vacuously true: unreachable state
    for v in s.basis:
        if dot(c, v) != 0:
            return False
    return dot(c, s.p) == d
