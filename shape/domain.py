"""
Custom lightweight relational shape domain for acyclic/unshared singly
linked lists (see plan doc section 2 for the full design rationale).

State is BOTTOM, or a tuple of tables all indexed by program-variable
index (0..n-1), n fixed per program:

  nullity[x]  in {NULL, NONNULL, TOP}
  eq[x][y]    in {TRUE, FALSE, UNKNOWN}   (symmetric; eq[x][x] = TRUE)
  succ[x]     in {var-index, 'NULL', None}   (None = unknown; EXACT when known)
  reach[x][y] subset of {NO, ODD, EVEN}, represented by one of:
              NO, ODD, EVEN, EITHER({ODD,EVEN}), TOP({NO,ODD,EVEN})
              -- directed: "does/would a list segment go x -> y, and parity"

Plus two global sticky flags carried alongside the state during analysis
(reported, not part of the join lattice per se -- see analysis.py):
  a memory-safety violation, a cycle violation, a sharing violation.
These are detected AT the mutating command (assign_deref / field_assign)
by analysis.py using the tables below; the domain module only provides the
table primitives.
"""

from dataclasses import dataclass
from typing import Tuple

BOTTOM = "BOTTOM"

NULL, NONNULL, TOP_NULLITY = "NULL", "NONNULL", "TOP"
TRUE, FALSE, UNKNOWN = "TRUE", "FALSE", "UNKNOWN"
NO, ODD, EVEN, EITHER, TOP_REACH = "NO", "ODD", "EVEN", "EITHER", "TOP"

_REACH_SETS = {
    NO: frozenset({"NO"}),
    ODD: frozenset({"ODD"}),
    EVEN: frozenset({"EVEN"}),
    EITHER: frozenset({"ODD", "EVEN"}),
    TOP_REACH: frozenset({"NO", "ODD", "EVEN"}),
}
_SET_TO_REACH = {v: k for k, v in _REACH_SETS.items()}


def reach_join(a, b):
    if a == b:
        return a
    u = _REACH_SETS[a] | _REACH_SETS[b]
    # Our domain only names 5 outcome-sets (NO / ODD / EVEN / EITHER={ODD,EVEN}
    # / TOP={NO,ODD,EVEN}). {NO,ODD} and {NO,EVEN} have no dedicated name;
    # rounding up to TOP is a sound (if occasionally less precise) choice.
    return _SET_TO_REACH.get(frozenset(u), TOP_REACH)


def reach_leq(a, b):
    return _REACH_SETS[a] <= _REACH_SETS[b]


def reach_shift(a):
    """Extend a path by exactly one more hop/node (flips parity; NO stays NO;
    EITHER/TOP unaffected in kind)."""
    return {NO: NO, ODD: EVEN, EVEN: ODD, EITHER: EITHER, TOP_REACH: TOP_REACH}[a]


def combine_seq(a, b):
    """Splice two path facts sharing an endpoint (z->mid, mid->w) into a
    fact for z->w. NO takes priority (broken path); else TOP if either
    side unknown; else EITHER if either side ambiguous; else definite
    ODD/EVEN by parity arithmetic (same-parity halves -> ODD overall,
    since the shared midpoint is counted once, see plan doc 2.3)."""
    if a == NO or b == NO:
        return NO
    if a == TOP_REACH or b == TOP_REACH:
        return TOP_REACH
    if a == EITHER or b == EITHER:
        return EITHER
    return ODD if a == b else EVEN


def reach_meet(a, b):
    """Intersection of outcome-sets; sound refinement used by assume(x=y)."""
    inter = _REACH_SETS[a] & _REACH_SETS[b]
    if not inter:
        return NO  # defensive fallback; genuine contradictions are caught
        # earlier by explicit nullity/eq checks in analysis.py for the
        # cases these test programs actually exercise.
    return _SET_TO_REACH[inter]


@dataclass(frozen=True)
class State:
    n: int
    nullity: Tuple[str, ...]
    eq: Tuple[Tuple[str, ...], ...]
    succ: Tuple[object, ...]
    reach: Tuple[Tuple[str, ...], ...]

    def with_(self, **kw):
        d = dict(nullity=self.nullity, eq=self.eq, succ=self.succ, reach=self.reach, n=self.n)
        d.update(kw)
        return State(**d)


def top(n: int) -> State:
    """The lattice TOP element: everything unconstrained. Used as the
    generic 'no information' value, e.g. inside domain-internal resets --
    NOT as the analysis's initial state (see initial_state below)."""
    nullity = tuple(TOP_NULLITY for _ in range(n))
    eq = tuple(
        tuple(TRUE if i == j else UNKNOWN for j in range(n)) for i in range(n)
    )
    succ = tuple(None for _ in range(n))
    reach = tuple(
        tuple(TOP_REACH for _ in range(n)) for _ in range(n)
    )
    return State(n=n, nullity=nullity, eq=eq, succ=succ, reach=reach)


def initial_state(n: int) -> State:
    """The analysis's actual entry state: per the language spec (2.1,
    simplifying assumption 2), ALL pointer variables -- including ones the
    program hasn't explicitly assigned yet -- start out equal to NULL, not
    merely 'unconstrained'. Getting this wrong caused a real bug during
    development: a `new`-node's "I'm fresh, distinct from everything"
    fact was derived relative to variables that were technically still
    TOP/unconstrained (not yet used in the program text), and that stale
    NO-reachability fact then propagated onto them once they were finally
    assigned later, incorrectly suppressing real reachability facts. Since
    every never-yet-assigned variable is actually NULL (not merely
    unknown) from the start, that fact was simply true the whole time and
    gets correctly superseded once the variable is genuinely assigned."""
    nullity = tuple(NULL for _ in range(n))
    eq = tuple(
        tuple(TRUE for _ in range(n)) for _ in range(n)
    )  # all NULL == all equal to each other (the null value)
    succ = tuple(None for _ in range(n))
    reach = tuple(
        tuple(NO for _ in range(n)) for _ in range(n)
    )  # NULL can neither reach nor be reached
    return State(n=n, nullity=nullity, eq=eq, succ=succ, reach=reach)


def is_bottom(s) -> bool:
    return s == BOTTOM


# ---------- small table-update helpers (functional, return new tuples) ----------

def _set1(tup, i, val):
    lst = list(tup)
    lst[i] = val
    return tuple(lst)


def _set2(tup2, i, j, val):
    lst = [list(row) for row in tup2]
    lst[i][j] = val
    return tuple(tuple(row) for row in lst)


def get_eq(s, x, y):
    if x == y:
        return TRUE
    return s.eq[x][y]


def get_reach(s, x, y):
    if x == y:
        return s.reach[x][y]
    return s.reach[x][y]


def set_nullity(s, x, val):
    return s.with_(nullity=_set1(s.nullity, x, val))


def set_eq(s, x, y, val):
    e = _set2(s.eq, x, y, val)
    e = _set2(e, y, x, val)
    return s.with_(eq=e)


def set_reach(s, x, y, val):
    return s.with_(reach=_set2(s.reach, x, y, val))


def set_succ(s, x, val):
    return s.with_(succ=_set1(s.succ, x, val))


def join(a, b) -> State:
    if is_bottom(a):
        return b
    if is_bottom(b):
        return a
    n = a.n
    nullity = tuple(
        a.nullity[i] if a.nullity[i] == b.nullity[i] else TOP_NULLITY
        for i in range(n)
    )
    eq = tuple(
        tuple(
            a.eq[i][j] if a.eq[i][j] == b.eq[i][j] else UNKNOWN
            for j in range(n)
        )
        for i in range(n)
    )
    succ = tuple(
        a.succ[i] if a.succ[i] == b.succ[i] else None
        for i in range(n)
    )
    reach = tuple(
        tuple(reach_join(a.reach[i][j], b.reach[i][j]) for j in range(n))
        for i in range(n)
    )
    return State(n=n, nullity=nullity, eq=eq, succ=succ, reach=reach)


def leq(a, b) -> bool:
    if is_bottom(a):
        return True
    if is_bottom(b):
        return False
    n = a.n
    for i in range(n):
        if b.nullity[i] != TOP_NULLITY and a.nullity[i] != b.nullity[i]:
            return False
        for j in range(n):
            if b.eq[i][j] != UNKNOWN and a.eq[i][j] != b.eq[i][j]:
                return False
            if not reach_leq(a.reach[i][j], b.reach[i][j]):
                return False
        if b.succ[i] is not None and a.succ[i] != b.succ[i]:
            return False
    return True
