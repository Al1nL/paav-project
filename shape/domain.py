"""
Custom lightweight relational shape domain for acyclic/unshared singly
linked lists (see plan doc section 2 for the full design rationale).

State is BOTTOM, or a tuple of tables. The variable-index space has n
"ordinary" coordinates (0..n-1: named program variables + scratch aux
slots), PLUS one extra reserved coordinate, index n itself, which stands
for the literal NULL value (see NULL_IDX below). eq/reach/succ are all
generalized to (n+1) x (n+1) *except* succ/reach conceptually don't need a
NULL row (NULL has no n-field and reaches nothing), but for uniformity the
tables are still sized against total dimension n+1; the NULL row of
succ/reach is simply never written or consulted.

  eq[x][y]    in {TRUE, FALSE, UNKNOWN}   (symmetric; eq[x][x] = TRUE)
              -- x == n (NULL_IDX) folded in as an ordinary coordinate:
              eq[x][NULL_IDX] IS "x's nullity" (TRUE=NULL, FALSE=NONNULL,
              UNKNOWN=TOP). There is deliberately no separate nullity
              table: nullity used to be stored redundantly alongside eq
              and reconciled only when READ (via a since-removed
              eq_effective helper), which is exactly what let the two
              views drift out of sync across a transformer and broke the
              monotonicity the fixpoint's soundness argument depends on
              (see report_v1's Section on the known limitation, and
              report_v2 for this fix). Folding NULL into eq as one more
              tracked identity removes the redundancy outright: there is
              only one fact, stored once.
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

_NULLITY_OF_EQ = {TRUE: NULL, FALSE: NONNULL, UNKNOWN: TOP_NULLITY}
_EQ_OF_NULLITY = {NULL: TRUE, NONNULL: FALSE, TOP_NULLITY: UNKNOWN}


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
    """Intersection of outcome-sets; sound refinement used by assume(x=y).
    Returns None on a genuine contradiction (disjoint outcome-sets) --
    e.g. one side claims a node trivially reaches itself (ODD) while the
    other claims it definitely doesn't (NO): if x and y are truly the
    same node, both facts must hold simultaneously, and there is no
    concrete store where an ODD-length self-segment and no self-segment
    at all are both true. Silently falling back to NO here (an earlier
    version of this function did) is unsound: NO is not a valid
    over-approximation of an empty set of possibilities, and letting a
    caller treat "contradiction" as if it were "provably NO" is exactly
    the kind of undetected-impossible-state bug that broke monotonicity
    elsewhere in this module (see _combine_prefer_derived, which had and
    was fixed for the identical issue). The caller must treat None as
    BOTTOM instead of writing it into the state."""
    inter = _REACH_SETS[a] & _REACH_SETS[b]
    if not inter:
        return None
    return _SET_TO_REACH[inter]


@dataclass(frozen=True)
class State:
    n: int  # count of ORDINARY coordinates (named vars + aux slots); the
             # eq table additionally has one reserved NULL row/col at index n
    eq: Tuple[Tuple[str, ...], ...]  # sized (n+1) x (n+1); see module docstring
    succ: Tuple[object, ...]         # sized n
    reach: Tuple[Tuple[str, ...], ...]  # sized n x n
    claimed: Tuple[bool, ...]        # sized n; see module docstring / find_sharing_conflict

    def with_(self, **kw):
        d = dict(eq=self.eq, succ=self.succ, reach=self.reach, n=self.n, claimed=self.claimed)
        d.update(kw)
        return State(**d)

    @property
    def nullity(self):
        """Derived, not stored: nullity[x] is exactly eq[x][NULL_IDX],
        recomputed on every access so it is definitionally impossible for
        it to disagree with eq (there is nothing else it could read)."""
        null_idx = self.n
        return tuple(_NULLITY_OF_EQ[self.eq[x][null_idx]] for x in range(self.n))


def null_idx(s) -> int:
    """The reserved eq-table coordinate standing for the literal NULL
    value. Always the last index, one past the ordinary variables/aux
    slots."""
    return s.n


def top(n: int) -> State:
    """The lattice TOP element: everything unconstrained. Used as the
    generic 'no information' value, e.g. inside domain-internal resets --
    NOT as the analysis's initial state (see initial_state below)."""
    m = n + 1
    eq = tuple(
        tuple(TRUE if i == j else UNKNOWN for j in range(m)) for i in range(m)
    )
    succ = tuple(None for _ in range(n))
    reach = tuple(
        tuple(TOP_REACH for _ in range(n)) for _ in range(n)
    )
    claimed = tuple(True for _ in range(n))  # unknown-but-treat-as-relevant is the safe default
    return State(n=n, eq=eq, succ=succ, reach=reach, claimed=claimed)


def initial_state(n: int, n_real: int = None) -> State:
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
    gets correctly superseded once the variable is genuinely assigned.

    `n_real` (defaults to n): only the first n_real coordinates are real,
    NULL-per-the-language-assumption program variables; any remaining
    coordinates up to n are internal scratch/aux slots (one per CFG edge
    that can overwrite a variable's identity -- see analysis.py's
    free_var_slot) and are NOT subject to that assumption -- they don't
    represent anything until first claimed by free_var_slot, so they
    start fully unconstrained (like top()), not NULL. Treating them as
    NULL like real variables was harmless under the old design (nothing
    ever checked for a contradiction on overwrite) but would spuriously
    look like a contradiction the first time such a slot is claimed, now
    that set_eq below actively detects contradictions.

    `claimed[i]`: True for real variables from the start (they are live,
    meaningful program variables from time zero, even while their value
    is NULL); False for aux slots until analysis.py's free_var_slot first
    populates one. This is what lets find_sharing_conflict/
    would_create_cycle soundly skip a candidate variable that provably
    isn't standing in for any real heap identity yet, instead of treating
    every one of the many still-untouched scratch slots as a plausible
    (if unknown) alias -- see find_sharing_conflict's docstring for why
    that mattered for precision."""
    if n_real is None:
        n_real = n
    m = n + 1
    null_idx_ = n

    def _is_real(i):
        return i < n_real or i == null_idx_  # the NULL coordinate is always "real"

    eq = tuple(
        tuple(
            TRUE if i == j else (TRUE if (_is_real(i) and _is_real(j)) else UNKNOWN)
            for j in range(m)
        )
        for i in range(m)
    )
    succ = tuple(None for _ in range(n))
    reach = tuple(
        tuple(NO for _ in range(n)) for _ in range(n)
    )  # NULL can neither reach nor be reached; harmless placeholder for
       # not-yet-claimed aux slots too, since set_reach never contradiction-checks
    claimed = tuple(i < n_real for i in range(n))
    return State(n=n, eq=eq, succ=succ, reach=reach, claimed=claimed)


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


def get_nullity(s, x):
    return _NULLITY_OF_EQ[get_eq(s, x, null_idx(s))]


def set_nullity(s, x, val):
    """Set x's nullity by setting eq[x][NULL_IDX] -- there is no separate
    table to keep in sync."""
    return set_eq(s, x, null_idx(s), _EQ_OF_NULLITY[val])


def set_eq(s, x, y, val):
    """Overwrite eq[x][y] (and its symmetric mirror), unconditionally.
    This intentionally does NOT check for contradiction against the old
    value: set_eq/set_nullity double as both a REFINEMENT operation
    (narrowing an unknown fact -- always safe) and a REASSIGNMENT
    operation (a variable's identity changing because the program
    genuinely reassigned it, e.g. reset_as_null/reset_as_fresh_new/
    free_var_slot -- where overwriting a stale old fact is exactly the
    point, not a contradiction to reject). Callers for which overwriting
    really would be a logical contradiction (specifically assume_eq,
    which represents a sound NARROWING and must reject narrowing to
    something already known false) are responsible for checking the old
    value themselves before calling this -- see make_transfer's
    assume_eq handling."""
    if is_bottom(s):
        return s
    e = _set2(s.eq, x, y, val)
    e = _set2(e, y, x, val)
    return s.with_(eq=e)


def set_reach(s, x, y, val):
    if is_bottom(s):
        return s
    return s.with_(reach=_set2(s.reach, x, y, val))


def set_succ(s, x, val):
    if is_bottom(s):
        return s
    return s.with_(succ=_set1(s.succ, x, val))


def set_claimed(s, x, val=True):
    """Mark coordinate x as denoting a real, live identity from here on
    (see State.claimed's docstring / find_sharing_conflict). Only ever
    needs to be called with val=True in practice (once claimed, a slot
    stays claimed until the whole state resets, and joins take care of
    combining across branches -- see join())."""
    if is_bottom(s):
        return s
    return s.with_(claimed=_set1(s.claimed, x, val))


def close_eq(s):
    """Close a state's eq table (which now also carries nullity, via the
    reserved NULL_IDX coordinate -- see module docstring) under the two
    inference rules that make "equals" behave like a genuine equivalence
    relation with substitution, over ALL tracked coordinates (not just the
    NULL one): for any x, y, z,
        eq[x][y]=TRUE and eq[y][z]=TRUE  => eq[x][z]=TRUE   (transitivity)
        eq[x][y]=TRUE and eq[y][z]=FALSE => eq[x][z]=FALSE  (substitution)
    iterated to a fixpoint (bounded: only ever adds information, table is
    finite, so this always terminates). If the same cell is ever forced to
    both TRUE and FALSE, the table describes a contradiction -- no
    concrete store fits -- and the whole state collapses to BOTTOM.

    This subsumes and replaces the narrower, NULL-specific patch attempted
    in an earlier revision (report_v1's "known limitation" section), which
    only propagated consistency between eq and a separately-stored
    nullity table and, being incomplete relative to full closure, itself
    introduced new monotonicity violations. Full closure is what a proper
    equality theory requires; a partial approximation of it does not
    inherit its soundness/monotonicity guarantees for free.

    Called once, at the true exit point of every function that returns a
    "finished" state (do_field_assign, do_assign_deref, copy_row,
    free_var_slot, reset_as_null, reset_as_fresh_new, join, and each
    assume_* transfer) -- NOT after every individual low-level set_eq call,
    both because that would be far more work than necessary (closure is
    idempotent: closing an already-closed table plus one new fact is what
    matters, not closing after every single cell write) and because
    "forgetting" a variable's identity (resetting a whole row/column to
    UNKNOWN, e.g. in free_var_slot) is a valid intermediate step that must
    NOT be immediately re-closed against stale neighbouring facts before
    the rest of that same reset has finished."""
    if is_bottom(s):
        return s
    m = s.n + 1
    eq = [list(row) for row in s.eq]
    changed = True
    while changed:
        changed = False
        for x in range(m):
            for y in range(m):
                # Only an EQUALITY hub (eq[x][y] = TRUE, i.e. x and y are
                # the same value) licenses propagation: whatever holds of
                # (y, z) then holds of (x, z) too, for either polarity.
                # eq[x][y] = FALSE (x != y) licenses nothing about z on
                # its own -- "x != y and y != z" does NOT imply x = z or
                # x != z in general (this domain has more than two
                # possible values); an earlier version of this closure
                # wrongly treated FALSE as a hub too and derived TRUE from
                # two unrelated disequalities, which produced spurious
                # BOTTOMs having nothing to do with a real contradiction.
                if x == y or eq[x][y] != TRUE:
                    continue
                for z in range(m):
                    if z == x or z == y:
                        continue
                    want = eq[y][z]
                    if want == UNKNOWN:
                        continue
                    if eq[x][z] == UNKNOWN:
                        eq[x][z] = want
                        eq[z][x] = want
                        changed = True
                    elif eq[x][z] != want:
                        return BOTTOM  # contradiction: no concrete store fits
    return s.with_(eq=tuple(tuple(r) for r in eq))


def _succ_alias_eq(eq_table, a_succ, b_succ):
    """Do two succ values denote the same fact, given a table that already
    knows which indices are aliases of each other? Exact match is the
    common case; two different indices also count if the table says
    they're the same node (TRUE, not just possibly/UNKNOWN). The literal
    "NULL" marker and a variable index whose nullity is provably NULL
    (eq[idx][NULL_IDX] == TRUE) are likewise just two spellings of the
    same fact -- both mean "this field is null" -- and must be recognized
    as such for the identical reason the rest of this alias-awareness
    exists: two states that reached the same real conclusion through
    different (but equally valid) representations must not look like
    they disagree (fuzzer-caught)."""
    if a_succ == b_succ:
        return True
    if isinstance(a_succ, int) and isinstance(b_succ, int):
        return eq_table[a_succ][b_succ] == TRUE
    null_idx_ = len(eq_table) - 1
    if a_succ == "NULL" and isinstance(b_succ, int):
        return eq_table[b_succ][null_idx_] == TRUE
    if b_succ == "NULL" and isinstance(a_succ, int):
        return eq_table[a_succ][null_idx_] == TRUE
    return False


def join(a, b) -> State:
    if is_bottom(a):
        return b
    if is_bottom(b):
        return a
    n = a.n
    m = n + 1
    eq = tuple(
        tuple(
            a.eq[i][j] if a.eq[i][j] == b.eq[i][j] else UNKNOWN
            for j in range(m)
        )
        for i in range(m)
    )
    # Close eq BEFORE deciding how succ merges: two succ values that look
    # different by raw index (a.succ[i]=3, b.succ[i]=5) must still be kept,
    # not dropped to "unknown", if the (closed) eq table already knows 3
    # and 5 are the same node -- otherwise a state that merely names the
    # same fact through a different index looks like it disagrees with
    # itself, which breaks monotonicity the same way the old separate
    # nullity table did (see report_v2's known-limitation writeup: this is
    # that exact gap, now closed the same way).
    closed_eq_state = close_eq(State(
        n=n, eq=eq, succ=tuple(None for _ in range(n)),
        reach=tuple(tuple(NO for _ in range(n)) for _ in range(n)),
        claimed=tuple(True for _ in range(n)),  # placeholder; unused by close_eq
    ))
    if is_bottom(closed_eq_state):
        return BOTTOM
    eq = closed_eq_state.eq
    succ = tuple(
        a.succ[i] if _succ_alias_eq(eq, a.succ[i], b.succ[i]) else None
        for i in range(n)
    )
    reach = tuple(
        tuple(reach_join(a.reach[i][j], b.reach[i][j]) for j in range(n))
        for i in range(n)
    )
    # claimed is a simple two-point lattice: False ("provably not yet a
    # real identity") is the precise/informative value, True (the
    # default/unknown-but-safe-to-check assumption) is its top. A branch
    # that has claimed[i] while its sibling hasn't means we can no longer
    # be SURE i is unclaimed in every represented execution, so the join
    # must keep it claimed (OR, not AND) -- the same direction as every
    # other "weaken on disagreement" rule in this function, just phrased
    # for a flag whose "more information" value happens to be False
    # instead of True.
    claimed = tuple(a.claimed[i] or b.claimed[i] for i in range(n))
    return State(n=n, eq=eq, succ=succ, reach=reach, claimed=claimed)


def leq(a, b) -> bool:
    if is_bottom(a):
        return True
    if is_bottom(b):
        return False
    n = a.n
    m = n + 1
    for i in range(m):
        for j in range(m):
            if b.eq[i][j] != UNKNOWN and a.eq[i][j] != b.eq[i][j]:
                return False
    for i in range(n):
        for j in range(n):
            if not reach_leq(a.reach[i][j], b.reach[i][j]):
                return False
        # b.succ[i] might be phrased through a different (but, according
        # to a's own more-informed eq table, already-known-aliased) index
        # than a.succ[i] -- that's not a disagreement, just two names for
        # the same fact. Only a genuine mismatch (not explained by a's own
        # aliasing knowledge) violates a <= b.
        if b.succ[i] is not None and not _succ_alias_eq(a.eq, a.succ[i], b.succ[i]):
            return False
        # claimed: b confidently saying "not yet a real identity" (False)
        # is the strong/precise claim (see join's comment); a, being at
        # least as precise, must agree if b makes that claim. If b's
        # claimed[i] is True (the default), a's value is unconstrained.
        if not b.claimed[i] and a.claimed[i]:
            return False
    return True

class DisjunctiveShapeState:
    def __init__(self, states=None, max_disjuncts=4):
        self.states = states if states is not None else []
        self.max_disjuncts = max_disjuncts

    def join(self, other):
        combined = list(self.states)
        for s in other.states:
            if s not in combined:
                combined.append(s)
        
        result = DisjunctiveShapeState(combined, self.max_disjuncts)
        result._reduce()
        return result

    def _reduce(self):
        # Use the standalone join() function defined in domain.py
        while len(self.states) > self.max_disjuncts:
            s1 = self.states.pop(0)
            s2 = self.states.pop(0)
            self.states.append(join(s1, s2)) 

    def __le__(self, other):
        # Use the standalone leq() function defined in domain.py
        for s1 in self.states:
            if not any(leq(s1, s2) for s2 in other.states):
                return False
        return True