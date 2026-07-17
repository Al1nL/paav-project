import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.cfg import parse_cfg
from common.fixpoint import run_fixpoint
from shape import domain as D
from shape.lang import parse_command


class Violation(Exception):
    def __init__(self, kind, edge):
        self.kind = kind
        self.edge = edge
        super().__init__(f"{kind} at: {edge.raw}")


def build_var_index(cfg):
    return {v: i for i, v in enumerate(cfg.variables)}


def eq_effective(s, x, y):
    """x=y taking nullity into account (NULL=NULL is True, NULL vs NONNULL
    is False) as well as the stored eq table."""
    if x == y:
        return D.TRUE
    nx, ny = s.nullity[x], s.nullity[y]
    if nx == D.NULL and ny == D.NULL:
        return D.TRUE
    if (nx == D.NULL) != (ny == D.NULL) and nx != D.TOP_NULLITY and ny != D.TOP_NULLITY:
        return D.FALSE
    return s.eq[x][y]


def free_var_slot(s, x, aux):
    """Before OVERWRITING variable x's identity (x:=y, x:=NULL, x:=new,
    x:=y.n), move x's current row/col to the dedicated scratch slot `aux`
    and redirect any dangling succ[w]==x references to aux too, then reset
    x to a fresh/unconstrained slot.

    Why this is needed (found via testing, not anticipated in the plan
    doc): `t.n := x; x := t` records succ[t] = (index of x). If we then
    overwrite x directly, succ[t] silently becomes self-referential
    nonsense (t.n ends up "pointing at whatever x is now" instead of the
    node x used to designate). This is the exact same aliasing hazard as
    Parity's `i := i + 1`, but arising here indirectly through the succ
    table across two separate commands rather than within one.

    Each CFG edge that overwrites a variable is given its OWN dedicated
    aux slot (computed once in analyze(), not round-robin/global), so the
    transfer function stays a pure function of (edge, input state) as
    required for a sound fixpoint iteration."""
    n = s.n
    s = D.set_nullity(s, aux, s.nullity[x])
    for z in range(n):
        if z == x or z == aux:
            continue
        s = D.set_eq(s, aux, z, s.eq[x][z])
    s = D.set_eq(s, aux, x, D.UNKNOWN)
    for z in range(n):
        if z == x or z == aux:
            continue
        s = D.set_reach(s, aux, z, s.reach[x][z])
        s = D.set_reach(s, z, aux, s.reach[z][x])
    s = D.set_reach(s, aux, aux, s.reach[x][x])
    s = D.set_succ(s, aux, s.succ[x])
    for w in range(n):
        if w != x and w != aux and s.succ[w] == x:
            s = D.set_succ(s, w, aux)
    # free slot x: fresh/unconstrained, ready for the caller's new value
    s = D.set_nullity(s, x, D.TOP_NULLITY)
    for z in range(n):
        if z == x:
            continue
        s = D.set_eq(s, x, z, D.UNKNOWN)
        s = D.set_reach(s, x, z, D.TOP_REACH)
        s = D.set_reach(s, z, x, D.TOP_REACH)
    s = D.set_reach(s, x, x, D.TOP_REACH)
    s = D.set_succ(s, x, None)
    return s


def copy_row(s, dst, src, aux=None):
    """dst becomes an alias of src: copy nullity/eq/reach/succ from src to
    dst (used by `x := y` and by `x := y.n` when y.n is a known variable).
    `aux`: dedicated scratch slot for this edge (see free_var_slot); pass
    None only when dst is known not to be dangling-referenced elsewhere."""
    if dst == src:
        return s  # already aliased; true no-op
    if aux is not None:
        # This also redirects any succ[w] == dst (including w == src) to
        # aux, so the copy below reads already-corrected values.
        s = free_var_slot(s, dst, aux)
    n = s.n
    nullity = D._set1(s.nullity, dst, s.nullity[src])
    eq = s.eq
    reach = s.reach
    for z in range(n):
        val = D.TRUE if z == src else s.eq[src][z]
        eq = D._set2(eq, dst, z, val)
        eq = D._set2(eq, z, dst, val)
        reach = D._set2(reach, dst, z, s.reach[src][z] if z != src else s.reach[src][src])
        reach = D._set2(reach, z, dst, s.reach[z][src])
    eq = D._set2(eq, dst, src, D.TRUE)
    eq = D._set2(eq, src, dst, D.TRUE)
    reach = D._set2(reach, dst, dst, s.reach[src][src])
    succ = D._set1(s.succ, dst, s.succ[src])
    return D.State(n=n, nullity=nullity, eq=eq, succ=succ, reach=reach)


def reset_as_null(s, x, aux=None):
    """x := NULL semantics on the tables (also reused when x.n turns out to
    be exactly NULL)."""
    if aux is not None:
        s = free_var_slot(s, x, aux)
    n = s.n
    s = D.set_nullity(s, x, D.NULL)
    s = D.set_succ(s, x, None)
    for z in range(n):
        if z == x:
            continue
        s = D.set_eq(s, x, z, D.UNKNOWN)
        s = D.set_reach(s, x, z, D.NO)
        s = D.set_reach(s, z, x, D.NO)
    s = D.set_reach(s, x, x, D.NO)
    return s


def reset_as_fresh_new(s, x, aux=None):
    """x := new: fresh node, provably distinct from every other currently
    tracked identity (see analysis rationale)."""
    if aux is not None:
        s = free_var_slot(s, x, aux)
    n = s.n
    s = D.set_nullity(s, x, D.NONNULL)
    s = D.set_succ(s, x, "NULL")
    for z in range(n):
        if z == x:
            continue
        s = D.set_eq(s, x, z, D.FALSE)
        s = D.set_reach(s, x, z, D.NO)
        s = D.set_reach(s, z, x, D.NO)
    s = D.set_reach(s, x, x, D.ODD)
    return s


def check_memory_safety(s, ptr_idx):
    """True iff dereferencing ptr_idx is NOT proven safe (nullity isn't
    definitely NONNULL) -- i.e. a violation."""
    return s.nullity[ptr_idx] != D.NONNULL


def would_create_cycle(s, x, y):
    """Would writing x.n := y close a cycle? True/False/'MAYBE' (unknown)."""
    if eq_effective(s, x, y) == D.TRUE:
        return True
    r = s.reach[y][x]
    if r in (D.ODD, D.EVEN, D.EITHER):
        return True
    if r == D.TOP_REACH:
        return "MAYBE"
    return False


def find_sharing_conflict(s, x, y):
    """Would writing x.n := y give node y a second incoming n-edge from a
    node other than x's? Returns the conflicting var index, or None."""
    n = s.n
    for z in range(n):
        if z == x or eq_effective(s, z, x) == D.TRUE:
            continue
        sz = s.succ[z]
        if sz is None or sz == "NULL":
            continue
        if sz == y or eq_effective(s, sz, y) == D.TRUE:
            return z
    return None


def do_field_assign(s, x, y):
    """Pure state transformer -- NEVER raises. If a precondition for a
    clean/sound mutation isn't proven, it falls back to returning the
    state unchanged (sound: claims nothing new, rather than claiming
    something false). Safety/cycle/sharing are reported separately by
    check_field_assign_issues(), evaluated once against the converged
    fixpoint (see module docstring / run notes)."""
    if check_memory_safety(s, x):
        return s  # can't soundly do anything; don't claim a mutation happened

    if y is None:  # x.n := NULL
        n = s.n
        s = D.set_succ(s, x, "NULL")
        for w in range(n):
            if w != x and eq_effective(s, x, w) == D.TRUE:
                s = D.set_succ(s, w, "NULL")
        return s

    if would_create_cycle(s, x, y) is not False:
        return s  # unsound to keep mutating a would-be-cyclic structure; skip
    if find_sharing_conflict(s, x, y) is not None:
        return s  # likewise for sharing

    n = s.n
    s = D.set_succ(s, x, y)
    for w in range(n):
        if w != x and eq_effective(s, w, x) == D.TRUE:
            s = D.set_succ(s, w, y)

    n = s.n
    if s.nullity[y] == D.NONNULL:
        s = _apply_splice(s, x, y)
    elif s.nullity[y] == D.TOP_NULLITY:
        # y's nullity is itself ambiguous (e.g. loop-iteration-count
        # uncertainty, not staleness -- see _apply_splice's docstring for
        # the distinction). Compute both possibilities and join them:
        # sound, and precise whenever both branches happen to agree.
        spliced = _apply_splice(s, x, y)
        s = _join_states_reach(s, spliced)
    return s


def _join_states_reach(a, b):
    n = a.n
    reach = tuple(
        tuple(D.reach_join(a.reach[i][j], b.reach[i][j]) for j in range(n))
        for i in range(n)
    )
    return a.with_(reach=reach)


def _apply_splice(s, x, y):
    n = s.n
    new_reach = [list(row) for row in s.reach]
    for z in range(n):
        rzx = s.reach[z][x]
        if rzx == D.NO:
            continue
        rzy_new = D.reach_shift(rzx)  # path z -> x -> y (one more hop)
        for w in range(n):
            ryw = s.reach[y][w]
            if ryw == D.NO:
                continue
            combined = D.combine_seq(rzy_new, ryw)
            # NOT a join with the old value: given the unshared-list
            # invariant (enforced by the sharing check above) plus the
            # "field was just reset to NULL" assumption, the old
            # reach[z][w] for exactly these (z reaches x, w reached
            # from y) pairs is PROVABLY NO -- z's forward chain was
            # fixed and dead-ended at x (null), so w (only reachable
            # via y's separate, unshared chain) couldn't have been
            # reachable from z before. Joining with that stale NO
            # would wrongly turn e.g. NO-join-EVEN into TOP; a direct
            # overwrite is exact here. (Found via testing: an earlier
            # join-based version silently destroyed precision through
            # every loop in the given example.)
            new_reach[z][w] = combined
    return s.with_(reach=tuple(tuple(r) for r in new_reach))


def check_field_assign_issues(s, x, y):
    """Post-hoc (converged-state) safety check for `x.n := y` (y=None
    means NULL). Returns a list of violation-kind strings (empty if OK)."""
    if D.is_bottom(s):
        return []  # unreachable code: vacuously safe
    issues = []
    if check_memory_safety(s, x):
        issues.append("MEMORY SAFETY (deref of possibly-NULL pointer)")
        return issues  # can't meaningfully check further
    if y is not None:
        cyc = would_create_cycle(s, x, y)
        if cyc is True:
            issues.append("CYCLE (would create a cyclic list)")
        elif cyc == "MAYBE":
            issues.append("POSSIBLE CYCLE (cannot rule out with this abstraction)")
        conflict = find_sharing_conflict(s, x, y)
        if conflict is not None:
            issues.append("SHARING (node would get a second incoming n-edge)")
    return issues


def _combine_prefer_derived(copied, derived):
    """Combine a value copied from an aliased slot with a value freshly
    re-derived from an already-established precise fact. If compatible
    (their outcome-sets overlap), intersect (sharper). If they genuinely
    disagree, trust `derived`: it's grounded directly in a fact we've
    already verified precise (e.g. reach[z][xx] right after xx was just
    assigned), whereas the copied value can carry staleness inherited
    from unrelated loop-iteration mixing elsewhere in the aliased slot's
    own history. (Found via testing: using a plain intersection/"meet"
    here silently fell back to NO on disagreement, destroying precision
    the same way the field-write splice bug did.)"""
    inter = D._REACH_SETS[copied] & D._REACH_SETS[derived]
    if inter:
        return D._SET_TO_REACH[frozenset(inter)]
    return derived


def do_assign_deref(s, x, y, aux=None):
    if check_memory_safety(s, y):
        return s  # unsafe to dereference y; don't claim a mutation happened

    succ_y = s.succ[y]
    if succ_y == "NULL":
        return reset_as_null(s, x, aux)
    if succ_y is not None:
        # x becomes an alias of the already-known successor `succ_y`. In
        # addition to copying succ_y's own facts, ALSO derive reach purely
        # from y's own (already-established) reach column via the shift
        # rule, and take the MEET (intersection) of both -- both are sound,
        # and whichever is sharper wins. This matters in practice: a node
        # discovered via one path (e.g. during list construction) can end
        # up with a less-precise row than what's re-derivable at the
        # point of use from the pointer we're dereferencing right now.
        n = s.n
        old_reach_col_y = [s.reach[z][y] for z in range(n)]
        out = copy_row(s, x, succ_y, aux)
        for z in range(n):
            derived = D.reach_shift(old_reach_col_y[z])
            out = D.set_reach(out, z, x, _combine_prefer_derived(out.reach[z][x], derived))
        return out

    # unknown successor: materialize x as the (fresh, newly-named) node y.n
    n = s.n
    old_reach_col_y = [s.reach[z][y] for z in range(n)]  # snapshot before any reset
    if aux is not None:
        s = free_var_slot(s, x, aux)
    s = D.set_nullity(s, x, D.TOP_NULLITY)
    s = D.set_succ(s, x, None)
    for z in range(n):
        if z != x:
            s = D.set_eq(s, x, z, D.UNKNOWN)
    for z in range(n):
        s = D.set_reach(s, z, x, D.reach_shift(old_reach_col_y[z]))
    for w in range(n):
        if w != x:
            s = D.set_reach(s, x, w, D.TOP_REACH)
    s = D.set_reach(s, x, x, D.TOP_REACH)  # nullity of x unknown; refined by later assume
    s = D.set_succ(s, y, x)
    return s


def check_assign_deref_issues(s, y):
    if D.is_bottom(s):
        return []
    if check_memory_safety(s, y):
        return ["MEMORY SAFETY (deref of possibly-NULL pointer)"]
    return []


def make_transfer(var_index, edge_aux):
    def transfer(edge, s):
        if D.is_bottom(s):
            return D.BOTTOM
        cmd = edge.cmd
        kind = cmd[0]
        aux = edge_aux.get(id(edge))
        if kind == "skip":
            return s
        if kind == "assign_null":
            _, x = cmd
            return reset_as_null(s, var_index[x], aux)
        if kind == "assign_new":
            _, x = cmd
            return reset_as_fresh_new(s, var_index[x], aux)
        if kind == "assign_var":
            _, x, y = cmd
            if x == y:
                return s
            return copy_row(s, var_index[x], var_index[y], aux)
        if kind == "assign_deref":
            _, x, y = cmd
            return do_assign_deref(s, var_index[x], var_index[y], aux)
        if kind == "field_assign":
            _, x, y = cmd
            yidx = None if y == "NULL" else var_index[y]
            return do_field_assign(s, var_index[x], yidx)
        if kind == "assume_true":
            return s
        if kind == "assume_false":
            return D.BOTTOM
        if kind == "assume_eq_null":
            _, x = cmd
            xi = var_index[x]
            if s.nullity[xi] == D.NONNULL:
                return D.BOTTOM
            return reset_as_null(s, xi)
        if kind == "assume_neq_null":
            _, x = cmd
            xi = var_index[x]
            if s.nullity[xi] == D.NULL:
                return D.BOTTOM
            s = D.set_nullity(s, xi, D.NONNULL)
            s = D.set_reach(s, xi, xi, D.ODD)
            return s
        if kind == "assume_eq":
            _, x, y = cmd
            xi, yi = var_index[x], var_index[y]
            n = s.n
            s = D.set_eq(s, xi, yi, D.TRUE)
            for z in range(n):
                s = D.set_reach(s, xi, z, D.reach_meet(s.reach[xi][z], s.reach[yi][z]))
                s = D.set_reach(s, z, xi, D.reach_meet(s.reach[z][xi], s.reach[z][yi]))
            if s.succ[xi] is None:
                s = D.set_succ(s, xi, s.succ[yi])
            elif s.succ[yi] is None:
                s = D.set_succ(s, yi, s.succ[xi])
            if s.nullity[xi] == D.TOP_NULLITY:
                s = D.set_nullity(s, xi, s.nullity[yi])
            elif s.nullity[yi] == D.TOP_NULLITY:
                s = D.set_nullity(s, yi, s.nullity[xi])
            return s
        if kind == "assume_neq":
            return s  # no general sound refinement
        if kind == "assert":
            return s
        raise ValueError(f"Unknown command kind: {kind}")

    return transfer


# ---------------------------------------------------------------------------
# assert checking
#
# Same idea as the Parity analysis's coset enumeration (see parity/analysis.py
# docstring): checking each disjunct's conjunction for entailment separately
# is sound but incomplete -- e.g. `assert (ODD x xx)(EVEN x xx)` just means
# "x reaches xx, parity unspecified" (reach[x][xx] == EITHER), and neither
# disjunct alone is entailed even though the OR is a tautology given that
# fact. We generalize: gather the *cells* (reach/nullity/eq entries) the
# formula actually depends on, enumerate every concrete combination each
# cell's abstract value admits, and require the ORC formula to hold for
# every combination. This stays exact because each cell already stores its
# outcome-set as one of a handful of named lattice elements (NO/ODD/EVEN/
# EITHER/TOP, etc.) rather than a bare "may/must" flag.
# ---------------------------------------------------------------------------

import itertools


def _gather_cells(orc, s, var_index):
    cells = {}

    def add_reach(x, y):
        key = ("reach", x, y)
        cells.setdefault(key, set(D._REACH_SETS[s.reach[x][y]]))

    def add_nullity(x):
        key = ("nullity", x)
        if key not in cells:
            v = s.nullity[x]
            cells[key] = {v} if v != D.TOP_NULLITY else {D.NULL, D.NONNULL}

    def add_eq(x, y):
        key = ("eq",) + tuple(sorted((x, y)))
        if key not in cells:
            v = eq_effective(s, x, y)
            cells[key] = {"EQ"} if v == D.TRUE else {"NEQ"} if v == D.FALSE else {"EQ", "NEQ"}

    def add_deref(x, y):
        sy = s.succ[y]
        if sy == "NULL":
            add_nullity(x)
        elif sy is not None:
            add_eq(x, sy)
        else:
            cells.setdefault(("deref", x, y), {"EQ", "NEQ"})

    for conj in orc:
        for kind, a, b in conj:
            if kind in ("LS", "NOLS", "ODD", "EVEN"):
                add_reach(var_index[a], var_index[b])
            elif kind in ("EQ_NULL", "NEQ_NULL"):
                add_nullity(var_index[a])
            elif kind in ("EQ", "NEQ"):
                add_eq(var_index[a], var_index[b])
            elif kind in ("EQ_DEREF", "NEQ_DEREF"):
                add_deref(var_index[a], var_index[b])
            else:
                raise ValueError(f"Unknown atom kind {kind}")
    return cells


def _atom_truth(atom, assignment, s, var_index):
    kind, a, b = atom
    if kind == "EQ_NULL":
        return assignment[("nullity", var_index[a])] == D.NULL
    if kind == "NEQ_NULL":
        return assignment[("nullity", var_index[a])] == D.NONNULL
    if kind == "EQ":
        x, y = var_index[a], var_index[b]
        return assignment[("eq",) + tuple(sorted((x, y)))] == "EQ"
    if kind == "NEQ":
        x, y = var_index[a], var_index[b]
        return assignment[("eq",) + tuple(sorted((x, y)))] == "NEQ"
    if kind == "LS":
        x, y = var_index[a], var_index[b]
        return assignment[("reach", x, y)] in (D.ODD, D.EVEN)
    if kind == "NOLS":
        x, y = var_index[a], var_index[b]
        return assignment[("reach", x, y)] == D.NO
    if kind in (D.ODD, D.EVEN):
        x, y = var_index[a], var_index[b]
        return assignment[("reach", x, y)] == kind
    if kind == "EQ_DEREF":
        x, y = var_index[a], var_index[b]
        sy = s.succ[y]
        if sy == "NULL":
            return assignment[("nullity", x)] == D.NULL
        elif sy is not None:
            return assignment[("eq",) + tuple(sorted((x, sy)))] == "EQ"
        return assignment[("deref", x, y)] == "EQ"
    if kind == "NEQ_DEREF":
        return not _atom_truth(("EQ_DEREF", a, b), assignment, s, var_index)
    raise ValueError(f"Unknown atom kind {kind}")


def _orc_holds(orc, assignment, s, var_index):
    for conj in orc:
        if all(_atom_truth(atom, assignment, s, var_index) for atom in conj):
            return True
    return False


def check_assert(s, orc, var_index):
    if D.is_bottom(s):
        return True
    cells = _gather_cells(orc, s, var_index)
    keys = list(cells.keys())
    domains = [sorted(cells[k]) for k in keys]

    # Consistency links: if ('reach', a, b1) and ('reach', a, b2) are both
    # being enumerated and succ[b1] == b2 exactly (b2 is b1's known,
    # single-hop successor), then reach[a][b2] must equal
    # shift(reach[a][b1]) in every real concrete state -- the two cells
    # aren't actually independent, even though they're stored as separate
    # table entries. Without this, enumeration invents impossible
    # combinations (found via testing: this alone caused a false
    # "possibly violated" on `assert (ODD x xx)(ODD x z)` in the given
    # example, since reach[x][xx] and reach[x][z] -- z being xx's direct
    # successor -- always move together but were checked independently).
    reach_keys = [k for k in keys if k[0] == "reach"]
    links = []
    for k1 in reach_keys:
        _, a1, b1 = k1
        for k2 in reach_keys:
            _, a2, b2 = k2
            if a1 == a2 and s.succ[b1] == b2:
                links.append((k1, k2))

    for combo in itertools.product(*domains):
        assignment = dict(zip(keys, combo))
        consistent = all(
            assignment[k2] == D.reach_shift(assignment[k1]) for k1, k2 in links
        )
        if not consistent:
            continue  # impossible combination; not a real counterexample
        if not _orc_holds(orc, assignment, s, var_index):
            return False
    return True


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def analyze(text: str):
    cfg = parse_cfg(text, parse_command)
    var_index = build_var_index(cfg)
    n_vars = len(cfg.variables)

    # One dedicated scratch slot per edge that overwrites a variable's
    # identity (assign_var/assign_null/assign_new/assign_deref) -- see
    # free_var_slot's docstring for why this must be per-edge (not a
    # single reused slot, and not a runtime round-robin counter).
    overwriting_kinds = {"assign_var", "assign_null", "assign_new", "assign_deref"}
    edge_aux = {}
    next_aux = n_vars
    for e in cfg.edges:
        if e.cmd[0] in overwriting_kinds:
            edge_aux[id(e)] = next_aux
            next_aux += 1
    n = next_aux  # total table dimension: named variables + all scratch slots

    transfer = make_transfer(var_index, edge_aux)
    pre, post_per_edge, iterations = run_fixpoint(
        cfg,
        bottom=lambda: D.BOTTOM,
        entry_state=D.initial_state(n),
        transfer=transfer,
        join=D.join,
        leq=D.leq,
    )

    assert_results = []
    safety_results = []
    for e in cfg.edges:
        cmd = e.cmd
        pre_state = pre[e.src]
        if cmd[0] == "assert":
            ok = check_assert(pre_state, cmd[1], var_index)
            assert_results.append((e, ok))
        elif cmd[0] == "field_assign":
            _, x, y = cmd
            yidx = None if y == "NULL" else var_index[y]
            issues = check_field_assign_issues(pre_state, var_index[x], yidx)
            if issues:
                safety_results.append((e, issues))
        elif cmd[0] == "assign_deref":
            _, x, y = cmd
            issues = check_assign_deref_issues(pre_state, var_index[y])
            if issues:
                safety_results.append((e, issues))

    return cfg, pre, assert_results, safety_results, iterations


def run_file(path):
    with open(path) as f:
        text = f.read()
    cfg, pre, assert_results, safety_results, iterations = analyze(text)
    print(f"=== {path} ===")
    print(f"(fixpoint reached in {iterations} worklist steps)")
    all_ok = True
    for e, issues in safety_results:
        all_ok = False
        for issue in issues:
            print(f"  [VIOLATION] {issue}: {e.raw}")
    for e, ok in assert_results:
        status = "VERIFIED" if ok else "POSSIBLY VIOLATED"
        if not ok:
            all_ok = False
        print(f"  [{status}] {e.raw}")
    print("RESULT:", "all safe & verified" if all_ok else "some issues found")
    print()
    return all_ok


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run_file(p)
