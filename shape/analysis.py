import glob
import os
import sys

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
    """x=y, read straight from the (closed) eq table. Nullity is folded
    into this same table via the reserved NULL coordinate (see
    domain.py's module docstring), so there is no separate fact to
    reconcile here anymore -- this function is kept only so existing call
    sites don't need to change."""
    return D.get_eq(s, x, y)


def free_var_slot(s, x, aux):
    """Save variable x's identity into scratch slot `aux` before x is overwritten."""
    if D.is_bottom(s):
        return D.BOTTOM
    n = s.n
    # aux is reused across fixpoint iterations of the SAME edge, so it may
    # still carry a stale definite fact from a previous iteration; clear it
    # to UNKNOWN before copying x's CURRENT identity in, or that stale
    # leftover could spuriously look like a contradiction against the new
    # (possibly different) value being copied in.
    for z in range(n + 1):
        if z == x or z == aux:
            continue
        s = D.set_eq(s, aux, z, D.UNKNOWN)
    for z in range(n + 1):
        if z == x or z == aux:
            continue
        s = D.set_eq(s, aux, z, D.get_eq(s, x, z))
    s = D.set_eq(s, aux, x, D.UNKNOWN)
    for z in range(n):
        if z == x or z == aux:
            continue
        s = D.set_reach(s, aux, z, s.reach[x][z])
        s = D.set_reach(s, z, aux, s.reach[z][x])
    s = D.set_reach(s, aux, aux, s.reach[x][x])
    s = D.set_succ(s, aux, s.succ[x])
    s = D.set_claimed(s, aux, True)  # Mark scratch slot as claimed (active identity)
    for w in range(n):
        if w != x and w != aux and s.succ[w] == x:
            s = D.set_succ(s, w, aux)
    # free slot x: fresh/unconstrained, ready for the caller's new value
    for z in range(n + 1):
        if z == x:
            continue
        s = D.set_eq(s, x, z, D.UNKNOWN)
    for z in range(n):
        if z == x:
            continue
        s = D.set_reach(s, x, z, D.TOP_REACH)
        s = D.set_reach(s, z, x, D.TOP_REACH)
    s = D.set_reach(s, x, x, D.TOP_REACH)
    s = D.set_succ(s, x, None)
    return D.close_eq(s)


def copy_row(s, dst, src, aux=None):
    """Copy pointer identity from src to dst (`dst := src` or `dst := src.n`)."""
    if D.is_bottom(s):
        return D.BOTTOM
    if dst == src:
        # Populate aux as an exact snapshot of dst's unchanged identity to preserve its slot allocation.
        if aux is not None:
            n = s.n
            for z in range(n + 1):
                if z == dst or z == aux:
                    continue
                s = D.set_eq(s, aux, z, D.UNKNOWN)
            for z in range(n + 1):
                if z == dst or z == aux:
                    continue
                s = D.set_eq(s, aux, z, D.get_eq(s, dst, z))
            # aux is an exact alias of dst when dst is not overwritten
            s = D.set_eq(s, aux, dst, D.TRUE)
            for z in range(n):
                if z == dst or z == aux:
                    continue
                s = D.set_reach(s, aux, z, s.reach[dst][z])
                s = D.set_reach(s, z, aux, s.reach[z][dst])
            s = D.set_reach(s, aux, aux, s.reach[dst][dst])
            s = D.set_reach(s, aux, dst, s.reach[dst][dst])
            s = D.set_reach(s, dst, aux, s.reach[dst][dst])
            s = D.set_succ(s, aux, s.succ[dst])
            s = D.set_claimed(s, aux, True)
            s = D.close_eq(s)
            if D.is_bottom(s):
                return s
        return s
    if aux is not None:
        # Redirect dangling succ references of dst to aux before overwrite
        s = free_var_slot(s, dst, aux)
        if D.is_bottom(s):
            return s
    n = s.n
    for z in range(n + 1):  # defensively clear dst first (see free_var_slot)
        if z == dst:
            continue
        s = D.set_eq(s, dst, z, D.UNKNOWN)
    for z in range(n + 1):  # + NULL coordinate: copies src's nullity too
        if z == dst:
            continue
        s = D.set_eq(s, dst, z, D.get_eq(s, src, z))
    for z in range(n):
        if z == dst:
            continue
        s = D.set_reach(s, dst, z, s.reach[src][z] if z != src else s.reach[src][src])
        s = D.set_reach(s, z, dst, s.reach[z][src])
    s = D.set_reach(s, dst, dst, s.reach[src][src])
    s = D.set_succ(s, dst, s.succ[src])
    return D.close_eq(s)


def reset_as_null(s, x, aux=None):
    """x := NULL semantics on the tables (also reused when x.n turns out to
    be exactly NULL)."""
    if D.is_bottom(s):
        return D.BOTTOM
    if aux is not None:
        s = free_var_slot(s, x, aux)
        if D.is_bottom(s):
            return s
    n = s.n
    for z in range(n):
        if z == x:
            continue
        s = D.set_eq(s, x, z, D.UNKNOWN)  # forget stale relations first
    s = D.set_nullity(s, x, D.NULL)
    s = D.set_succ(s, x, None)
    for z in range(n):
        if z == x:
            continue
        s = D.set_reach(s, x, z, D.NO)
        s = D.set_reach(s, z, x, D.NO)
    s = D.set_reach(s, x, x, D.NO)
    # close_eq re-derives "x equals every other NULL var" (and "x differs
    # from every other NONNULL var") automatically from the single fact
    # eq[x][NULL] = TRUE just established -- no need to loop over every
    # other variable's nullity by hand.
    return D.close_eq(s)


def reset_as_fresh_new(s, x, aux=None):
    """x := new: initialize x as a fresh, distinct node."""
    if D.is_bottom(s):
        return D.BOTTOM
    if aux is not None:
        s = free_var_slot(s, x, aux)
        if D.is_bottom(s):
            return s
    n = s.n
    for z in range(n):
        if z == x:
            continue
        s = D.set_eq(s, x, z, D.UNKNOWN)  # forget stale relations first
    for z in range(n):
        if z == x:
            continue
        # Mark coordinate x distinct from all other coordinates z
        s = D.set_eq(s, x, z, D.FALSE)
        s = D.set_reach(s, x, z, D.NO)
        s = D.set_reach(s, z, x, D.NO)
    s = D.set_nullity(s, x, D.NONNULL)
    s = D.set_succ(s, x, "NULL")
    s = D.set_reach(s, x, x, D.ODD)
    return D.close_eq(s)


def check_memory_safety(s, ptr_idx):
    """True iff dereferencing ptr_idx is NOT proven safe (nullity isn't
    definitely NONNULL) -- i.e. a violation."""
    return s.nullity[ptr_idx] != D.NONNULL


def _resolve_succ(s, w):
    """Get successor of w, resolving variable indices that are NULL to "NULL"."""
    v = s.succ[w]
    if isinstance(v, int) and s.nullity[v] == D.NULL:
        return "NULL"
    return v


def would_create_cycle(s, x, y):
    """Check if writing x.n := y would create a cycle (True, False, or 'MAYBE')."""
    ex = eq_effective(s, x, y)
    if ex == D.TRUE:
        return True
    r = s.reach[y][x]
    if r in (D.ODD, D.EVEN, D.EITHER):
        return True
    if r == D.TOP_REACH or ex == D.UNKNOWN:
        return "MAYBE"
    return False


def _compute_live(s, n_vars):
    """Compute the set of coordinates reachable from a named program variable via eq-aliasing."""
    n = s.n
    live = set(range(min(n_vars, n)))
    changed = True
    while changed:
        changed = False
        for z in list(live):
            for w in range(n):
                if w not in live and D.get_eq(s, z, w) != D.FALSE:
                    live.add(w)
                    changed = True
    return live


def find_sharing_conflict(s, x, y, n_vars=None):
    """Check if writing x.n := y introduces a sharing conflict (var index, 'MAYBE', or None)."""
    n = s.n
    live = _compute_live(s, n_vars) if n_vars is not None else set(range(n))
    maybe = False
    for z in range(n):
        if z == x or eq_effective(s, z, x) == D.TRUE:
            continue
        if s.nullity[z] == D.NULL or not s.claimed[z]:
            continue
        if z not in live:
            continue
        sz = _resolve_succ(s, z)
        if sz == "NULL":
            continue
        if sz is None:
            maybe = True
            continue
        if sz == y or eq_effective(s, sz, y) == D.TRUE:
            return z
        if eq_effective(s, sz, y) == D.UNKNOWN:
            maybe = True
    return "MAYBE" if maybe else None


def _propagate_succ_to_aliases(s, x, new_val):
    """After x's node's n-field is set to `new_val`, every OTHER variable
    w that denotes the *same* node also has w.n == new_val now. If w is a
    PROVEN alias of x (eq_effective TRUE), set succ[w] := new_val
    directly. If it's merely POSSIBLE that w aliases x (eq_effective
    UNKNOWN), we can't soundly assert succ[w] := new_val outright --
    that's only true on the "w is x" branch, and simply leaving succ[w]
    untouched implicitly assumes the opposite ("w isn't x") branch
    instead, which is just as unsound the other way. The sound answer is
    the join of both branches: new_val if that already happens to match
    w's old succ (both branches agree), else "unknown" (None). Skipping
    this and leaving a stale concrete succ[w] behind on the UNKNOWN case
    breaks monotonicity: a state that has since forgotten whether w
    aliases x must not produce a MORE certain-looking result than one
    that still remembers it does (caught by exhaustive fuzzing)."""
    n = s.n
    for w in range(n):
        if w == x:
            continue
        ew = eq_effective(s, x, w)
        if ew == D.TRUE:
            s = D.set_succ(s, w, new_val)
        elif ew == D.UNKNOWN and s.succ[w] != new_val:
            s = D.set_succ(s, w, None)
    return s


def do_field_assign(s, x, y, n_vars=None):
    """State transformer for field assignment `x.n := y` (y=None for NULL)."""
    if D.is_bottom(s):
        return D.BOTTOM
    if s.nullity[x] == D.NULL:
        return D.BOTTOM  # every execution reaching here crashes
    if s.nullity[x] == D.TOP_NULLITY:
        s = D.set_nullity(s, x, D.NONNULL)  # only the non-crash branch survives
        s = D.close_eq(s)
        if D.is_bottom(s):
            return s

    if y is None:  # x.n := NULL
        n = s.n
        old_reach_row_x = [s.reach[x][w] for w in range(n)]  # snapshot before cutting
        s = D.set_succ(s, x, "NULL")
        s = _propagate_succ_to_aliases(s, x, "NULL")
        # Sever x's outgoing reachability, preserving reflexive facts for aliases of x.
        for w in range(n):
            if w == x:
                continue
            alias = D.get_eq(s, x, w)
            if alias == D.TRUE:
                continue
            elif alias == D.FALSE:
                s = D.set_reach(s, x, w, D.NO)
            else:
                s = D.set_reach(s, x, w, D.TOP_REACH)
        # Update reachability for variables z that previously reached nodes downstream of x.
        for z in range(n):
            if z == x:
                continue
            rzx = s.reach[z][x]
            if rzx == D.NO:
                continue
            z_certainly_reaches_x = rzx != D.TOP_REACH
            for w in range(n):
                if w == x or w == z:
                    continue
                # Preserve reflexive reachability if w is an alias of x
                eq_xw = D.get_eq(s, x, w)
                if eq_xw == D.TRUE:
                    continue
                eq_zw = D.get_eq(s, z, w)
                if eq_zw == D.TRUE:
                    continue  # w IS z; reach[z][w] is the trivial reflexive fact, unaffected
                if old_reach_row_x[w] == D.NO:
                    continue  # x's old chain never reached w; genuinely nothing to cut
                # Only sever reachability if z's path to w definitely passed through x
                old_certainly_reached = old_reach_row_x[w] != D.TOP_REACH
                certain = (z_certainly_reaches_x and eq_xw == D.FALSE
                           and eq_zw == D.FALSE and old_certainly_reached)
                if certain:
                    s = D.set_reach(s, z, w, D.NO)
                else:
                    s = D.set_reach(s, z, w, D.reach_join(s.reach[z][w], D.NO))
        return s

    n = s.n
    s = D.set_succ(s, x, y)
    s = _propagate_succ_to_aliases(s, x, y)

    n = s.n
    cyc = would_create_cycle(s, x, y)
    conflict = find_sharing_conflict(s, x, y, n_vars)
    definite_break = (cyc is True) or (conflict is not None and conflict != "MAYBE")
    maybe_break = (cyc == "MAYBE") or (conflict == "MAYBE")

    if definite_break or maybe_break:
        # Weaken reachability if this write breaks acyclic/unshared invariants
        s = _weaken_for_uncertain_write(s, x, y, n)
    else:
        s = _apply_splice_best_effort(s, x, y)
    return s


def _apply_splice_best_effort(s, x, y):
    """Apply the exact splice for x.n := y, additionally accounting for
    y's own nullity uncertainty (join both branches when y's nullity is
    TOP) but NOT for cycle/sharing uncertainty; the caller handles that
    dimension separately, since the two kinds of uncertainty are
    independent and each needs its own join."""
    if s.nullity[y] == D.NONNULL:
        return _apply_splice(s, x, y)
    if s.nullity[y] == D.TOP_NULLITY:
        # y's nullity is itself ambiguous. Compute both possibilities and join them
        spliced = _apply_splice(s, x, y)
        return _join_states_reach(s, spliced)
    return s  # y is NULL: nothing to splice


def _weaken_for_uncertain_write(s, x, y, n):
    """Weaken reachability table when a field write breaks acyclic/unshared invariants."""
    old_reach_col_x = [s.reach[z][x] for z in range(n)]
    old_reach_row_y = [s.reach[y][w] for w in range(n)]
    for z in range(n):
        s = D.set_reach(s, x, z, D.TOP_REACH)
        s = D.set_reach(s, z, x, D.TOP_REACH)
    for z in range(n):
        if old_reach_col_x[z] == D.NO:
            continue
        for w in range(n):
            if old_reach_row_y[w] == D.NO:
                continue
            s = D.set_reach(s, z, w, D.TOP_REACH)
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
            new_reach[z][w] = combined
    return s.with_(reach=tuple(tuple(r) for r in new_reach))


def check_field_assign_issues(s, x, y, n_vars=None):
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
        conflict = find_sharing_conflict(s, x, y, n_vars)
        if conflict == "MAYBE":
            issues.append("POSSIBLE SHARING (cannot rule out with this abstraction)")
        elif conflict is not None:
            issues.append("SHARING (node would get a second incoming n-edge)")
    return issues


def _combine_prefer_derived(copied, derived):
    """Combine a copied reachability value with a freshly re-derived value."""
    inter = D._REACH_SETS[copied] & D._REACH_SETS[derived]
    if inter:
        return D._SET_TO_REACH[frozenset(inter)]
    return None


def do_assign_deref(s, x, y, aux=None):
    if D.is_bottom(s):
        return D.BOTTOM
    # Same memory-safety refinement as do_field_assign: a provably-NULL y
    # means every execution reaching this edge crashes (BOTTOM); a merely
    # unproven-safe y means the only surviving branch requires y to
    # actually be NONNULL, so narrow to that instead of freezing the
    # whole state.
    if s.nullity[y] == D.NULL:
        return D.BOTTOM
    if s.nullity[y] == D.TOP_NULLITY:
        s = D.set_nullity(s, y, D.NONNULL)
        s = D.close_eq(s)
        if D.is_bottom(s):
            return s

    succ_y = s.succ[y]
    if succ_y == "NULL":
        out = reset_as_null(s, x, aux)
        if D.is_bottom(out):
            return out
        # Record `succ[y] = x` even though x is NULL on this path, so the
        # exact `y.n == x` fact is representation-identical to the other two
        # branches and survives their CFG join. _resolve_succ collapses this
        # back to "NULL" for every consumer that branches on succ.
        return D.set_succ(out, y, x)
    if succ_y is not None:
        return _do_assign_deref_known_succ(s, x, y, succ_y, aux)

    # Unknown successor: join fresh node treatment with hypothetical coincidence candidates
    n = s.n
    old_reach_col_y = [s.reach[z][y] for z in range(n)]  # snapshot before any reset
    if aux is not None:
        fresh = free_var_slot(s, x, aux)
        if D.is_bottom(fresh):
            return fresh
    else:
        fresh = s
    for z in range(n + 1):  # + NULL coordinate: x's own nullity becomes TOP too
        if z != x:
            fresh = D.set_eq(fresh, x, z, D.UNKNOWN)
    fresh = D.set_succ(fresh, x, None)
    for z in range(n):
        # x's nullity is genuinely TOP here (y.n's successor being
        # unknown includes "y.n is NULL"), so the sound answer for
        # z -> x must cover BOTH "x turns out to be a real node"
        # (shift of the old z -> y fact) AND "x turns out NULL" (NO) --
        # i.e. their join, not just the shifted value alone.
        fresh = D.set_reach(fresh, z, x, D.reach_join(D.reach_shift(old_reach_col_y[z]), D.NO))
    for w in range(n):
        if w != x:
            fresh = D.set_reach(fresh, x, w, D.TOP_REACH)
    fresh = D.set_reach(fresh, x, x, D.TOP_REACH)  # nullity of x unknown; refined by later assume
    fresh = D.set_succ(fresh, y, x)
    result = D.close_eq(fresh)
    for w in range(n):
        # Skip self-loop hypothesis when x == y (acyclic invariant)
        if x == y and w == x:
            continue
        candidate = _do_assign_deref_known_succ(s, x, y, w, aux)
        result = D.join(result, candidate)
    return result


def _do_assign_deref_known_succ(s, x, y, succ_y, aux):
    """Transformer for `x := y.n` when `y.n` is equal to `succ_y`."""
    # Derive reachability via y's reach column and intersect with copied facts
    n = s.n
    old_reach_col_y = [s.reach[z][y] for z in range(n)]
    out = copy_row(s, x, succ_y, aux)
    if D.is_bottom(out):
        return out
    for z in range(n):
        # The "via y, then one more hop" derivation only models chains
        # through a node genuinely distinct from the one x is being
        # aliased to. For z == x itself or z == succ_y (the very node
        # x now aliases), reach[z][x] is already exactly pinned by
        # copy_row's aliasing (trivial self-reach); re-deriving it
        # from the stale pre-overwrite `reach[z][y]` snapshot doesn't
        # apply here and spuriously "contradicts" the correct copied
        # value, so leave those two reflexive cells untouched.
        if z == x or z == succ_y:
            continue
        # Skip shift if z does not reach y
        if old_reach_col_y[z] == D.NO:
            continue
        # `succ_y` being a *known* alias does not mean it is *non-null*:
        # it can itself be an alias to a NULL-valued slot (e.g. a chain
        # of free_var_slot redirects that eventually bottoms out at NULL).
        # Shifting reach[z][y] by one hop asserts "z reaches x", which
        # is only sound reasoning when x is confirmed to actually BE a
        # node. When x's nullity isn't confirmed NONNULL, the true answer
        # must also cover "x turns out NULL, so z does not reach it at
        # all" (NO) -- mirroring the "unknown successor" branch above,
        # which already gets this right via reach_join(shift(...), NO).
        # Without this, z == y hits it hardest: reach[y][y] is always the
        # trivial reflexive ODD, so its shift is unconditionally EVEN,
        # directly contradicting a correctly-copied NO whenever x turns out
        # NULL -- a false contradiction, not a real one, that made a perfectly
        # reachable program point look like BOTTOM (unreachable).
        shifted = D.reach_shift(old_reach_col_y[z])
        derived = shifted if out.nullity[x] == D.NONNULL else D.reach_join(shifted, D.NO)
        combined = _combine_prefer_derived(out.reach[z][x], derived)
        if combined is None:
            return D.BOTTOM  # the two soundly-derived facts contradict
        out = D.set_reach(out, z, x, combined)
    # x is now the node y.n (aliased to succ_y): record `succ[y] = x`
    # too, matching the other two branches, so the exact `y.n == x` fact
    # is representation-identical on all paths and survives their join
    # (the assert checker reads it back through _resolve_succ).
    # this is only correct when y != x. When x == y (a self-dereference,
    # `x := x.n`), `out` already has slot y == x OVERWRITTEN by copy_row
    # above to represent the NEW value (x's old content).
    if y == x:
        return out
    return D.set_succ(out, y, x)


def check_assign_deref_issues(s, y):
    if D.is_bottom(s):
        return []
    if check_memory_safety(s, y):
        return ["MEMORY SAFETY (deref of possibly-NULL pointer)"]
    return []

def make_transfer(var_index, edge_aux):
    
    def _base_transfer(edge, s):
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
            return do_field_assign(s, var_index[x], yidx, len(var_index))
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
            s = D.close_eq(s)
            if D.is_bottom(s):
                return s
            s = D.set_reach(s, xi, xi, D.ODD)
            return s
        if kind == "assume_eq":
            _, x, y = cmd
            xi, yi = var_index[x], var_index[y]
            n = s.n
            if D.get_eq(s, xi, yi) == D.FALSE:
                return D.BOTTOM  # already known distinct; this branch is dead
            s = D.set_eq(s, xi, yi, D.TRUE)
            s = D.close_eq(s)
            if D.is_bottom(s):
                return s
            for z in range(n):
                m1 = D.reach_meet(s.reach[xi][z], s.reach[yi][z])
                if m1 is None:
                    return D.BOTTOM  # contradiction: no concrete store fits
                s = D.set_reach(s, xi, z, m1)
                m2 = D.reach_meet(s.reach[z][xi], s.reach[z][yi])
                if m2 is None:
                    return D.BOTTOM
                s = D.set_reach(s, z, xi, m2)
            if s.succ[xi] is None:
                s = D.set_succ(s, xi, s.succ[yi])
            elif s.succ[yi] is None:
                s = D.set_succ(s, yi, s.succ[xi])
            elif s.succ[xi] != s.succ[yi]:
                sx, sy = s.succ[xi], s.succ[yi]
                if isinstance(sx, int) and isinstance(sy, int):
                    if D.get_eq(s, sx, sy) == D.FALSE:
                        return D.BOTTOM 
                    s = D.set_eq(s, sx, sy, D.TRUE)
                    s = D.close_eq(s)
                    if D.is_bottom(s):
                        return s  
                    s = D.set_succ(s, xi, sx)
                    s = D.set_succ(s, yi, sx)
                elif sx == "NULL" and isinstance(sy, int):
                    if s.nullity[sy] == D.NONNULL:
                        return D.BOTTOM  
                    s = D.set_nullity(s, sy, D.NULL)
                    s = D.close_eq(s)
                    if D.is_bottom(s):
                        return s
                    s = D.set_succ(s, xi, sx)
                    s = D.set_succ(s, yi, sx)
                elif sy == "NULL" and isinstance(sx, int):
                    if s.nullity[sx] == D.NONNULL:
                        return D.BOTTOM 
                    s = D.set_nullity(s, sx, D.NULL)
                    s = D.close_eq(s)
                    if D.is_bottom(s):
                        return s
                    s = D.set_succ(s, xi, sx)
                    s = D.set_succ(s, yi, sx)
                else:
                    s = D.set_succ(s, xi, None)
                    s = D.set_succ(s, yi, None)
            return s
        if kind == "assume_neq":
            return s  # no general sound refinement
        if kind == "assert":
            return s
        raise ValueError(f"Unknown command kind: {kind}")

    # resolve _base_transfer
    def transfer(edge, disjunctive_state):
        new_states = []
        for s in disjunctive_state.states:
            res = _base_transfer(edge, s)
            if res is not D.BOTTOM:
                new_states.append(res)
        return D.DisjunctiveShapeState(new_states)

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
        sy = _resolve_succ(s, y)
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
        sy = _resolve_succ(s, y)
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
        bottom=lambda: D.DisjunctiveShapeState([]),
        entry_state=D.DisjunctiveShapeState([D.initial_state(n, n_real=n_vars)]),
        transfer=transfer,
        join=lambda a, b: a.join(b),
        leq=lambda a, b: a <= b,
    )

    assert_results = []
    safety_results = []
    # Iterate through the states in the wrapper for the final checks
    for e in cfg.edges:
        cmd = e.cmd
        disjunctive_pre = pre[e.src]
        
        if cmd[0] == "assert":
            # True only if the assert holds across EVERY possible path
            ok = all(check_assert(s, cmd[1], var_index) for s in disjunctive_pre.states)
            assert_results.append((e, ok))
            
        elif cmd[0] == "field_assign":
            _, x, y = cmd
            yidx = None if y == "NULL" else var_index[y]
            issues = []
            for s in disjunctive_pre.states:
                issues.extend(check_field_assign_issues(s, var_index[x], yidx, len(var_index)))
            if issues:
                safety_results.append((e, list(set(issues)))) # deduplicate issues
                
        elif cmd[0] == "assign_deref":
            _, x, y = cmd
            issues = []
            for s in disjunctive_pre.states:
                issues.extend(check_assign_deref_issues(s, var_index[y]))
            if issues:
                safety_results.append((e, list(set(issues))))

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
    for arg in sys.argv[1:]:
        # Expand wildcard patterns like 'interesting-tests/*.txt'
        matched_files = glob.glob(arg)

        if not matched_files:
            # If glob finds no matches, pass the raw argument so Python raises a standard error
            run_file(arg)
        else:
            # Run all matching files in sorted order
            for p in sorted(matched_files):
                run_file(p)
