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
    required for a sound fixpoint iteration.

    The eq loops below run over range(n + 1), not range(n): index n is
    the reserved NULL coordinate (see domain.py), and x's/aux's nullity
    is just one more cell of the same table now, not a separate one."""
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
    s = D.set_claimed(s, aux, True)  # aux now holds a real (possibly-stale
    # but real) identity -- see find_sharing_conflict's docstring for why
    # this matters: an unclaimed slot is soundly skippable as a sharing/
    # cycle candidate (nothing lives there yet), a claimed one is not.
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
    """dst becomes an alias of src: copy nullity/eq/reach/succ from src to
    dst (used by `x := y` and by `x := y.n` when y.n is a known variable).
    `aux`: dedicated scratch slot for this edge (see free_var_slot); pass
    None only when dst is known not to be dangling-referenced elsewhere."""
    if D.is_bottom(s):
        return D.BOTTOM
    if dst == src:
        # dst doesn't actually change (it's already exactly src). But
        # `aux` was still allocated for this edge and must not be left
        # permanently unclaimed: FIX (found by differential/ground-
        # truth fuzzing -- see report.pdf) -- an unclaimed aux slot can
        # carry a stale fact from whenever its index was first marked
        # distinct from something else (e.g. reset_as_fresh_new, which
        # correctly marks EVERY coordinate including not-yet-relevant
        # future aux slots -- see that function's docstring), and
        # nothing ever corrects it if this shortcut skips claiming the
        # slot entirely. Populate aux as an exact, permanent snapshot
        # of dst's (unchanged) current identity -- like free_var_slot,
        # but WITHOUT resetting dst afterward, since dst genuinely isn't
        # being overwritten here.
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
            # Unlike free_var_slot (where dst is about to become
            # something else, so its future relationship to aux is
            # UNKNOWN), dst does NOT change here -- aux really is an
            # exact, permanent alias of dst, so record that directly.
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
        # This also redirects any succ[w] == dst (including w == src) to
        # aux, so the copy below reads already-corrected values.
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
    """x := new: fresh node, provably distinct from every other currently
    tracked identity (see analysis rationale)."""
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
        # NOTE: this must mark EVERY z distinct, not just claimed ones
        # -- claimed[z] == False ("provably not yet a real identity")
        # is itself the MORE precise/informative value in this domain's
        # own ordering (see domain.py's module docstring / the
        # monotonicity check's v3 note: weakening claimed only ever
        # flips False -> True), so behaving differently for claimed vs
        # unclaimed z would make this function's output depend on that
        # field in a non-monotone way -- confirmed directly by fuzzing
        # an earlier version that special-cased unclaimed z here. The
        # real hazard that version was working around (an unclaimed aux
        # slot marked FALSE here, then never reclaimed/corrected later
        # because copy_row's dst==src shortcut skipped free_var_slot
        # entirely) is fixed at its actual source instead: copy_row now
        # always properly claims and populates its aux slot, even in
        # the dst==src case, so no slot is ever left both marked here
        # AND permanently unclaimed.
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
    """s.succ[w], except a succ entry that points at a variable slot now
    proven NULL is reported as the literal "NULL" -- both describe the same
    heap fact ("w.n is the null value").

    Two things create such entries:
      * `t.n := x` recorded while x was still NULL (early in the fixpoint,
        before x's list was built) -- a genuine pre-existing case;
      * do_assign_deref now records `succ[y] = x` on *every* branch, including
        the one where `y.n` turned out to be NULL (x is then a NULL slot), so
        the exact `y.n == x` relationship is representation-identical on all
        paths and survives their CFG join -- without this, a join of the
        "y.n is NULL" path (succ[y] == "NULL") with a "y.n is a real node"
        path collapsed succ[y] to None and drove reach[y][y.n] to TOP, which
        is what left the y-traversal assertions (`LS y yy`, `t = yy.n`)
        unprovable on the main example.

    Consumers that branch on succ (sharing check, deref-atom evaluation,
    assert consistency links) must treat these exactly like "NULL" rather
    than as a live forward n-pointer; that is what this resolver is for.
    `free_var_slot` already redirects such an entry to the scratch slot the
    moment its target variable is reassigned, so it can never go stale."""
    v = s.succ[w]
    if isinstance(v, int) and s.nullity[v] == D.NULL:
        return "NULL"
    return v


def would_create_cycle(s, x, y):
    """Would writing x.n := y close a cycle? True/False/'MAYBE' (unknown).
    An UNKNOWN (not proven false) x==y is itself a "maybe": if x and y
    turn out to be the same node, x.n := y is a self-loop, and silently
    falling through to the reach check (which says nothing about x==y)
    would wrongly report False instead of at least MAYBE."""
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
    """The set of coordinates reachable from a NAMED program variable via
    a chain of eq-aliasing. A coordinate outside this set (almost always
    a `free_var_slot` scratch aux slot whose node has since become fully
    disconnected from every live variable) cannot be dereferenced,
    aliased, or otherwise observed by any future command in the program
    -- nothing left in the program text can ever name it again. This is
    the same "garbage is out of scope for the given assertion language"
    argument already used to justify the NULL/unclaimed skips below.

    IMPORTANT (found by exhaustive_monotonicity_check-style fuzzing on
    this specific addition -- see report.pdf): growth here MUST use
    `eq != FALSE` (TRUE *or* UNKNOWN both count as "possibly aliased,
    cannot rule out"), never `eq == TRUE` alone. A state weakening from
    eq[z][w]=TRUE to eq[z][w]=UNKNOWN carries LESS information, so the
    live-set computed from it must never *shrink* relative to the
    stronger state's -- but "== TRUE" does shrink it (TRUE triggers
    growth, UNKNOWN doesn't), which broke monotonicity. "!= FALSE" is
    stable under that exact weakening direction (both TRUE and UNKNOWN
    trigger growth), so it doesn't.

    A second, analogous idea -- also growing `live` by following succ
    chains forward (z live and succ[z]=w known => w live) -- was tried
    and DROPPED after the same fuzzing caught it as unsound for the same
    reason: succ[z] going from a known index to None under weakening
    loses that growth trigger with no compensating alternative (unlike
    eq, there's no "possibly-succ" value to fall back to that stays
    stable under weakening), so it cannot be made monotone-safe without
    effectively disabling the filter whenever any live coordinate has an
    unresolved successor -- which is the common case inside a loop, i.e.
    exactly where the filter is supposed to help. Only the aliasing-based
    (eq) growth survived fuzzing; this is a narrower, more conservative
    filter than originally attempted, and is documented as such."""
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
    """Would writing x.n := y give node y a second incoming n-edge from a
    node other than x's? Returns a definite conflicting var index,
    "MAYBE" (cannot rule it out), or None (definitely no conflict).

    An unresolved successor (sz is None, i.e. we don't know what z.n is)
    must NOT be silently treated as "z.n definitely isn't y" -- that's
    exactly the unsound shortcut that let real sharing go undetected:
    z.n could well turn out to be y in the concrete run we just can't
    prove it either way, so it downgrades this to "can't rule out"
    rather than "no conflict". Likewise an unresolved eq(sz, y).

    Three candidates are soundly skippable outright, though (found via
    precision testing: without these, this function answered "can't rule
    it out" almost everywhere, since a handful of genuinely-relevant
    unknowns were swamped by dozens of irrelevant ones):
      - z provably NULL: a NULL value denotes no node at all, so there is
        no "z's n-field" to possibly conflict with anything.
      - z not yet `claimed` (see State.claimed's docstring): an internal
        scratch/aux slot that free_var_slot has never actually populated
        doesn't stand in for any real heap identity in any execution
        reaching this point -- there's nothing there to conflict either.
      - z not LIVE (see _compute_live): a scratch aux slot that WAS
        populated at some point but has since become unreachable from
        every named program variable (no live variable is an alias of z,
        and no live-reachable coordinate's succ points at z) cannot be
        dereferenced or otherwise observed again by anything the program
        can still name -- a real second in-edge dangling only from such
        a slot can never be witnessed by any future command or assert,
        so it is out of scope for the same reason garbage nodes are
        (see report's note on the sharing check's scope). This is the
        one addition beyond the original two skips above: those two
        catch immediately-inert candidates, this one additionally catches
        candidates that were relevant once but have since gone dead --
        the dominant source of aux-slot noise inside loops, where each
        loop iteration mints a fresh aux slot per reassignment.
    None of the three loses real precision: all are cases where z
    provably cannot be a second, distinct in-edge to any node THE
    PROGRAM CAN STILL OBSERVE, not cases where we're merely optimistic
    about a genuinely-relevant unknown."""
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
    """Pure state transformer -- NEVER raises.

    Memory safety: if x is PROVABLY null, every execution reaching this
    edge crashes -- there is no live continuation, so the sound abstract
    result is BOTTOM (not "unchanged": leaving s as-is would keep
    propagating a dead, crashed path's facts as if they were live). If x
    is merely UNPROVEN safe (nullity TOP), the only surviving (non-crash)
    branch requires x to actually be NONNULL, so we narrow to that and
    proceed with the write, rather than freezing the whole state -- that
    narrowing is itself a sound refinement, not a guess.

    A *predicted* cycle/sharing conflict must NOT stop us from modeling
    the write itself. `x.n := y` always mutates the heap in the concrete
    semantics -- the language has no runtime check that aborts on
    creating a cycle or a shared node; those are just properties the
    analysis is supposed to prove absent (and does, separately, via
    check_field_assign_issues against the pre-state). If we returned `s`
    unchanged instead, every fact this write should update (chiefly
    succ[x], i.e. "what x.n now denotes") would go stale and keep
    describing the *old* value of x.n forever afterwards -- unsound: it
    lets later asserts about x.n "verify" against a value that's no
    longer real. So succ[x] is always updated. However, once a cycle or
    sharing conflict is real (or merely possible), the acyclic/unshared
    invariant that `_apply_splice`'s exactness argument leans on is
    broken, so its precise ODD/EVEN arithmetic is no longer trustworthy;
    every reach fact touching x's node is instead soundly weakened to
    TOP rather than kept at its old, now-unjustified, precise value."""
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
        # Only x's OWN outgoing chain is cut (reach[x][w], w != x): x's
        # node still exists and is still reachable from wherever it was
        # before (reach[z][x] unaffected), and x still trivially
        # "reaches itself" as a bare 1-node segment (reach[x][x]
        # unaffected -- nulling the outgoing field doesn't erase x's own
        # node).
        #
        # FIX (found by differential/ground-truth fuzzing -- see
        # report.pdf): "w != x" is not the same condition as "w is a
        # genuinely different node from x". A w that is merely a
        # DIFFERENT INDEX but a KNOWN ALIAS of x (eq[x][w] == TRUE) is
        # every bit as reflexive as w == x itself -- cutting x's own
        # outgoing field cannot un-reach x's own node no matter which
        # name currently refers to it, so reach[x][w] must stay
        # whatever the correct reflexive fact already is (left
        # untouched here, exactly like reach[x][x]), not be forced to
        # NO. When it's merely UNCERTAIN whether w aliases x
        # (eq[x][w] == UNKNOWN), forcing NO is unsound for the same
        # reason on the branch where it turns out to be an alias; the
        # domain has no exact representation for "NO-or-reflexive-ODD",
        # so TOP_REACH is the closest sound over-approximation.
        # Confirmed: without this, `d.n := NULL` on a `d` that is a
        # known alias of some other live variable `b` (e.g. after
        # `b := d`) forced reach[d][b] to NO while reach[b][d] correctly
        # stayed ODD -- eq[d][b] == TRUE with reach[d][b] != reach[b][d]
        # is a direct internal contradiction, not just imprecision.
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
        # FIX (found by differential/ground-truth fuzzing -- see
        # report.pdf): cutting x's outgoing edge must ALSO invalidate
        # any OTHER variable z's reach to whatever was only reachable
        # by continuing past x -- x's own row is not the only thing
        # this write affects. Given the unshared-list invariant, a
        # forward chain is unique: if z reaches x and w was reachable
        # from x's (old) successor, z's only route to w necessarily
        # passed through the edge just severed, so z can no longer
        # reach w at all -- and this is CERTAIN (not just weakened),
        # by the same "provably NO" argument _apply_splice already
        # makes for the symmetric attach case, whenever z is certain to
        # reach x in the first place. When that's merely possible
        # (reach[z][x] == TOP_REACH), only join in the NO possibility
        # rather than overwriting, since z might not reach x at all, in
        # which case its route to w (if any) never depended on x.
        # Without this, a stale "z reaches w" fact from before the cut
        # survives indefinitely: confirmed by a case where `b.n := NULL`
        # left `reach[d][a] == EVEN` (from the since-severed d -> b -> a
        # chain) even though `d` provably no longer reaches `a` at all.
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
                # Same aliasing gap as the fix just above, applied here
                # too (found the same way): w being merely a different
                # INDEX from x doesn't mean it's a different NODE. If w
                # is a known alias of x (eq[x][w] == TRUE), it IS x, so
                # it's not "downstream of x's old outgoing edge" at all
                # -- it's the very node whose outgoing edge is being
                # cut, and z's reach to it is exactly z's reach to x,
                # unaffected. Confirmed by a case where a 12 that
                # legitimately reached d (== an alias `10`, via
                # eq[10][d]==TRUE) had reach[12][10] wrongly cut to NO,
                # even though 12 continued to reach d/10 the entire
                # time -- only d's OWN outgoing edge was cut, and this
                # pair never went through it.
                eq_xw = D.get_eq(s, x, w)
                if eq_xw == D.TRUE:
                    continue
                eq_zw = D.get_eq(s, z, w)
                if eq_zw == D.TRUE:
                    continue  # w IS z; reach[z][w] is the trivial reflexive fact, unaffected
                if old_reach_row_x[w] == D.NO:
                    continue  # x's old chain never reached w; genuinely nothing to cut
                # FIX (found by monotonicity fuzzing, immediately after
                # the fix just above -- see report.pdf): the same
                # certain/uncertain distinction that fix already applies
                # to eq_xw/eq_zw also has to apply to old_reach_row_x[w]
                # itself. Only the `== NO` case above is certain enough
                # to justify skipping outright; TOP_REACH here means
                # "x's old chain MIGHT have reached w", not "did" -- so
                # it must not license a confident NO any more than an
                # uncertain eq does. Confirmed directly: weakening a
                # state's reach[x][w] from a certain NO to TOP (strictly
                # LESS information) flipped this function's output for
                # reach[z][w] from an unchanged EVEN (correctly left
                # alone, since x's old chain provably never reached w)
                # to a confidently wrong NO.
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
        # The invariant _apply_splice relies on is broken by this write
        # (for sure, or possibly -- see the note below on why "possibly"
        # doesn't currently earn anything sharper): every reach fact
        # touching x is no longer trustworthy.
        #
        # Tried and reverted: computing the precise splice AND the
        # weakened result and joining them (the same "compute both
        # possibilities" technique that works for y's nullity
        # uncertainty just below). That technique only helps when the
        # two branches disagree on something less than everything;
        # here, _weaken_for_uncertain_write sets every cell it touches
        # to TOP outright, and join(anything, TOP) is always TOP by
        # construction -- so joining just silently reproduces this same
        # branch, confirmed by direct testing (byte-identical output,
        # traced end to end). The real remaining lever for recovering
        # this precision is elsewhere: find_sharing_conflict currently
        # answers "maybe" the moment ANY tracked variable anywhere has a
        # completely unestablished successor, which in practice is true
        # for nearly every untouched scratch slot simultaneously, not
        # just ones plausibly relevant to this specific write. Narrowing
        # that safely needs the state to distinguish "never going to be
        # a live pointer here" from "currently unknown but could still
        # alias" -- which it currently cannot (an untouched scratch slot
        # and a touched-but-forgotten one are indistinguishable in the
        # table) -- and is a real design change to the conflict-
        # detection logic, not a local patch to this function.
        s = _weaken_for_uncertain_write(s, x, y, n)
    else:
        s = _apply_splice_best_effort(s, x, y)
    return s


def _apply_splice_best_effort(s, x, y):
    """Apply the exact splice for x.n := y, additionally accounting for
    y's own nullity uncertainty (join both branches when y's nullity is
    TOP -- see the docstring on the y-nullity handling this factors out
    of) but NOT for cycle/sharing uncertainty; the caller handles that
    dimension separately, since the two kinds of uncertainty are
    independent and each needs its own join."""
    if s.nullity[y] == D.NONNULL:
        return _apply_splice(s, x, y)
    if s.nullity[y] == D.TOP_NULLITY:
        # y's nullity is itself ambiguous (e.g. loop-iteration-count
        # uncertainty, not staleness -- see _apply_splice's docstring for
        # the distinction). Compute both possibilities and join them:
        # sound, and precise whenever both branches happen to agree.
        spliced = _apply_splice(s, x, y)
        return _join_states_reach(s, spliced)
    return s  # y is NULL: nothing to splice


def _weaken_for_uncertain_write(s, x, y, n):
    """Give up on x's reach facts (and the downstream splice-affected
    cells reach[z][w] for z reaching x and w reached from y) because this
    write may -- or, when called from the definite-break case, definitely
    does -- break the acyclic/unshared invariant _apply_splice's
    exactness argument relies on. Must ALSO cover those downstream cells,
    not just x's own row/column: the concrete write still happens, so a
    genuine new path z -> x -> y -> w may now exist, and leaving
    reach[z][w] at its stale pre-write value would silently understate
    reachability (unsound: could make a real LS/ODD/EVEN fact falsely
    read as NO)."""
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
    """Combine a value copied from an aliased slot with a value freshly
    re-derived from an already-established precise fact. If compatible
    (their outcome-sets overlap), intersect (sharper). If they genuinely
    disagree, trust `derived`: it's grounded directly in a fact we've
    already verified precise (e.g. reach[z][xx] right after xx was just
    assigned), whereas the copied value can carry staleness inherited
    from unrelated loop-iteration mixing elsewhere in the aliased slot's
    own history. (Found via testing: using a plain intersection/"meet"
    here silently fell back to NO on disagreement, destroying precision
    the same way the field-write splice bug did.) If the two outcome-sets
    are genuinely disjoint, that isn't "trust one side" territory either
    -- both were derived soundly from the same real state, so a real
    disagreement means the state itself is contradictory (unreachable).
    Returns None in that case; the caller must treat it as BOTTOM rather
    than picking either value."""
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

    # unknown successor: y.n could turn out to be a genuinely fresh node
    # we've never named before, OR it could coincidentally already equal
    # some OTHER already-tracked variable's current identity -- including
    # (but not limited to) x's own. The "succ_y is already known" branch
    # above handles that precisely via copy_row's aliasing whenever we
    # DO know which variable it is; here we don't, so the sound answer
    # joins the "fresh" treatment with the "as if we'd learned succ_y is
    # w" result for every already-tracked real variable w. Committing
    # only to "fresh" silently discards every one of those coincidence
    # possibilities, each of which a sibling state that happens to
    # retain the concrete (coincidentally-aliased) succ_y can legitimately
    # reach via the branch above -- an asymmetry a fuzzer catches as a
    # monotonicity violation (this generalizes an earlier, narrower fix
    # that only special-cased "coincides with x").
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
        # FIX (found by differential/ground-truth fuzzing -- see
        # report.pdf): when x == y (a self-dereference, `x := x.n`),
        # the hypothesis w == x means "x.n already equals x itself" --
        # a self-loop that would already be a cycle in the PRE-state,
        # which cannot be true of a state reachable under this
        # language's acyclic invariant. Exploring it anyway (as one of
        # the joined hypotheses) let a spurious `succ[x] == x` fact
        # survive into the result even though no reachable concrete
        # state has one. This is specific to x == y: when x != y, w ==
        # x is a perfectly ordinary, legitimate hypothesis (y.n
        # happening to already equal x's current value) already handled
        # correctly elsewhere (copy_row's dst == src case).
        if x == y and w == x:
            continue
        candidate = _do_assign_deref_known_succ(s, x, y, w, aux)
        result = D.join(result, candidate)
    return result


def _do_assign_deref_known_succ(s, x, y, succ_y, aux):
    """x := y.n, where y.n is definitely (or, from do_assign_deref's
    "unknown successor" branch, hypothetically) the variable `succ_y`.
    Factored out so that branch can invoke this once per hypothetical
    possibility and join the results, exactly reusing the same precise
    logic as the case where succ_y is genuinely known."""
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
        # FIX (found by differential/ground-truth fuzzing -- see
        # report.pdf): when z does NOT reach y at all
        # (old_reach_col_y[z] == NO), there is no "via y" path to shift
        # in the first place -- shifting NO and combining it with
        # copy_row's already-correct out.reach[z][x] treats "no route
        # through y" as if it were an independent, equally-valid claim
        # "z definitely does not reach x at all", which can directly
        # (and wrongly) contradict a real route to x that z has through
        # some entirely different, already-established path (e.g. z
        # already reaches succ_y directly, and x was just aliased to
        # succ_y by copy_row above). _apply_splice already gets this
        # right with the exact same guard (`if rzx == NO: continue`);
        # this branch needs it too, for the same reason.
        if old_reach_col_y[z] == D.NO:
            continue
        # FIX (found by differential/ground-truth fuzzing -- see
        # report.pdf): `succ_y` being a *known* alias does not mean it
        # is *non-null* -- it can itself be an alias to a NULL-valued
        # slot (e.g. a chain of free_var_slot redirects that eventually
        # bottoms out at NULL), exactly as happened here. Shifting
        # reach[z][y] by one hop asserts "z reaches x", which is only
        # sound reasoning when x is confirmed to actually BE a node.
        # When x's nullity isn't confirmed NONNULL, the true answer
        # must also cover "x turns out NULL, so z does not reach it at
        # all" (NO) -- mirroring the "unknown successor" branch above,
        # which already gets this right via reach_join(shift(...), NO).
        # Without this, z == y hits it hardest: reach[y][y] is always
        # the trivial reflexive ODD, so its shift is unconditionally
        # EVEN, directly contradicting a correctly-copied NO whenever
        # x turns out NULL -- a false contradiction, not a real one,
        # that made a perfectly reachable program point look like
        # BOTTOM (unreachable).
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
    #
    # FIX (found by differential/ground-truth fuzzing -- see
    # report.pdf): this is only correct when y != x. When x == y (a
    # self-dereference, `x := x.n`), `out` already has slot y == x
    # OVERWRITTEN by copy_row above to represent the NEW value (x's
    # alias of succ_y) -- it no longer refers to the OLD y whose
    # successor this fact is actually about. Recording succ[y] = x
    # unconditionally then asserts "x.n == x", a self-loop that cannot
    # exist in any reachable concrete state under this language's
    # acyclic invariant, purely as an artifact of reusing the same slot
    # for both the old and new meaning. When x == y there is no longer
    # a live coordinate correctly denoting "the old y" to attach this
    # fact to (its data was moved to `aux` by free_var_slot, which
    # isn't a named, further-queried coordinate), so the fact is simply
    # not recorded in that case -- everything else this function
    # already derived (x's nullity/eq/reach, aliased from succ_y) is
    # still correct and sufficient.
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
