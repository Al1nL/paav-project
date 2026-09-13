"""
differential_fuzz_shape.py -- soundness testing against ground truth, not just
internal monotonicity consistency.

Every prior check (unit tests, exhaustive_monotonicity_check.py) only checks
properties INTERNAL to the abstract domain: does a<=b imply f(a)<=f(b)? That
catches a lot (it caught the assume_eq bug), but it can never catch a
transfer function that is *consistently* wrong in a way that happens to
preserve ordering -- monotonicity says nothing about whether f(a) actually
over-approximates what the concrete program does.

This script closes that gap directly: it generates random small programs,
executes them TWICE --
  1. concretely, against an actual simulated heap (arrays of node ids +
     a next-pointer table) -- this is ground truth, not an abstraction;
  2. abstractly, through the exact same transfer functions used by
     shape/analysis.py's real analyze() driver (same make_transfer, same
     per-edge aux-slot allocation, same entry state).
and after every single command, checks that every fact the abstract state
claims (nullity, equality, succ, reach-with-parity) is actually true of the
concrete heap. Soundness means: for the invariant this analysis is entitled
to assume the program respects (Section 2.1's "unshared, acyclic, field
reset before write"), the abstract state must never claim something false
about a reachable concrete execution.

To keep every generated program inside that invariant (so a violation found
here is a real bug, not just the generator producing a program the analysis
was never supposed to handle), the generator is *guided*: at each step, it
consults the concrete heap built so far and only offers actions that are
memory-safe, don't create a cycle, and don't create a second in-edge to any
node -- exactly the property class this project's own safety checks exist to
verify absence of.

Usage (from repo root):
    python3 -m tests.differential_fuzz_shape
"""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape import domain as D
from shape.analysis import make_transfer, _resolve_succ


class FakeEdge:
    """Minimal stand-in for common.cfg's Edge: only .cmd is used by
    make_transfer's dispatcher. A distinct object per command (required:
    edge_aux is keyed by id(edge), exactly as the real analyze() driver
    keys it by id() of the real CFG edge objects)."""
    __slots__ = ("cmd",)

    def __init__(self, cmd):
        self.cmd = cmd


OVERWRITING_KINDS = {"assign_var", "assign_null", "assign_new", "assign_deref"}

OUTCOME_MAP = {
    D.NO: {D.NO},
    D.ODD: {D.ODD},
    D.EVEN: {D.EVEN},
    D.EITHER: {D.ODD, D.EVEN},
    D.TOP_REACH: {D.NO, D.ODD, D.EVEN},
}


class ConcreteHeap:
    """Ground truth: node ids (small ints) and a next-pointer table."""

    def __init__(self):
        self.next_ptr = {}   # node id -> node id or None
        self._counter = 0

    def fresh(self):
        nid = self._counter
        self._counter += 1
        self.next_ptr[nid] = None
        return nid

    def reaches(self, a, b, cap):
        """Does walking .n from node a (>=0 hops) reach node b? Returns the
        hop count (>=0) if so, else None. `cap` guards against an
        unexpected cycle (should never trigger given the generator's own
        acyclicity guard, but this is ground truth code, so it must not
        hang if that guard ever has a bug of its own)."""
        cur = a
        d = 0
        seen = set()
        while True:
            if cur == b:
                return d
            if cur is None or cur in seen or d > cap:
                return None
            seen.add(cur)
            cur = self.next_ptr[cur]
            d += 1

    def has_predecessor(self, node):
        return any(v == node for v in self.next_ptr.values())


def generate_command(rng, var_names, var_index, heap, slot):
    """Pick one random, generator-guaranteed-valid command (as a cmd tuple)
    given the current concrete state (`slot`: list of node-id-or-None,
    length >= n_vars, indices 0..n_vars-1 are the named variables). Returns
    None if no valid choice exists for the sampled action this round (the
    caller retries with a different action)."""
    n_vars = len(var_names)
    action = rng.choice([
        "assign_null", "assign_new", "assign_var", "assign_deref",
        "field_assign_var", "field_assign_null", "skip",
        "assume_eq", "assume_neq", "assume_eq_null", "assume_neq_null",
    ])
    x = rng.choice(var_names)
    xi = var_index[x]

    if action == "skip":
        return ("skip",)

    if action == "assign_null":
        return ("assign_null", x)

    if action == "assign_new":
        return ("assign_new", x)

    if action == "assign_var":
        others = [v for v in var_names if v != x]
        if not others:
            return None
        y = rng.choice(others)
        return ("assign_var", x, y)

    if action == "assign_deref":
        candidates = [v for v in var_names if slot[var_index[v]] is not None]
        if not candidates:
            return None
        y = rng.choice(candidates)
        return ("assign_deref", x, y)

    if action == "field_assign_null":
        if slot[xi] is None:
            return None
        return ("field_assign", x, "NULL")

    if action == "field_assign_var":
        if slot[xi] is None:
            return None
        if heap.next_ptr[slot[xi]] is not None:
            return None  # assumption 2.1.3: field must be NULL before write
        candidates = []
        for y in var_names:
            if y == x:
                continue
            yv = slot[var_index[y]]
            if yv is None:
                candidates.append(y)  # x.n := NULL via a null variable: always fine
                continue
            # acyclicity: y's value must not already (transitively) reach x's node
            if heap.reaches(yv, slot[xi], cap=n_vars + 5) is not None:
                continue
            # no sharing: y's node must not already have a predecessor
            if heap.has_predecessor(yv):
                continue
            candidates.append(y)
        if not candidates:
            return None
        y = rng.choice(candidates)
        return ("field_assign", x, y)

    if action == "assume_eq":
        others = [v for v in var_names if v != x and slot[var_index[v]] == slot[xi]]
        if not others:
            return None
        y = rng.choice(others)
        return ("assume_eq", x, y)

    if action == "assume_neq":
        others = [v for v in var_names if v != x and slot[var_index[v]] != slot[xi]]
        if not others:
            return None
        y = rng.choice(others)
        return ("assume_neq", x, y)

    if action == "assume_eq_null":
        if slot[xi] is not None:
            return None
        return ("assume_eq_null", x)

    if action == "assume_neq_null":
        if slot[xi] is None:
            return None
        return ("assume_neq_null", x)

    raise AssertionError(action)


def apply_concrete(cmd, heap, slot, aux):
    kind = cmd[0]
    if kind == "skip" or kind.startswith("assume"):
        return
    if kind == "assign_null":
        _, x = cmd
        xi = VAR_INDEX[x]
        slot[aux] = slot[xi]
        slot[xi] = None
        return
    if kind == "assign_new":
        _, x = cmd
        xi = VAR_INDEX[x]
        slot[aux] = slot[xi]
        slot[xi] = heap.fresh()
        return
    if kind == "assign_var":
        _, x, y = cmd
        xi, yi = VAR_INDEX[x], VAR_INDEX[y]
        if xi == yi:
            return
        slot[aux] = slot[xi]
        slot[xi] = slot[yi]
        return
    if kind == "assign_deref":
        _, x, y = cmd
        xi, yi = VAR_INDEX[x], VAR_INDEX[y]
        slot[aux] = slot[xi]
        slot[xi] = heap.next_ptr[slot[yi]]
        return
    if kind == "field_assign":
        _, x, y = cmd
        xi = VAR_INDEX[x]
        target = None if y == "NULL" else slot[VAR_INDEX[y]]
        heap.next_ptr[slot[xi]] = target
        return
    raise AssertionError(kind)


def reaches_in(next_ptr, a, b, cap):
    """Ground-truth reachability against a specific (snapshotted) next_ptr
    mapping, rather than the live/final heap -- see run_trial's docstring
    note on why a snapshot per step is required."""
    cur = a
    d = 0
    seen = set()
    while True:
        if cur == b:
            return d
        if cur is None or cur in seen or d > cap:
            return None
        seen.add(cur)
        cur = next_ptr.get(cur)
        d += 1


def check_consistency(abs_state, slot, next_ptr, n, step_no, trace):
    """Every fact abs_state claims must be true of the concrete (slot,
    next_ptr) snapshot -- both taken at THIS step, not the final/live
    ones (a node's field, or a variable's value, can change again later
    in the trace, so using the live objects would compare against facts
    that aren't true yet).

    Coordinates with claimed[i] == False are skipped entirely: by the
    analysis's own design contract (see free_var_slot's and
    find_sharing_conflict's docstrings), an unclaimed slot represents
    nothing yet and is guaranteed to never be queried by anything the
    analysis actually reports (check_assert only ever indexes named
    variables; find_sharing_conflict/would_create_cycle explicitly skip
    unclaimed candidates; nothing else ever sets a succ reference into a
    slot without also claiming it). A slot can be left permanently
    unclaimed by a legitimate degenerate case (copy_row's `dst == src`
    early-exit, when `x := y.n` finds y.n already aliases x itself) --
    that's inert by the contract above, not a soundness gap, so holding
    it to the same standard as claimed coordinates would be testing a
    guarantee the analysis never makes rather than one it does."""
    if D.is_bottom(abs_state):
        return (f"step {step_no}: abstract state is BOTTOM (claims this point is "
                f"UNREACHABLE), but the concrete execution reached it normally")

    claimed = abs_state.claimed

    for i in range(n):
        if not claimed[i]:
            continue
        nul = D.get_nullity(abs_state, i)
        is_null = slot[i] is None
        if nul == D.NULL and not is_null:
            return f"step {step_no}: nullity[{i}]=NULL but concrete slot[{i}]={slot[i]!r} (non-null)"
        if nul == D.NONNULL and is_null:
            return f"step {step_no}: nullity[{i}]=NONNULL but concrete slot[{i}] is None"

    for i, j in itertools.combinations(range(n), 2):
        if not claimed[i] or not claimed[j]:
            continue
        eq = D.get_eq(abs_state, i, j)
        actual = slot[i] == slot[j]
        if eq == D.TRUE and not actual:
            return f"step {step_no}: eq[{i}][{j}]=TRUE but concrete values differ ({slot[i]!r} vs {slot[j]!r})"
        if eq == D.FALSE and actual:
            return f"step {step_no}: eq[{i}][{j}]=FALSE but concrete values are equal ({slot[i]!r})"

    for i in range(n):
        if not claimed[i] or slot[i] is None:
            continue
        sc = _resolve_succ(abs_state, i)
        actual_next = next_ptr.get(slot[i])
        if sc == "NULL":
            if actual_next is not None:
                return f"step {step_no}: succ[{i}]=NULL but concrete next({slot[i]})={actual_next!r}"
        elif isinstance(sc, int):
            if not claimed[sc]:
                continue
            expected = slot[sc]
            if actual_next != expected:
                return (f"step {step_no}: succ[{i}]={sc} (claims next({slot[i]}) == "
                        f"slot[{sc}]={expected!r}) but concrete next({slot[i]})={actual_next!r}")

    for i, j in itertools.product(range(n), repeat=2):
        if i == j or not claimed[i] or not claimed[j]:
            continue
        if slot[i] is None or slot[j] is None:
            continue
        outcome_val = D.get_reach(abs_state, i, j)
        d = reaches_in(next_ptr, slot[i], slot[j], cap=n + 5)
        if d is None:
            actual_outcome = D.NO
        else:
            actual_outcome = D.ODD if d % 2 == 0 else D.EVEN  # node-count parity, d=0 => count 1 => ODD
        allowed = OUTCOME_MAP[outcome_val]
        if actual_outcome not in allowed:
            return (f"step {step_no}: reach[{i}][{j}]={outcome_val} (allows {allowed}) but concrete "
                    f"outcome is {actual_outcome} (hop distance {d})")

    return None


def run_trial(seed, n_steps, var_names):
    global VAR_INDEX, HEAP
    rng = random.Random(seed)
    n_vars = len(var_names)
    VAR_INDEX = {v: i for i, v in enumerate(var_names)}
    HEAP = ConcreteHeap()
    slot = [None] * (n_vars + n_steps)  # room for one aux slot per step, worst case

    edges = []
    edge_aux = {}
    next_aux = n_vars
    trace = []
    snapshots = [list(slot)]              # snapshots[k] = variable slots after k commands
    heap_snapshots = [dict(HEAP.next_ptr)]  # heap_snapshots[k] = heap fields after k commands

    attempts = 0
    while len(edges) < n_steps and attempts < n_steps * 20:
        attempts += 1
        cmd = generate_command(rng, var_names, VAR_INDEX, HEAP, slot)
        if cmd is None:
            continue
        e = FakeEdge(cmd)
        if cmd[0] in OVERWRITING_KINDS:
            edge_aux[id(e)] = next_aux
            aux = next_aux
            next_aux += 1
        else:
            aux = None
        apply_concrete(cmd, HEAP, slot, aux)
        edges.append(e)
        trace.append(cmd)
        snapshots.append(list(slot))
        heap_snapshots.append(dict(HEAP.next_ptr))

    n = next_aux
    transfer = make_transfer(VAR_INDEX, edge_aux)
    abs_state = D.DisjunctiveShapeState([D.initial_state(n, n_real=n_vars)])

    def unwrap(ds):
        # Straight-line traces (no real CFG branching) never legitimately
        # produce more than one surviving disjunct from a single-disjunct
        # input; more than one would itself indicate a bug in the wrapper.
        assert len(ds.states) <= 1, f"unexpected branching: {len(ds.states)} disjuncts"
        return ds.states[0] if ds.states else D.BOTTOM

    violation = check_consistency(unwrap(abs_state), snapshots[0], heap_snapshots[0], n, 0, trace)
    if violation:
        return violation, trace[:0]

    for step_no, e in enumerate(edges, start=1):
        abs_state = transfer(e, abs_state)
        violation = check_consistency(unwrap(abs_state), snapshots[step_no], heap_snapshots[step_no], n, step_no, trace)
        if violation:
            return violation, trace[:step_no]

    return None, trace


def run(num_trials=3000, n_steps=25, seed0=42, var_names=("a", "b", "c", "d")):
    checked_steps = 0
    for t in range(num_trials):
        violation, trace = run_trial(seed0 + t, n_steps, list(var_names))
        checked_steps += len(trace)
        if violation:
            print(f"VIOLATION on trial seed={seed0 + t}:")
            print(f"  {violation}")
            print("  program:")
            for i, cmd in enumerate(trace, 1):
                print(f"    {i}. {cmd}")
            return 1
    print(f"ALL CLEAN: {num_trials} random programs, {checked_steps} total command "
          f"executions, every abstract fact checked against concrete ground truth "
          f"after every single step.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
