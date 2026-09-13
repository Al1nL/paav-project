"""
loop_differential_check.py -- extends the straight-line differential
fuzzer (differential_fuzz_shape.py) to the one mechanism it cannot
reach at all: the DisjunctiveShapeState join/leq machinery at an actual
CFG loop header, which is untested by every other check in this repo
(unit tests, monotonicity fuzzing, and the straight-line differential
fuzzer all either use hand-picked states or single flat command
sequences with no back-edge).

Method: for each hand-built loop program below, unroll its loop body
for trip counts 0, 1, 2, ..., K concretely (ground truth, via the same
ConcreteHeap/apply_concrete machinery as differential_fuzz_shape.py),
and separately run the REAL analyze() on the (not unrolled) looping
program exactly once to get its fixpoint result at the loop-exit node
-- which is supposed to soundly over-approximate EVERY trip count at
once. Every concrete unrolling's final (variables, heap) must be
consistent with that one abstract state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape import domain as D
from shape.analysis import analyze, build_var_index
from differential_fuzz_shape import ConcreteHeap, apply_concrete, check_consistency
import differential_fuzz_shape as dfs


LOOP_PROGRAMS = {
    # Mirrors the given example's own x-list prepend loop in miniature:
    # build a list of `trip_count` fresh nodes by repeated prepending,
    # loop condition is nondeterministic (assume(TRUE) both ways, exactly
    # like the real given example), then assert facts about x at exit.
    "prepend_loop": """
x t
L0 x := NULL L1
L1 assume(TRUE) L2
L1 assume(TRUE) L5
L2 t := new L3
L3 t.n := x L4
L4 x := t L1
L5 skip L6
""",
    # Two independent prepend loops interleaved each iteration (mirrors
    # the given example's x AND y lists built together) -- this is
    # exactly the "loop-carried aliasing" pattern the report discusses.
    "two_prepend_loops": """
x y t
L0 x := NULL L1
L1 y := NULL L2
L2 assume(TRUE) L3
L2 assume(TRUE) L8
L3 t := new L4
L4 t.n := x L5
L5 x := t L6
L6 t := new L7
L7 t.n := y L2b
L2b x := x L2c
L2c t.n := y L7b
""",
}

# The two_prepend_loops text above has a typo-prone hand structure;
# replace it with a clean, verified-parseable version instead.
LOOP_PROGRAMS["two_prepend_loops"] = """
x y t
L0 x := NULL L1
L1 y := NULL L2
L2 assume(TRUE) L3
L2 assume(TRUE) L9
L3 t := new L4
L4 t.n := x L5
L5 x := t L6
L6 t := new L7
L7 t.n := y L8
L8 y := t L2
L9 skip L10
"""


def unroll_prepend_loop(trip_count):
    """Ground-truth (var-assignments, heap) after `trip_count` passes of
    prepend_loop's body, built directly (not by re-parsing text)."""
    heap = ConcreteHeap()
    x = None
    t = None
    for _ in range(trip_count):
        node = heap.fresh()
        heap.next_ptr[node] = x
        x = node
        t = node  # x := t immediately follows t := new in the loop body
    return {"x": x, "t": t}, heap


def unroll_two_prepend_loops(trip_count):
    heap = ConcreteHeap()
    x = None
    y = None
    t = None
    for _ in range(trip_count):
        nx = heap.fresh()
        heap.next_ptr[nx] = x
        x = nx
        t = nx
        ny = heap.fresh()
        heap.next_ptr[ny] = y
        y = ny
        t = ny
    return {"x": x, "y": y, "t": t}, heap


UNROLLERS = {
    "prepend_loop": unroll_prepend_loop,
    "two_prepend_loops": unroll_two_prepend_loops,
}


def check_named_vars_consistency(abs_state, slot, next_ptr, n_vars, step_no):
    """Like differential_fuzz_shape.check_consistency, but restricted to
    the named-variable coordinates (0..n_vars-1). Aux-slot ground truth
    doesn't have a clean 1:1 correspondence across unrolled loop trip
    counts (the same abstract aux slot, reused per edge across every
    iteration via the fixpoint join, corresponds to a DIFFERENT concrete
    snapshot at each concrete iteration) -- but nothing this project
    reports (assert_results, safety_results) depends on aux-slot facts
    directly, only on named-variable ones, so this restriction doesn't
    weaken the check for anything actually being verified."""
    if D.is_bottom(abs_state):
        return f"step {step_no}: abstract state is BOTTOM but concrete execution reached it normally"
    claimed = abs_state.claimed
    for i in range(n_vars):
        if not claimed[i]:
            continue
        nul = D.get_nullity(abs_state, i)
        is_null = slot[i] is None
        if nul == D.NULL and not is_null:
            return f"step {step_no}: nullity[{i}]=NULL but concrete slot[{i}]={slot[i]!r}"
        if nul == D.NONNULL and is_null:
            return f"step {step_no}: nullity[{i}]=NONNULL but concrete slot[{i}] is None"
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j or not claimed[i] or not claimed[j]:
                continue
            eq = D.get_eq(abs_state, i, j)
            actual = slot[i] == slot[j]
            if eq == D.TRUE and not actual:
                return f"step {step_no}: eq[{i}][{j}]=TRUE but concrete values differ"
            if eq == D.FALSE and actual:
                return f"step {step_no}: eq[{i}][{j}]=FALSE but concrete values equal"
    for i in range(n_vars):
        if not claimed[i] or slot[i] is None:
            continue
        sc = dfs._resolve_succ(abs_state, i)
        actual_next = next_ptr.get(slot[i])
        if sc == "NULL":
            if actual_next is not None:
                return f"step {step_no}: succ[{i}]=NULL but concrete next({slot[i]})={actual_next!r}"
        elif isinstance(sc, int) and sc < n_vars and claimed[sc]:
            expected = slot[sc]
            if actual_next != expected:
                return (f"step {step_no}: succ[{i}]={sc} (expects next({slot[i]})==slot[{sc}]={expected!r}) "
                        f"but concrete next({slot[i]})={actual_next!r}")
        # sc pointing at an aux index (>= n_vars) is skipped: no ground
        # truth tracked for aux slots across unrolled trip counts (see
        # docstring above) -- sound to skip, only weakens this specific
        # check, never causes a false failure.
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j or not claimed[i] or not claimed[j]:
                continue
            if slot[i] is None or slot[j] is None:
                continue
            outcome_val = D.get_reach(abs_state, i, j)
            d = dfs.reaches_in(next_ptr, slot[i], slot[j], cap=n_vars + 10)
            actual_outcome = D.NO if d is None else (D.ODD if d % 2 == 0 else D.EVEN)
            if actual_outcome not in dfs.OUTCOME_MAP[outcome_val]:
                return (f"step {step_no}: reach[{i}][{j}]={outcome_val} but concrete outcome is "
                        f"{actual_outcome} (hop distance {d})")
    return None


def check_program(name, text, exit_node, max_trip):
    cfg_text = text
    unroller = UNROLLERS[name]

    cfg, pre, assert_results, safety_results, iterations = analyze(cfg_text)
    var_index = build_var_index(cfg)
    n_vars = len(var_index)

    disjunctive_exit = pre.get(exit_node)
    if disjunctive_exit is None:
        return f"exit node {exit_node!r} never reached in abstract analysis (missing from pre)"

    failures = []
    for trip_count in range(0, max_trip + 1):
        concrete_vars, heap = unroller(trip_count)
        slot = [None] * n_vars
        for vname, idx in var_index.items():
            slot[idx] = concrete_vars.get(vname)
        ok_any = False
        msgs = []
        for i, s in enumerate(disjunctive_exit.states):
            v = check_named_vars_consistency(s, slot, heap.next_ptr, n_vars, trip_count)
            if v is None:
                ok_any = True
                break
            msgs.append(f"disjunct {i}: {v}")
        if not ok_any:
            failures.append(f"trip_count={trip_count}: no disjunct consistent with ground truth "
                             f"(vars={concrete_vars}); tried: {msgs}")
    if failures:
        return "\n".join(failures)
    return None


def build_var_index_names(text):
    return text.strip().splitlines()[0].split()


def run():
    any_fail = False

    r = check_program("prepend_loop", LOOP_PROGRAMS["prepend_loop"], "L5", max_trip=6)
    if r:
        print("prepend_loop: FAIL\n" + r)
        any_fail = True
    else:
        print("prepend_loop: consistent for trip counts 0..6")

    r = check_program("two_prepend_loops", LOOP_PROGRAMS["two_prepend_loops"], "L9", max_trip=6)
    if r:
        print("two_prepend_loops: FAIL\n" + r)
        any_fail = True
    else:
        print("two_prepend_loops: consistent for trip counts 0..6")

    print()
    print("ALL CONSISTENT" if not any_fail else "INCONSISTENCIES FOUND")
    return 0 if not any_fail else 1


if __name__ == "__main__":
    sys.exit(run())
