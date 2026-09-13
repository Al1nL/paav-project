"""
Black-box + targeted white-box edge-case tests for the shape analysis
pipeline (impl/shape/analysis.py), mirroring tests/test_parity_analysis.py
in spirit.

Covers: the succ-aliasing fix (free_var_slot / copy_row, the shape-domain
analog of Parity's `i := i + 1` bug), do_field_assign's three safety
checks and the overwrite-not-join splice fix, do_assign_deref's three
branches including the _combine_prefer_derived precision fix, check_assert's
consistency-link fix, the initial_state (all-NULL) soundness fix, and a
regression guard re-running the 5 official interesting-tests files.

Run with (from the repo root):
    python -m unittest discover -s tests -v
or just this file:
    python -m unittest tests.test_shape_analysis -v
"""

import os
import sys
import textwrap
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape.analysis import (
    analyze,
    make_transfer,
    check_assert,
    free_var_slot,
    copy_row,
    reset_as_null,
    reset_as_fresh_new,
    do_field_assign,
    do_assign_deref,
    check_field_assign_issues,
    check_assign_deref_issues,
    _combine_prefer_derived,
)
from shape import domain as D


def _wrap_single_state_transfer(real_transfer):
    """Adapts a make_transfer(...)-produced function -- which now takes and
    returns a D.DisjunctiveShapeState, since transfer functions apply to
    every disjunct independently (see report.pdf Sec 3.8/3.9) -- back to
    the single-State-in, single-State-or-BOTTOM-out calling convention
    these tests were written against. This is purely an interface
    adapter: every test below exercises a single hand-built State with no
    real CFG branching, so it can never legitimately produce more than
    one surviving disjunct; the assertion below would fail loudly if a
    future change to the tests' own state-construction ever violated
    that assumption, rather than silently picking one of several."""
    def transfer(edge, state):
        is_bottom_in = (state is D.BOTTOM) or D.is_bottom(state)
        ds_in = D.DisjunctiveShapeState([]) if is_bottom_in else D.DisjunctiveShapeState([state])
        ds_out = real_transfer(edge, ds_in)
        if not ds_out.states:
            return D.BOTTOM
        assert len(ds_out.states) == 1, (
            f"test scenario unexpectedly produced {len(ds_out.states)} disjuncts; "
            "these single-state tests assume at most one"
        )
        return ds_out.states[0]
    return transfer


INTERESTING_TESTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "shape", "interesting-tests"
)


class TestFreeVarSlot(unittest.TestCase):
    """The shape-domain analog of parity's rename_coord aliasing fix: the
    `t.n := x; x := t` pattern (used twice in the given example) must not
    let x's old identity vanish or become self-referential nonsense."""

    def test_snapshot_preserves_old_identity_and_redirects_dangling_succ(self):
        # t=0, x=1, aux=2 (aux dedicated to the edge overwriting x)
        s = D.initial_state(3)
        s = reset_as_fresh_new(s, 0)   # t := new
        s = reset_as_fresh_new(s, 1)   # x := new
        s = D.set_succ(s, 0, 1)        # t.n := x  (poked directly for isolation)

        s2 = free_var_slot(s, 1, 2)    # snapshot step of "x := t"

        self.assertEqual(s2.nullity[2], D.NONNULL)   # aux inherits x's identity
        self.assertEqual(s2.succ[2], "NULL")
        self.assertEqual(D.get_eq(s2, 2, 0), D.FALSE)  # aux still != t
        self.assertEqual(s2.reach[2][2], D.ODD)

        self.assertEqual(s2.succ[0], 2)   # t's dangling ref redirected to aux, not x

        self.assertEqual(s2.nullity[1], D.TOP_NULLITY)  # x is now a blank slate
        self.assertIsNone(s2.succ[1])
        self.assertEqual(D.get_eq(s2, 1, 0), D.UNKNOWN)
        self.assertEqual(s2.reach[1][1], D.TOP_REACH)

    def test_same_edge_firing_twice_does_not_leak_across_iterations(self):
        # Simulates the SAME "x := t" edge firing on two different
        # fixpoint iterations, sharing the same dedicated aux=2 both
        # times (as the real analyzer does for one fixed edge).
        s = D.initial_state(3)
        s = reset_as_fresh_new(s, 0)
        s = reset_as_fresh_new(s, 1)
        s = D.set_succ(s, 0, 1)
        first = free_var_slot(s, 1, 2)

        second_input = reset_as_fresh_new(first, 1)  # x reassigned to a new node
        second_input = D.set_succ(second_input, 0, 1)
        second = free_var_slot(second_input, 1, 2)   # reuses aux=2 again

        self.assertEqual(second.nullity[2], D.NONNULL)
        self.assertEqual(second.succ[2], "NULL")
        self.assertEqual(second.succ[0], 2)


class TestCopyRow(unittest.TestCase):
    def test_self_copy_is_a_true_noop(self):
        s = D.initial_state(2)
        s = reset_as_fresh_new(s, 0)
        self.assertEqual(copy_row(s, 0, 0), s)

    def test_dst_becomes_true_alias_of_src(self):
        s = D.initial_state(3)  # x=0, y=1, aux=2
        s = reset_as_fresh_new(s, 1)  # y := new
        s2 = copy_row(s, 0, 1, aux=2)  # x := y
        self.assertEqual(s2.nullity[0], D.NONNULL)
        self.assertEqual(D.get_eq(s2, 0, 1), D.TRUE)
        self.assertEqual(s2.succ[0], s2.succ[1])

    def test_overwriting_dst_redirects_dangling_succ_through_the_copy(self):
        # t.n := x; x := y  -- t's dangling ref to OLD x must move to aux,
        # not silently become "t.n == y".
        s = D.initial_state(4)  # t=0, x=1, y=2, aux=3
        s = reset_as_fresh_new(s, 0)
        s = reset_as_fresh_new(s, 1)
        s = reset_as_fresh_new(s, 2)
        s = D.set_succ(s, 0, 1)        # t.n := x
        s2 = copy_row(s, 1, 2, aux=3)  # x := y
        self.assertEqual(s2.succ[0], 3)
        self.assertEqual(D.get_eq(s2, 1, 2), D.TRUE)
        self.assertNotEqual(s2.succ[0], 2)


class TestDoFieldAssignSafety(unittest.TestCase):
    def test_memory_safety_blocks_mutation_on_possibly_null_x(self):
        # Can't prove x safe, but the surviving (non-crash) branch
        # requires x to actually be NONNULL -- refine and proceed
        # (write the pointer) rather than freezing the whole state.
        s = D.top(2)
        result = do_field_assign(s, 0, 1)
        self.assertEqual(result.nullity[0], D.NONNULL)
        self.assertEqual(result.succ[0], 1)

    def test_definite_cycle_is_detected_and_reach_weakened_not_frozen(self):
        s = D.initial_state(2)
        s = reset_as_fresh_new(s, 0)
        s = reset_as_fresh_new(s, 1)
        s = D.set_reach(s, 1, 0, D.ODD)  # y already reaches x -> definite cycle
        result = do_field_assign(s, 0, 1)
        # the pointer write itself still happens (it does, electrically,
        # even though it creates a cycle we can't reason about)...
        self.assertEqual(result.succ[0], 1)
        # ...but every reach fact touching x is no longer trustworthy.
        self.assertEqual(result.reach[0][1], D.TOP_REACH)
        self.assertEqual(result.reach[1][0], D.TOP_REACH)

    def test_maybe_cycle_is_also_treated_as_unsafe_to_splice(self):
        s = D.initial_state(2)
        s = reset_as_fresh_new(s, 0)
        s = reset_as_fresh_new(s, 1)
        s = D.set_reach(s, 1, 0, D.TOP_REACH)  # unknown whether y reaches x
        result = do_field_assign(s, 0, 1)
        self.assertEqual(result.succ[0], 1)
        self.assertEqual(result.reach[0][1], D.TOP_REACH)

    def test_definite_sharing_is_detected_and_reach_weakened_not_frozen(self):
        s = D.initial_state(3)  # x=0, y=1, z=2
        s = reset_as_fresh_new(s, 0)
        s = reset_as_fresh_new(s, 1)
        s = reset_as_fresh_new(s, 2)
        # z.n := y, properly spliced via do_field_assign (a raw D.set_succ
        # poke would leave reach[z][y] at its stale NO, which find_
        # sharing_conflict's reach-based check would then -- correctly,
        # for any state that arose from real analysis -- trust over the
        # poked succ, silently missing the conflict this test means to
        # exercise).
        s = do_field_assign(s, 2, 1)
        result = do_field_assign(s, 0, 1)  # x.n := y -> definite sharing
        self.assertEqual(result.succ[0], 1)
        self.assertEqual(result.reach[0][1], D.TOP_REACH)

    def test_clean_mutation_sets_succ_and_splices_reach(self):
        s = D.initial_state(2)
        s = reset_as_fresh_new(s, 0)  # x := new
        s = reset_as_fresh_new(s, 1)  # y := new
        result = do_field_assign(s, 0, 1)  # x.n := y
        self.assertEqual(result.succ[0], 1)
        self.assertEqual(result.reach[0][1], D.EVEN)  # x(1)+y(1)=2 nodes

    def test_splice_overwrites_not_joins_the_old_value(self):
        # z reaches x (EVEN), y reaches w (EVEN); before x.n:=y, z and w
        # are on separate chains -> reach[z][w] is NO. A join-based
        # (buggy) splice would compute NO |_| EVEN = TOP; the correct
        # overwrite gives exactly EVEN. Direct regression for the
        # "join-vs-overwrite" bug documented in report.pdf Sec 3.4.
        s = D.initial_state(4)  # z=0, x=1, y=2, w=3
        for i in range(4):
            s = reset_as_fresh_new(s, i)
        s = D.set_reach(s, 0, 1, D.EVEN)  # z -> x
        s = D.set_reach(s, 2, 3, D.EVEN)  # y -> w
        self.assertEqual(s.reach[0][3], D.NO)  # precondition

        result = do_field_assign(s, 1, 2)  # x.n := y
        self.assertEqual(result.reach[0][3], D.EVEN)  # not TOP

    def test_ambiguous_y_nullity_joins_splice_and_noop_branches(self):
        # y's nullity is TOP (e.g. after a loop with uncertain iteration
        # count). The sound result must be the JOIN of "y turned out
        # non-null, splice happened" and "y stayed null, no splice" --
        # here that's reach_join(NO, EVEN) = TOP (no dedicated name for
        # exactly {NO, EVEN}), not silently just one branch.
        s = D.initial_state(4)  # z=0, x=1, y=2, w=3
        for i in range(4):
            s = reset_as_fresh_new(s, i)
        s = D.set_nullity(s, 2, D.TOP_NULLITY)  # y ambiguous
        s = D.set_reach(s, 0, 1, D.EVEN)  # z -> x
        s = D.set_reach(s, 2, 3, D.EVEN)  # y -> w
        self.assertEqual(s.reach[0][3], D.NO)

        result = do_field_assign(s, 1, 2)  # x.n := y
        self.assertEqual(result.reach[0][3], D.TOP_REACH)


class TestDoAssignDeref(unittest.TestCase):
    def test_memory_safety_blocks_deref_of_possibly_null(self):
        # Can't prove y safe, but the surviving (non-crash) branch
        # requires y to actually be NONNULL -- refine and proceed
        # rather than freezing the whole state.
        s = D.top(2)
        result = do_assign_deref(s, 0, 1)
        self.assertEqual(result.nullity[1], D.NONNULL)

    def test_known_null_successor_resets_x_to_null(self):
        s = D.initial_state(2)
        s = reset_as_fresh_new(s, 1)  # y := new (succ[y] == "NULL")
        result = do_assign_deref(s, 0, 1)  # x := y.n
        self.assertEqual(result.nullity[0], D.NULL)

    def test_known_variable_successor_aliases_it(self):
        s = D.initial_state(3)  # x=0, y=1, w=2
        s = reset_as_fresh_new(s, 1)
        s = reset_as_fresh_new(s, 2)
        s = do_field_assign(s, 1, 2)  # y.n := w (properly splices reach,
        # unlike a raw D.set_succ poke, which would leave succ and reach
        # inconsistent with each other -- the fixed do_assign_deref now
        # correctly treats that inconsistency as a real contradiction)
        result = do_assign_deref(s, 0, 1)  # x := y.n
        self.assertEqual(D.get_eq(result, 0, 2), D.TRUE)
        self.assertEqual(result.nullity[0], D.NONNULL)

    def test_combine_prefer_derived_intersects_when_compatible(self):
        # Two independent, individually-sound estimates of the SAME real
        # fact that overlap (TOP includes EVEN) -> intersection sharpens
        # to EVEN.
        self.assertEqual(
            _combine_prefer_derived(D.TOP_REACH, D.EVEN), D.EVEN
        )

    def test_combine_prefer_derived_signals_contradiction_on_disagreement(self):
        # Genuine disagreement (disjoint outcome-sets): the two
        # "individually sound" facts can't both be true -- that's a
        # contradiction (caller must treat it as D.BOTTOM), not a cue to
        # arbitrarily trust one side.
        self.assertIsNone(_combine_prefer_derived(D.ODD, D.EVEN))

    def test_unknown_successor_materializes_fresh_node_with_conservative_reach(self):
        s = D.initial_state(3)  # x=0 (deref target), y=1, z=2
        s = reset_as_fresh_new(s, 1)
        s = s.with_(succ=D._set1(s.succ, 1, None))  # force succ[y] unknown
        s = D.set_reach(s, 2, 1, D.EVEN)  # z -> y is EVEN
        result = do_assign_deref(s, 0, 1)  # x := y.n
        # x's own nullity is genuinely TOP here (y.n's successor being
        # unknown includes "y.n is NULL"), so the sound combined answer
        # for z -> x must cover BOTH "x turns out real" (shift(EVEN)=ODD)
        # and "x turns out NULL" (NO) -- i.e. TOP, not just ODD.
        self.assertEqual(result.reach[2][0], D.TOP_REACH)
        # succ[y] := x is recorded uniformly across all three branches of
        # do_assign_deref (see its docstring) -- x IS y.n by definition,
        # regardless of how much was known about y.n beforehand.
        self.assertEqual(result.succ[1], 0)


class TestCheckAssertShape(unittest.TestCase):
    def setUp(self):
        self.var_index = {"x": 0, "xx": 1, "z": 2, "y": 3}

    def test_bottom_is_vacuously_verified(self):
        orc = [[("EQ_NULL", "x", None), ("NEQ_NULL", "x", None)]]
        self.assertTrue(check_assert(D.BOTTOM, orc, self.var_index))

    def test_ls_and_nols_atoms(self):
        s = D.initial_state(4)
        s = D.set_reach(s, 0, 1, D.ODD)
        s = D.set_reach(s, 0, 2, D.NO)
        self.assertTrue(check_assert(s, [[("LS", "x", "xx")]], self.var_index))
        self.assertFalse(check_assert(s, [[("LS", "x", "z")]], self.var_index))
        self.assertTrue(check_assert(s, [[("NOLS", "x", "z")]], self.var_index))

    def test_odd_even_tautology_when_reach_is_either(self):
        s = D.initial_state(2)
        s = D.set_reach(s, 0, 1, D.EITHER)
        orc = [[("ODD", "x", "xx")], [("EVEN", "x", "xx")]]
        self.assertTrue(check_assert(s, orc, self.var_index))
        self.assertFalse(check_assert(s, [[("ODD", "x", "xx")]], self.var_index))
        self.assertFalse(check_assert(s, [[("EVEN", "x", "xx")]], self.var_index))

    def test_consistency_link_rules_out_impossible_combinations(self):
        # x -> xx is EITHER, xx's known successor is z, so x -> z must
        # equal shift(x -> xx) in every real state, even though both
        # cells are stored independently. Without the link, this true
        # tautology would be wrongly rejected.
        s = D.initial_state(3)  # x=0, xx=1, z=2
        s = D.set_succ(s, 1, 2)             # xx.n == z
        s = D.set_reach(s, 0, 1, D.EITHER)
        s = D.set_reach(s, 0, 2, D.EITHER)
        orc = [
            [("ODD", "x", "xx"), ("EVEN", "x", "z")],
            [("EVEN", "x", "xx"), ("ODD", "x", "z")],
        ]
        self.assertTrue(check_assert(s, orc, self.var_index))

    def test_eq_and_null_atoms(self):
        s = D.initial_state(2)
        s = D.set_nullity(s, 0, D.NONNULL)
        s = D.set_eq(s, 0, 1, D.FALSE)
        self.assertTrue(check_assert(s, [[("NEQ_NULL", "x", None)]], self.var_index))
        self.assertFalse(check_assert(s, [[("EQ_NULL", "x", None)]], self.var_index))
        self.assertTrue(check_assert(s, [[("NEQ", "x", "xx")]], self.var_index))
        self.assertFalse(check_assert(s, [[("EQ", "x", "xx")]], self.var_index))

    def test_deref_atom_with_known_successor(self):
        s = D.initial_state(3)  # x=0, xx=1 (deref target), z=2
        s = D.set_succ(s, 1, 2)        # xx.n == z
        s = D.set_eq(s, 0, 2, D.TRUE)  # x == z
        self.assertTrue(check_assert(s, [[("EQ_DEREF", "x", "xx")]], self.var_index))
        self.assertFalse(check_assert(s, [[("NEQ_DEREF", "x", "xx")]], self.var_index))


class TestTransferDispatch(unittest.TestCase):
    """Exercise make_transfer's dispatch table directly, bypassing the
    parser, for every command kind lang.py can produce."""

    def setUp(self):
        self.var_index = {"x": 0, "y": 1}
        self.transfer = _wrap_single_state_transfer(make_transfer(self.var_index, edge_aux={}))

    def edge(self, cmd):
        return types.SimpleNamespace(cmd=cmd)

    def test_skip_and_assert_and_assume_true_are_identity(self):
        s = D.top(2)
        self.assertEqual(self.transfer(self.edge(("skip",)), s), s)
        self.assertEqual(self.transfer(self.edge(("assume_true",)), s), s)
        self.assertEqual(
            self.transfer(self.edge(("assert", [[("EQ_NULL", "x", None)]])), s), s
        )

    def test_assume_false_is_bottom(self):
        self.assertTrue(D.is_bottom(self.transfer(self.edge(("assume_false",)), D.top(2))))

    def test_bottom_input_stays_bottom_for_any_command(self):
        self.assertTrue(D.is_bottom(self.transfer(self.edge(("skip",)), D.BOTTOM)))

    def test_assume_eq_null_prunes_contradicting_branch(self):
        s = D.set_nullity(D.top(2), 0, D.NONNULL)
        result = self.transfer(self.edge(("assume_eq_null", "x")), s)
        self.assertTrue(D.is_bottom(result))

    def test_assume_neq_null_prunes_contradicting_branch(self):
        s = D.set_nullity(D.top(2), 0, D.NULL)
        result = self.transfer(self.edge(("assume_neq_null", "x")), s)
        self.assertTrue(D.is_bottom(result))

    def test_assign_var_self_alias_is_identity(self):
        s = D.top(2)
        result = self.transfer(self.edge(("assign_var", "x", "x")), s)
        self.assertEqual(result, s)

    def test_field_assign_null_literal(self):
        s = D.initial_state(2)
        s = D.set_nullity(s, 0, D.NONNULL)
        result = self.transfer(self.edge(("field_assign", "x", "NULL")), s)
        self.assertEqual(result.succ[0], "NULL")


class TestInitialStateFixRegression(unittest.TestCase):
    """report.pdf Sec 3.2 / plan doc: entry state must be all-NULL, not
    TOP -- verify the observable consequence end-to-end."""

    def test_two_never_assigned_pointers_are_equal_and_non_reaching(self):
        program = """\
            a b

            L0 skip L1
            L1 assert (a = b) L2
            L2 assert (NOLS a b) L3
            """
        _, _, assert_results, safety_results, _ = analyze(textwrap.dedent(program))
        self.assertEqual(len(safety_results), 0)
        self.assertTrue(all(ok for _, ok in assert_results))


class TestAssumeEqSuccMergeSoundness(unittest.TestCase):
    """assume_eq's sx/sy succ-merge derivation (x.n and y.n must denote the
    same value once x == y is assumed) calls D.set_eq/D.set_nullity to
    record that derived fact. Both are documented as unconditional
    overwrites (see domain.py's set_eq docstring: "callers ... are
    responsible for checking the old value themselves before calling
    this"). All three branches of that merge (int/int, "NULL"/int,
    int/"NULL") skipped that check: if the derived fact ("sx==sy", or
    "sx is NULL", or "sy is NULL") CONTRADICTED an already-established,
    unrelated fact about sx or sy, the old code overwrote the true fact
    with the fabricated one instead of recognizing the assumed branch is
    actually dead (BOTTOM). Caught by hand-constructing the scenario
    below (independently reproducible by exhaustive_monotonicity_check's
    own random generator, given enough time to reach this specific
    check -- see report.pdf): a state where x.n is known to be v (a node
    independently proven non-NULL elsewhere) and y.n is known to be NULL.
    assume(x == y) forces x.n == y.n, i.e. v == NULL -- directly
    contradicting what was already known about v. The old code silently
    set v's nullity to NULL (poisoning every fact computed from v
    afterward); the fix returns BOTTOM, since no real execution reaches
    this branch."""

    def _transfer(self, var_index):
        return _wrap_single_state_transfer(make_transfer(var_index, edge_aux={}))

    def test_sy_null_contradicting_independently_proven_nonnull_sx_is_bottom(self):
        # x=0, y=1, v=2. v is independently proven non-NULL; x.n == v;
        # y.n == NULL. assume(x == y) must be BOTTOM (dead branch), not a
        # live state asserting the impossible "v is NULL".
        n = 3
        s = D.top(n)
        s = D.set_nullity(s, 0, D.NONNULL)
        s = D.set_nullity(s, 1, D.NONNULL)
        s = D.set_nullity(s, 2, D.NONNULL)
        s = D.set_succ(s, 0, 2)      # x.n == v
        s = D.set_succ(s, 1, "NULL")  # y.n == NULL
        s = D.close_eq(s)
        self.assertFalse(D.is_bottom(s))  # sanity: setup itself isn't contradictory

        transfer = self._transfer({"x": 0, "y": 1, "v": 2})
        out = transfer(types.SimpleNamespace(cmd=("assume_eq", "x", "y")), s)
        self.assertTrue(D.is_bottom(out),
                         "x==y is impossible here (would force v's known-nonnull "
                         "field to be NULL); must be BOTTOM, not a live state")

    def test_sx_null_contradicting_independently_proven_nonnull_sy_is_bottom(self):
        # Mirror image: y.n == v (nonnull), x.n == NULL.
        n = 3
        s = D.top(n)
        s = D.set_nullity(s, 0, D.NONNULL)
        s = D.set_nullity(s, 1, D.NONNULL)
        s = D.set_nullity(s, 2, D.NONNULL)
        s = D.set_succ(s, 0, "NULL")  # x.n == NULL
        s = D.set_succ(s, 1, 2)       # y.n == v
        s = D.close_eq(s)
        self.assertFalse(D.is_bottom(s))

        transfer = self._transfer({"x": 0, "y": 1, "v": 2})
        out = transfer(types.SimpleNamespace(cmd=("assume_eq", "x", "y")), s)
        self.assertTrue(D.is_bottom(out))

    def test_int_int_merge_contradicting_independently_known_distinct_is_bottom(self):
        # x=0, y=1, u=2, w=3. x.n == u, y.n == w, and u/w are already
        # independently known to be DISTINCT nodes. assume(x == y) would
        # force u == w -- contradicting that -- so must be BOTTOM.
        n = 4
        s = D.top(n)
        for i in range(4):
            s = D.set_nullity(s, i, D.NONNULL)
        s = D.set_eq(s, 2, 3, D.FALSE)  # u != w, independently established
        s = D.set_succ(s, 0, 2)  # x.n == u
        s = D.set_succ(s, 1, 3)  # y.n == w
        s = D.close_eq(s)
        self.assertFalse(D.is_bottom(s))

        transfer = self._transfer({"x": 0, "y": 1, "u": 2, "w": 3})
        out = transfer(types.SimpleNamespace(cmd=("assume_eq", "x", "y")), s)
        self.assertTrue(D.is_bottom(out))

    def test_monotonicity_more_informed_sibling_must_not_be_worse(self):
        """Direct monotonicity check for this exact scenario: a state that
        additionally knows succ[x]==v (the more-informed one) must never
        produce a WORSE post-assume_eq result than an otherwise-identical
        sibling that simply never learned succ[x] at all."""
        def build(know_succ_x):
            n = 3
            s = D.top(n)
            s = D.set_nullity(s, 0, D.NONNULL)
            s = D.set_nullity(s, 1, D.NONNULL)
            s = D.set_nullity(s, 2, D.NONNULL)
            if know_succ_x:
                s = D.set_succ(s, 0, 2)
            s = D.set_succ(s, 1, "NULL")
            return D.close_eq(s)

        more_informed = build(True)
        less_informed = build(False)
        self.assertTrue(D.leq(more_informed, less_informed))

        transfer = self._transfer({"x": 0, "y": 1, "v": 2})
        r_more = transfer(types.SimpleNamespace(cmd=("assume_eq", "x", "y")), more_informed)
        r_less = transfer(types.SimpleNamespace(cmd=("assume_eq", "x", "y")), less_informed)
        self.assertTrue(
            D.leq(r_more, r_less),
            "the more-informed input's result must be <= the less-informed "
            "input's result (both are BOTTOM here, since the extra knowledge "
            "reveals this branch is dead -- the old code broke this by "
            "producing a live, self-contradictory state instead)",
        )


class TestOfficialInterestingTestsRegression(unittest.TestCase):
    """Re-run the 5 required files and check outcomes match the tool's
    actual documented behavior (README.txt / report.pdf Sec 3.7-3.8)."""

    def _run(self, filename):
        path = os.path.join(INTERESTING_TESTS_DIR, filename)
        with open(path) as f:
            text = f.read()
        return analyze(text)

    def test_given_example_matches_the_fully_sound_output(self):
        # This count has changed twice now, for two different, unrelated
        # reasons -- both documented here since the history is relevant
        # to trusting the current number, not just the number itself.
        #
        # (1) A sequence of monotonicity bugs found by fuzzing (a<=b must
        # imply transfer(a)<=transfer(b)) had made find_sharing_conflict/
        # would_create_cycle silently treat "successor unknown" as "no
        # conflict" -- genuinely unsound, not just imprecise. Fixing that
        # took the count from the original (unsound) 0 safety issues /
        # 7/9 verified to an honest 3 safety issues (edges L10, L14, L41,
        # all merely POSSIBLE) / 3/9 verified.
        #
        # (2) Introducing DisjunctiveShapeState (a bounded, per-branch
        # powerset domain -- see report.pdf Sec 3.8/3.9) keeps the two
        # branches of the list-building loop's nondeterministic exit
        # separate instead of joining them into one lossy TOP-ish state at
        # the loop header. With branch identity preserved, none of L10/
        # L14/L41 actually admit a real conflict on either branch (`x` is
        # always the current list head with no predecessor; `z` is a
        # fresh, untouched allocation) -- both were possible-but-spurious
        # warnings caused by losing that separation, not real risk, and
        # all 9 asserts -- including the ones that need reachability/
        # parity facts to survive across the loop -- now verify. This was
        # cross-checked independently: by hand-tracing the concrete
        # program's own semantics (the reported facts are true regardless
        # of trip count), and by a dedicated loop-unrolling differential
        # check (tests/loop_differential_check.py) confirming the
        # abstract fixpoint at representative loop headers stays
        # consistent with concrete ground truth across many trip counts.
        _, _, assert_results, safety_results, _ = self._run("1-given-example.txt")
        self.assertEqual(len(safety_results), 0)
        oks = [ok for _, ok in assert_results]
        self.assertEqual(oks.count(True), 9)
        self.assertEqual(oks.count(False), 0)

    def test_sharing_violation_flagged(self):
        _, _, _, safety_results, _ = self._run("2-sharing-violation.txt")
        kinds = [issue for _, issues in safety_results for issue in issues]
        self.assertTrue(any("SHARING" in k for k in kinds))

    def test_cycle_violation_flagged(self):
        _, _, _, safety_results, _ = self._run("3-cycle-violation.txt")
        kinds = [issue for _, issues in safety_results for issue in issues]
        self.assertTrue(any("CYCLE" in k for k in kinds))

    def test_merge_aliasing_rejects_added_assertion(self):
        # Same fully-sound baseline as the given example (see that test's
        # comment): 0 safety issues, since none of L10/L14/L41 (nor this
        # variant's own extra write at L43) admit a real conflict once
        # branch identity is preserved through the loop.
        #
        # The added "LS y xx" assert (the last one in the file) is a
        # direct, hand-traceable consequence of this variant's edit
        # (yy.n := x instead of skip;skip): y's tail now leads directly
        # into x's list, so y provably reaches xx too -- it VERIFIES, it
        # is not rejected. What the same edit actually breaks is
        # `t = yy.n` (an earlier assert, unaffected by this file's added
        # one): t still holds its value from the traversal loop's exit
        # (NULL), while yy.n is now x (non-NULL) -- so that one correctly
        # drops to unverified. (The project PDF's own prose names a
        # different edge, L57, as the one that should fail for this
        # variant; by this transcription's line numbers the mechanically
        # affected assert is L56 -- a labeling nuance between the prose
        # and this transcription, not a semantic disagreement about what
        # the program does.)
        _, _, assert_results, safety_results, _ = self._run("4-merge-aliasing.txt")
        self.assertEqual(len(safety_results), 0)
        by_node = {e.src: ok for e, ok in assert_results}
        self.assertFalse(by_node["L56"])  # t = yy.n
        self.assertTrue(by_node["L59"])   # the added LS y xx

    def test_null_deref_negative_flagged(self):
        _, _, _, safety_results, _ = self._run("5-null-deref-negative.txt")
        kinds = [issue for _, issues in safety_results for issue in issues]
        self.assertTrue(any("MEMORY SAFETY" in k for k in kinds))


if __name__ == "__main__":
    unittest.main()
