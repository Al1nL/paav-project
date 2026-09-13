"""
Black-box / end-to-end edge-case tests for the parity analysis pipeline
(CFG parser -> transfer functions -> fixpoint -> assert checker).

These build small inline CFG-text programs (same textual format as
parity/interesting-tests/*.txt) to pin down specific behaviors discussed
while designing the analysis: the i:=i+1 aliasing fix working through the
*real* parser and transfer dispatch (not just the domain primitive), an
assume() that proves a branch unreachable, and the conservative i:=j-1
choice. It also re-runs the project's 5 required interesting-tests files
and checks their outcome matches what README.txt documents, as a
regression guard.

Run with (from the repo root):
    python -m unittest discover -s tests -v
or just this file:
    python -m unittest tests.test_parity_analysis -v
"""

import os
import sys
import textwrap
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parity.analysis import analyze, make_transfer, check_assert, format_state
from parity import domain as D

INTERESTING_TESTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "parity", "interesting-tests"
)


def pin(state, idx, value):
    """Helper: meet state with 'variable idx has parity `value`'."""
    return D.meet_equation(state, 1 << idx, value)


def same_set(a, b):
    """Same solution set, regardless of which base point each state uses."""
    if D.is_bottom(a) or D.is_bottom(b):
        return D.is_bottom(a) and D.is_bottom(b)
    return D.leq(a, b) and D.leq(b, a)


def results_of(text):
    """Run analyze() on inline program text; return list of bool verdicts
    in the order the assert edges appear in the file."""
    _, _, results, _ = analyze(textwrap.dedent(text))
    return [ok for _, ok in results]


class TestSelfIncrementAliasingEndToEnd(unittest.TestCase):
    """The 'i := i + 1' bug (report Sec 2.3 / slides 'Trap 1'), exercised
    through the real parser + transfer dispatch, not just domain.py
    directly. If the forget-before-read bug were reintroduced, the first
    assert below would wrongly fail to verify."""

    def test_increment_makes_i_and_j_opposite_not_unrelated(self):
        program = """\
            i j

            L0 j := i L1
            L1 i := i + 1 L2
            L2 assert (ODD i EVEN j) (EVEN i ODD j) L3
            L3 assert (EVEN i EVEN j) (ODD i ODD j) L4
            """
        opposite_parity_holds, same_parity_holds = results_of(program)
        self.assertTrue(opposite_parity_holds)
        self.assertFalse(same_parity_holds)


class TestUnreachableBranchIsVacuouslyVerified(unittest.TestCase):
    """assume(i=j) after the analysis has proven i,j have permanently
    different parity must drive the state to BOTTOM (the branch is dead
    code), and any assert past that point is vacuously VERIFIED --
    including one that would be a nonsensical claim if the branch were
    actually reachable."""

    def test_dead_branch_after_impossible_assume(self):
        program = """\
            i j

            L0 i := 0 L1
            L1 j := 1 L2
            L2 assume(i = j) L3
            L3 assert (ODD i) L4
            """
        (ok,) = results_of(program)
        self.assertTrue(ok)


class TestDecrementIsConservative(unittest.TestCase):
    """i := j-1 must forget i entirely (report Sec 2.3): even though j is
    only ever the literal constant 0 here (so i really is always Even at
    runtime), the domain only tracks j's *parity*, not its exact value,
    so it cannot soundly conclude i is anything -- this is a genuine,
    documented precision loss, not a bug."""

    def test_single_disjunct_even_is_not_provable(self):
        program = """\
            i j

            L0 j := 0 L1
            L1 i := j - 1 L2
            L2 assert (EVEN i) L3
            """
        (ok,) = results_of(program)
        self.assertFalse(ok)

    def test_tautological_disjunction_still_verifies(self):
        program = """\
            i j

            L0 j := 0 L1
            L1 i := j - 1 L2
            L2 assert (EVEN i) (ODD i) L3
            """
        (ok,) = results_of(program)
        self.assertTrue(ok)


class TestRequiredInterestingTestsRegression(unittest.TestCase):
    """Re-run the 5 files required by the project spec and check their
    outcome matches what README.txt documents -- a regression guard so a
    future change to domain.py/analysis.py can't silently break one of
    the deliverable examples without a test failing."""

    EXPECTATIONS = {
        "1-given-example.txt": True,
        "2-independent-tautology.txt": True,
        "3-negative-unconditional-even.txt": False,
        "4-multihop-chain.txt": True,
        "5-double-increment-nonrelational-ok.txt": True,
    }

    def test_each_interesting_test_matches_documented_outcome(self):
        for filename, expected_all_ok in self.EXPECTATIONS.items():
            path = os.path.join(INTERESTING_TESTS_DIR, filename)
            with self.subTest(filename=filename):
                with open(path) as f:
                    text = f.read()
                _, _, results, _ = analyze(text)
                all_ok = all(ok for _, ok in results)
                self.assertEqual(all_ok, expected_all_ok)


class TestAllTransferKindsDispatch(unittest.TestCase):
    """Exercise make_transfer's dispatch table directly, for every command
    kind lang.py's parser can produce, bypassing the parser itself -- this
    catches a bug in the dispatch table (wrong index, wrong flip flag,
    wrong domain call) even if the parser's own output looks fine."""

    def setUp(self):
        self.var_index = {"i": 0, "j": 1}
        self.aux_idx = 2
        self.transfer = make_transfer(self.var_index, self.aux_idx)

    def edge(self, cmd):
        return types.SimpleNamespace(cmd=cmd)

    def test_skip_is_identity(self):
        s = D.top(3)
        self.assertIs(self.transfer(self.edge(("skip",)), s), s)

    def test_assert_is_identity(self):
        s = D.top(3)
        cmd = ("assert", [[("EVEN", "i")]])
        self.assertIs(self.transfer(self.edge(cmd), s), s)

    def test_assume_true_is_identity(self):
        s = D.top(3)
        self.assertIs(self.transfer(self.edge(("assume_true",)), s), s)

    def test_assume_false_is_bottom(self):
        s = D.top(3)
        self.assertTrue(D.is_bottom(self.transfer(self.edge(("assume_false",)), s)))

    def test_assume_neq_is_identity(self):
        s = pin(D.top(3), 0, 0)
        self.assertIs(self.transfer(self.edge(("assume_neq", "i", "j")), s), s)

    def test_assume_neq_const_is_identity(self):
        s = pin(D.top(3), 0, 0)
        self.assertIs(self.transfer(self.edge(("assume_neq_const", "i", 5)), s), s)

    def test_assign_nondet_forgets(self):
        s = pin(D.top(3), 0, 0)
        result = self.transfer(self.edge(("assign_nondet", "i")), s)
        self.assertTrue(same_set(result, D.forget(s, 0)))

    def test_assign_decr_forgets_conservatively(self):
        s = pin(D.top(3), 0, 0)
        result = self.transfer(self.edge(("assign_decr", "i", "j")), s)
        self.assertTrue(same_set(result, D.forget(s, 0)))

    def test_assign_const_dispatches_correct_parity(self):
        s = D.top(3)
        even_result = self.transfer(self.edge(("assign_const", "i", 4)), s)
        self.assertTrue(D.entails(even_result, 1 << 0, 0))
        odd_result = self.transfer(self.edge(("assign_const", "i", 7)), s)
        self.assertTrue(D.entails(odd_result, 1 << 0, 1))

    def test_assign_var_and_assign_incr_dispatch_correct_flip(self):
        s = pin(D.top(3), 1, 0)  # j = Even
        copy_result = self.transfer(self.edge(("assign_var", "i", "j")), s)
        self.assertTrue(D.entails(copy_result, 1 << 0, 0))  # i = Even (copy)
        incr_result = self.transfer(self.edge(("assign_incr", "i", "j")), s)
        self.assertTrue(D.entails(incr_result, 1 << 0, 1))  # i = Odd (flip)

    def test_assume_eq_and_assume_eq_const_dispatch(self):
        s = D.top(3)
        eq_result = self.transfer(self.edge(("assume_eq", "i", "j")), s)
        c = (1 << 0) ^ (1 << 1)
        self.assertTrue(D.entails(eq_result, c, 0))
        eqk_result = self.transfer(self.edge(("assume_eq_const", "i", 3)), s)
        self.assertTrue(D.entails(eqk_result, 1 << 0, 1))

    def test_unknown_command_kind_raises(self):
        with self.assertRaises(ValueError):
            self.transfer(self.edge(("not_a_real_kind",)), D.top(3))


class TestCheckAssertDirectly(unittest.TestCase):
    """Exercise check_assert/_orc_holds directly with hand-built states and
    ORC structures, isolating AND-within-conjunct / OR-across-disjuncts
    semantics and the complementary-disjunct tautology case (report Sec
    2.4) from the rest of the pipeline."""

    def setUp(self):
        self.var_index = {"i": 0, "j": 1, "k": 2}

    def test_single_atom_pass_and_fail(self):
        s = pin(D.top(3), 0, 0)  # i = Even
        self.assertTrue(check_assert(s, [[("EVEN", "i")]], self.var_index))
        self.assertFalse(check_assert(s, [[("ODD", "i")]], self.var_index))

    def test_conjunct_requires_all_atoms(self):
        s = pin(pin(D.top(3), 0, 0), 1, 0)  # i=Even, j=Even
        self.assertTrue(
            check_assert(s, [[("EVEN", "i"), ("EVEN", "j")]], self.var_index)
        )
        # one atom of the conjunct is false -> the whole conjunct fails
        self.assertFalse(
            check_assert(s, [[("EVEN", "i"), ("ODD", "j")]], self.var_index)
        )

    def test_disjunction_succeeds_if_any_disjunct_holds(self):
        s = pin(D.top(3), 0, 0)  # i = Even
        orc = [[("ODD", "i")], [("EVEN", "i")]]  # first fails, second holds
        self.assertTrue(check_assert(s, orc, self.var_index))

    def test_complementary_disjuncts_tautology_when_related(self):
        # i == k, both individually free: (EVEN i EVEN k) or (ODD i ODD k)
        # is a tautology over the state, though NEITHER disjunct alone is
        # entailed -- this is report Sec 2.4's precision fix.
        c = (1 << 0) ^ (1 << 2)
        s = D.meet_equation(D.top(3), c, 0)
        orc = [[("EVEN", "i"), ("EVEN", "k")], [("ODD", "i"), ("ODD", "k")]]
        self.assertTrue(check_assert(s, orc, self.var_index))

    def test_same_orc_fails_when_variables_are_forced_opposite(self):
        # Contrast: if i and k are forced to OPPOSITE parity, the same
        # "same-parity" disjunction must correctly fail.
        c = (1 << 0) ^ (1 << 2)
        s = D.meet_equation(D.top(3), c, 1)  # i != k
        orc = [[("EVEN", "i"), ("EVEN", "k")], [("ODD", "i"), ("ODD", "k")]]
        self.assertFalse(check_assert(s, orc, self.var_index))

    def test_bottom_state_is_vacuously_verified_for_any_orc(self):
        orc = [[("EVEN", "i"), ("ODD", "i")]]  # nonsensical if reachable
        self.assertTrue(check_assert(D.BOTTOM, orc, self.var_index))

    def test_oversized_coset_raises_instead_of_hanging(self):
        big_var_index = {f"v{n}": n for n in range(25)}
        s = D.top(25)
        with self.assertRaises(RuntimeError):
            check_assert(s, [[("EVEN", "v0")]], big_var_index)


class TestBranchMergeWithDeadPath(unittest.TestCase):
    """A diamond CFG where one branch is provably infeasible: the merge
    point must reflect ONLY the live branch's facts, not be weakened by
    joining with the dead branch's (BOTTOM) contribution."""

    def test_dead_branch_does_not_weaken_the_merge(self):
        program = """\
            i j

            L0 i := 0 L1
            L1 assume(i = 5) L2
            L1 assume(i != 5) L3
            L2 j := 1 L4
            L3 j := 0 L4
            L4 assert (EVEN j) L5
            """
        # i is pinned Even by L0; assume(i=5) demands parity(i)=Odd (5 is
        # odd) -> contradiction -> BOTTOM down the L2 branch. Only L3's
        # j:=0 should reach L4.
        (ok,) = results_of(program)
        self.assertTrue(ok)

    def test_control_case_two_live_branches_do_weaken_the_merge(self):
        # Same shape, but BOTH branches are now genuinely reachable (i is
        # left nondeterministic instead of pinned), so the merge legally
        # sees both j=Odd and j=Even -> must NOT verify EVEN j. This
        # confirms the test above is actually precision-sensitive, not
        # vacuously true regardless of how joins are implemented.
        program = """\
            i j

            L0 i := ? L1
            L1 assume(i = 5) L2
            L1 assume(i != 5) L3
            L2 j := 1 L4
            L3 j := 0 L4
            L4 assert (EVEN j) L5
            """
        (ok,) = results_of(program)
        self.assertFalse(ok)


class TestUnreachableDisconnectedComponent(unittest.TestCase):
    """A CFG node with no path from the entry node at all must sit at
    BOTTOM for its whole life, and any assert on it is vacuously VERIFIED
    -- even one asserting something that would obviously be false in any
    real execution reaching it."""

    def test_disconnected_component_assert_is_vacuously_verified(self):
        program = """\
            i

            L0 i := 0 L1
            L1 skip L1x

            Z0 skip Z1
            Z1 assert (ODD i) Z2
            """
        # Two assert-free structural notes: L1x is just a dead end for the
        # real (reachable) component; Z0/Z1/Z2 are entirely disconnected
        # from L0. "ODD i" would contradict the reachable component's
        # invariant (i is always Even) if it were ever reachable -- but
        # it's never reached at all.
        _, _, results, _ = analyze(textwrap.dedent(program))
        self.assertEqual(len(results), 1)
        _, ok = results[0]
        self.assertTrue(ok)


class TestFormatState(unittest.TestCase):
    """format_state() is currently unused by the CLI (see review), but
    it's real, reachable code -- lock down its behavior so wiring it into
    run_file later (or leaving it as a debug helper) doesn't silently
    regress."""

    def test_shows_relation_when_neither_side_individually_pinned(self):
        program = """\
            i j

            L0 i := ? L1
            L1 j := i L2
            """
        cfg, pre, _, _ = analyze(textwrap.dedent(program))
        text = format_state(pre["L2"], cfg)
        self.assertIn("parity(i)=parity(j)", text)
        self.assertNotIn("Even", text)
        self.assertNotIn("Odd", text)

    def test_shows_individual_facts_instead_of_relation_when_pinned(self):
        program = """\
            i j k

            L0 i := 0 L1
            L1 j := i L2
            L2 k := ? L3
            """
        cfg, pre, _, _ = analyze(textwrap.dedent(program))
        text = format_state(pre["L3"], cfg)
        self.assertIn("i=Even", text)
        self.assertIn("j=Even", text)
        self.assertNotIn("k=Even", text)
        self.assertNotIn("k=Odd", text)
        self.assertNotIn("parity(i)=parity(j)", text)  # both already pinned

    def test_top_state_message(self):
        program = """\
            i

            L0 i := ? L1
            """
        cfg, pre, _, _ = analyze(textwrap.dedent(program))
        self.assertEqual(format_state(pre["L1"], cfg), "TOP (no constraints)")

    def test_unreachable_state_message(self):
        program = """\
            i

            L0 assume(FALSE) L1
            """
        cfg, pre, _, _ = analyze(textwrap.dedent(program))
        self.assertEqual(format_state(pre["L1"], cfg), "UNREACHABLE")


if __name__ == "__main__":
    unittest.main()
