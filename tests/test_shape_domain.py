"""
White-box edge-case tests for the shape domain (impl/shape/domain.py).

Mirrors tests/test_parity_domain.py in spirit: these poke the abstract
domain's primitives directly, independent of the CFG parser. The `reach`
lattice (NO/ODD/EVEN/EITHER/TOP, i.e. named subsets of {NO,ODD,EVEN}) is
tested exhaustively against hand-derived truth tables, since it's small
enough to enumerate completely and is the one place a typo would be easy
to miss by inspection alone.

Run with (from the repo root):
    python -m unittest discover -s tests -v
or just this file:
    python -m unittest tests.test_shape_domain -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape import domain as D

NO, ODD, EVEN, EITHER, TOP = D.NO, D.ODD, D.EVEN, D.EITHER, D.TOP_REACH
ALL_REACH = [NO, ODD, EVEN, EITHER, TOP]


class TestReachJoin(unittest.TestCase):
    EXPECTED = {
        (NO, NO): NO, (ODD, ODD): ODD, (EVEN, EVEN): EVEN,
        (EITHER, EITHER): EITHER, (TOP, TOP): TOP,
        (NO, ODD): TOP, (NO, EVEN): TOP, (NO, EITHER): TOP, (NO, TOP): TOP,
        (ODD, EVEN): EITHER, (ODD, EITHER): EITHER, (ODD, TOP): TOP,
        (EVEN, EITHER): EITHER, (EVEN, TOP): TOP,
        (EITHER, TOP): TOP,
    }

    def test_full_truth_table_both_orders(self):
        for (a, b), expected in self.EXPECTED.items():
            with self.subTest(a=a, b=b):
                self.assertEqual(D.reach_join(a, b), expected)
                self.assertEqual(D.reach_join(b, a), expected)  # commutative

    def test_covers_every_unordered_pair(self):
        seen = {frozenset((a, b)) for a, b in self.EXPECTED} | {
            frozenset((a, a)) for a in ALL_REACH
        }
        all_pairs = {frozenset((a, b)) for a in ALL_REACH for b in ALL_REACH}
        self.assertEqual(seen, all_pairs)


class TestReachLeq(unittest.TestCase):
    # (a, b) -> a <= b
    EXPECTED = {
        (NO, NO): True, (NO, ODD): False, (NO, EVEN): False,
        (NO, EITHER): False, (NO, TOP): True,
        (ODD, NO): False, (ODD, ODD): True, (ODD, EVEN): False,
        (ODD, EITHER): True, (ODD, TOP): True,
        (EVEN, NO): False, (EVEN, ODD): False, (EVEN, EVEN): True,
        (EVEN, EITHER): True, (EVEN, TOP): True,
        (EITHER, NO): False, (EITHER, ODD): False, (EITHER, EVEN): False,
        (EITHER, EITHER): True, (EITHER, TOP): True,
        (TOP, NO): False, (TOP, ODD): False, (TOP, EVEN): False,
        (TOP, EITHER): False, (TOP, TOP): True,
    }

    def test_full_truth_table(self):
        for (a, b), expected in self.EXPECTED.items():
            with self.subTest(a=a, b=b):
                self.assertEqual(D.reach_leq(a, b), expected)

    def test_covers_every_ordered_pair(self):
        self.assertEqual(set(self.EXPECTED), {(a, b) for a in ALL_REACH for b in ALL_REACH})

    def test_leq_agrees_with_join_as_upper_bound(self):
        # a <= join(a,b) and b <= join(a,b) for every pair (standard join law)
        for a in ALL_REACH:
            for b in ALL_REACH:
                j = D.reach_join(a, b)
                with self.subTest(a=a, b=b):
                    self.assertTrue(D.reach_leq(a, j))
                    self.assertTrue(D.reach_leq(b, j))


class TestReachShift(unittest.TestCase):
    EXPECTED = {NO: NO, ODD: EVEN, EVEN: ODD, EITHER: EITHER, TOP: TOP}

    def test_full_table(self):
        for a, expected in self.EXPECTED.items():
            with self.subTest(a=a):
                self.assertEqual(D.reach_shift(a), expected)

    def test_shift_is_its_own_inverse_on_parity(self):
        # shifting twice returns to the original (extra hop, then another
        # extra hop, flips parity twice)
        for a in ALL_REACH:
            self.assertEqual(D.reach_shift(D.reach_shift(a)), a)


class TestCombineSeq(unittest.TestCase):
    EXPECTED = {}
    for x in ALL_REACH:
        EXPECTED[(NO, x)] = NO
        EXPECTED[(x, NO)] = NO
    EXPECTED[(ODD, ODD)] = ODD
    EXPECTED[(ODD, EVEN)] = EVEN
    EXPECTED[(EVEN, ODD)] = EVEN
    EXPECTED[(EVEN, EVEN)] = ODD
    for x in (ODD, EVEN, EITHER):
        EXPECTED[(EITHER, x)] = EITHER
        EXPECTED[(x, EITHER)] = EITHER
    EXPECTED[(EITHER, EITHER)] = EITHER
    for x in (ODD, EVEN, EITHER, TOP):
        EXPECTED[(TOP, x)] = TOP
        EXPECTED[(x, TOP)] = TOP
    EXPECTED[(TOP, TOP)] = TOP

    def test_full_truth_table(self):
        for (a, b), expected in self.EXPECTED.items():
            with self.subTest(a=a, b=b):
                self.assertEqual(D.combine_seq(a, b), expected)

    def test_covers_every_ordered_pair(self):
        self.assertEqual(set(self.EXPECTED), {(a, b) for a in ALL_REACH for b in ALL_REACH})

    def test_two_definite_segments_compose_by_parity_arithmetic(self):
        # z->mid is ODD (e.g. 1 node: z==mid) and mid->w is ODD (1 node:
        # mid==w) -- concretely z==mid==w, so z->w is trivially 1 node
        # (ODD), matching "same-parity halves -> ODD (shared node counted
        # once)".
        self.assertEqual(D.combine_seq(ODD, ODD), ODD)
        # z->mid 2 nodes (EVEN), mid->w 2 nodes (EVEN): concrete chain
        # z,a,mid,b,w minus the double-counted mid = 2+2-1 = 3 nodes = ODD.
        self.assertEqual(D.combine_seq(EVEN, EVEN), ODD)
        # z->mid 2 nodes (EVEN), mid->w 1 node (ODD, mid==w): total is
        # just z->mid's 2 nodes = EVEN.
        self.assertEqual(D.combine_seq(EVEN, ODD), EVEN)


class TestReachMeet(unittest.TestCase):
    EXPECTED = {
        (NO, NO): NO, (NO, ODD): None, (NO, EVEN): None, (NO, EITHER): None, (NO, TOP): NO,
        (ODD, ODD): ODD, (ODD, EVEN): None, (ODD, EITHER): ODD, (ODD, TOP): ODD,
        (EVEN, EVEN): EVEN, (EVEN, EITHER): EVEN, (EVEN, TOP): EVEN,
        (EITHER, EITHER): EITHER, (EITHER, TOP): EITHER,
        (TOP, TOP): TOP,
    }

    def test_full_truth_table_both_orders(self):
        for (a, b), expected in self.EXPECTED.items():
            with self.subTest(a=a, b=b):
                self.assertEqual(D.reach_meet(a, b), expected)
                self.assertEqual(D.reach_meet(b, a), expected)

    def test_meet_of_disjoint_singles_signals_contradiction(self):
        # ODD and EVEN share no concrete outcome: if x and y are truly the
        # same node, both facts can't hold simultaneously, so this is a
        # genuine contradiction, not "provably NO". An earlier version of
        # reach_meet silently fell back to NO here, which is unsound (NO
        # is not a valid over-approximation of an empty set of
        # possibilities) and was a real, fuzzer-caught monotonicity bug
        # in transfer assume_eq(x,y) -- see analysis.py's assume_eq
        # handling, which now treats this None as BOTTOM.
        self.assertIsNone(D.reach_meet(ODD, EVEN))


class TestTopAndInitialState(unittest.TestCase):
    def test_top_is_fully_unconstrained(self):
        s = D.top(3)
        for i in range(3):
            self.assertEqual(s.nullity[i], D.TOP_NULLITY)
            self.assertIsNone(s.succ[i])
            for j in range(3):
                if i != j:
                    self.assertEqual(s.eq[i][j], D.UNKNOWN)
                    self.assertEqual(s.reach[i][j], TOP)
                else:
                    self.assertEqual(s.eq[i][j], D.TRUE)

    def test_initial_state_is_all_null(self):
        s = D.initial_state(3)
        for i in range(3):
            self.assertEqual(s.nullity[i], D.NULL)
            self.assertIsNone(s.succ[i])
            for j in range(3):
                self.assertEqual(s.eq[i][j], D.TRUE)  # every NULL == every NULL
                self.assertEqual(s.reach[i][j], NO)    # NULL reaches nothing


class TestStateJoinAndLeq(unittest.TestCase):
    def test_join_with_bottom_is_identity(self):
        s = D.top(2)
        self.assertEqual(D.join(D.BOTTOM, s), s)
        self.assertEqual(D.join(s, D.BOTTOM), s)
        self.assertTrue(D.is_bottom(D.join(D.BOTTOM, D.BOTTOM)))

    def test_join_agrees_on_nullity_keeps_it_disagrees_goes_top(self):
        a = D.set_nullity(D.top(2), 0, D.NONNULL)
        b = D.set_nullity(D.top(2), 0, D.NONNULL)
        j = D.join(a, b)
        self.assertEqual(j.nullity[0], D.NONNULL)

        c = D.set_nullity(D.top(2), 0, D.NULL)
        j2 = D.join(a, c)
        self.assertEqual(j2.nullity[0], D.TOP_NULLITY)

    def test_join_agrees_on_succ_keeps_it_disagrees_goes_none(self):
        a = D.set_succ(D.top(2), 0, "NULL")
        b = D.set_succ(D.top(2), 0, "NULL")
        self.assertEqual(D.join(a, b).succ[0], "NULL")

        c = D.set_succ(D.top(2), 0, 1)
        self.assertIsNone(D.join(a, c).succ[0])

    def test_leq_reflexive_and_bottom(self):
        s = D.top(2)
        self.assertTrue(D.leq(s, s))
        self.assertTrue(D.leq(D.BOTTOM, s))
        self.assertFalse(D.leq(s, D.BOTTOM))

    def test_leq_false_when_nullity_disagrees_on_a_pinned_field(self):
        a = D.set_nullity(D.top(2), 0, D.NULL)
        b = D.set_nullity(D.top(2), 0, D.NONNULL)
        self.assertFalse(D.leq(a, b))
        self.assertTrue(D.leq(a, D.top(2)))  # top has no opinion -> always ok

    def test_leq_false_when_succ_disagrees_on_a_pinned_field(self):
        a = D.set_succ(D.top(2), 0, 1)
        b = D.set_succ(D.top(2), 0, "NULL")
        self.assertFalse(D.leq(a, b))
        self.assertTrue(D.leq(a, D.top(2)))


if __name__ == "__main__":
    unittest.main()
