"""
White-box edge-case tests for the parity domain (impl/parity/domain.py).

These are *not* the project's required "interesting test programs" (those
live in parity/interesting-tests/ and are exercised end-to-end through the
CFG parser). This file instead pokes the abstract domain's primitives
directly -- the goal is to pin down exactly the tricky cases discussed
while designing it: BOTTOM propagation, the aliasing/rename fix, forget as
existential quantification, and join both losing and keeping relations.

Run with (from the repo root):
    python -m unittest discover -s tests -v
or just this file:
    python -m unittest tests.test_parity_domain -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parity import domain as D


def pin(state, idx, value):
    """Helper: meet state with 'variable idx has parity `value`'."""
    return D.meet_equation(state, 1 << idx, value)


def same_set(a, b):
    """Two states represent the same solution set, regardless of which
    concrete (p, basis) representative each happens to use. Prefer this
    over dataclass equality whenever comparing states built via different
    code paths (e.g. join(a,b) vs join(b,a)) -- they can legitimately pick
    different base points for the same underlying flat."""
    if D.is_bottom(a) or D.is_bottom(b):
        return D.is_bottom(a) and D.is_bottom(b)
    return D.leq(a, b) and D.leq(b, a)


class TestBasisAndSpan(unittest.TestCase):
    def test_top_has_full_independent_basis(self):
        s = D.top(3)
        self.assertEqual(len(s.basis), 3)
        # every single variable is unconstrained
        for i in range(3):
            self.assertFalse(D.entails(s, 1 << i, 0))
            self.assertFalse(D.entails(s, 1 << i, 1))

    def test_reduce_basis_dedups_and_drops_dependent_vectors(self):
        # 0b01, 0b10, and 0b11 (= 01 ^ 10) are only rank 2; a zero vector
        # and a duplicate must also vanish.
        basis = D.reduce_basis([0b01, 0b10, 0b11, 0b01, 0b00])
        self.assertEqual(len(basis), 2)
        for v in [0b01, 0b10, 0b11]:
            self.assertTrue(D.in_span(v, basis))

    def test_in_span_trivial_zero_vector(self):
        self.assertTrue(D.in_span(0, (0b01, 0b10)))
        self.assertFalse(D.in_span(0b11, (0b01,)))


class TestMeetEquation(unittest.TestCase):
    def test_pin_single_variable(self):
        s = pin(D.top(2), 0, 0)  # i := Even
        self.assertTrue(D.entails(s, 1 << 0, 0))
        self.assertFalse(D.entails(s, 1 << 0, 1))
        # j (index 1) must remain completely untouched
        self.assertFalse(D.entails(s, 1 << 1, 0))
        self.assertFalse(D.entails(s, 1 << 1, 1))

    def test_re_asserting_same_fact_is_a_noop(self):
        s = pin(D.top(2), 0, 0)
        s2 = pin(s, 0, 0)
        self.assertEqual(s, s2)

    def test_contradicting_a_pinned_fact_is_bottom(self):
        s = pin(D.top(2), 0, 0)  # i = Even
        s2 = pin(s, 0, 1)        # now demand i = Odd
        self.assertTrue(D.is_bottom(s2))

    def test_relational_fact_pins_neither_variable_alone(self):
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(2), c, 0)  # parity(i) = parity(j)
        self.assertTrue(D.entails(s, c, 0))
        for idx in (0, 1):
            self.assertFalse(D.entails(s, 1 << idx, 0))
            self.assertFalse(D.entails(s, 1 << idx, 1))

    def test_pinning_one_side_of_a_relation_pins_the_other(self):
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(2), c, 0)  # i == j (parity)
        s = pin(s, 0, 1)                      # now force i = Odd
        self.assertTrue(D.entails(s, 1 << 1, 1))  # j must be Odd too

    def test_relation_plus_opposite_pin_is_contradiction(self):
        # i == j, but also i = Even and j = Odd -> impossible
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(2), c, 0)
        s = pin(s, 0, 0)   # i = Even
        s = pin(s, 1, 1)   # j = Odd  (contradicts i == j)
        self.assertTrue(D.is_bottom(s))


class TestForget(unittest.TestCase):
    def test_forget_undoes_a_pin(self):
        s = pin(D.top(1), 0, 0)
        s = D.forget(s, 0)
        self.assertFalse(D.entails(s, 1 << 0, 0))
        self.assertFalse(D.entails(s, 1 << 0, 1))
        self.assertEqual(s, D.top(1))

    def test_forgetting_one_side_of_a_relation_frees_both(self):
        # Existentially quantifying j out of "i == j" must yield TRUE
        # (i unconstrained): for any i there EXISTS a matching j.
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(2), c, 0)
        s = D.forget(s, 1)  # forget j
        self.assertEqual(s, D.top(2))

    def test_forget_on_bottom_is_bottom(self):
        self.assertTrue(D.is_bottom(D.forget(D.BOTTOM, 0)))


class TestAssignLinearAliasing(unittest.TestCase):
    """Regression tests for the 'i := i + 1' aliasing bug (report Sec 2.3)."""

    def test_self_increment_flips_a_pinned_parity(self):
        # n=1 real var + 1 aux slot -> aux_idx = 1
        s = pin(D.top(2), 0, 0)  # i = Even
        s = D.assign_linear(s, 0, 0, flip=True, aux_idx=1)  # i := i + 1
        self.assertTrue(D.entails(s, 1 << 0, 1))  # i must now be Odd

    def test_self_increment_preserves_relation_to_other_variable(self):
        # THE key regression test: i and j start related (i == j). After
        # i := i + 1, the naive "forget i, then relate to j" bug would
        # destroy this relation entirely (i becomes fully free). The
        # correct behavior is that i and j become OPPOSITE parity.
        n = 2  # i=0, j=1, aux=2
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(n + 1), c, 0)  # i == j
        s = D.assign_linear(s, 0, 0, flip=True, aux_idx=n)  # i := i + 1
        self.assertTrue(D.entails(s, c, 1))   # now i and j must DIFFER
        self.assertFalse(D.entails(s, c, 0))
        # neither is individually pinned -- only the relation is known
        for idx in (0, 1):
            self.assertFalse(D.entails(s, 1 << idx, 0))
            self.assertFalse(D.entails(s, 1 << idx, 1))

    def test_non_aliased_assignment_still_works(self):
        # sanity check: i := j + 1 for DISTINCT i, j (no aliasing at all)
        n = 2  # i=0, j=1, aux=2
        s = pin(D.top(n + 1), 1, 0)  # j = Even
        s = D.assign_linear(s, 0, 1, flip=True, aux_idx=n)  # i := j + 1
        self.assertTrue(D.entails(s, 1 << 0, 1))  # i must be Odd

    def test_scratch_coordinate_is_free_again_afterwards(self):
        # The aux slot must never leak a stale constraint into later use.
        n = 2
        aux = n
        s = pin(D.top(n + 1), 1, 1)
        s = D.assign_linear(s, 0, 1, flip=False, aux_idx=aux)
        self.assertFalse(D.entails(s, 1 << aux, 0))
        self.assertFalse(D.entails(s, 1 << aux, 1))

    def test_assign_const(self):
        s = pin(D.top(2), 0, 1)               # i = Odd
        s = D.assign_linear(s, 0, None, flip=True)  # i := (some odd K)
        self.assertTrue(D.entails(s, 1 << 0, 1))
        s = D.assign_linear(s, 0, None, flip=False)  # i := (some even K)
        self.assertTrue(D.entails(s, 1 << 0, 0))  # fully overwritten, no BOTTOM


class TestJoin(unittest.TestCase):
    def test_join_with_bottom_is_identity(self):
        s = pin(D.top(2), 0, 0)
        self.assertEqual(D.join(D.BOTTOM, s), s)
        self.assertEqual(D.join(s, D.BOTTOM), s)
        self.assertTrue(D.is_bottom(D.join(D.BOTTOM, D.BOTTOM)))

    def test_join_of_disjoint_parities_loses_the_individual_fact(self):
        # This is the textbook per-variable-domain failure, reproduced at
        # the primitive level: Even ⊔ Odd = unconstrained.
        even = pin(D.top(1), 0, 0)
        odd = pin(D.top(1), 0, 1)
        joined = D.join(even, odd)
        self.assertFalse(D.entails(joined, 1 << 0, 0))
        self.assertFalse(D.entails(joined, 1 << 0, 1))

    def test_join_preserves_a_relation_present_on_both_sides(self):
        # THE motivating example from report Sec 2.1: i,j individually
        # flip between Even/Even and Odd/Odd across iterations, but
        # parity(i) == parity(j) must survive the join.
        c = (1 << 0) ^ (1 << 1)
        even_even = pin(pin(D.top(2), 0, 0), 1, 0)
        odd_odd = pin(pin(D.top(2), 0, 1), 1, 1)
        joined = D.join(even_even, odd_odd)
        self.assertTrue(D.entails(joined, c, 0))       # i == j survives
        for idx in (0, 1):                              # but neither alone
            self.assertFalse(D.entails(joined, 1 << idx, 0))
            self.assertFalse(D.entails(joined, 1 << idx, 1))


class TestLeq(unittest.TestCase):
    def test_reflexive(self):
        for s in (D.top(2), pin(D.top(2), 0, 0), D.BOTTOM):
            self.assertTrue(D.leq(s, s))

    def test_bottom_is_leq_everything(self):
        self.assertTrue(D.leq(D.BOTTOM, D.top(2)))
        self.assertFalse(D.leq(D.top(2), D.BOTTOM))

    def test_more_constrained_is_leq_less_constrained(self):
        pinned = pin(D.top(2), 0, 0)
        self.assertTrue(D.leq(pinned, D.top(2)))
        self.assertFalse(D.leq(D.top(2), pinned))


class TestEntailsOnBottom(unittest.TestCase):
    def test_bottom_entails_everything_vacuously(self):
        self.assertTrue(D.entails(D.BOTTOM, 1 << 0, 0))
        self.assertTrue(D.entails(D.BOTTOM, 1 << 0, 1))


class TestJoinAlgebraicLaws(unittest.TestCase):
    """Sanity checks that `join` behaves like a real set-join regardless of
    which side's base point the implementation happens to anchor on."""

    def _sample_states(self):
        c = (1 << 0) ^ (1 << 1)
        return [
            D.top(2),
            pin(D.top(2), 0, 0),
            pin(D.top(2), 0, 1),
            D.meet_equation(D.top(2), c, 0),   # i == j
            D.meet_equation(D.top(2), c, 1),   # i != j
            pin(pin(D.top(2), 0, 0), 1, 1),
            D.BOTTOM,
        ]

    def test_join_is_commutative_as_a_set(self):
        states = self._sample_states()
        for a in states:
            for b in states:
                with self.subTest(a=a, b=b):
                    self.assertTrue(same_set(D.join(a, b), D.join(b, a)))

    def test_join_is_idempotent(self):
        for s in self._sample_states():
            with self.subTest(s=s):
                self.assertTrue(same_set(D.join(s, s), s))

    def test_leq_is_antisymmetric_via_same_set(self):
        # Two syntactically different representatives of the same flat
        # (built via different meet orders) must be mutually <=.
        c = (1 << 0) ^ (1 << 1)
        s1 = pin(D.meet_equation(D.top(2), c, 0), 0, 0)          # i==j, then i=Even
        s2 = D.meet_equation(pin(D.top(2), 1, 0), c, 0)          # j=Even, then i==j
        self.assertTrue(D.leq(s1, s2))
        self.assertTrue(D.leq(s2, s1))
        self.assertTrue(same_set(s1, s2))


class TestRepeatedAliasing(unittest.TestCase):
    """The rename_coord bug (see report / conversation) only showed up
    when the aliased variable already carried a relation. These push that
    scenario harder: repeated increments, a genuine no-op self-copy, and
    reusing the *same* aux slot across *different* variables in sequence
    -- mirroring how analysis.py shares one aux_idx for the whole program,
    not just one statement."""

    def test_self_copy_i_equals_i_is_a_true_noop(self):
        # i := i (flip=False, var_idx == other_idx) must change nothing,
        # including a relation to another variable.
        n, aux = 2, 2
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(n + 1), c, 0)  # i == j
        s2 = D.assign_linear(s, 0, 0, flip=False, aux_idx=aux)
        self.assertTrue(same_set(s, s2))

    def test_double_increment_returns_to_original_relation(self):
        # i==j -> (i:=i+1) -> i!=j -> (i:=i+1) -> i==j again.
        n, aux = 2, 2
        c = (1 << 0) ^ (1 << 1)
        s0 = D.meet_equation(D.top(n + 1), c, 0)  # i == j
        s1 = D.assign_linear(s0, 0, 0, flip=True, aux_idx=aux)
        self.assertTrue(D.entails(s1, c, 1))      # i != j after one flip
        s2 = D.assign_linear(s1, 0, 0, flip=True, aux_idx=aux)
        self.assertTrue(D.entails(s2, c, 0))      # back to i == j
        self.assertTrue(same_set(s0, s2))

    def test_triple_increment_is_odd_number_of_flips(self):
        n, aux = 2, 2
        c = (1 << 0) ^ (1 << 1)
        s = D.meet_equation(D.top(n + 1), c, 0)  # i == j
        for _ in range(3):
            s = D.assign_linear(s, 0, 0, flip=True, aux_idx=aux)
        self.assertTrue(D.entails(s, c, 1))  # odd number of flips -> i != j

    def test_aux_slot_reused_across_different_variables_no_leakage(self):
        # i,j start related (i==j); k is a third, totally unrelated
        # variable. i := i+1 uses the shared aux slot; then k := k+1 uses
        # the *same* aux slot. Neither the i/j relation nor k's
        # independence should be disturbed by the aux slot's reuse.
        i, j, k, aux = 0, 1, 2, 3
        c_ij = (1 << i) ^ (1 << j)
        s = D.meet_equation(D.top(4), c_ij, 0)  # i == j, k and aux free
        s = D.assign_linear(s, i, i, flip=True, aux_idx=aux)   # i := i+1
        self.assertTrue(D.entails(s, c_ij, 1))                  # i != j now
        s = D.assign_linear(s, k, k, flip=True, aux_idx=aux)    # k := k+1
        self.assertTrue(D.entails(s, c_ij, 1))                  # untouched
        # k remains fully independent (still unconstrained on its own)
        self.assertFalse(D.entails(s, 1 << k, 0))
        self.assertFalse(D.entails(s, 1 << k, 1))
        # aux is free again for whatever comes next
        self.assertFalse(D.entails(s, 1 << aux, 0))
        self.assertFalse(D.entails(s, 1 << aux, 1))


class TestAssignLinearOnBottom(unittest.TestCase):
    def test_var_to_var_assignment_on_bottom_stays_bottom(self):
        self.assertTrue(
            D.is_bottom(D.assign_linear(D.BOTTOM, 0, 1, flip=True, aux_idx=2))
        )

    def test_aliased_assignment_on_bottom_stays_bottom(self):
        self.assertTrue(
            D.is_bottom(D.assign_linear(D.BOTTOM, 0, 0, flip=True, aux_idx=1))
        )

    def test_const_assignment_on_bottom_stays_bottom(self):
        self.assertTrue(
            D.is_bottom(D.assign_linear(D.BOTTOM, 0, None, flip=False))
        )


if __name__ == "__main__":
    unittest.main()
