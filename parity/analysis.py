import glob
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.cfg import parse_cfg
from common.fixpoint import run_fixpoint
from parity import domain as D
from parity.lang import parse_command


def build_var_index(cfg):
    return {v: i for i, v in enumerate(cfg.variables)}


def make_transfer(var_index, aux_idx):
    def transfer(edge, s):
        cmd = edge.cmd
        kind = cmd[0]
        if kind == "skip":
            return s
        if kind == "assign_var":
            _, i, j = cmd
            return D.assign_linear(s, var_index[i], var_index[j], flip=False, aux_idx=aux_idx)
        if kind == "assign_incr":
            _, i, j = cmd
            return D.assign_linear(s, var_index[i], var_index[j], flip=True, aux_idx=aux_idx)
        if kind == "assign_decr":
            _, i, j = cmd
            j_idx = var_index[j]
            if D.entails(s, 1 << j_idx, 1):   # j provably odd => j != 0, flip is sound
                return D.assign_linear(s, var_index[i], j_idx, flip=True, aux_idx=aux_idx)
            return D.forget(s, var_index[i])  # j possibly 0 (even/unknown), forget i to be sound
        if kind == "assign_const":
            _, i, k = cmd
            return D.assign_linear(s, var_index[i], None, flip=bool(k % 2))
        if kind == "assign_nondet":
            _, i = cmd
            return D.forget(s, var_index[i])
        if kind == "assume_eq":
            _, i, j = cmd
            c = (1 << var_index[i]) ^ (1 << var_index[j])
            return D.meet_equation(s, c, 0)
        if kind == "assume_neq":
            return s  # no sound parity refinement from disequality
        if kind == "assume_eq_const":
            _, i, k = cmd
            c = 1 << var_index[i]
            return D.meet_equation(s, c, k % 2)
        if kind == "assume_neq_const":
            return s
        if kind == "assume_true":
            return s
        if kind == "assume_false":
            return D.BOTTOM
        if kind == "assert":
            return s  # asserts don't transform state; checked separately
        raise ValueError(f"Unknown command kind: {kind}")

    return transfer


def check_assert(state, orc, var_index):
    """Check if the ORC formula holds for all concrete points in state.
    Enumerates the GF(2) affine coset (2**k points) to evaluate exact disjunctive truth."""
    if D.is_bottom(state):
        return True  # unreachable program point: vacuously safe

    k = len(state.basis)
    if k > 20:
        # Should not happen for these programs (k <= #variables), but
        # guard against blow-up rather than hang.
        raise RuntimeError("Coset too large to enumerate exactly")

    for mask in range(1 << k):
        val = state.p
        for bit_i, v in enumerate(state.basis):
            if (mask >> bit_i) & 1:
                val ^= v
        if not _orc_holds(val, orc, var_index):
            return False
    return True


def _orc_holds(val: int, orc, var_index) -> bool:
    """val: a concrete parity-assignment bitmask. Evaluate the ORC formula
    (disjunction of conjunctions of EVEN/ODD atoms) directly against it."""
    for conjunct in orc:
        if all(
            ((val >> var_index[var]) & 1) == (1 if pred == "ODD" else 0)
            for pred, var in conjunct
        ):
            return True
    return False


def analyze(text: str):
    cfg = parse_cfg(text, parse_command)
    var_index = build_var_index(cfg)
    n = len(cfg.variables)
    aux_idx = n  # one reserved scratch coordinate beyond the named variables

    transfer = make_transfer(var_index, aux_idx)
    pre, post_per_edge, iterations = run_fixpoint(
        cfg,
        bottom=lambda: D.BOTTOM,
        entry_state=D.top(n + 1),
        transfer=transfer,
        join=D.join,
        leq=D.leq,
    )

    results = []
    for e in cfg.edges:
        if e.cmd[0] == "assert":
            ok = check_assert(pre[e.src], e.cmd[1], var_index)
            results.append((e, ok))

    return cfg, pre, results, iterations


def format_state(state, cfg):
    if D.is_bottom(state):
        return "UNREACHABLE"
    var_index = build_var_index(cfg)
    facts = []
    for v, idx in var_index.items():
        e = 1 << idx
        if D.entails(state, e, 0):
            facts.append(f"{v}=Even")
        elif D.entails(state, e, 1):
            facts.append(f"{v}=Odd")
    # pairwise relational facts not individually determined
    names = list(var_index)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            ia, ib = var_index[a], var_index[b]
            e = (1 << ia) ^ (1 << ib)
            if D.entails(state, e, 0) and not (
                D.entails(state, 1 << ia, 0) or D.entails(state, 1 << ia, 1)
            ):
                facts.append(f"parity({a})=parity({b})")
            elif D.entails(state, e, 1) and not (
                D.entails(state, 1 << ia, 0) or D.entails(state, 1 << ia, 1)
            ):
                facts.append(f"parity({a})!=parity({b})")
    return ", ".join(facts) if facts else "TOP (no constraints)"


def run_file(path):
    with open(path) as f:
        text = f.read()
    cfg, pre, results, iterations = analyze(text)
    print(f"=== {path} ===")
    print(f"(fixpoint reached in {iterations} worklist steps)")
    all_ok = True
    for e, ok in results:
        status = "VERIFIED" if ok else "POSSIBLY VIOLATED"
        if not ok:
            all_ok = False
        print(f"  [{status}] {e.raw}")
    print("RESULT:", "all asserts verified" if all_ok else "some asserts not verified")
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
