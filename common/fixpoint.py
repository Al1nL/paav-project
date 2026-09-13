"""
Generic chaotic-iteration / worklist fixpoint engine.

Both analyses supply:
  - cfg:        a common.cfg.CFG
  - bottom():   the domain's bottom element (unreachable)
  - entry_state: the abstract state to install at cfg.entry
  - transfer(edge, state) -> new_state   (must be monotone)
  - join(a, b) -> state
  - leq(a, b) -> bool                    (a <= b in the lattice)

No widening is used: both domains are designed to have finite height for
any fixed program (see report.pdf, sections on termination), so plain
Kleene iteration over a worklist terminates.
"""

from collections import deque


def run_fixpoint(cfg, bottom, entry_state, transfer, join, leq):
    """Run worklist chaotic iteration over cfg until fixpoint convergence.
    Returns (pre_states, post_per_edge, iterations)."""
    pre = {n: bottom() for n in cfg.nodes}
    pre[cfg.entry] = entry_state

    worklist = deque(cfg.nodes)
    in_worklist = set(cfg.nodes)

    # cache of post-state per edge for later assert-checking / reporting
    post_per_edge = {}

    iterations = 0
    while worklist:
        n = worklist.popleft()
        in_worklist.discard(n)
        iterations += 1

        cur_in = pre[n]
        for e in cfg.successors(n):
            new_out = transfer(e, cur_in)
            post_per_edge[id(e)] = new_out
            old = pre[e.dst]
            joined = join(old, new_out)
            if not leq(joined, old):
                pre[e.dst] = joined
                if e.dst not in in_worklist:
                    worklist.append(e.dst)
                    in_worklist.add(e.dst)

    return pre, post_per_edge, iterations
