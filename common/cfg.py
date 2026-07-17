"""
Generic CFG representation + parser for the project's input file format:

    var1 var2 ... varN

    Ls  <command text>  Lt
    Ls  <command text>  Lt
    ...

- Blank lines are ignored.
- A node with two outgoing edges must have both edges be `assume` commands
  (checked by caller / not enforced here, per the simplifying assumptions).
- The command text itself is language-specific (parity vs shape commands),
  so this module takes a `parse_command` callback and returns opaque
  command objects; it only knows about the *graph* structure.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any


@dataclass
class Edge:
    src: str
    cmd: Any          # language-specific parsed command
    dst: str
    raw: str           # original text, for error messages


@dataclass
class CFG:
    variables: List[str]
    edges: List[Edge]
    nodes: List[str] = field(default_factory=list)
    out_edges: Dict[str, List[Edge]] = field(default_factory=dict)
    in_edges: Dict[str, List[Edge]] = field(default_factory=dict)
    entry: str = None

    def successors(self, n):
        return self.out_edges.get(n, [])

    def predecessors(self, n):
        return self.in_edges.get(n, [])


def parse_cfg(text: str, parse_command: Callable[[str], Any]) -> CFG:
    lines = [l.rstrip("\n") for l in text.splitlines()]
    # first non-blank line = variable declarations
    lines_iter = iter(lines)
    var_line = None
    rest = []
    for l in lines:
        if l.strip() == "":
            continue
        var_line = l
        break
    started = False
    for l in lines:
        if not started:
            if l is var_line:
                started = True
            continue
        rest.append(l)

    variables = var_line.split()

    edges = []
    node_set = []

    def add_node(n):
        if n not in node_set:
            node_set.append(n)

    for l in rest:
        if l.strip() == "":
            continue
        raw = l.strip()
        # format: <src>  <command...>  <dst>
        # src = first whitespace-separated token, dst = last token
        toks = raw.split()
        src = toks[0]
        dst = toks[-1]
        cmd_text = raw[len(src):-len(dst)].strip() if len(toks) > 2 else ""
        # more robust: reconstruct command text by removing first/last token
        # occurrences positionally (avoids issues if src==dst text overlaps)
        first_sp = raw.find(src) + len(src)
        last_sp = raw.rfind(dst)
        cmd_text = raw[first_sp:last_sp].strip()

        cmd = parse_command(cmd_text)
        add_node(src)
        add_node(dst)
        edges.append(Edge(src=src, cmd=cmd, dst=dst, raw=raw))

    cfg = CFG(variables=variables, edges=edges, nodes=node_set)
    cfg.out_edges = {n: [] for n in node_set}
    cfg.in_edges = {n: [] for n in node_set}
    for e in edges:
        cfg.out_edges[e.src].append(e)
        cfg.in_edges[e.dst].append(e)

    # Entry node: per the simplifying assumptions this should be the unique
    # node with no incoming edges. In practice one of the given example
    # programs (the shape-analysis example) loops back to its own starting
    # node (L6 -> L8), so that node does have an incoming edge. We fall back
    # to "the source of the first edge in the file" when there isn't a
    # unique no-incoming-edges node, which matches the intended reading
    # order for every test program in this project.
    roots = [n for n in node_set if len(cfg.in_edges[n]) == 0]
    if len(roots) == 1:
        cfg.entry = roots[0]
    else:
        cfg.entry = edges[0].src
    return cfg
