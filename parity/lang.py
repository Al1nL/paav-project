"""
Parser for the integer-program command language (project PDF, Section 1.1):

    C ::= skip | i := j | i := K | i := ? | i := j + 1 | i := j - 1
        | assume E | assert ORC
    E ::= i = j | i != j | i = K | i != K | TRUE | FALSE
    ORC ::= (ANDC) | (ANDC) ORC
    ANDC ::= b | b ANDC
    b ::= EVEN i | ODD i        (parity-analysis specific atomic predicates)

Returns opaque tuples consumed by parity/analysis.py.
"""

import re

ASSIGN_RE = re.compile(r"^(\w+)\s*:=\s*(.+)$")
ASSUME_RE = re.compile(r"^assume\s*\((.+)\)$")
ASSERT_RE = re.compile(r"^assert\s*(.+)$")
PAREN_GROUP_RE = re.compile(r"\(([^()]*)\)")


def parse_command(text: str):
    text = text.strip()
    if text == "skip":
        return ("skip",)

    m = ASSIGN_RE.match(text)
    if m:
        lhs, rhs = m.group(1), m.group(2).strip()
        if rhs == "?":
            return ("assign_nondet", lhs)
        pm = re.match(r"^(\w+)\s*\+\s*1$", rhs)
        if pm:
            return ("assign_incr", lhs, pm.group(1))
        mm = re.match(r"^(\w+)\s*-\s*1$", rhs)
        if mm:
            return ("assign_decr", lhs, mm.group(1))
        if re.match(r"^\d+$", rhs):
            return ("assign_const", lhs, int(rhs))
        if re.match(r"^\w+$", rhs):
            return ("assign_var", lhs, rhs)
        raise ValueError(f"Cannot parse assignment RHS: {rhs!r} in {text!r}")

    m = ASSUME_RE.match(text)
    if m:
        cond = m.group(1).strip()
        if cond == "TRUE":
            return ("assume_true",)
        if cond == "FALSE":
            return ("assume_false",)
        nm = re.match(r"^(\w+)\s*!=\s*(\w+)$", cond)
        if nm:
            lhs, rhs = nm.group(1), nm.group(2)
            if re.match(r"^\d+$", rhs):
                return ("assume_neq_const", lhs, int(rhs))
            return ("assume_neq", lhs, rhs)
        em = re.match(r"^(\w+)\s*=\s*(\w+)$", cond)
        if em:
            lhs, rhs = em.group(1), em.group(2)
            if re.match(r"^\d+$", rhs):
                return ("assume_eq_const", lhs, int(rhs))
            return ("assume_eq", lhs, rhs)
        raise ValueError(f"Cannot parse assume condition: {cond!r}")

    m = ASSERT_RE.match(text)
    if m:
        body = m.group(1).strip()
        groups = PAREN_GROUP_RE.findall(body)
        if not groups:
            raise ValueError(f"assert with no parenthesized groups: {text!r}")
        orc = []
        for g in groups:
            atoms = []
            toks = g.split()
            i = 0
            while i < len(toks):
                pred = toks[i]
                var = toks[i + 1]
                if pred not in ("EVEN", "ODD"):
                    raise ValueError(f"Unknown atomic predicate {pred!r} in {text!r}")
                atoms.append((pred, var))
                i += 2
            orc.append(atoms)
        return ("assert", orc)

    raise ValueError(f"Cannot parse command: {text!r}")
