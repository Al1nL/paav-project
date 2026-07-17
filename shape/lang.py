"""
Parser for the acyclic/unshared-list command language (project PDF, Section 2.1):

    C ::= skip | x := y | x := NULL | x := y.n | x.n := y | x := new
        | assume(E) | assert(ORC)
    E ::= x = y | x != y | x = NULL | x != NULL | TRUE | FALSE
    ORC ::= (ANDC) | (ANDC) ORC
    ANDC ::= b | b ANDC
    b ::= x = NULL | x != NULL | x = y | x != y | x = y.n | x != y.n
        | LS x y | NOLS x y | ODD x y | EVEN x y
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
        if rhs == "NULL":
            return ("assign_null", lhs)
        if rhs == "new":
            return ("assign_new", lhs)
        dm = re.match(r"^(\w+)\.n$", rhs)
        if dm:
            return ("assign_deref", lhs, dm.group(1))
        # x.n := y  (lhs itself has ".n")
        if re.match(r"^\w+$", rhs):
            return ("assign_var", lhs, rhs)
        raise ValueError(f"Cannot parse assignment RHS: {rhs!r} in {text!r}")

    # x.n := y  (must check before generic assign, lhs contains ".n")
    fm = re.match(r"^(\w+)\.n\s*:=\s*(\w+)$", text)
    if fm:
        return ("field_assign", fm.group(1), fm.group(2))

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
            if rhs == "NULL":
                return ("assume_neq_null", lhs)
            return ("assume_neq", lhs, rhs)
        em = re.match(r"^(\w+)\s*=\s*(\w+)$", cond)
        if em:
            lhs, rhs = em.group(1), em.group(2)
            if rhs == "NULL":
                return ("assume_eq_null", lhs)
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
                if toks[i] in ("LS", "NOLS", "ODD", "EVEN"):
                    atoms.append((toks[i], toks[i + 1], toks[i + 2]))
                    i += 3
                elif toks[i + 1] == "=" and toks[i + 2] == "NULL":
                    atoms.append(("EQ_NULL", toks[i], None))
                    i += 3
                elif toks[i + 1] == "!=" and toks[i + 2] == "NULL":
                    atoms.append(("NEQ_NULL", toks[i], None))
                    i += 3
                elif toks[i + 1] == "=" and re.match(r"^\w+\.n$", toks[i + 2] or ""):
                    atoms.append(("EQ_DEREF", toks[i], toks[i + 2].split(".")[0]))
                    i += 3
                elif toks[i + 1] == "!=" and re.match(r"^\w+\.n$", toks[i + 2] or ""):
                    atoms.append(("NEQ_DEREF", toks[i], toks[i + 2].split(".")[0]))
                    i += 3
                elif toks[i + 1] == "=":
                    atoms.append(("EQ", toks[i], toks[i + 2]))
                    i += 3
                elif toks[i + 1] == "!=":
                    atoms.append(("NEQ", toks[i], toks[i + 2]))
                    i += 3
                else:
                    raise ValueError(f"Cannot parse assert atom near {toks[i:]!r} in {text!r}")
            orc.append(atoms)
        return ("assert", orc)

    raise ValueError(f"Cannot parse command: {text!r}")
