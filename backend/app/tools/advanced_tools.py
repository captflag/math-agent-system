"""Advanced SymPy tools: matrices, vectors, complex numbers, systems,
inequalities — the JEE-syllabus areas the basic single-variable tools miss.

Matrices and vectors are passed as JSON arrays of expression strings, so each
element goes through the same validated `_parse` as everything else and the
bracket characters never reach the parser.
"""
from __future__ import annotations

import sympy

from app.tools.math_tools import _parse


def solve_system(equations: list[str], variables: list[str]) -> str:
    """Solve a system of equations (linear or nonlinear)."""
    try:
        eqs = []
        for eq in equations:
            if "=" in eq:
                lhs, rhs = eq.split("=", 1)
                eqs.append(sympy.Eq(_parse(lhs), _parse(rhs)))
            else:
                eqs.append(sympy.Eq(_parse(eq), 0))
        vars_ = [sympy.Symbol(v) for v in variables]
        solutions = sympy.solve(eqs, vars_, dict=True)
        return f"solutions: {solutions}"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def solve_inequality(inequality: str, variable: str = "x") -> str:
    """Solve an inequality like 'x**2 - 4 > 0' for one variable."""
    try:
        for op in ("<=", ">=", "<", ">"):
            if op in inequality:
                lhs, rhs = inequality.split(op, 1)
                rel = {"<": sympy.Lt, ">": sympy.Gt, "<=": sympy.Le, ">=": sympy.Ge}[op]
                expr = rel(_parse(lhs), _parse(rhs))
                break
        else:
            return "error: no inequality operator (<, >, <=, >=) found"
        result = sympy.solve(expr, sympy.Symbol(variable))
        return f"solution set: {result}"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def _to_matrix(rows: list[list[str]]) -> sympy.Matrix:
    return sympy.Matrix([[_parse(str(el)) for el in row] for row in rows])


def matrix_operation(matrix: list[list[str]], operation: str,
                     other: list[list[str]] | None = None) -> str:
    """Matrix ops: determinant, inverse, rank, transpose, eigenvalues, multiply, add."""
    try:
        m = _to_matrix(matrix)
        if operation == "determinant":
            return f"det = {sympy.simplify(m.det())}"
        if operation == "inverse":
            return f"inverse = {m.inv().tolist()}"
        if operation == "rank":
            return f"rank = {m.rank()}"
        if operation == "transpose":
            return f"transpose = {m.T.tolist()}"
        if operation == "eigenvalues":
            return f"eigenvalues (value: multiplicity) = {m.eigenvals()}"
        if operation in ("multiply", "add"):
            if other is None:
                return "error: 'other' matrix required for multiply/add"
            n = _to_matrix(other)
            result = m * n if operation == "multiply" else m + n
            return f"result = {result.tolist()}"
        return f"error: unknown operation '{operation}'"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def vector_operation(a: list[str], operation: str, b: list[str] | None = None) -> str:
    """Vector ops: dot, cross, magnitude, angle, unit."""
    try:
        va = sympy.Matrix([_parse(str(x)) for x in a])
        if operation == "magnitude":
            return f"|a| = {sympy.simplify(va.norm())}"
        if operation == "unit":
            return f"unit vector = {sympy.simplify(va / va.norm()).T.tolist()[0]}"
        if b is None:
            return f"error: 'b' vector required for {operation}"
        vb = sympy.Matrix([_parse(str(x)) for x in b])
        if operation == "dot":
            return f"a·b = {sympy.simplify(va.dot(vb))}"
        if operation == "cross":
            return f"a×b = {sympy.simplify(va.cross(vb)).T.tolist()[0]}"
        if operation == "angle":
            cos_t = sympy.simplify(va.dot(vb) / (va.norm() * vb.norm()))
            return f"cos(angle) = {cos_t}, angle = {sympy.acos(cos_t)}"
        return f"error: unknown operation '{operation}'"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def complex_operation(expression: str, operation: str = "simplify") -> str:
    """Complex number ops (use I for the imaginary unit):
    simplify, modulus, argument, conjugate, rectangular (a+bi form)."""
    try:
        z = _parse(expression)
        if operation == "simplify":
            return f"simplified = {sympy.simplify(z)}"
        if operation == "modulus":
            return f"|z| = {sympy.simplify(sympy.Abs(z))}"
        if operation == "argument":
            return f"arg(z) = {sympy.simplify(sympy.arg(z))}"
        if operation == "conjugate":
            return f"conjugate = {sympy.simplify(sympy.conjugate(z))}"
        if operation == "rectangular":
            expanded = sympy.expand_complex(z)
            return f"a+bi form = {expanded} (Re = {sympy.re(z)}, Im = {sympy.im(z)})"
        return f"error: unknown operation '{operation}'"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


ADVANCED_TOOLS = {
    "solve_system": (
        solve_system,
        {
            "name": "solve_system",
            "description": "Solve a system of equations (linear or nonlinear) with SymPy. "
                           "Use for simultaneous equations, intersection points, etc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "equations": {"type": "array", "items": {"type": "string"},
                                  "description": "e.g. ['2*x + y = 7', 'x - y = 2']"},
                    "variables": {"type": "array", "items": {"type": "string"},
                                  "description": "e.g. ['x', 'y']"},
                },
                "required": ["equations", "variables"],
            },
        },
    ),
    "solve_inequality": (
        solve_inequality,
        {
            "name": "solve_inequality",
            "description": "Solve a single-variable inequality with SymPy, e.g. 'x**2 - 4 > 0'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "inequality": {"type": "string"},
                    "variable": {"type": "string", "default": "x"},
                },
                "required": ["inequality"],
            },
        },
    ),
    "matrix_operation": (
        matrix_operation,
        {
            "name": "matrix_operation",
            "description": "Matrix computations with SymPy: determinant, inverse, rank, "
                           "transpose, eigenvalues, multiply, add. Matrix entries may be "
                           "symbolic expressions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                               "description": "Row-major, e.g. [['1','2'],['3','4']]"},
                    "operation": {"type": "string",
                                  "enum": ["determinant", "inverse", "rank", "transpose",
                                           "eigenvalues", "multiply", "add"]},
                    "other": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                              "description": "Second matrix for multiply/add"},
                },
                "required": ["matrix", "operation"],
            },
        },
    ),
    "vector_operation": (
        vector_operation,
        {
            "name": "vector_operation",
            "description": "Vector computations with SymPy: dot, cross, magnitude, angle, unit. "
                           "Use for 3D geometry (direction ratios, normals, projections).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "string"},
                          "description": "Components, e.g. ['1', '2', '-1']"},
                    "operation": {"type": "string",
                                  "enum": ["dot", "cross", "magnitude", "angle", "unit"]},
                    "b": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["a", "operation"],
            },
        },
    ),
    "complex_operation": (
        complex_operation,
        {
            "name": "complex_operation",
            "description": "Complex number computations with SymPy (imaginary unit is I): "
                           "simplify, modulus, argument, conjugate, rectangular form.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(1 + I)**10'"},
                    "operation": {"type": "string",
                                  "enum": ["simplify", "modulus", "argument", "conjugate",
                                           "rectangular"],
                                  "default": "simplify"},
                },
                "required": ["expression"],
            },
        },
    ),
}
