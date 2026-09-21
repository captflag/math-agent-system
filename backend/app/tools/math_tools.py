"""Deterministic SymPy tools the agent uses to compute and verify answers.

All parsing goes through sympy's parse_expr with a restricted local namespace —
never eval(). Every function returns a plain string (or an error string) so
results can be dropped straight into a tool_result block.
"""
from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

_LOCALS = {
    name: getattr(sympy, name)
    for name in (
        "pi", "E", "I", "oo", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "exp", "log", "sqrt", "Abs", "factorial",
        "binomial", "gamma", "floor", "ceiling", "Rational", "GoldenRatio",
    )
}


# parse_expr ultimately eval()s, so expressions must be validated first:
# whitelist the character set, forbid dunder access, and only allow identifiers
# that are known functions/constants or short variable names.
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/^().,= \t]*$")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_DENY_IDENTS = {"exec", "eval", "open", "input", "print", "type", "vars",
                "dir", "help", "chr", "ord", "getattr", "setattr", "lambda"}


def _validate(expression: str) -> None:
    if not _ALLOWED_CHARS.match(expression):
        raise ValueError("expression contains disallowed characters")
    if "__" in expression:
        raise ValueError("expression contains disallowed token '__'")
    for ident in _IDENT.findall(expression):
        if ident in _LOCALS:
            continue
        if ident in _DENY_IDENTS or len(ident) > 5:
            raise ValueError(f"unknown identifier '{ident}' is not allowed")


def _parse(expression: str) -> sympy.Expr:
    _validate(expression)
    return parse_expr(expression, transformations=_TRANSFORMS, local_dict=dict(_LOCALS))


def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve an equation for a variable. Accepts 'lhs = rhs' or an expression assumed = 0."""
    try:
        var = sympy.Symbol(variable)
        if "=" in equation:
            lhs, rhs = equation.split("=", 1)
            expr = sympy.Eq(_parse(lhs), _parse(rhs))
        else:
            expr = sympy.Eq(_parse(equation), 0)
        solutions = sympy.solve(expr, var)
        return f"solutions for {variable}: {solutions}"
    except Exception as e:  # noqa: BLE001 — errors go back to the model as text
        return f"error: {e}"


def differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
    """Differentiate an expression with respect to a variable."""
    try:
        result = sympy.diff(_parse(expression), sympy.Symbol(variable), order)
        return f"d^{order}/d{variable}^{order} = {result}"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def integrate_expression(expression: str, variable: str = "x",
                         lower: str | None = None, upper: str | None = None) -> str:
    """Integrate an expression; definite if bounds are given, else indefinite."""
    try:
        var = sympy.Symbol(variable)
        expr = _parse(expression)
        if lower is not None and upper is not None:
            result = sympy.integrate(expr, (var, _parse(lower), _parse(upper)))
            return f"definite integral over [{lower}, {upper}] = {result}"
        result = sympy.integrate(expr, var)
        return f"indefinite integral = {result} + C"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def evaluate_expression(expression: str, precision: int = 12) -> str:
    """Numerically evaluate an expression (also simplifies symbolically first)."""
    try:
        expr = sympy.simplify(_parse(expression))
        numeric = expr.evalf(precision)
        if expr == numeric:
            return f"value = {numeric}"
        return f"symbolic = {expr}, numeric ≈ {numeric}"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def simplify_expression(expression: str) -> str:
    """Simplify / factor / expand an expression and report all three forms."""
    try:
        expr = _parse(expression)
        return (
            f"simplified = {sympy.simplify(expr)}; "
            f"factored = {sympy.factor(expr)}; "
            f"expanded = {sympy.expand(expr)}"
        )
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


# name -> (callable, JSON schema) used by the agent loop
SYMPY_TOOLS = {
    "solve_equation": (
        solve_equation,
        {
            "name": "solve_equation",
            "description": "Solve an equation for a variable using SymPy (exact, symbolic). "
                           "Use this to compute or verify roots. Format: 'x**2 + 5*x + 6 = 0'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string"},
                    "variable": {"type": "string", "default": "x"},
                },
                "required": ["equation"],
            },
        },
    ),
    "differentiate": (
        differentiate,
        {
            "name": "differentiate",
            "description": "Differentiate an expression symbolically with SymPy. "
                           "Use to compute or verify derivatives.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "variable": {"type": "string", "default": "x"},
                    "order": {"type": "integer", "default": 1},
                },
                "required": ["expression"],
            },
        },
    ),
    "integrate_expression": (
        integrate_expression,
        {
            "name": "integrate_expression",
            "description": "Integrate an expression symbolically with SymPy. Definite if "
                           "'lower' and 'upper' bounds are provided.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "variable": {"type": "string", "default": "x"},
                    "lower": {"type": "string"},
                    "upper": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    ),
    "evaluate_expression": (
        evaluate_expression,
        {
            "name": "evaluate_expression",
            "description": "Numerically evaluate any arithmetic/symbolic expression with SymPy. "
                           "ALWAYS use this instead of doing arithmetic in your head.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "precision": {"type": "integer", "default": 12},
                },
                "required": ["expression"],
            },
        },
    ),
    "simplify_expression": (
        simplify_expression,
        {
            "name": "simplify_expression",
            "description": "Simplify, factor and expand an expression with SymPy.",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    ),
}
