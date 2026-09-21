"""Step-level verification: numeric spot-checking of claimed equalities.

`check_step` substitutes random rational values for every free symbol on both
sides of a claimed identity/equation-step and compares numerically. Three
random trials at 1e-6 tolerance catch virtually all algebra slips while never
false-failing a true identity (up to removable singularities, which are
retried at fresh points).
"""
from __future__ import annotations

import random

import sympy

from app.tools.math_tools import _parse

_TRIALS = 3
_TOL = 1e-6
_MAX_RESAMPLES = 12


def check_step(lhs: str, rhs: str) -> str:
    """Numerically verify that lhs == rhs for all values of the free variables."""
    try:
        left = _parse(lhs)
        right = _parse(rhs)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"

    symbols = sorted(left.free_symbols | right.free_symbols, key=str)

    # No variables: direct numeric comparison
    if not symbols:
        try:
            diff = complex((left - right).evalf(20))
            if abs(diff) <= _TOL:
                return f"step VERIFIED: {lhs} = {rhs} (numeric check)"
            return f"step MISMATCH: {lhs} != {rhs} (difference = {diff})"
        except Exception as e:  # noqa: BLE001
            return f"error: could not evaluate numerically: {e}"

    rng = random.Random(42)  # deterministic for reproducibility
    passed = 0
    attempts = 0
    tried_points: list[dict] = []
    while passed < _TRIALS and attempts < _MAX_RESAMPLES:
        attempts += 1
        point = {s: sympy.Rational(rng.randint(2, 19), rng.randint(2, 7)) for s in symbols}
        try:
            lv = complex(left.subs(point).evalf(20))
            rv = complex(right.subs(point).evalf(20))
        except Exception:  # noqa: BLE001 — singular point, resample
            continue
        scale = max(abs(lv), abs(rv), 1.0)
        if abs(lv - rv) / scale > _TOL:
            pt = {str(k): str(v) for k, v in point.items()}
            return (f"step MISMATCH at {pt}: LHS = {lv}, RHS = {rv}. "
                    f"The claimed equality is FALSE — recheck this step.")
        passed += 1
        tried_points.append({str(k): str(v) for k, v in point.items()})

    if passed >= _TRIALS:
        return (f"step VERIFIED: '{lhs}' equals '{rhs}' at {passed} random points "
                f"of {[s for s in map(str, symbols)]}")
    return "error: could not find enough valid evaluation points (singularities?)"


CHECK_STEP_SCHEMA = {
    "name": "check_step",
    "description": "Numerically verify one algebraic step: that LHS equals RHS as "
                   "expressions (checked at random points for every free variable). "
                   "Use this on EACH nontrivial algebraic manipulation in your "
                   "solution — a step that fails this check is wrong.",
    "input_schema": {
        "type": "object",
        "properties": {
            "lhs": {"type": "string", "description": "Left side, e.g. '(x+1)**2'"},
            "rhs": {"type": "string", "description": "Right side, e.g. 'x**2 + 2*x + 1'"},
        },
        "required": ["lhs", "rhs"],
    },
}
