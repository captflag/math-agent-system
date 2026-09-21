"""Function plotting tool: renders graphs for solutions (matplotlib, headless)."""
from __future__ import annotations

import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy  # noqa: E402

from app.config import settings  # noqa: E402
from app.tools.math_tools import _parse  # noqa: E402


def plot_function(expressions: list[str], x_min: float = -10.0,
                  x_max: float = 10.0) -> str:
    """Plot one or more functions of x on a shared axis; returns the plot URL."""
    if not expressions:
        return "error: no expressions given"
    if x_min >= x_max:
        return "error: x_min must be less than x_max"
    try:
        x = sympy.Symbol("x")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = np.linspace(float(x_min), float(x_max), 600)
        for expr_str in expressions[:6]:
            expr = _parse(expr_str)
            extra = expr.free_symbols - {x}
            if extra:
                plt.close(fig)
                return f"error: only 'x' may be free; found {sorted(map(str, extra))}"
            fn = sympy.lambdify(x, expr, modules=["numpy"])
            with np.errstate(all="ignore"):
                ys = np.asarray(fn(xs), dtype=complex)
            ys = np.where(np.abs(ys.imag) < 1e-9, ys.real, np.nan)
            ys = np.where(np.abs(ys) > 1e6, np.nan, ys)  # clip singularities
            ax.plot(xs, ys.astype(float), label=str(expr))
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.grid(True, alpha=0.25)
        ax.legend()
        settings.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex[:12]}.png"
        fig.savefig(settings.PLOTS_DIR / name, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return f"plot saved: /plots/{name}"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


PLOT_TOOL_SCHEMA = {
    "name": "plot_function",
    "description": "Plot one or more functions of x (e.g. for calculus, coordinate "
                   "geometry, or areas between curves). The plot is shown to the "
                   "student alongside the solution — use it whenever a graph aids "
                   "understanding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expressions": {"type": "array", "items": {"type": "string"},
                            "description": "Functions of x, e.g. ['x**2', 'x']"},
            "x_min": {"type": "number", "default": -10},
            "x_max": {"type": "number", "default": 10},
        },
        "required": ["expressions"],
    },
}
