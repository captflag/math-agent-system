"""Calibrated confidence from benchmark results.

After a benchmark run we know empirically how often SymPy-verified answers were
correct vs unverified ones. Those measured rates become the confidence numbers
shown with every future answer — real calibration, not vibes.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings

_MIN_SAMPLES = 5  # below this, a rate is too noisy to show


def save_from_rows(rows: list[dict], path: Path | None = None) -> dict:
    """Compute and persist verified/unverified accuracy from benchmark rows."""
    verified = [r for r in rows if r.get("sympy_verified")]
    unverified = [r for r in rows if not r.get("sympy_verified")]

    def rate(rs: list[dict]) -> float | None:
        if len(rs) < _MIN_SAMPLES:
            return None
        return round(sum(bool(r.get("correct")) for r in rs) / len(rs), 3)

    data = {
        "n_total": len(rows),
        "n_verified": len(verified),
        "n_unverified": len(unverified),
        "verified_accuracy": rate(verified),
        "unverified_accuracy": rate(unverified),
    }
    target = path or settings.CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def load(path: Path | None = None) -> dict | None:
    target = path or settings.CALIBRATION_PATH
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def confidence_for(sympy_verified: bool, path: Path | None = None) -> float | None:
    """Historical accuracy for answers with this verification status, or None
    if no benchmark has been run yet."""
    cal = load(path)
    if not cal:
        return None
    key = "verified_accuracy" if sympy_verified else "unverified_accuracy"
    return cal.get(key)
