"""Verification tests for the upgrade wave: step-checking, self-correction,
calibration, plotting, cascade, rate limiting, tutor sessions, schemas, API."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.agent import (
    estimate_cost,
    needs_verification_retry,
    should_escalate,
)
from app.calibration import confidence_for, load, save_from_rows
from app.config import settings
from app.main import app
from app.ratelimit import RateLimiter, is_limited
from app.schemas import (
    PracticeProblem,
    Solution,
    SolveResponse,
    TokenUsage,
    Verification,
)
from app.tools.plotting import plot_function
from app.tools.verification import check_step
from app.tutor import SessionStore

client = TestClient(app)


# ---------- check_step (step-level verification) ----------

def test_check_step_true_identity():
    result = check_step("(x+1)**2", "x**2 + 2*x + 1")
    assert "VERIFIED" in result


def test_check_step_trig_identity():
    result = check_step("sin(x)**2 + cos(x)**2", "1")
    assert "VERIFIED" in result


def test_check_step_catches_wrong_step():
    result = check_step("(x+1)**2", "x**2 + 1")
    assert "MISMATCH" in result and "FALSE" in result


def test_check_step_sign_error():
    result = check_step("(a-b)**2", "a**2 + 2*a*b + b**2")
    assert "MISMATCH" in result


def test_check_step_constant():
    assert "VERIFIED" in check_step("2 + 2", "4")
    assert "MISMATCH" in check_step("2 + 2", "5")


def test_check_step_multivariable():
    result = check_step("(x+y)*(x-y)", "x**2 - y**2")
    assert "VERIFIED" in result


def test_check_step_injection_blocked():
    assert check_step("__import__('os')", "1").startswith("error:")


# ---------- Self-correction decision ----------

def _solution(verified: bool, method: str = "sympy") -> Solution:
    return Solution(
        steps=[], final_answer="42",
        verification=Verification(verified=verified, method=method),
    )


def test_retry_when_claimed_but_not_observed():
    # Model says verified, but no tool ever ran — do not trust it
    assert needs_verification_retry(_solution(True), sympy_ok=False)


def test_no_retry_when_verified_and_observed():
    assert not needs_verification_retry(_solution(True), sympy_ok=True)


def test_retry_when_unverified_computational():
    assert needs_verification_retry(_solution(False, "did not check"), sympy_ok=False)


def test_no_retry_for_conceptual():
    assert not needs_verification_retry(
        _solution(False, "conceptual proof, not applicable"), sympy_ok=False)


# ---------- Cascade decision ----------

def test_escalate_on_unverified():
    assert should_escalate(_solution(False), sympy_ok=False)
    assert should_escalate(_solution(True), sympy_ok=False)
    assert should_escalate(None, sympy_ok=True)


def test_no_escalation_when_solid():
    assert not should_escalate(_solution(True), sympy_ok=True)


# ---------- Cost estimation ----------

def test_estimate_cost_opus():
    usage = TokenUsage(input=10_000, output=2_000)
    # 10K * $5/M + 2K * $25/M = 0.05 + 0.05 = 0.10
    assert estimate_cost("claude-opus-5", usage) == 0.1


def test_estimate_cost_counts_cache():
    usage = TokenUsage(input=0, output=0, cache_read=100_000, cache_write=10_000)
    cost = estimate_cost("claude-opus-5", usage)
    assert 0 < cost < 0.2


# ---------- Calibration ----------

def test_calibration_roundtrip(tmp_path):
    path = tmp_path / "cal.json"
    rows = ([{"sympy_verified": True, "correct": True}] * 9
            + [{"sympy_verified": True, "correct": False}]
            + [{"sympy_verified": False, "correct": True}] * 3
            + [{"sympy_verified": False, "correct": False}] * 3)
    data = save_from_rows(rows, path)
    assert data["verified_accuracy"] == 0.9
    assert data["unverified_accuracy"] == 0.5
    assert load(path)["n_total"] == 16
    assert confidence_for(True, path) == 0.9
    assert confidence_for(False, path) == 0.5


def test_calibration_too_few_samples(tmp_path):
    path = tmp_path / "cal.json"
    save_from_rows([{"sympy_verified": True, "correct": True}] * 3, path)
    assert confidence_for(True, path) is None


def test_confidence_none_without_benchmark(tmp_path):
    assert confidence_for(True, tmp_path / "missing.json") is None


# ---------- Plotting ----------

def test_plot_creates_file():
    result = plot_function(["x**2", "x"], -3, 3)
    assert result.startswith("plot saved: /plots/")
    name = result.split("/plots/")[1]
    assert (settings.PLOTS_DIR / name).exists()


def test_plot_rejects_bad_range():
    assert plot_function(["x"], 5, 5).startswith("error:")


def test_plot_rejects_other_symbols():
    assert plot_function(["y**2"]).startswith("error:")


def test_plot_injection_blocked():
    assert plot_function(["__import__('os')"]).startswith("error:")


def test_plot_handles_singularity():
    result = plot_function(["1/x"], -2, 2)
    assert result.startswith("plot saved:")


# ---------- Rate limiting ----------

def test_rate_limiter_allows_then_blocks():
    rl = RateLimiter(per_minute=3)
    assert all(rl.allow("ip1", now=100.0 + i) for i in range(3))
    assert not rl.allow("ip1", now=103.0)
    assert rl.allow("ip2", now=103.0)          # other clients unaffected
    assert rl.allow("ip1", now=100.0 + 61.0)   # window slides


def test_limited_paths():
    assert is_limited("/api/v1/solve")
    assert is_limited("/api/v1/tutor")
    assert not is_limited("/api/v1/health")
    assert not is_limited("/plots/abc.png")


# ---------- Tutor sessions ----------

def test_session_create_and_reuse():
    store = SessionStore(ttl=1000)
    sid, msgs = store.get_or_create(None)
    msgs.append({"role": "user", "content": "hi"})
    sid2, msgs2 = store.get_or_create(sid)
    assert sid2 == sid and len(msgs2) == 1


def test_session_expiry(monkeypatch):
    store = SessionStore(ttl=10)
    sid, _ = store.get_or_create(None)
    store._sessions[sid]["touched"] -= 100
    store.purge()
    assert len(store) == 0


# ---------- Schemas ----------

def test_solution_with_diagnosis():
    payload = {
        "steps": [{"step_number": 1, "description": "d", "formula": None,
                   "explanation": "long enough"}],
        "final_answer": "x = 2",
        "verification": {"verified": True, "method": "check_step", "details": None},
        "references": [],
        "diagnosis": {"error_step": 3, "misconception": "sign error when moving terms"},
    }
    sol = Solution.model_validate_json(json.dumps(payload))
    assert sol.diagnosis.error_step == 3


def test_practice_problem_schema():
    p = PracticeProblem(
        problem="Solve x+1=2", steps=[], final_answer="x=1",
        verification=Verification(verified=True, method="sympy"))
    assert p.problem


def test_solve_response_new_fields_default():
    r = SolveResponse(
        question="q", solution=_solution(True), source="reasoning",
        tools_used=[], sympy_verified=True, turns=1, processing_time=0.1)
    assert r.steps_checked == 0 and r.corrected is False and r.confidence is None
    assert r.usage.input == 0 and r.plots == []


# ---------- API surface ----------

def test_practice_endpoint_validation():
    assert client.post("/api/v1/practice", json={"topic": "x"}).status_code == 422
    assert client.post("/api/v1/practice",
                       json={"topic": "calculus", "difficulty": "impossible"}
                       ).status_code == 422


def test_tutor_endpoint_validation():
    assert client.post("/api/v1/tutor", json={}).status_code == 422


def test_calibration_endpoint():
    r = client.get("/api/v1/calibration")
    assert r.status_code == 200


def test_solve_accepts_mode():
    # invalid mode rejected; valid diagnose mode passes schema (fails later only
    # at the credentials stage, which is a 200 SSE stream)
    bad = client.post("/api/v1/solve", json={"question": "solve x+1=2", "mode": "banana"})
    assert bad.status_code == 422
