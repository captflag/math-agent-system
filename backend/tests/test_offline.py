"""Offline tests — everything deterministic runs without an API key."""
import json

import pytest
from fastapi.testclient import TestClient

from app.guardrails import validate_input
from app.main import app
from app.schemas import Solution
from app.tools.knowledge_base import knowledge_base, search_knowledge_base
from app.tools.math_tools import (
    differentiate,
    evaluate_expression,
    integrate_expression,
    simplify_expression,
    solve_equation,
)

client = TestClient(app)


# ---------- SymPy tools ----------

def test_solve_quadratic():
    result = solve_equation("x**2 + 5*x + 6 = 0")
    assert "-2" in result and "-3" in result


def test_solve_without_equals():
    assert "6" in solve_equation("3*x - 18", "x")


def test_differentiate():
    result = differentiate("x**3 + 2*x**2 - 5*x + 1")
    assert "3*x**2" in result and "4*x" in result


def test_integrate_definite():
    assert "= 9" in integrate_expression("2*x", "x", "0", "3")


def test_integrate_indefinite():
    assert "x**3/3" in integrate_expression("x**2")


def test_evaluate():
    assert "153.9" in evaluate_expression("pi * 7**2", precision=6)


def test_simplify():
    result = simplify_expression("sin(x)**2 + cos(x)**2")
    assert "simplified = 1" in result


def test_tool_error_is_returned_not_raised():
    result = solve_equation("x^^^bad(((")
    assert result.startswith("error:")


def test_no_code_injection():
    result = evaluate_expression("__import__('os').system('echo pwned')")
    assert result.startswith("error:")


# ---------- Knowledge base ----------

def test_kb_loaded():
    assert len(knowledge_base.entries) >= 20


def test_kb_retrieval_quadratic():
    results = knowledge_base.search("solve the quadratic equation x^2 + 5x + 6 = 0")
    assert results and results[0]["id"] == "alg-001"


def test_kb_retrieval_derivative():
    results = knowledge_base.search("derivative of x^3 + 2x^2")
    assert results and results[0]["topic"] == "calculus"


def test_kb_tool_formats_text():
    text = search_knowledge_base("area of a circle radius 7")
    assert "geo-001" in text


def test_kb_no_match():
    assert "No relevant" in search_knowledge_base("zxqv wvutk")


# ---------- Guardrails ----------

def test_guardrail_accepts_math():
    r = validate_input("Solve x^2 - 4 = 0")
    assert r.ok and r.sanitized


def test_guardrail_rejects_offtopic():
    r = validate_input("What is the best pizza topping in town today?")
    assert not r.ok and r.guardrail == "topic"


def test_guardrail_rejects_too_long():
    r = validate_input("solve " + "x" * 3000)
    assert not r.ok and r.guardrail == "length"


def test_guardrail_scrubs_email():
    r = validate_input("I am john.doe@example.com, solve x + 1 = 2")
    assert r.ok and "john.doe@example.com" not in r.sanitized


# ---------- Schemas ----------

def test_solution_schema_roundtrip():
    payload = {
        "steps": [{"step_number": 1, "description": "Factor", "formula": "(x+2)(x+3)",
                   "explanation": "Split the quadratic into linear factors."}],
        "final_answer": "x = -2 or x = -3",
        "verification": {"verified": True, "method": "sympy solve_equation", "details": None},
        "references": ["alg-001"],
    }
    sol = Solution.model_validate_json(json.dumps(payload))
    assert sol.verification.verified


# ---------- API (no LLM calls) ----------

def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["kb_entries"] >= 20


def test_solve_guardrail_422():
    r = client.post("/api/v1/solve", json={"question": "tell me a story about a dragon"})
    assert r.status_code == 422


def test_feedback_roundtrip(tmp_path, monkeypatch):
    from app import feedback as fb
    from app.config import settings
    monkeypatch.setattr(settings, "FEEDBACK_DB", tmp_path / "fb.sqlite3")

    fid = fb.save_feedback("Q1", "42", accuracy=5, clarity=5)
    assert fid == 1
    stats = fb.get_stats()
    assert stats["count"] == 1 and stats["avg_accuracy"] == 5.0
    shots = fb.get_few_shot_examples()
    assert shots and shots[0]["final_answer"] == "42"


def test_feedback_api_validation():
    r = client.post("/api/v1/feedback", json={
        "question": "q", "final_answer": "a", "accuracy": 9, "clarity": 3,
    })
    assert r.status_code == 422
