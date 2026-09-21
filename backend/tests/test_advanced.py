"""Offline tests for the advanced tools, grader, and new API fields."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.guardrails import validate_input
from app.tools.advanced_tools import (
    complex_operation,
    matrix_operation,
    solve_inequality,
    solve_system,
    vector_operation,
)
from benchmarks.grader import extract_letters, extract_number, grade


# ---------- Systems & inequalities ----------

def test_solve_system_linear():
    result = solve_system(["2*x + y = 7", "x - y = 2"], ["x", "y"])
    assert "x: 3" in result and "y: 1" in result


def test_solve_system_nonlinear():
    result = solve_system(["x**2 + y**2 = 25", "x - y = 1"], ["x", "y"])
    assert "solutions:" in result and "error" not in result


def test_solve_inequality():
    result = solve_inequality("x**2 - 4 > 0")
    assert "2" in result and "error" not in result


def test_solve_inequality_le():
    result = solve_inequality("2*x + 3 <= 7")
    assert "error" not in result


# ---------- Matrices ----------

def test_matrix_determinant():
    assert "det = -2" in matrix_operation([["1", "2"], ["3", "4"]], "determinant")


def test_matrix_inverse():
    result = matrix_operation([["2", "1"], ["1", "1"]], "inverse")
    assert "[1, -1]" in result and "[-1, 2]" in result


def test_matrix_eigenvalues():
    result = matrix_operation([["2", "0"], ["0", "3"]], "eigenvalues")
    assert "2" in result and "3" in result


def test_matrix_multiply():
    result = matrix_operation([["1", "0"], ["0", "1"]], "multiply",
                              other=[["5", "6"], ["7", "8"]])
    assert "[5, 6]" in result


def test_matrix_symbolic():
    result = matrix_operation([["a", "b"], ["c", "d"]], "determinant")
    assert "a*d - b*c" in result


def test_matrix_injection_blocked():
    result = matrix_operation([["__import__('os')"]], "determinant")
    assert result.startswith("error:")


# ---------- Vectors ----------

def test_vector_dot():
    assert "a·b = 0" in vector_operation(["1", "1"], "dot", b=["1", "-1"])


def test_vector_cross():
    result = vector_operation(["1", "1", "1"], "cross", b=["1", "2", "3"])
    assert "[1, -2, 1]" in result


def test_vector_magnitude():
    assert "sqrt(6)" in vector_operation(["1", "2", "-1"], "magnitude")


# ---------- Complex numbers ----------

def test_complex_power():
    assert "= 16" in complex_operation("(1 + I)**8", "simplify")


def test_complex_modulus():
    assert "sqrt(2)" in complex_operation("1 + I", "modulus")


def test_complex_rectangular():
    result = complex_operation("(2 + 3*I)/(1 - I)", "rectangular")
    assert "-1/2" in result and "5/2" in result


# ---------- Grader ----------

def test_extract_letters():
    assert extract_letters("The answer is C") == ["C"]
    assert extract_letters("A, D are correct") == ["A", "D"]
    assert extract_letters("no options here") == []


def test_extract_number():
    assert extract_number("The value is 0.5") == 0.5
    assert extract_number("answer: -3") == -3.0
    assert extract_number("1,234.5 units") == 1234.5


def test_grade_mcq():
    assert grade("MCQ", "C", "C) -2 tan θ")["correct"]
    assert not grade("MCQ", "C", "B")["correct"]


def test_grade_mcq_multiple():
    assert grade("MCQ(multiple)", "AD", "A, D")["correct"]
    r = grade("MCQ(multiple)", "ABD", "A, D")
    assert not r["correct"] and r["partial"]
    assert not grade("MCQ(multiple)", "AD", "A, B")["partial"]


def test_grade_integer():
    assert grade("Integer", "5", "5")["correct"]
    assert not grade("Integer", "5", "6")["correct"]


def test_grade_numeric_tolerance():
    assert grade("Numeric", "0.5", "0.50")["correct"]
    assert grade("Numeric", "0.5", "0.51")["correct"]
    assert not grade("Numeric", "0.5", "0.53")["correct"]


# ---------- Guardrails with image ----------

def test_image_bypasses_topic_check():
    r = validate_input("Solve the problem shown in the attached figure", has_image=True)
    assert r.ok


def test_no_image_still_enforces_topic():
    r = validate_input("Describe the picture of my cat please", has_image=False)
    assert not r.ok
