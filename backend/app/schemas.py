"""Pydantic schemas shared by the agent and the API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SolutionStep(BaseModel):
    step_number: int
    description: str
    formula: Optional[str] = None
    explanation: str


class Verification(BaseModel):
    verified: bool
    method: str = Field(description="How the answer was checked, e.g. 'sympy solve_equation cross-check'")
    details: Optional[str] = None


class Diagnosis(BaseModel):
    """Where a student's own attempt went wrong (diagnose mode)."""
    error_step: Optional[int] = Field(None, description="1-based step in the student's work")
    misconception: Optional[str] = None


class Solution(BaseModel):
    """The structured solution the agent must produce."""
    steps: list[SolutionStep]
    final_answer: str
    verification: Verification
    references: list[str] = []
    diagnosis: Optional[Diagnosis] = None


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    image_base64: Optional[str] = Field(
        None, max_length=8_000_000,
        description="Optional figure/diagram as base64 (no data: prefix)")
    image_media_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"] = "image/png"
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = Field(
        None, description="Reasoning effort; use 'xhigh' for hard (JEE Advanced) problems")
    mode: Literal["solve", "diagnose"] = Field(
        "solve", description="'diagnose' finds the error in a student's own attempt")


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0


class SolveResponse(BaseModel):
    question: str
    solution: Solution
    source: Literal["knowledge_base", "web_search", "reasoning", "hybrid"]
    tools_used: list[str]
    sympy_verified: bool          # server-observed: a sympy tool ran without error
    steps_checked: int = 0        # successful check_step spot-checks
    corrected: bool = False       # self-correction retry was triggered
    confidence: Optional[float] = None  # calibrated from benchmark runs, if available
    plots: list[str] = []         # URLs of rendered graphs
    model: str = ""
    usage: TokenUsage = TokenUsage()
    est_cost: float = 0.0         # USD estimate for this solve
    turns: int
    processing_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PracticeRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=100)
    difficulty: Literal["JEE Main", "JEE Advanced"] = "JEE Main"


class PracticeProblem(BaseModel):
    """A generated practice problem with a machine-verified answer key."""
    problem: str
    steps: list[SolutionStep]
    final_answer: str
    verification: Verification
    topic: str = ""
    difficulty: str = ""


class TutorRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    image_base64: Optional[str] = Field(None, max_length=8_000_000)
    image_media_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"] = "image/png"


class FeedbackRequest(BaseModel):
    question: str
    final_answer: str
    accuracy: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=2000)


class GuardrailError(BaseModel):
    detail: str
    guardrail: str
