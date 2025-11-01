"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class SourceType(str, Enum):
    """Source of the answer"""
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"

class QuestionRequest(BaseModel):
    """Request model for submitting a question"""
    question: str = Field(..., max_length=1000, description="Mathematical question")
    context: Optional[str] = Field(None, description="Additional context")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class Step(BaseModel):
    """Individual step in solution"""
    step_number: int
    description: str
    formula: Optional[str] = None
    explanation: str

class QuestionResponse(BaseModel):
    """Response model for question answer"""
    question_id: str
    question: str
    source: SourceType
    confidence: float
    steps: List[Step]
    final_answer: str
    references: Optional[List[str]] = []
    processing_time: float
    timestamp: datetime

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback"""
    question_id: str
    accuracy_rating: int = Field(..., ge=1, le=5, description="Accuracy rating (1-5)")
    clarity_rating: int = Field(..., ge=1, le=5, description="Clarity rating (1-5)")
    step_helpfulness: int = Field(..., ge=1, le=5, description="Step-by-step helpfulness (1-5)")
    improvements: Optional[str] = Field(None, description="Suggested improvements")
    user_id: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    feedback_id: str
    question_id: str
    message: str
    timestamp: datetime

class StatusResponse(BaseModel):
    """Response model for question processing status"""
    question_id: str
    source: SourceType
    confidence: float
    processing_time: float
    retrieval_details: Dict[str, any]

class GuardrailViolation(BaseModel):
    """Guardrail violation details"""
    type: str
    message: str
    severity: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, bool]
    timestamp: datetime
