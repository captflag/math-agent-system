"""API routes: solve (SSE + sync), feedback, health."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent import math_agent
from app.calibration import load as load_calibration
from app.feedback import get_stats, save_feedback
from app.guardrails import validate_input
from app.schemas import FeedbackRequest, PracticeRequest, SolveRequest, TutorRequest
from app.tools.knowledge_base import knowledge_base
from app.tutor import tutor_events

router = APIRouter()


def _sse(gen):
    async def event_stream():
        async for event in gen:
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _check_guardrails(req: SolveRequest) -> str:
    result = validate_input(req.question, has_image=bool(req.image_base64))
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"detail": result.error, "guardrail": result.guardrail},
        )
    return result.sanitized


@router.post("/solve")
async def solve_stream(req: SolveRequest):
    """Stream agent progress and the final solution as Server-Sent Events."""
    sanitized = _check_guardrails(req)
    return _sse(math_agent.solve_events(
        sanitized,
        image_base64=req.image_base64,
        image_media_type=req.image_media_type,
        effort=req.effort,
        mode=req.mode,
    ))


@router.post("/tutor")
async def tutor_stream(req: TutorRequest):
    """Socratic tutoring turn (multi-turn via session_id) as SSE."""
    return _sse(tutor_events(
        req.message,
        session_id=req.session_id,
        image_base64=req.image_base64,
        image_media_type=req.image_media_type,
    ))


@router.post("/practice")
async def practice_stream(req: PracticeRequest):
    """Generate a practice problem with a machine-verified answer key (SSE)."""
    return _sse(math_agent.practice_events(req.topic, req.difficulty))


@router.get("/calibration")
async def calibration():
    """Benchmark-measured accuracy for verified vs unverified answers."""
    return load_calibration() or {"message": "No benchmark run yet — run "
                                  "benchmarks/run_jee_benchmark.py to calibrate."}


@router.post("/solve/sync")
async def solve_sync(req: SolveRequest):
    """Non-streaming variant: returns only the final SolveResponse."""
    sanitized = _check_guardrails(req)
    final = None
    async for event in math_agent.solve_events(
        sanitized,
        image_base64=req.image_base64,
        image_media_type=req.image_media_type,
        effort=req.effort,
        mode=req.mode,
    ):
        if event["type"] == "error":
            raise HTTPException(status_code=502, detail=event["message"])
        if event["type"] == "solution":
            final = event["data"]
    if final is None:
        raise HTTPException(status_code=502, detail="Agent produced no solution.")
    return final


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    feedback_id = save_feedback(
        question=req.question,
        final_answer=req.final_answer,
        accuracy=req.accuracy,
        clarity=req.clarity,
        comments=req.comments,
    )
    return {"feedback_id": feedback_id, "message": "Feedback recorded — thank you."}


@router.get("/feedback/stats")
async def feedback_stats():
    return get_stats()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "kb_entries": len(knowledge_base.entries),
        "version": "2.0.0",
    }
