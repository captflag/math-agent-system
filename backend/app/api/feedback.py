from fastapi import APIRouter

router = APIRouter()

@router.post("/feedback")
async def submit_feedback(feedback: dict):
    return {"message": "Feedback received", "feedback": feedback}
