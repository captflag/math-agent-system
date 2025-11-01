from fastapi import APIRouter

router = APIRouter()

@router.post("/question")
async def submit_question(question: dict):
    return {"message": "Question received", "question": question}
