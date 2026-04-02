"""
Challenge API Routes

GET  /api/challenge  — Get a set of personalized quiz questions
"""
from fastapi import APIRouter

from app.core.models import ChallengeRequest, ChallengeResponse
from app.agents.orchestrator import run_challenge_pipeline

router = APIRouter(prefix="/api", tags=["challenge"])


@router.post("/challenge", response_model=ChallengeResponse)
async def get_challenge(request: ChallengeRequest):
    """
    Get a set of personalised quiz questions.

    Questions are tailored to the user's skill level and weak areas.
    """
    result = await run_challenge_pipeline(
        user_id=request.user_id,
        num_questions=5,
    )

    return ChallengeResponse(
        questions=result.get("quiz_questions", []),
    )
