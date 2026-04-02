"""
Challenge API Routes

POST /api/challenge  — Get a set of personalized quiz questions
"""
import json
from fastapi import APIRouter

from app.core.models import ChallengeRequest, ChallengeResponse, SkillProfile
from app.agents.orchestrator import run_challenge_pipeline
from app.core.database import get_or_create_user

router = APIRouter(prefix="/api", tags=["challenge"])


@router.post("/challenge", response_model=ChallengeResponse)
async def get_challenge(request: ChallengeRequest):
    # Build real skill profile from DB
    user = get_or_create_user(request.user_id)
    profile = SkillProfile(
        user_id=user["user_id"],
        total_guesses=user["total_guesses"],
        average_accuracy=user["average_accuracy"],
        level=user["level"],
        streak=user["streak"],
        weak_categories=json.loads(user["weak_categories"]),
        strong_categories=json.loads(user["strong_categories"]),
    )

    result = await run_challenge_pipeline(
        user_id=request.user_id,
        skill_profile=profile if user["total_guesses"] > 0 else None,
        num_questions=5,
    )

    return ChallengeResponse(
        questions=result.get("quiz_questions", []),
    )
