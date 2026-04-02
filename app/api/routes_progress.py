"""
Progress API Routes

GET /api/progress/:user_id  — Get full progress data for the dashboard
"""
from fastapi import APIRouter

from app.core.database import get_user_progress

router = APIRouter(prefix="/api", tags=["progress"])


@router.get("/progress/{user_id}")
async def get_progress(user_id: str):
    """Get user's full progress data for the Progress page."""
    return get_user_progress(user_id)
