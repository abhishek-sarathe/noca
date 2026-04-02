"""
Guess API Routes

POST /api/guess  — Submit a food description + calorie guess, get back the full analysis
"""
from fastapi import APIRouter, HTTPException

from app.core.models import GuessRequest, GuessResponse, FoodItem, NutritionInfo
from app.agents.orchestrator import run_guess_pipeline

router = APIRouter(prefix="/api", tags=["guess"])


@router.post("/guess", response_model=GuessResponse)
async def submit_guess(request: GuessRequest):
    """
    Submit a food guess and get back the full analysis.

    Send a food description (text) or image, along with your calorie guess.
    The system will:
    1. Identify the food items
    2. Look up accurate calorie data
    3. Compare to your guess
    4. Give you personalised feedback
    """
    if not request.food_text and not request.food_image_base64:
        raise HTTPException(
            status_code=400,
            detail="Please provide either food_text or food_image_base64",
        )

    # Run the full agent pipeline
    result = await run_guess_pipeline(
        food_text=request.food_text,
        food_image_base64=request.food_image_base64,
        user_guess_calories=request.user_guess_calories,
        user_id=request.user_id,
    )

    # Check for errors
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Build response
    return GuessResponse(
        identified_foods=result.get("identified_foods", []),
        nutrition_data=result.get("nutrition_data", []),
        total_actual_calories=result.get("total_actual_calories", 0),
        user_guess_calories=request.user_guess_calories,
        accuracy_percentage=result.get("accuracy_percentage", 0),
        feedback_message=result.get("feedback_message", ""),
        fun_facts=result.get("fun_facts", []),
        tips=result.get("tips", []),
    )
