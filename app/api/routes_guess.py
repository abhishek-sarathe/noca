"""
Guess API Routes

POST /api/guess  — Submit a food description + calorie guess, get back the full analysis
"""
from fastapi import APIRouter, HTTPException

from app.core.models import GuessRequest, GuessResponse
from app.agents.orchestrator import run_guess_pipeline
from app.core.database import save_guess

router = APIRouter(prefix="/api", tags=["guess"])


@router.post("/guess", response_model=GuessResponse)
async def submit_guess(request: GuessRequest):
    if not request.food_text and not request.food_image_base64:
        raise HTTPException(
            status_code=400,
            detail="Please provide either food_text or food_image_base64",
        )

    result = await run_guess_pipeline(
        food_text=request.food_text,
        food_image_base64=request.food_image_base64,
        user_guess_calories=request.user_guess_calories,
        user_id=request.user_id,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Calculate total protein
    nutrition_data = result.get("nutrition_data", [])
    total_protein = sum(n.protein_g for n in nutrition_data)
    identified_foods = result.get("identified_foods", [])

    # Determine primary category from first identified food
    category = ""
    if identified_foods:
        name = identified_foods[0].name.lower()
        # Simple category mapping
        breakfast_foods = {"idli", "dosa", "poha", "upma", "paratha", "egg_bhurji"}
        street_foods = {"samosa", "pani_puri", "vada_pav", "pav_bhaji"}
        curries = {"dal", "butter_chicken", "paneer_butter_masala", "rajma", "chole", "palak_paneer", "aloo_gobi"}
        rice_dishes = {"rice", "chicken_biryani", "biryani"}
        if name in breakfast_foods:
            category = "breakfast"
        elif name in street_foods:
            category = "street_food"
        elif name in curries:
            category = "curry"
        elif name in rice_dishes:
            category = "rice_dish"

    # Save to database
    try:
        save_guess(
            user_id=request.user_id,
            food_description=request.food_text or "Image upload",
            identified_foods=identified_foods,
            user_guess_calories=request.user_guess_calories,
            user_guess_protein=getattr(request, "user_guess_protein", 0) or 0,
            actual_calories=result.get("total_actual_calories", 0),
            actual_protein=total_protein,
            accuracy_pct=result.get("accuracy_percentage", 0),
            category=category,
            feedback=result.get("feedback_message", ""),
            fun_facts=result.get("fun_facts", []),
        )
    except Exception as e:
        print(f"[DB] Failed to save guess: {e}")

    return GuessResponse(
        identified_foods=identified_foods,
        nutrition_data=nutrition_data,
        total_actual_calories=result.get("total_actual_calories", 0),
        user_guess_calories=request.user_guess_calories,
        accuracy_percentage=result.get("accuracy_percentage", 0),
        feedback_message=result.get("feedback_message", ""),
        fun_facts=result.get("fun_facts", []),
        tips=result.get("tips", []),
    )
