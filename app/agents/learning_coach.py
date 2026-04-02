"""
Learning Coach Agent

Analyses the user's guess vs actual calories and provides:
- Accuracy feedback
- Personalised tips based on their mistake patterns
- Fun facts to help remember
- Graduation status updates

This is the agent that makes the app feel like a teacher, not a tracker.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.models import FoodItem, NutritionInfo, GuessRecord, SkillProfile
from app.agents.nutrition_analyst import get_fun_facts_for_food, get_common_mistakes_for_food


COACH_PROMPT = """You are a friendly, encouraging nutrition coach for an Indian food calorie learning app.
The user just guessed the calories for their meal. Give them feedback.

Meal: {food_description}
User's guess: {user_guess} calories
Actual calories: {actual_calories} calories
Accuracy: {accuracy_pct}% (100% = perfect guess)
User level: {level}/5
Total guesses so far: {total_guesses}

{mistake_context}

Generate a SHORT, engaging feedback response as JSON:
{{
  "feedback_message": "2-3 sentences. Be encouraging but honest. If they were close, celebrate. If way off, explain why in a memorable way. Use Indian food references.",
  "tips": ["1-2 practical tips specific to this food to help them guess better next time"],
  "encouragement": "A short motivational line based on their level and progress"
}}

TONE GUIDELINES:
- Level 1-2: Very encouraging, celebrate small wins, focus on learning
- Level 3-4: More specific feedback, point out patterns, challenge them
- Level 5: Peer-level feedback, subtle insights, graduation hints

Respond with ONLY JSON. No markdown, no extra text."""


def get_coach_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.6,
        max_tokens=512,
    )


def calculate_accuracy(guess: float, actual: float) -> float:
    """
    Calculate accuracy as a percentage.
    100% = perfect guess, 0% = completely off.

    Uses a generous curve — being within 20% is still pretty good.
    """
    if actual == 0:
        return 0.0

    error_ratio = abs(guess - actual) / actual
    # Convert to accuracy: 0% error = 100%, 50% error = 50%, 100% error = 0%
    accuracy = max(0, (1 - error_ratio)) * 100
    return round(accuracy, 1)


async def generate_feedback(
    food_items: list[FoodItem],
    nutrition_data: list[NutritionInfo],
    user_guess: float,
    actual_total: float,
    skill_profile: SkillProfile | None = None,
) -> dict:
    """
    Generate personalized feedback for a guess attempt.

    Returns dict with: feedback_message, tips, fun_facts, accuracy_pct
    """
    accuracy = calculate_accuracy(user_guess, actual_total)

    # Gather fun facts from knowledge base
    fun_facts = []
    for item in food_items:
        facts = get_fun_facts_for_food(item.name)
        fun_facts.extend(facts)
    fun_facts = fun_facts[:3]  # Cap at 3

    # Check if this matches common mistakes
    mistake_context = ""
    for item in food_items:
        mistakes = get_common_mistakes_for_food(item.name)
        if mistakes:
            mistake_context += f"Common mistake for {item.display_name}: {mistakes[0]}\n"

    # Build food description
    food_desc = ", ".join([
        f"{item.display_name} ({item.portion_description})"
        for item in food_items
    ])

    profile = skill_profile or SkillProfile()

    llm = get_coach_llm()
    prompt = COACH_PROMPT.format(
        food_description=food_desc,
        user_guess=round(user_guess),
        actual_calories=round(actual_total),
        accuracy_pct=round(accuracy),
        level=profile.level,
        total_guesses=profile.total_guesses,
        mistake_context=mistake_context or "No specific common mistakes apply here.",
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Clean markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```")

        data = json.loads(raw)

        tips = data.get("tips", [])
        if isinstance(tips, str):
            tips = [tips]

        return {
            "feedback_message": data.get("feedback_message", f"You guessed {round(user_guess)} cal — the actual was {round(actual_total)} cal."),
            "tips": tips,
            "fun_facts": fun_facts,
            "accuracy_pct": accuracy,
            "encouragement": data.get("encouragement", "Keep going!"),
        }

    except Exception as e:
        print(f"[Learning Coach] LLM feedback failed: {e}")
        # Fallback: generate basic feedback without LLM
        diff = user_guess - actual_total
        if accuracy >= 90:
            msg = f"Incredible! You guessed {round(user_guess)} cal and the actual was {round(actual_total)} cal. That's {round(accuracy)}% accurate!"
        elif accuracy >= 70:
            msg = f"Pretty good! You guessed {round(user_guess)} cal vs actual {round(actual_total)} cal. {'You overestimated a bit.' if diff > 0 else 'You underestimated slightly.'}"
        elif accuracy >= 50:
            hint = "Try accounting for oil/ghee next time." if diff < 0 else "This food is lighter than it looks!"
            msg = f"Not bad! You guessed {round(user_guess)} cal but it was actually {round(actual_total)} cal. {hint}"
        else:
            hint = "The cooking fat and accompaniments add up more than you think!" if diff < 0 else "This dish is lighter than its reputation suggests!"
            msg = f"You guessed {round(user_guess)} cal but it was actually {round(actual_total)} cal. {hint}"

        return {
            "feedback_message": msg,
            "tips": ["Try using your fist as a portion reference — 1 fist of rice ≈ 130 cal."],
            "fun_facts": fun_facts,
            "accuracy_pct": accuracy,
            "encouragement": "Every guess makes you better!",
        }


def create_guess_record(
    food_items: list[FoodItem],
    user_guess: float,
    actual_total: float,
    accuracy: float,
) -> GuessRecord:
    """Create a GuessRecord for the user's history."""
    from datetime import datetime

    primary_food = food_items[0] if food_items else FoodItem(name="unknown", display_name="Unknown")
    return GuessRecord(
        food_name=primary_food.display_name or primary_food.name,
        guessed_calories=user_guess,
        actual_calories=actual_total,
        accuracy_pct=accuracy,
        category=food_items[0].name.split("_")[0] if food_items else "unknown",
        timestamp=datetime.now().isoformat(),
    )
