"""
LangGraph Orchestrator

Wires all 4 agents into a StateGraph pipeline.
This is the "brain" that routes requests through the right agents
in the right order.

Two pipelines:
1. Guess Pipeline: Food Identifier → Nutrition Analyst → Learning Coach
2. Challenge Pipeline: Quiz Generator (with skill profile input)
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from app.agents.food_identifier import identify_food
from app.agents.nutrition_analyst import analyse_nutrition
from app.agents.learning_coach import generate_feedback, calculate_accuracy
from app.agents.quiz_generator import generate_quiz
from app.core.models import FoodItem, NutritionInfo, SkillProfile, QuizQuestion


# ═══════════════════════════════════════════
# Guess Pipeline State
# ═══════════════════════════════════════════

class GuessGraphState(TypedDict):
    """State flowing through the Guess pipeline."""
    # Inputs
    food_text: str
    food_image_base64: str
    user_guess_calories: float
    user_id: str

    # Agent outputs (accumulated through the pipeline)
    identified_foods: list[FoodItem]
    nutrition_data: list[NutritionInfo]
    total_actual_calories: float
    accuracy_percentage: float
    feedback_message: str
    fun_facts: list[str]
    tips: list[str]

    # Control
    error: str


# ═══════════════════════════════════════════
# Guess Pipeline Nodes
# ═══════════════════════════════════════════

async def identify_food_node(state: GuessGraphState) -> dict:
    """Node 1: Identify foods from text/image."""
    try:
        foods = await identify_food(
            food_text=state.get("food_text"),
            food_image_base64=state.get("food_image_base64"),
        )
        return {"identified_foods": foods}
    except Exception as e:
        return {"error": f"Food identification failed: {str(e)}"}


async def analyse_nutrition_node(state: GuessGraphState) -> dict:
    """Node 2: Get nutrition data for identified foods."""
    try:
        foods = state.get("identified_foods", [])
        if not foods:
            return {"error": "No foods identified to analyse."}

        nutrition = await analyse_nutrition(foods)
        total_cal = sum(n.calories for n in nutrition)
        return {
            "nutrition_data": nutrition,
            "total_actual_calories": total_cal,
        }
    except Exception as e:
        return {"error": f"Nutrition analysis failed: {str(e)}"}


async def generate_feedback_node(state: GuessGraphState) -> dict:
    """Node 3: Generate learning feedback based on guess accuracy."""
    try:
        foods = state.get("identified_foods", [])
        nutrition = state.get("nutrition_data", [])
        user_guess = state.get("user_guess_calories", 0)
        actual = state.get("total_actual_calories", 0)

        feedback = await generate_feedback(
            food_items=foods,
            nutrition_data=nutrition,
            user_guess=user_guess,
            actual_total=actual,
            skill_profile=None,  # TODO: Load from DB
        )

        return {
            "accuracy_percentage": feedback["accuracy_pct"],
            "feedback_message": feedback["feedback_message"],
            "fun_facts": feedback["fun_facts"],
            "tips": feedback["tips"],
        }
    except Exception as e:
        return {"error": f"Feedback generation failed: {str(e)}"}


def should_continue_after_identify(state: GuessGraphState) -> str:
    """Conditional edge: continue to nutrition or stop on error."""
    if state.get("error"):
        return "end"
    if not state.get("identified_foods"):
        return "end"
    return "analyse_nutrition"


def should_continue_after_nutrition(state: GuessGraphState) -> str:
    """Conditional edge: continue to feedback or stop on error."""
    if state.get("error"):
        return "end"
    return "generate_feedback"


# ═══════════════════════════════════════════
# Build the Guess Pipeline Graph
# ═══════════════════════════════════════════

def build_guess_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the Guess pipeline.

    Flow:
    START → identify_food → analyse_nutrition → generate_feedback → END
           (with error short-circuits at each step)
    """
    graph = StateGraph(GuessGraphState)

    # Add nodes
    graph.add_node("identify_food", identify_food_node)
    graph.add_node("analyse_nutrition", analyse_nutrition_node)
    graph.add_node("generate_feedback", generate_feedback_node)

    # Set entry point
    graph.set_entry_point("identify_food")

    # Add conditional edges
    graph.add_conditional_edges(
        "identify_food",
        should_continue_after_identify,
        {
            "analyse_nutrition": "analyse_nutrition",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "analyse_nutrition",
        should_continue_after_nutrition,
        {
            "generate_feedback": "generate_feedback",
            "end": END,
        },
    )

    # Feedback → END
    graph.add_edge("generate_feedback", END)

    return graph


# Compile the graph (singleton)
guess_pipeline = build_guess_graph().compile()


# ═══════════════════════════════════════════
# Challenge Pipeline State
# ═══════════════════════════════════════════

class ChallengeGraphState(TypedDict):
    user_id: str
    skill_profile: SkillProfile | None
    num_questions: int
    challenge_type: str  # specific type or empty for mix
    quiz_questions: list[QuizQuestion]
    error: str


async def generate_quiz_node(state: ChallengeGraphState) -> dict:
    """Generate quiz questions based on skill profile."""
    try:
        questions = await generate_quiz(
            skill_profile=state.get("skill_profile"),
            num_questions=state.get("num_questions", 5),
            challenge_type=state.get("challenge_type", ""),
        )
        return {"quiz_questions": questions}
    except Exception as e:
        return {"error": f"Quiz generation failed: {str(e)}"}


def build_challenge_graph() -> StateGraph:
    """Build the Challenge pipeline graph."""
    graph = StateGraph(ChallengeGraphState)
    graph.add_node("generate_quiz", generate_quiz_node)
    graph.set_entry_point("generate_quiz")
    graph.add_edge("generate_quiz", END)
    return graph


challenge_pipeline = build_challenge_graph().compile()


# ═══════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════

async def run_guess_pipeline(
    food_text: str | None = None,
    food_image_base64: str | None = None,
    user_guess_calories: float = 0,
    user_id: str = "anonymous",
) -> dict:
    """
    Run the full guess pipeline.

    Returns the final state with all agent outputs.
    """
    initial_state = {
        "food_text": food_text or "",
        "food_image_base64": food_image_base64 or "",
        "user_guess_calories": user_guess_calories,
        "user_id": user_id,
        "identified_foods": [],
        "nutrition_data": [],
        "total_actual_calories": 0.0,
        "accuracy_percentage": 0.0,
        "feedback_message": "",
        "fun_facts": [],
        "tips": [],
        "error": "",
    }

    result = await guess_pipeline.ainvoke(initial_state)
    return result


async def run_challenge_pipeline(
    user_id: str = "anonymous",
    skill_profile: SkillProfile | None = None,
    num_questions: int = 5,
    challenge_type: str | None = None,
) -> dict:
    """Run the challenge/quiz pipeline."""
    initial_state = {
        "user_id": user_id,
        "skill_profile": skill_profile,
        "num_questions": num_questions,
        "challenge_type": challenge_type or "",
        "quiz_questions": [],
        "error": "",
    }

    result = await challenge_pipeline.ainvoke(initial_state)
    return result
