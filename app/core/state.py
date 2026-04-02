"""
LangGraph shared state schema.

This is the single source of truth that flows through the agent pipeline.
Every agent reads from and writes to this state.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from app.core.models import FoodItem, NutritionInfo, QuizQuestion, SkillProfile


class GuessState(BaseModel):
    """State that flows through the Guess pipeline."""

    # ── Input ──
    user_id: str = "anonymous"
    food_image_base64: Optional[str] = None
    food_text: Optional[str] = None
    user_guess_calories: Optional[float] = None

    # ── Food Identifier Agent output ──
    identified_foods: list[FoodItem] = Field(default_factory=list)
    identification_confidence: float = 0.0

    # ── Nutrition Analyst Agent output ──
    nutrition_data: list[NutritionInfo] = Field(default_factory=list)
    total_actual_calories: float = 0.0

    # ── Learning Coach Agent output ──
    accuracy_percentage: float = 0.0
    feedback_message: str = ""
    fun_facts: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)

    # ── Control flow ──
    current_step: str = "start"
    error: Optional[str] = None


class ChallengeState(BaseModel):
    """State for the Challenge/Quiz pipeline."""

    user_id: str = "anonymous"
    skill_profile: Optional[SkillProfile] = None
    quiz_questions: list[QuizQuestion] = Field(default_factory=list)
    current_step: str = "start"
    error: Optional[str] = None
