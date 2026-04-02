"""
Pydantic models — the data contracts between agents, API, and frontend.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════
# Food & Nutrition Models
# ═══════════════════════════════════════════

class FoodItem(BaseModel):
    """A single identified food item with portion estimate."""
    name: str                                    # canonical name, e.g. "poha"
    display_name: str = ""                       # user-friendly name, e.g. "Poha (Flattened Rice)"
    portion_description: str = ""                # e.g. "1 medium plate"
    estimated_weight_grams: float = 0.0          # e.g. 250.0
    preparation_style: str = "standard"          # e.g. "with peanuts and curry leaves"
    confidence: float = 0.0                      # 0.0 to 1.0


class NutritionInfo(BaseModel):
    """Nutrition breakdown for a single food item."""
    food_name: str
    calories: float
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    portion_grams: float = 0.0
    source: str = "knowledge_base"               # "knowledge_base" | "llm_estimate"
    confidence: float = 0.0


# ═══════════════════════════════════════════
# Quiz & Challenge Models
# ═══════════════════════════════════════════

class QuizQuestion(BaseModel):
    """A single quiz/challenge question."""
    question_type: str                           # "quick_guess" | "higher_lower" | "portion_detective" | "calorie_swap"
    question_text: str
    food_items: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)     # for multiple choice
    correct_answer: str = ""
    explanation: str = ""
    difficulty: int = 1                          # 1-5
    calories_info: dict = Field(default_factory=dict)    # supporting calorie data


# ═══════════════════════════════════════════
# User Profile & Progress Models
# ═══════════════════════════════════════════

class GuessRecord(BaseModel):
    """A single guess attempt record."""
    food_name: str
    guessed_calories: float
    actual_calories: float
    accuracy_pct: float                          # how close the guess was (100 = perfect)
    category: str = ""                           # "breakfast", "street_food", etc.
    timestamp: str = ""


class SkillProfile(BaseModel):
    """User's learning progress and skill assessment."""
    user_id: str = "anonymous"
    total_guesses: int = 0
    average_accuracy: float = 0.0                # rolling average accuracy %
    streak: int = 0                              # current daily streak
    weak_categories: list[str] = Field(default_factory=list)   # categories they struggle with
    strong_categories: list[str] = Field(default_factory=list)
    guess_history: list[GuessRecord] = Field(default_factory=list)
    graduation_ready: bool = False               # True when accuracy > threshold
    level: int = 1                               # 1=Beginner, 5=Expert

    def update_with_guess(self, record: GuessRecord) -> None:
        """Update profile with a new guess result."""
        self.guess_history.append(record)
        self.total_guesses += 1

        # Rolling average of last 20 guesses
        recent = self.guess_history[-20:]
        self.average_accuracy = sum(g.accuracy_pct for g in recent) / len(recent)

        # Update level based on accuracy
        if self.average_accuracy >= 90:
            self.level = 5
            self.graduation_ready = True
        elif self.average_accuracy >= 75:
            self.level = 4
        elif self.average_accuracy >= 60:
            self.level = 3
        elif self.average_accuracy >= 40:
            self.level = 2
        else:
            self.level = 1

        # Recalculate weak/strong categories
        self._update_category_skills()

    def _update_category_skills(self) -> None:
        """Identify weak and strong food categories."""
        category_scores: dict[str, list[float]] = {}
        for g in self.guess_history[-50:]:
            if g.category:
                category_scores.setdefault(g.category, []).append(g.accuracy_pct)

        self.weak_categories = []
        self.strong_categories = []
        for cat, scores in category_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 50:
                self.weak_categories.append(cat)
            elif avg > 75:
                self.strong_categories.append(cat)


# ═══════════════════════════════════════════
# API Request/Response Models
# ═══════════════════════════════════════════

class GuessRequest(BaseModel):
    """What the frontend sends when user submits a guess."""
    food_text: Optional[str] = None              # text description of food
    food_image_base64: Optional[str] = None      # base64 image data
    user_guess_calories: float                   # the user's estimate
    user_id: str = "anonymous"


class GuessResponse(BaseModel):
    """What we send back after processing a guess."""
    identified_foods: list[FoodItem]
    nutrition_data: list[NutritionInfo]
    total_actual_calories: float
    user_guess_calories: float
    accuracy_percentage: float
    feedback_message: str
    fun_facts: list[str]
    tips: list[str]


class ChallengeRequest(BaseModel):
    user_id: str = "anonymous"


class ChallengeResponse(BaseModel):
    questions: list[QuizQuestion]
