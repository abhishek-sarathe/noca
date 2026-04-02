"""
Quiz Generator Agent

Creates personalised quiz questions and challenges based on:
- User's skill profile (weak areas get more questions)
- Difficulty scaling based on user level
- Multiple quiz types for variety

Quiz Types:
1. Quick Guess — See food, estimate calories
2. Higher or Lower — Compare two foods
3. Portion Detective — Match portion to calorie count
4. Calorie Swap — "What else could X calories be?"
"""
import json
import random

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.models import QuizQuestion, SkillProfile
from app.core.rag import query_knowledge_base


QUIZ_SYSTEM_PROMPT = """You are a fun, engaging quiz creator for an Indian food calorie learning app.
Create quiz questions that teach users to estimate calories better.

The user's skill level is {level}/5. Weak categories: {weak_categories}.

Generate {num_questions} quiz questions as a JSON array. Mix the question types.

Question types:
1. "quick_guess" — Show a food with portion, user guesses calories
2. "higher_lower" — Two foods, which has more calories?
3. "portion_detective" — Same food, different portions, match to calorie count
4. "calorie_swap" — Given X calories, what other food has similar calories?

For each question provide:
{{
  "question_type": "quick_guess|higher_lower|portion_detective|calorie_swap",
  "question_text": "The question to show the user",
  "food_items": ["food1", "food2"],
  "options": ["option A", "option B", "option C", "option D"],
  "correct_answer": "the correct option",
  "explanation": "A fun, memorable explanation with the actual calorie info",
  "difficulty": 1-5,
  "calories_info": {{"food1": 300, "food2": 250}}
}}

IMPORTANT:
- Use REAL Indian food items and accurate calorie data
- Make explanations fun and memorable, use comparisons
- For level 1-2: common foods, obvious differences
- For level 3-4: similar calorie foods, tricky preparations
- For level 5: restaurant vs homemade, subtle portion differences
- Focus extra questions on the user's weak categories

Respond with ONLY a JSON array. No markdown, no extra text."""


def get_quiz_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.7,  # Higher temp for creative variety
        max_tokens=2048,
    )


async def generate_quiz(
    skill_profile: SkillProfile | None = None,
    num_questions: int = 5,
) -> list[QuizQuestion]:
    """
    Generate personalized quiz questions.

    Args:
        skill_profile: User's current skill profile (or None for default)
        num_questions: How many questions to generate

    Returns:
        List of QuizQuestion objects
    """
    level = skill_profile.level if skill_profile else 1
    weak_cats = skill_profile.weak_categories if skill_profile else ["general"]

    # Get some food knowledge to ground the LLM
    rag_context_parts = []
    for cat in weak_cats[:3]:
        results = query_knowledge_base(
            f"{cat} Indian food calories",
            n_results=3,
            doc_type="portions",
        )
        for r in results:
            rag_context_parts.append(r["text"])

    # Also get some random foods for variety
    random_queries = ["popular Indian snack calories", "Indian breakfast calories", "Indian dessert calories"]
    for q in random.sample(random_queries, min(2, len(random_queries))):
        results = query_knowledge_base(q, n_results=2)
        for r in results:
            rag_context_parts.append(r["text"])

    rag_context = "\n".join(rag_context_parts[:10])

    llm = get_quiz_llm()

    prompt = QUIZ_SYSTEM_PROMPT.format(
        level=level,
        weak_categories=", ".join(weak_cats) if weak_cats else "none identified yet",
        num_questions=num_questions,
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Here's some food data to use:\n\n{rag_context}\n\nNow generate {num_questions} quiz questions."),
    ]

    response = await llm.ainvoke(messages)
    raw_text = response.content.strip()

    try:
        # Clean markdown fences
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip().rstrip("```")

        questions_data = json.loads(raw_text)
        questions = [QuizQuestion(**q) for q in questions_data]
        return questions

    except Exception as e:
        print(f"[Quiz Generator] Failed to parse LLM response: {e}")
        print(f"[Quiz Generator] Raw: {raw_text[:500]}")
        # Return a fallback question
        return [
            QuizQuestion(
                question_type="quick_guess",
                question_text="How many calories are in 1 regular samosa?",
                food_items=["samosa"],
                options=["150 cal", "250 cal", "350 cal", "450 cal"],
                correct_answer="250 cal",
                explanation="A regular samosa (~100g) has about 250 calories. The deep-fried pastry shell absorbs a lot of oil!",
                difficulty=1,
                calories_info={"samosa": 250},
            )
        ]
