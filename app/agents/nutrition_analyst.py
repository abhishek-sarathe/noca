"""
Nutrition Analyst Agent

Takes identified food items and retrieves accurate nutrition data
using RAG against the Indian Food Knowledge Base.

Flow:
1. Try direct lookup by canonical name (fastest, most accurate)
2. If not found, use RAG similarity search
3. Apply portion size and preparation multipliers
4. Return structured NutritionInfo
"""
import json
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.models import FoodItem, NutritionInfo
from app.core.rag import get_food_by_name, query_knowledge_base


ESTIMATION_PROMPT = """You are a nutrition analyst specializing in Indian food. 
Given the food item details and knowledge base context, provide your best calorie estimate.

Food item: {food_name}
Portion: {portion_description} (~{weight_grams}g)
Preparation: {preparation_style}

Knowledge base context:
{rag_context}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "calories": <number>,
  "protein_g": <number>,
  "carbs_g": <number>,
  "fat_g": <number>,
  "fiber_g": <number>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief one-line explanation of your estimate>"
}}"""


def get_nutrition_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.1,
        max_tokens=512,
    )


def _calculate_from_knowledge_base(food_data: dict, food_item: FoodItem) -> Optional[NutritionInfo]:
    """
    Calculate calories directly from knowledge base data.
    This is the preferred path — no LLM call needed, most accurate.
    """
    base_cal = food_data["base_calories_per_100g"]
    weight = food_item.estimated_weight_grams

    # Find best matching portion
    best_portion_cal = None
    for portion_name, portion_data in food_data.get("portions", {}).items():
        # If weights are close, use the predefined calorie value
        if abs(portion_data["grams"] - weight) < 50:
            best_portion_cal = portion_data["calories"]
            break

    if best_portion_cal is not None:
        calories = best_portion_cal
    else:
        # Calculate from base rate
        calories = (base_cal * weight) / 100

    # Apply preparation multiplier
    prep_style = food_item.preparation_style.lower()
    for prep_name, prep_data in food_data.get("preparations", {}).items():
        if prep_name.lower() in prep_style or prep_style in prep_name.lower():
            calories *= prep_data["multiplier"]
            break

    # Rough macro estimates (can be refined with actual IFCT data later)
    # For now, use general Indian food macro ratios
    return NutritionInfo(
        food_name=food_item.name,
        calories=round(calories, 1),
        protein_g=round(calories * 0.12 / 4, 1),    # ~12% protein
        carbs_g=round(calories * 0.55 / 4, 1),       # ~55% carbs
        fat_g=round(calories * 0.33 / 9, 1),         # ~33% fat
        fiber_g=round(weight * 0.02, 1),              # rough estimate
        portion_grams=weight,
        source="knowledge_base",
        confidence=0.85,
    )


async def _estimate_with_llm(food_item: FoodItem, rag_context: str) -> NutritionInfo:
    """
    Fallback: Use LLM with RAG context to estimate nutrition.
    Used when the food item isn't in our knowledge base.
    """
    llm = get_nutrition_llm()

    prompt = ESTIMATION_PROMPT.format(
        food_name=food_item.display_name or food_item.name,
        portion_description=food_item.portion_description,
        weight_grams=food_item.estimated_weight_grams,
        preparation_style=food_item.preparation_style,
        rag_context=rag_context,
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw_text = response.content.strip()

    try:
        # Clean markdown fences
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip().rstrip("```")

        data = json.loads(raw_text)
        return NutritionInfo(
            food_name=food_item.name,
            calories=round(float(data.get("calories", 0)), 1),
            protein_g=round(float(data.get("protein_g", 0)), 1),
            carbs_g=round(float(data.get("carbs_g", 0)), 1),
            fat_g=round(float(data.get("fat_g", 0)), 1),
            fiber_g=round(float(data.get("fiber_g", 0)), 1),
            portion_grams=food_item.estimated_weight_grams,
            source="llm_estimate",
            confidence=round(float(data.get("confidence", 0.5)), 2),
        )
    except Exception as e:
        print(f"[Nutrition Analyst] LLM estimation failed: {e}")
        # Last resort: rough estimate from weight
        return NutritionInfo(
            food_name=food_item.name,
            calories=round(food_item.estimated_weight_grams * 1.5, 1),  # ~150 cal/100g average
            portion_grams=food_item.estimated_weight_grams,
            source="fallback_estimate",
            confidence=0.3,
        )


async def analyse_nutrition(food_items: list[FoodItem]) -> list[NutritionInfo]:
    """
    Get nutrition data for a list of identified food items.

    Strategy:
    1. Direct knowledge base lookup (fastest, most accurate)
    2. RAG-enhanced LLM estimation (for items not in KB)
    """
    results = []

    for item in food_items:
        # Step 1: Try direct lookup
        food_data = get_food_by_name(item.name)

        if food_data:
            # Direct calculation from knowledge base
            nutrition = _calculate_from_knowledge_base(food_data, item)
            if nutrition:
                results.append(nutrition)
                continue

        # Step 2: RAG search for similar foods + LLM estimation
        query = f"{item.display_name} {item.preparation_style} calories nutrition"
        rag_results = query_knowledge_base(query, n_results=3)
        rag_context = "\n".join([r["text"] for r in rag_results])

        nutrition = await _estimate_with_llm(item, rag_context)
        results.append(nutrition)

    return results


def get_fun_facts_for_food(food_name: str) -> list[str]:
    """Get fun facts and learning content for a food item."""
    food_data = get_food_by_name(food_name)
    if food_data:
        return food_data.get("fun_facts", [])

    # Fallback: RAG search for learning content
    results = query_knowledge_base(
        f"{food_name} fun facts calories",
        n_results=2,
        doc_type="learning",
    )
    facts = []
    for r in results:
        meta = r.get("metadata", {})
        if "fun_facts_json" in meta:
            facts.extend(json.loads(meta["fun_facts_json"]))
    return facts[:3]


def get_common_mistakes_for_food(food_name: str) -> list[str]:
    """Get common estimation mistakes for a food item."""
    food_data = get_food_by_name(food_name)
    if food_data:
        return food_data.get("common_mistakes", [])
    return []
