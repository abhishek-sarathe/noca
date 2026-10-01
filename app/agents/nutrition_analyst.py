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

    Strategy (in order of reliability):
    1. Extract piece count from portion_description → interpolate from our portion data
    2. Find closest matching portion by weight
    3. Last resort: base_cal × weight / 100
    """
    portions = food_data.get("portions", {})
    base_cal = food_data["base_calories_per_100g"]
    llm_weight = food_item.estimated_weight_grams

    # ── Strategy 1: Extract piece count and interpolate ──
    # Parse "8 pieces", "2 pieces", "1 plate" etc from the LLM's portion description
    import re
    piece_match = re.search(r'(\d+)\s*(?:piece|pcs|pc)', food_item.portion_description.lower())

    if piece_match:
        user_pieces = int(piece_match.group(1))

        # Extract piece counts and calories from our portion data
        # e.g., "6 pieces (1 plate)" → 6, "1 piece (~40g)" → 1
        portion_points = []  # [(pieces, calories, grams)]
        for portion_name, portion_data in portions.items():
            pm = re.search(r'(\d+)\s*(?:piece|pcs|pc)', portion_name.lower())
            if pm:
                portion_points.append((
                    int(pm.group(1)),
                    portion_data["calories"],
                    portion_data["grams"],
                ))

        if portion_points:
            portion_points.sort(key=lambda x: x[0])

            # Exact match
            for pieces, cal, grams in portion_points:
                if pieces == user_pieces:
                    calories = cal
                    actual_weight = grams
                    break
            else:
                # Interpolate: find the two closest portion points
                if user_pieces <= portion_points[0][0]:
                    # Below smallest — scale down from smallest
                    per_piece_cal = portion_points[0][1] / portion_points[0][0]
                    per_piece_g = portion_points[0][2] / portion_points[0][0]
                    calories = per_piece_cal * user_pieces
                    actual_weight = per_piece_g * user_pieces
                elif user_pieces >= portion_points[-1][0]:
                    # Above largest — scale up from largest
                    per_piece_cal = portion_points[-1][1] / portion_points[-1][0]
                    per_piece_g = portion_points[-1][2] / portion_points[-1][0]
                    calories = per_piece_cal * user_pieces
                    actual_weight = per_piece_g * user_pieces
                else:
                    # Between two points — linear interpolation
                    for i in range(len(portion_points) - 1):
                        low = portion_points[i]
                        high = portion_points[i + 1]
                        if low[0] <= user_pieces <= high[0]:
                            ratio = (user_pieces - low[0]) / (high[0] - low[0])
                            calories = low[1] + ratio * (high[1] - low[1])
                            actual_weight = low[2] + ratio * (high[2] - low[2])
                            break
                    else:
                        calories = (base_cal * llm_weight) / 100
                        actual_weight = llm_weight

            # Apply preparation + build NutritionInfo (calories derive from macros)
            return _build_nutrition_info(
                food_name=food_item.name,
                food_data=food_data,
                weight_grams=actual_weight,
                prep_style=food_item.preparation_style,
                confidence=0.9,
            )

    # ── Strategy 2: Find closest portion by weight ──
    best_match = None
    best_diff = float("inf")
    for portion_name, portion_data in portions.items():
        diff = abs(portion_data["grams"] - llm_weight)
        if diff < best_diff:
            best_diff = diff
            best_match = portion_data

    # Use closest portion if within 80g (generous tolerance)
    if best_match and best_diff < 80:
        calories = best_match["calories"]
        actual_weight = best_match["grams"]
    else:
        # ── Strategy 3: Last resort — use base rate × weight ──
        # But cap weight to the largest known portion to prevent absurd numbers
        max_portion_grams = max((p["grams"] for p in portions.values()), default=llm_weight)
        capped_weight = min(llm_weight, max_portion_grams * 1.2)
        calories = (base_cal * capped_weight) / 100
        actual_weight = capped_weight

    return _build_nutrition_info(
        food_name=food_item.name,
        food_data=food_data,
        weight_grams=actual_weight,
        prep_style=food_item.preparation_style,
        confidence=0.85,
    )


def _calc_base_macros(food_data: dict, weight_grams: float) -> tuple[float, float, float]:
    """
    Calculate base protein, carbs, fat for a given weight from per-100g data.
    This is BEFORE any preparation multiplier — raw base values scaled to portion.
    """
    protein_100 = food_data.get("protein_per_100g")
    carbs_100 = food_data.get("carbs_per_100g")
    fat_100 = food_data.get("fat_per_100g")

    if protein_100 is not None:
        factor = weight_grams / 100
        return (
            round(protein_100 * factor, 1),
            round(carbs_100 * factor, 1),
            round(fat_100 * factor, 1),
        )

    # Fallback: rough estimate (should rarely hit this with 82 foods)
    base_cal = food_data.get("base_calories_per_100g", 150)
    cal = base_cal * weight_grams / 100
    return (
        round(cal * 0.12 / 4, 1),
        round(cal * 0.55 / 4, 1),
        round(cal * 0.33 / 9, 1),
    )


def _apply_preparation(
    protein: float, carbs: float, fat: float,
    prep_style: str, food_data: dict
) -> tuple[float, float, float]:
    """
    Apply preparation-specific multipliers to each macro independently.

    Supports two formats in JSON:
    - New format (per-macro): {"protein_multiplier": 1.1, "carbs_multiplier": 1.1, "fat_multiplier": 8.0}
    - Old format (single): {"multiplier": 1.8} — applied proportionally as fallback

    Frying increases fat dramatically but barely changes protein/carbs.
    Adding cream increases fat. Adding sugar increases carbs. Etc.
    """
    prep_lower = prep_style.lower()

    for prep_name, prep_data in food_data.get("preparations", {}).items():
        if prep_name.lower() in prep_lower or prep_lower in prep_name.lower():

            # New format: per-macro multipliers
            if "protein_multiplier" in prep_data:
                return (
                    round(protein * prep_data.get("protein_multiplier", 1.0), 1),
                    round(carbs * prep_data.get("carbs_multiplier", 1.0), 1),
                    round(fat * prep_data.get("fat_multiplier", 1.0), 1),
                )

            # Old format: single multiplier — infer per-macro behavior
            mult = prep_data.get("multiplier", 1.0)
            if mult == 1.0:
                return (protein, carbs, fat)

            # Smart inference based on common preparation keywords
            prep_combined = (prep_name + " " + prep_data.get("note", "")).lower()

            if any(kw in prep_combined for kw in ["fried", "fry", "deep"]):
                # Frying: fat increases a lot, protein/carbs barely change
                # Reverse-engineer: if total mult is 1.8, and P/C stay ~1.0,
                # the fat must absorb most of the extra calories
                extra_cal = (protein * 4 + carbs * 4 + fat * 9) * (mult - 1.0)
                extra_fat = extra_cal / 9  # Extra calories come from oil (fat)
                return (protein, carbs, round(fat + extra_fat, 1))

            elif any(kw in prep_combined for kw in ["ghee", "butter", "cream", "makhani"]):
                # Fat-based additions
                extra_cal = (protein * 4 + carbs * 4 + fat * 9) * (mult - 1.0)
                extra_fat = extra_cal / 9
                return (protein, carbs, round(fat + extra_fat, 1))

            elif any(kw in prep_combined for kw in ["sugar", "sweet", "syrup"]):
                # Sugar-based additions
                extra_cal = (protein * 4 + carbs * 4 + fat * 9) * (mult - 1.0)
                extra_carbs = extra_cal / 4
                return (protein, round(carbs + extra_carbs, 1), fat)

            else:
                # Generic: distribute proportionally
                return (
                    round(protein * mult, 1),
                    round(carbs * mult, 1),
                    round(fat * mult, 1),
                )

    return (protein, carbs, fat)


def _get_accompaniments(food_data: dict, food_item: FoodItem) -> list[NutritionInfo]:
    """
    Get accompaniment nutrition data that should be ADDED to the base food.

    Checks ALL text fields from the food item (display_name, portion_description,
    preparation_style) for mentions of accompaniments.
    
    Accompaniments are additive — never multiplied.
    """
    results = []
    accompaniments = food_data.get("accompaniments", {})

    if not accompaniments:
        return results

    # Build a single search string from all available text
    search_text = " ".join([
        food_item.display_name or "",
        food_item.portion_description or "",
        food_item.preparation_style or "",
        food_item.name or "",
    ]).lower()

    for acc_name, acc_data in accompaniments.items():
        acc_key = acc_name.lower().replace("_", " ")
        is_default = acc_data.get("default", False)

        # Match if: default, or any word from the accompaniment name appears in any text
        if is_default or any(kw in search_text for kw in acc_key.split()):
            results.append(NutritionInfo(
                food_name=acc_name,
                protein_g=acc_data.get("protein_g", 0),
                carbs_g=acc_data.get("carbs_g", 0),
                fat_g=acc_data.get("fat_g", 0),
                portion_grams=acc_data.get("grams", 0),
                source="knowledge_base",
                confidence=0.85,
                is_accompaniment=True,
            ))

    return results


def _build_nutrition_info(
    food_name: str,
    food_data: dict,
    weight_grams: float,
    prep_style: str,
    confidence: float = 0.85,
) -> NutritionInfo:
    """
    Build a NutritionInfo with the correct pipeline:
    1. Calculate base macros from per-100g data × weight
    2. Apply preparation multipliers (per-macro)
    3. Calories auto-derive from macros (Atwater)
    """
    protein, carbs, fat = _calc_base_macros(food_data, weight_grams)
    protein, carbs, fat = _apply_preparation(protein, carbs, fat, prep_style, food_data)

    return NutritionInfo(
        food_name=food_name,
        protein_g=protein,
        carbs_g=carbs,
        fat_g=fat,
        fiber_g=round(weight_grams * 0.02, 1),
        portion_grams=round(weight_grams, 1),
        source="knowledge_base",
        confidence=confidence,
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
        # Last resort: rough estimate from weight (~150 cal/100g average Indian food)
        weight = food_item.estimated_weight_grams
        est_cal = weight * 1.5  # ~150 cal/100g
        return NutritionInfo(
            food_name=food_item.name,
            protein_g=round(est_cal * 0.12 / 4, 1),
            carbs_g=round(est_cal * 0.55 / 4, 1),
            fat_g=round(est_cal * 0.33 / 9, 1),
            portion_grams=weight,
            source="fallback_estimate",
            confidence=0.3,
        )


async def analyse_nutrition(food_items: list[FoodItem]) -> list[NutritionInfo]:
    """
    Get nutrition data for a list of identified food items.

    Pipeline:
    1. Direct knowledge base lookup (fastest, most accurate)
       - Base macros × weight → per-macro preparation multipliers → Atwater calories
       - Accompaniments added as separate items (additive, not multiplied)
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

                # Add accompaniments (sambar, chutney, etc.) as separate entries
                accompaniments = _get_accompaniments(food_data, item)
                results.extend(accompaniments)
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
