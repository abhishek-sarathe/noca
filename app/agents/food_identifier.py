"""
Food Identifier Agent

Takes a food description (text or image) and identifies:
- What foods are present
- Estimated portion sizes
- Preparation style

Uses Groq's Llama model for structured extraction.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.models import FoodItem


SYSTEM_PROMPT = """You are an expert Indian food identifier. Given a description of a meal, identify each DISTINCT food item present.

For EACH food item, provide:
1. name: The canonical Indian name (lowercase, e.g., "poha", "butter_chicken", "roti")
2. display_name: User-friendly name (e.g., "Poha (Flattened Rice)")
3. portion_description: How much (e.g., "1 medium plate", "2 pieces")
4. estimated_weight_grams: Your best estimate in grams
5. preparation_style: How it's prepared (e.g., "with peanuts and curry leaves")
6. confidence: 0.0 to 1.0

CRITICAL RULES — READ CAREFULLY:

1. DO NOT split a dish into its sub-components.
   - "pani puri" = 1 item (includes the puri shells, water, fillings, chutneys)
   - "chole bhature" = 1 item (includes chole + bhatura together)
   - "idli sambar" = 1 item UNLESS user explicitly says "idli" AND "sambar" separately
   - "dal chawal" or "rajma chawal" = 1 combined item
   - "pav bhaji" = 1 item (includes pav + bhaji)

2. Only list SEPARATE dishes the user explicitly mentions.
   - "butter chicken with naan and lassi" = 3 items (butter chicken, naan, lassi)
   - "2 idli with sambar" = 1 item (idli with sambar — sambar is the standard accompaniment)
   - "dosa" = 1 item (assume standard accompaniments are included in the calorie count)

3. Standard accompaniments are INCLUDED in the main dish, NOT separate items:
   - Chutneys served with dosa/idli/vada → included
   - Raita served with biryani → included
   - Pani/water in pani puri → included
   - Pickle on the side → ignore (negligible calories)

4. Only create separate items when the user names them as distinct dishes:
   - "I had dosa and then some curd rice" = 2 items
   - "butter chicken, 2 naan, and dal" = 3 items

5. Use realistic Indian portion sizes. A "plate" of rice = 200-300g. A "bowl" of curry = 200-250g.

6. Keep the total number of items MINIMAL. Aim for 1-3 items for a typical meal. More than 4 items should be rare.

Respond with ONLY a JSON array. No explanations, no markdown.

Example: "2 idli with sambar"
[
  {
    "name": "idli",
    "display_name": "Idli with Sambar",
    "portion_description": "2 pieces with sambar",
    "estimated_weight_grams": 130,
    "preparation_style": "steamed idli with sambar",
    "confidence": 0.9
  }
]

Example: "butter chicken with 2 naan and a lassi"
[
  {
    "name": "butter_chicken",
    "display_name": "Butter Chicken",
    "portion_description": "1 bowl",
    "estimated_weight_grams": 250,
    "preparation_style": "restaurant style butter chicken",
    "confidence": 0.85
  },
  {
    "name": "naan",
    "display_name": "Butter Naan",
    "portion_description": "2 pieces",
    "estimated_weight_grams": 200,
    "preparation_style": "butter naan",
    "confidence": 0.85
  },
  {
    "name": "lassi",
    "display_name": "Sweet Lassi",
    "portion_description": "1 glass",
    "estimated_weight_grams": 250,
    "preparation_style": "sweet lassi",
    "confidence": 0.8
  }
]"""


def get_food_identifier_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.1,  # Low temp for structured output
        max_tokens=1024,
    )


async def identify_food(
    food_text: str | None = None,
    food_image_base64: str | None = None,
) -> list[FoodItem]:
    """
    Identify food items from text description or image.

    Args:
        food_text: Text description like "I had 2 rotis with dal and rice"
        food_image_base64: Base64 encoded image of food (future: use vision model)

    Returns:
        List of identified FoodItem objects
    """
    llm = get_food_identifier_llm()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if food_text:
        user_content = f"Identify the food items in this meal: {food_text}"
    elif food_image_base64:
        # For now, we'll describe that we'd use vision here
        # Groq's vision model can be integrated when needed
        user_content = (
            "A food image was provided. For now, please identify based on "
            "the most common Indian meal composition. "
            "Note: Vision integration coming in next iteration."
        )
    else:
        return []

    messages.append(HumanMessage(content=user_content))

    response = await llm.ainvoke(messages)
    raw_text = response.content.strip()

    # Parse JSON response
    try:
        # Clean up potential markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        items_data = json.loads(raw_text.strip())
        food_items = [FoodItem(**item) for item in items_data]
        return food_items

    except (json.JSONDecodeError, Exception) as e:
        print(f"[Food Identifier] Failed to parse LLM response: {e}")
        print(f"[Food Identifier] Raw response: {raw_text}")
        # Fallback: return a single generic item
        return [
            FoodItem(
                name="unknown_food",
                display_name="Unidentified Food",
                portion_description="1 serving",
                estimated_weight_grams=200,
                preparation_style="standard",
                confidence=0.3,
            )
        ]
