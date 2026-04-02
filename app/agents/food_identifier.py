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


SYSTEM_PROMPT = """You are an expert Indian food identifier. Given a description of a meal (text or image context), you must identify each food item present.

For EACH food item, provide:
1. name: The canonical Indian name (lowercase, e.g., "poha", "butter_chicken", "roti")
2. display_name: User-friendly name (e.g., "Poha (Flattened Rice)")
3. portion_description: How much appears to be there (e.g., "1 medium plate", "2 pieces")
4. estimated_weight_grams: Your best estimate of the weight in grams
5. preparation_style: How it appears to be prepared (e.g., "with peanuts and curry leaves", "butter naan")
6. confidence: How confident you are (0.0 to 1.0)

IMPORTANT RULES:
- Always think in terms of Indian food first. "Rice" = likely "steamed rice" (chawal), not fried rice.
- Account for typical accompaniments. If someone says "idli", they likely have sambar and chutney too.
- Be specific about preparation. "Dosa" could be plain, masala, ghee roast — ask or infer from context.
- Estimate realistic Indian portion sizes. A "plate" of rice in India is typically 200-300g cooked.

Respond with ONLY a JSON array of food items. No explanations, no markdown, no extra text.

Example response:
[
  {
    "name": "masala_dosa",
    "display_name": "Masala Dosa",
    "portion_description": "1 regular masala dosa",
    "estimated_weight_grams": 150,
    "preparation_style": "masala dosa with potato filling",
    "confidence": 0.85
  },
  {
    "name": "coconut_chutney",
    "display_name": "Coconut Chutney",
    "portion_description": "1 small bowl",
    "estimated_weight_grams": 40,
    "preparation_style": "standard coconut chutney",
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
