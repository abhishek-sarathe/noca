"""
RAG pipeline for Indian Food Knowledge Base.

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Each food item is chunked into searchable documents with rich metadata.
"""
import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


# ── Singleton ChromaDB client ──
_chroma_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the Indian foods collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="indian_foods",
            metadata={"description": "Indian food nutrition knowledge base"},
        )
    return _collection


def build_documents_from_food(food: dict) -> list[dict]:
    """
    Convert a single food entry into multiple searchable documents.

    Strategy: Create overlapping documents that capture different
    aspects of the food — name/aliases, nutrition, preparations,
    fun facts — so the RAG can find relevant info from different
    query angles.
    """
    name = food["canonical_name"]
    display = food["display_name"]
    aliases = ", ".join(food.get("aliases", []))
    category = food.get("category", "")
    region = ", ".join(food.get("region", []))

    documents = []

    # Doc 1: Overview + aliases (for food identification queries)
    overview = (
        f"{display} (also known as: {aliases}). "
        f"Category: {category}. Popular in: {region}. "
        f"Base calories: {food['base_calories_per_100g']} cal per 100g."
    )
    documents.append({
        "id": f"{name}_overview",
        "text": overview,
        "metadata": {
            "food_name": name,
            "display_name": display,
            "doc_type": "overview",
            "category": category,
            "base_cal_per_100g": food["base_calories_per_100g"],
        },
    })

    # Doc 2: Portion information (for calorie estimation queries)
    portions_text = f"Portion sizes for {display}: "
    for portion_name, portion_data in food.get("portions", {}).items():
        portions_text += f"{portion_name} = {portion_data['grams']}g = {portion_data['calories']} calories. "
    documents.append({
        "id": f"{name}_portions",
        "text": portions_text,
        "metadata": {
            "food_name": name,
            "display_name": display,
            "doc_type": "portions",
            "category": category,
            "portions_json": json.dumps(food.get("portions", {})),
        },
    })

    # Doc 3: Preparation variations
    preps_text = f"Preparation styles for {display}: "
    for prep_name, prep_data in food.get("preparations", {}).items():
        note = prep_data.get("note", "")
        if "protein_multiplier" in prep_data:
            p_m = prep_data["protein_multiplier"]
            c_m = prep_data["carbs_multiplier"]
            f_m = prep_data["fat_multiplier"]
            preps_text += (
                f"{prep_name} (protein x{p_m}, carbs x{c_m}, fat x{f_m} — {note}). "
            )
        else:
            multiplier = prep_data.get("multiplier", 1.0)
            preps_text += f"{prep_name} (calorie multiplier: {multiplier}x — {note}). "
    documents.append({
        "id": f"{name}_preparations",
        "text": preps_text,
        "metadata": {
            "food_name": name,
            "display_name": display,
            "doc_type": "preparations",
            "category": category,
            "preparations_json": json.dumps(food.get("preparations", {})),
        },
    })

    # Doc 4: Fun facts + common mistakes (for learning/quiz content)
    facts = food.get("fun_facts", [])
    mistakes = food.get("common_mistakes", [])
    visual_refs = food.get("visual_references", [])
    learning_text = f"Learning content for {display}: "
    for fact in facts:
        learning_text += f"Fun fact: {fact} "
    for mistake in mistakes:
        learning_text += f"Common mistake: {mistake} "
    for ref in visual_refs:
        learning_text += f"Visual reference: {ref} "
    documents.append({
        "id": f"{name}_learning",
        "text": learning_text,
        "metadata": {
            "food_name": name,
            "display_name": display,
            "doc_type": "learning",
            "category": category,
            "fun_facts_json": json.dumps(facts),
            "common_mistakes_json": json.dumps(mistakes),
        },
    })

    return documents


def seed_knowledge_base() -> int:
    """
    Load indian_foods.json and seed into ChromaDB.
    Returns the number of documents added.
    """
    data_path = Path(__file__).parent.parent / "data" / "indian_foods.json"
    with open(data_path, "r", encoding="utf-8") as f:
        foods = json.load(f)

    collection = get_collection()

    all_ids = []
    all_texts = []
    all_metadatas = []

    for food in foods:
        docs = build_documents_from_food(food)
        for doc in docs:
            all_ids.append(doc["id"])
            all_texts.append(doc["text"])
            all_metadatas.append(doc["metadata"])

    # Upsert (idempotent — safe to run multiple times)
    collection.upsert(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metadatas,
    )

    print(f"Seeded {len(all_ids)} documents from {len(foods)} food items.")
    return len(all_ids)


def query_knowledge_base(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    category: Optional[str] = None,
) -> list[dict]:
    """
    Query the knowledge base with optional filters.

    Args:
        query: Natural language query (e.g. "how many calories in butter chicken")
        n_results: Number of results to return
        doc_type: Filter by doc type: "overview", "portions", "preparations", "learning"
        category: Filter by food category: "breakfast", "curry", "snack", etc.

    Returns:
        List of dicts with 'text', 'metadata', and 'distance' keys.
    """
    collection = get_collection()

    where_filter = None
    if doc_type and category:
        where_filter = {"$and": [{"doc_type": doc_type}, {"category": category}]}
    elif doc_type:
        where_filter = {"doc_type": doc_type}
    elif category:
        where_filter = {"category": category}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["documents"]:
        for i, doc_text in enumerate(results["documents"][0]):
            output.append({
                "text": doc_text,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })

    return output


def get_food_by_name(food_name: str) -> Optional[dict]:
    """
    Direct lookup of a food item by canonical name.
    Returns the full food entry from JSON.
    """
    data_path = Path(__file__).parent.parent / "data" / "indian_foods.json"
    with open(data_path, "r", encoding="utf-8") as f:
        foods = json.load(f)

    food_name_lower = food_name.lower().strip()
    for food in foods:
        if food["canonical_name"] == food_name_lower:
            return food
        if food_name_lower in [a.lower() for a in food.get("aliases", [])]:
            return food

    return None
