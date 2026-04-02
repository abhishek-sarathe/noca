"""
Quick test: Run the guess pipeline directly without the server.

Usage: python -m tests.test_pipeline
"""
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.core.rag import seed_knowledge_base, query_knowledge_base
from app.agents.orchestrator import run_guess_pipeline, run_challenge_pipeline


async def test_rag():
    """Test the knowledge base."""
    print("\n" + "=" * 50)
    print("TEST: Knowledge Base RAG")
    print("=" * 50)

    # Seed if needed
    seed_knowledge_base()

    # Test queries
    queries = [
        "how many calories in butter chicken",
        "poha breakfast calories",
        "samosa deep fried snack",
        "biryani Swiggy delivery",
    ]

    for q in queries:
        results = query_knowledge_base(q, n_results=2)
        print(f"\nQuery: '{q}'")
        for r in results:
            food = r["metadata"].get("display_name", "?")
            doc_type = r["metadata"].get("doc_type", "?")
            dist = r["distance"]
            print(f"  → {food} ({doc_type}) [distance: {dist:.3f}]")


async def test_guess_pipeline():
    """Test the full guess pipeline."""
    print("\n" + "=" * 50)
    print("TEST: Guess Pipeline")
    print("=" * 50)

    test_cases = [
        {
            "description": "Simple: 2 idli with sambar",
            "food_text": "2 idli with sambar",
            "guess": 200,
        },
        {
            "description": "Complex: butter chicken + naan + lassi",
            "food_text": "butter chicken with 2 butter naan and a sweet lassi",
            "guess": 600,
        },
        {
            "description": "Street food: pani puri",
            "food_text": "1 plate pani puri, 8 pieces",
            "guess": 150,
        },
    ]

    for tc in test_cases:
        print(f"\n--- {tc['description']} ---")
        print(f"Your guess: {tc['guess']} cal")

        result = await run_guess_pipeline(
            food_text=tc["food_text"],
            user_guess_calories=tc["guess"],
        )

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            continue

        print(f"Identified: {[f.display_name for f in result.get('identified_foods', [])]}")
        print(f"Actual calories: {result.get('total_actual_calories', 0):.0f}")
        print(f"Accuracy: {result.get('accuracy_percentage', 0):.1f}%")
        print(f"Feedback: {result.get('feedback_message', 'N/A')}")
        print(f"Tips: {result.get('tips', [])}")
        print(f"Fun facts: {result.get('fun_facts', [])[:2]}")


async def test_challenge_pipeline():
    """Test the challenge/quiz pipeline."""
    print("\n" + "=" * 50)
    print("TEST: Challenge Pipeline")
    print("=" * 50)

    result = await run_challenge_pipeline(num_questions=3)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    questions = result.get("quiz_questions", [])
    print(f"\nGenerated {len(questions)} questions:\n")

    for i, q in enumerate(questions, 1):
        print(f"Q{i} [{q.question_type}] (difficulty {q.difficulty}/5)")
        print(f"   {q.question_text}")
        if q.options:
            print(f"   Options: {q.options}")
        print(f"   Answer: {q.correct_answer}")
        print(f"   Explanation: {q.explanation[:100]}...")
        print()


async def main():
    await test_rag()
    await test_guess_pipeline()
    await test_challenge_pipeline()
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
