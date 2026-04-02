"""
Seed the Indian Food Knowledge Base into ChromaDB.

Run with: python -m app.data.seed_knowledge_base
"""
from app.core.rag import seed_knowledge_base

if __name__ == "__main__":
    count = seed_knowledge_base()
    print(f"\nKnowledge base ready with {count} documents.")
    print("You can now start the server with: uvicorn app.main:app --reload")
