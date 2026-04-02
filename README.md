# NOCA — Not Only Calorie App

**Learn to estimate calories. Stop logging. Live free.**

A multi-agent AI system that teaches users to accurately estimate food calories, with deep Indian food expertise.

## Tech Stack
- **Backend**: FastAPI + Python 3.11 (⚠️ NOT 3.12+ — LangChain/LangGraph have compatibility issues)
- **Agents**: LangChain + LangGraph
- **LLM**: Groq (Llama 3.3 70B)
- **RAG**: ChromaDB + sentence-transformers
- **Database**: SQLite (dev) → PostgreSQL (prod)

## Setup

```bash
# 1. Make sure you're on Python 3.11
python3.11 --version  # should show 3.11.x

# 2. Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Initialize the knowledge base
python -m app.data.seed_knowledge_base

# 6. Run the server
uvicorn app.main:app --reload --port 8000
```

### Installing Python 3.11 (if needed)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# macOS (Homebrew)
brew install python@3.11
```

## Project Structure
```
noca/
├── app/
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Settings & env vars
│   ├── agents/
│   │   ├── orchestrator.py  # LangGraph StateGraph
│   │   ├── food_identifier.py
│   │   ├── nutrition_analyst.py
│   │   ├── quiz_generator.py
│   │   └── learning_coach.py
│   ├── core/
│   │   ├── state.py         # Shared state schema
│   │   ├── models.py        # Pydantic models
│   │   └── rag.py           # RAG pipeline
│   ├── api/
│   │   ├── routes_guess.py
│   │   ├── routes_challenge.py
│   │   └── routes_profile.py
│   └── data/
│       ├── indian_foods.json
│       └── seed_knowledge_base.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```
