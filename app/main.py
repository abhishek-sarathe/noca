"""
NOCA — Not Only Calorie App — FastAPI Application

Learn to estimate calories. Stop logging. Live free.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.rag import seed_knowledge_base, get_collection
from app.core.database import init_db
from app.api.routes_guess import router as guess_router
from app.api.routes_challenge import router as challenge_router
from app.api.routes_progress import router as progress_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: seed knowledge base + init database."""
    init_db()
    collection = get_collection()
    if collection.count() == 0:
        print("Knowledge base empty — seeding...")
        seed_knowledge_base()
    else:
        print(f"Knowledge base loaded: {collection.count()} documents.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="NOCA — Not Only Calorie App",
    description="An AI-powered app that teaches you to estimate calories — so you stop needing it.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(guess_router)
app.include_router(challenge_router)
app.include_router(progress_router)


@app.get("/")
async def root():
    return {
        "app": "NOCA — Not Only Calorie App",
        "tagline": "Learn to estimate. Stop logging. Live free.",
        "version": "0.1.0",
        "endpoints": {
            "POST /api/guess": "Submit a food + calorie guess → get full analysis",
            "POST /api/challenge": "Get personalised quiz questions",
            "GET /api/docs": "Interactive API docs (Swagger)",
        },
    }


@app.get("/health")
async def health():
    """Health check — also verifies knowledge base is loaded."""
    collection = get_collection()
    return {
        "status": "healthy",
        "knowledge_base_docs": collection.count(),
    }
