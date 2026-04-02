"""
Database layer — SQLite for dev, designed to swap to PostgreSQL later.

Tables:
  users         — skill profiles, streaks, levels
  guess_history — every guess attempt with accuracy data
"""
import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path("noca.db")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db_cursor():
    conn = get_db()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                total_guesses INTEGER DEFAULT 0,
                average_accuracy REAL DEFAULT 0.0,
                level INTEGER DEFAULT 1,
                streak INTEGER DEFAULT 0,
                last_guess_date TEXT DEFAULT '',
                weak_categories TEXT DEFAULT '[]',
                strong_categories TEXT DEFAULT '[]',
                graduation_ready INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS guess_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                food_description TEXT NOT NULL,
                identified_foods TEXT DEFAULT '[]',
                user_guess_calories REAL NOT NULL,
                user_guess_protein REAL DEFAULT 0,
                actual_calories REAL NOT NULL,
                actual_protein REAL DEFAULT 0,
                accuracy_pct REAL NOT NULL,
                category TEXT DEFAULT '',
                feedback TEXT DEFAULT '',
                fun_facts TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

    print("Database initialized.")


# ═══════════════════════════════════════════
# User operations
# ═══════════════════════════════════════════

def get_or_create_user(user_id: str) -> dict:
    """Get user profile, creating if needed."""
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()

        if row:
            return dict(row)

        cur.execute(
            "INSERT INTO users (user_id) VALUES (?)",
            (user_id,)
        )
        return {
            "user_id": user_id,
            "total_guesses": 0,
            "average_accuracy": 0.0,
            "level": 1,
            "streak": 0,
            "last_guess_date": "",
            "weak_categories": "[]",
            "strong_categories": "[]",
            "graduation_ready": 0,
        }


def update_user_after_guess(user_id: str):
    """
    Recalculate user stats from guess_history.
    Called after every new guess is saved.
    """
    with get_db_cursor() as cur:
        # Get last 20 guesses for rolling average
        cur.execute("""
            SELECT accuracy_pct, category, created_at
            FROM guess_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 20
        """, (user_id,))
        recent = cur.fetchall()

        if not recent:
            return

        # Rolling accuracy
        avg_accuracy = sum(r["accuracy_pct"] for r in recent) / len(recent)

        # Total guesses
        cur.execute(
            "SELECT COUNT(*) as cnt FROM guess_history WHERE user_id = ?",
            (user_id,)
        )
        total = cur.fetchone()["cnt"]

        # Level
        if avg_accuracy >= 90:
            level = 5
            graduation_ready = 1
        elif avg_accuracy >= 75:
            level = 4
            graduation_ready = 0
        elif avg_accuracy >= 60:
            level = 3
            graduation_ready = 0
        elif avg_accuracy >= 40:
            level = 2
            graduation_ready = 0
        else:
            level = 1
            graduation_ready = 0

        # Streak calculation
        cur.execute(
            "SELECT last_guess_date, streak FROM users WHERE user_id = ?",
            (user_id,)
        )
        user_row = cur.fetchone()
        today = date.today().isoformat()
        last_date = user_row["last_guess_date"] if user_row else ""
        old_streak = user_row["streak"] if user_row else 0

        if last_date == today:
            streak = old_streak  # Already guessed today
        elif last_date == str(date.today().replace(day=date.today().day - 1)) if date.today().day > 1 else "":
            streak = old_streak + 1  # Consecutive day
        else:
            streak = 1  # Streak broken or first guess

        # Category analysis (last 50 guesses)
        cur.execute("""
            SELECT category, accuracy_pct
            FROM guess_history
            WHERE user_id = ? AND category != ''
            ORDER BY created_at DESC
            LIMIT 50
        """, (user_id,))
        cat_rows = cur.fetchall()

        cat_scores = {}
        for r in cat_rows:
            cat = r["category"]
            cat_scores.setdefault(cat, []).append(r["accuracy_pct"])

        weak = [c for c, scores in cat_scores.items() if sum(scores) / len(scores) < 50]
        strong = [c for c, scores in cat_scores.items() if sum(scores) / len(scores) > 75]

        # Update user
        cur.execute("""
            UPDATE users SET
                total_guesses = ?,
                average_accuracy = ?,
                level = ?,
                streak = ?,
                last_guess_date = ?,
                weak_categories = ?,
                strong_categories = ?,
                graduation_ready = ?,
                updated_at = datetime('now')
            WHERE user_id = ?
        """, (
            total,
            round(avg_accuracy, 1),
            level,
            streak,
            today,
            json.dumps(weak),
            json.dumps(strong),
            graduation_ready,
            user_id,
        ))


# ═══════════════════════════════════════════
# Guess history operations
# ═══════════════════════════════════════════

def save_guess(
    user_id: str,
    food_description: str,
    identified_foods: list,
    user_guess_calories: float,
    user_guess_protein: float,
    actual_calories: float,
    actual_protein: float,
    accuracy_pct: float,
    category: str = "",
    feedback: str = "",
    fun_facts: list = None,
):
    """Save a guess attempt and update user stats."""
    # Ensure user exists
    get_or_create_user(user_id)

    foods_json = json.dumps([
        {"name": f.name if hasattr(f, "name") else f.get("name", ""),
         "display_name": f.display_name if hasattr(f, "display_name") else f.get("display_name", "")}
        for f in identified_foods
    ]) if identified_foods else "[]"

    with get_db_cursor() as cur:
        cur.execute("""
            INSERT INTO guess_history
                (user_id, food_description, identified_foods,
                 user_guess_calories, user_guess_protein,
                 actual_calories, actual_protein,
                 accuracy_pct, category, feedback, fun_facts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            food_description,
            foods_json,
            user_guess_calories,
            user_guess_protein,
            actual_calories,
            actual_protein,
            round(accuracy_pct, 1),
            category,
            feedback,
            json.dumps(fun_facts or []),
        ))

    # Recalculate user stats
    update_user_after_guess(user_id)


def get_guess_history(user_id: str, limit: int = 20) -> list[dict]:
    """Get recent guess history for a user."""
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT * FROM guess_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cur.fetchall()]


def get_user_progress(user_id: str) -> dict:
    """
    Get full progress data for the frontend Progress page.
    """
    user = get_or_create_user(user_id)
    history = get_guess_history(user_id, limit=20)

    # Category accuracy breakdown
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT category, AVG(accuracy_pct) as avg_acc, COUNT(*) as cnt
            FROM guess_history
            WHERE user_id = ? AND category != ''
            GROUP BY category
            ORDER BY cnt DESC
            LIMIT 8
        """, (user_id,))
        categories = {row["category"]: round(row["avg_acc"], 1) for row in cur.fetchall()}

    # Best guess ever
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT food_description, accuracy_pct
            FROM guess_history
            WHERE user_id = ?
            ORDER BY accuracy_pct DESC
            LIMIT 1
        """, (user_id,))
        best = cur.fetchone()

    return {
        "user_id": user["user_id"],
        "total_guesses": user["total_guesses"],
        "average_accuracy": user["average_accuracy"],
        "level": user["level"],
        "streak": user["streak"],
        "graduation_ready": bool(user["graduation_ready"]),
        "weak_categories": json.loads(user["weak_categories"]),
        "strong_categories": json.loads(user["strong_categories"]),
        "best_guess": {
            "food": best["food_description"],
            "accuracy": best["accuracy_pct"],
        } if best else None,
        "categories": categories,
        "history": [
            {
                "food": h["food_description"],
                "guessed": h["user_guess_calories"],
                "actual": h["actual_calories"],
                "accuracy": h["accuracy_pct"],
                "protein_guessed": h["user_guess_protein"],
                "protein_actual": h["actual_protein"],
            }
            for h in history
        ],
    }
