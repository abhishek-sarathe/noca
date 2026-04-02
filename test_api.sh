#!/bin/bash
# Quick test script for Not a Calorie Tracker API
# Run: bash test_api.sh

BASE_URL="http://localhost:8000"

echo "=========================="
echo "NOCA — API Tests"
echo "=========================="
echo ""

# 1. Health check
echo "1. Health Check"
echo "───────────────"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""
echo ""

# 2. Guess: Simple single food
echo "2. Guess: '2 idli with sambar' (guessing 200 cal)"
echo "──────────────────────────────────────────────────"
curl -s -X POST "$BASE_URL/api/guess" \
  -H "Content-Type: application/json" \
  -d '{
    "food_text": "2 idli with sambar",
    "user_guess_calories": 200
  }' | python3 -m json.tool
echo ""
echo ""

# 3. Guess: Complex meal
echo "3. Guess: 'butter chicken with 2 naan and a lassi' (guessing 600 cal)"
echo "─────────────────────────────────────────────────────────────────────"
curl -s -X POST "$BASE_URL/api/guess" \
  -H "Content-Type: application/json" \
  -d '{
    "food_text": "butter chicken with 2 naan and a sweet lassi",
    "user_guess_calories": 600
  }' | python3 -m json.tool
echo ""
echo ""

# 4. Guess: Street food
echo "4. Guess: '1 plate pani puri (8 pieces)' (guessing 150 cal)"
echo "────────────────────────────────────────────────────────────"
curl -s -X POST "$BASE_URL/api/guess" \
  -H "Content-Type: application/json" \
  -d '{
    "food_text": "1 plate pani puri, about 8 pieces",
    "user_guess_calories": 150
  }' | python3 -m json.tool
echo ""
echo ""

# 5. Challenge: Get quiz questions
echo "5. Challenge: Get daily quiz"
echo "────────────────────────────"
curl -s -X POST "$BASE_URL/api/challenge" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}' | python3 -m json.tool
echo ""
echo ""

echo "================================="
echo "Tests complete!"
echo "================================="
