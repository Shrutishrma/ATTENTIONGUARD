"""
One-time seeder. Run this after setting up .env with a valid MONGO_URI.
Safe to re-run — skips records that already exist.

Usage:
    python website/populate_db.py
"""

import sys
import os

# Resolve the project root so .env is found regardless of where the script is run from
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, '.env'))

from website import create_app
from website.models import add_user, add_teacher
from flask import current_app

app = create_app()

# ------------------------------------------------------------------
# Demo teacher accounts
# ------------------------------------------------------------------
demo_teachers = [
    {"register_number": "DEMOTEACHER", "password": "demo123"},
]

# ------------------------------------------------------------------
# Demo student accounts
# ------------------------------------------------------------------
demo_students = [
    {"register_number": "DEMO001", "password": "demo123"},
    {"register_number": "DEMO002", "password": "demo123"},
    {"register_number": "DEMO003", "password": "demo123"},
]

# Original seeded students (kept for backwards compatibility)
original_students = [
    {"register_number": "222BCAA49", "password": "password1"},
    {"register_number": "222BCAA50", "password": "password1"},
]

# ------------------------------------------------------------------
# Demo question set — 10 general knowledge questions, code: DEMO
# ------------------------------------------------------------------
demo_question_set = {
    "code": "DEMO",
    "title": "General Knowledge Demo Quiz",
    "created_by": "DEMOTEACHER",
    "questions": [
        {
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "answer": "Paris"
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Saturn"],
            "answer": "Mars"
        },
        {
            "question": "What is the chemical symbol for water?",
            "options": ["H2O", "CO2", "NaCl", "O2"],
            "answer": "H2O"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "options": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Leo Tolstoy"],
            "answer": "William Shakespeare"
        },
        {
            "question": "How many continents are there on Earth?",
            "options": ["5", "6", "7", "8"],
            "answer": "7"
        },
        {
            "question": "What is the largest ocean on Earth?",
            "options": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"],
            "answer": "Pacific Ocean"
        },
        {
            "question": "Which country invented the game of chess?",
            "options": ["China", "Persia", "India", "Greece"],
            "answer": "India"
        },
        {
            "question": "Approximately how fast does light travel?",
            "options": ["300 km/s", "3,000 km/s", "300,000 km/s", "3,000,000 km/s"],
            "answer": "300,000 km/s"
        },
        {
            "question": "Who painted the Mona Lisa?",
            "options": ["Vincent van Gogh", "Leonardo da Vinci", "Pablo Picasso", "Rembrandt"],
            "answer": "Leonardo da Vinci"
        },
        {
            "question": "What is the smallest country in the world?",
            "options": ["Monaco", "San Marino", "Vatican City", "Liechtenstein"],
            "answer": "Vatican City"
        },
    ]
}

with app.app_context():
    print("\n--- Seeding teachers ---")
    for t in demo_teachers:
        if not current_app.mongo.db.user.find_one({"register_number": t["register_number"]}):
            add_teacher(t["register_number"], t["password"])
            print(f"  Created teacher: {t['register_number']}")
        else:
            print(f"  Already exists:  {t['register_number']}")

    print("\n--- Seeding students ---")
    for s in demo_students + original_students:
        if not current_app.mongo.db.user.find_one({"register_number": s["register_number"]}):
            add_user(s["register_number"], s["password"])
            print(f"  Created student: {s['register_number']}")
        else:
            print(f"  Already exists:  {s['register_number']}")

    print("\n--- Seeding question sets ---")
    if not current_app.mongo.db.question_sets.find_one({"code": demo_question_set["code"]}):
        current_app.mongo.db.question_sets.insert_one(demo_question_set)
        print(f"  Created question set: {demo_question_set['code']} — {demo_question_set['title']}")
    else:
        print(f"  Already exists: {demo_question_set['code']}")

    print("\nDone.")
    print("\nDemo credentials:")
    print("  Student  -> Register: DEMO001 / DEMO002 / DEMO003  |  Password: demo123  |  Test Code: DEMO")
    print("  Teacher  -> Register: DEMOTEACHER                  |  Password: demo123")
