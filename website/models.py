from datetime import datetime, timedelta
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash


def add_user(register_number, password, name=""):
    current_app.mongo.db.user.insert_one({
        "register_number": register_number,
        "name": name.strip(),
        "password": generate_password_hash(password),
        "role": "student",
        "banned": False,
        "ban_timer": None,
        "completed_tests": {},
        "violations": {}
    })


def add_teacher(register_number, password):
    current_app.mongo.db.user.insert_one({
        "register_number": register_number,
        "password": generate_password_hash(password),
        "role": "teacher",
        "banned": False,
        "ban_timer": None,
        "completed_tests": {},
        "violations": {}
    })


import re

def find_user_by_reg(register_number):
    if not register_number:
        return None
    reg = str(register_number).strip()
    escaped = re.escape(reg)
    return current_app.mongo.db.user.find_one({"register_number": {"$regex": f"^{escaped}$", "$options": "i"}})


def ban_user(register_number, duration_minutes=5):
    ban_until = (datetime.now() + timedelta(minutes=duration_minutes)).replace(second=0, microsecond=0)
    current_app.mongo.db.user.update_one(
        {"register_number": register_number},
        {"$set": {"banned": True, "ban_timer": ban_until}}
    )


def is_user_banned(register_number):
    user = find_user_by_reg(register_number)
    if user and user.get("banned"):
        if user.get("ban_timer") and datetime.now() > user["ban_timer"]:
            current_app.mongo.db.user.update_one(
                {"register_number": register_number},
                {"$set": {"banned": False, "ban_timer": None}}
            )
            return [False, None]
        return [True, user["ban_timer"]]
    return [False, None]


def record_test_completion(register_number, question_set_code, score, correct_answers, time_taken=0, questions_detail=None):
    """Record test completion with score, time, details, and violation summary."""
    # Get current violation data for this test
    user = find_user_by_reg(register_number)
    test_violations = user.get("violations", {}).get(question_set_code, []) if user else []
    violation_count = len(test_violations)
    violation_types = list(set(v.get("type", "unknown") for v in test_violations))

    current_app.mongo.db.user.update_one(
        {"register_number": register_number},
        {"$set": {f"completed_tests.{question_set_code}": {
            "score": score,
            "correct_answers": correct_answers,
            "time_taken": time_taken,
            "questions_detail": questions_detail if questions_detail else [],
            "violation_count": violation_count,
            "violation_types": violation_types,
            "completed_at": datetime.utcnow()
        }}}
    )


def record_violation(register_number, question_set_code, violation_type, message):
    """Record an individual proctoring violation to the database."""
    violation_entry = {
        "type": violation_type,
        "message": message,
        "timestamp": datetime.utcnow()
    }
    current_app.mongo.db.user.update_one(
        {"register_number": register_number},
        {"$push": {f"violations.{question_set_code}": violation_entry}}
    )


def get_violation_count(register_number, question_set_code):
    """Get the number of violations for a user on a specific test."""
    user = find_user_by_reg(register_number)
    if user:
        return len(user.get("violations", {}).get(question_set_code, []))
    return 0


def has_taken_test(register_number, question_set_code):
    user = find_user_by_reg(register_number)
    if user:
        return user.get("completed_tests", {}).get(question_set_code, False)
    return False


def can_attempt_test(register_number, question_set_code, max_attempts=1):
    if max_attempts == 0:
        return True
    user = find_user_by_reg(register_number)
    if user:
        if question_set_code in user.get("completed_tests", {}):
            return False
    return True


def clear_violations(register_number, question_set_code):
    """Clear violations for a user on a specific test (e.g. before starting a retake)."""
    current_app.mongo.db.user.update_one(
        {"register_number": register_number},
        {"$set": {f"violations.{question_set_code}": []}}
    )


def check_password(register_number, password):
    user = find_user_by_reg(register_number)
    if user:
        return check_password_hash(user["password"], password)
    return False
