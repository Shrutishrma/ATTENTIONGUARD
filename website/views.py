import datetime
import random
from flask import Blueprint, render_template, request, session, redirect, url_for, flash, current_app, jsonify
from .models import record_test_completion, record_violation
from .utils import teacher_required

views = Blueprint('views', __name__)


@views.route('/test/<code>', methods=['GET', 'POST'])
def test(code):
    if 'regno' not in session or 'test_code' not in session:
        flash('User not logged in.', 'error')
        return redirect(url_for('auth.login'))

    if session.get('test_in_progress') is not True:
        flash('You cannot retake the test.', 'warning')
        return redirect(url_for('auth.login'))

    regno = session.get('regno')
    question_set = current_app.mongo.db.question_sets.find_one({"code": code})

    if not question_set:
        flash('Invalid test code. Please try again.', 'error')
        return redirect(url_for('auth.login'))

    # Shuffle questions per student to prevent cheating
    questions = list(question_set['questions'])
    random.shuffle(questions)
    question_dict = {q['question']: q['options'] for q in questions}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        time_taken = int(form_data.pop('time_taken', 0))
        
        # Calculate correct answers and build question-by-question analysis
        questions_detail = []
        correct_answers = 0
        
        for q in question_set['questions']:
            q_text = q['question']
            chosen = form_data.get(q_text)
            correct = q['answer']
            is_correct = str(chosen).strip() == str(correct).strip()
            if is_correct:
                correct_answers += 1
            questions_detail.append({
                "question": q_text,
                "chosen": chosen if chosen else "—",
                "correct": correct,
                "is_correct": is_correct
            })
            
        score = round(correct_answers / len(question_set['questions']) * 100, 2)
        record_test_completion(regno, code, score, correct_answers, time_taken, questions_detail)
        session.pop('test_in_progress', None)
        return redirect(url_for('views.score', code=code))

    duration_minutes = question_set.get('duration_minutes', 15)
    return render_template('test.html', question_dict=question_dict, code=code, regno=regno, duration_minutes=duration_minutes)


@views.route('/rules', methods=['GET', 'POST'])
def rules():
    if 'regno' not in session or 'test_code' not in session:
        flash('You need to log in first.', 'error')
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        if 'test_in_progress' in session:
            return redirect(url_for('views.test', code=session['test_code']))
        return redirect(url_for('auth.login'))

    code = session['test_code']
    question_set = current_app.mongo.db.question_sets.find_one({"code": code})
    duration_minutes = question_set.get('duration_minutes', 15) if question_set else 15

    return render_template('rules.html', duration_minutes=duration_minutes)


@views.route('/score/<code>')
def score(code):
    if 'regno' not in session:
        flash('You need to log in first.', 'error')
        return redirect(url_for('auth.login'))

    regno = session.get('regno')
    question_set = current_app.mongo.db.question_sets.find_one({"code": code})
    user = current_app.mongo.db.user.find_one({"register_number": regno})
    completed_test = user.get("completed_tests", {}).get(code)

    if not completed_test:
        return redirect(url_for('auth.login'))

    # Format time taken
    seconds_taken = completed_test.get("time_taken", 0)
    minutes = int(seconds_taken) // 60
    secs = int(seconds_taken) % 60
    formatted_time = f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"

    questions_detail = completed_test.get("questions_detail", [])

    return render_template('score.html',
                           code=code,
                           score=completed_test['score'],
                           correct_answers=completed_test['correct_answers'],
                           noQ=str(len(question_set['questions'])),
                           regno=regno,
                           time_taken=formatted_time,
                           questions_detail=questions_detail)


@views.route('/deny', methods=['POST'])
def deny():
    flash('You have declined the exam rules.', 'warning')
    session.pop('regno', None)
    session.pop('test_code', None)
    return redirect(url_for('auth.login'))


# -----------------------------------------------------------------------
# API: Log violations server-side (called by proctoring engine JS)
# -----------------------------------------------------------------------

@views.route('/api/violation', methods=['POST'])
def log_violation():
    """Receive and store proctoring violations from the client."""
    if 'regno' not in session or 'test_code' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    violation_type = data.get('type', 'unknown')
    message = data.get('message', '')

    record_violation(
        session['regno'],
        session['test_code'],
        violation_type,
        message
    )

    return jsonify({'status': 'recorded', 'type': violation_type})


# -----------------------------------------------------------------------
# Invigilator dashboard (teacher only)
# -----------------------------------------------------------------------

@views.route('/invigilator/<code>', methods=['GET'])
@teacher_required
def invi(code):
    test_code = code
    user_data = []

    # Get all users who have either completed the test or have violations for it
    users_completed = current_app.mongo.db.user.find(
        {"completed_tests." + test_code: {"$exists": True}}
    )
    users_with_violations = current_app.mongo.db.user.find(
        {"violations." + test_code: {"$exists": True}}
    )

    # Combine unique users
    seen_regnos = set()
    all_users = []
    for user in users_completed:
        seen_regnos.add(user.get("register_number", ""))
        all_users.append(user)
    for user in users_with_violations:
        if user.get("register_number", "") not in seen_regnos:
            seen_regnos.add(user.get("register_number", ""))
            all_users.append(user)

    for user in all_users:
        test_data = user.get("completed_tests", {}).get(test_code, {})
        violations = user.get("violations", {}).get(test_code, [])
        violation_count = len(violations)
        violation_types = list(set(v.get("type", "unknown") for v in violations))

        # Format time taken
        seconds_taken = test_data.get("time_taken")
        if seconds_taken is not None:
            minutes = int(seconds_taken) // 60
            secs = int(seconds_taken) % 60
            formatted_time = f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"
        else:
            formatted_time = "—"

        user_data.append({
            "reg_no": user.get("register_number", ""),
            "status": "submitted" if test_data else "ongoing",
            "ban": "yes" if user.get("banned", False) else "no",
            "grade": test_data.get("score", "—"),
            "violation_count": violation_count,
            "violation_types": ", ".join(violation_types) if violation_types else "—",
            "time_taken": formatted_time,
            "questions_detail": test_data.get("questions_detail", [])
        })

    return render_template('invigilator.html', user_data=user_data, test_code=test_code)


@views.route('/invigilator/<code>/export', methods=['GET'])
@teacher_required
def export_results(code):
    import csv
    import io
    from flask import Response

    test_code = code

    # Get all users who have either completed the test or have violations for it
    users_completed = current_app.mongo.db.user.find(
        {"completed_tests." + test_code: {"$exists": True}}
    )
    users_with_violations = current_app.mongo.db.user.find(
        {"violations." + test_code: {"$exists": True}}
    )

    seen_regnos = set()
    all_users = []
    for user in users_completed:
        seen_regnos.add(user.get("register_number", ""))
        all_users.append(user)
    for user in users_with_violations:
        if user.get("register_number", "") not in seen_regnos:
            seen_regnos.add(user.get("register_number", ""))
            all_users.append(user)

    # Create CSV file in memory
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Write header
    cw.writerow(['Reg No', 'Status', 'Banned', 'Grade (%)', 'Violations', 'Violation Types', 'Time Taken'])
    
    # Write rows
    for user in all_users:
        test_data = user.get("completed_tests", {}).get(test_code, {})
        violations = user.get("violations", {}).get(test_code, [])
        violation_count = len(violations)
        violation_types = list(set(v.get("type", "unknown") for v in violations))
        
        # Format time taken
        seconds_taken = test_data.get("time_taken")
        if seconds_taken is not None:
            minutes = int(seconds_taken) // 60
            secs = int(seconds_taken) % 60
            formatted_time = f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"
        else:
            formatted_time = "—"
            
        cw.writerow([
            user.get("register_number", ""),
            "submitted" if test_data else "ongoing",
            "yes" if user.get("banned", False) else "no",
            test_data.get("score", "—"),
            violation_count,
            ", ".join(violation_types) if violation_types else "—",
            formatted_time
        ])
        
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=results_{test_code}.csv"}
    )
