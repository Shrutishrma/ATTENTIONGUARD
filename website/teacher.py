import random
import string
from datetime import datetime

from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session, current_app)

from .utils import teacher_required
from .parser import parse_questions_from_file

teacher_bp = Blueprint('teacher', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv'}
_SAFE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'  # no 0/O, 1/I/L to avoid confusion


def _generate_code():
    """Return a unique 6-character alphanumeric test code."""
    while True:
        code = ''.join(random.choices(_SAFE_CHARS, k=6))
        if not current_app.mongo.db.question_sets.find_one({"code": code}):
            return code


# ---------------------------------------------------------------------------
# Dashboard — list all question sets created by this teacher
# ---------------------------------------------------------------------------

@teacher_bp.route('/teacher')
@teacher_required
def dashboard():
    question_sets = list(
        current_app.mongo.db.question_sets
        .find({"created_by": session['regno']})
        .sort("created_at", -1)
    )
    for qs in question_sets:
        qs['_id'] = str(qs['_id'])
        qs['attempt_count'] = current_app.mongo.db.user.count_documents(
            {f"completed_tests.{qs['code']}": {"$exists": True}}
        )
        created = qs.get('created_at')
        if isinstance(created, datetime):
            qs['created_at_display'] = created.strftime('%b %d, %Y')
        elif isinstance(created, str) and created.strip():
            try:
                dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                qs['created_at_display'] = dt.strftime('%b %d, %Y')
            except Exception:
                qs['created_at_display'] = created[:10]
        else:
            qs['created_at_display'] = '—'
    return render_template('teacher_dashboard.html',
                           question_sets=question_sets,
                           regno=session['regno'])


# ---------------------------------------------------------------------------
# Upload — parse file, validate, generate code, store in DB
# ---------------------------------------------------------------------------

@teacher_bp.route('/teacher/upload', methods=['GET', 'POST'])
@teacher_required
def upload():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        file = request.files.get('file')

        if not title:
            flash('Please provide a title for the question set.', 'error')
            return redirect(url_for('teacher.upload'))

        if not file or not file.filename:
            flash('Please select a file to upload.', 'error')
            return redirect(url_for('teacher.upload'))

        ext = file.filename.rsplit('.', 1)[-1] if '.' in file.filename else ''
        if ext.lower() not in ALLOWED_EXTENSIONS:
            flash(f'Unsupported file type ".{ext}". Allowed: PDF, DOCX, TXT, CSV.', 'error')
            return redirect(url_for('teacher.upload'))

        try:
            questions = parse_questions_from_file(file, ext)
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('teacher.upload'))

        allow_retake = 'allow_retake' in request.form
        max_attempts = 0 if allow_retake else 1
        try:
            duration_minutes = int(request.form.get('duration_minutes', 15))
        except ValueError:
            duration_minutes = 15

        code = _generate_code()
        current_app.mongo.db.question_sets.insert_one({
            "code": code,
            "title": title,
            "questions": questions,
            "created_by": session['regno'],
            "created_at": datetime.utcnow(),
            "max_attempts": max_attempts,
            "duration_minutes": duration_minutes,
        })

        flash(f'SUCCESS — Question set "{title}" created. Test code: {code}', 'success')
        return redirect(url_for('teacher.dashboard'))

    return render_template('upload.html')


# ---------------------------------------------------------------------------
# Delete a question set (only the creator can delete it)
# ---------------------------------------------------------------------------

@teacher_bp.route('/teacher/delete/<code>', methods=['POST'])
@teacher_required
def delete_questionset(code):
    result = current_app.mongo.db.question_sets.delete_one(
        {"code": code, "created_by": session['regno']}
    )
    if result.deleted_count:
        flash(f'Question set {code} has been deleted.', 'info')
    else:
        flash('Could not delete — question set not found or not owned by you.', 'error')
    return redirect(url_for('teacher.dashboard'))
