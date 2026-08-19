from functools import wraps
from flask import session, redirect, url_for, flash


def teacher_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'teacher':
            flash('Access denied. Please log in as a teacher.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated
