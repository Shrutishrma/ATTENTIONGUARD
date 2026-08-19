from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from .models import find_user_by_reg, is_user_banned, check_password, has_taken_test, ban_user, can_attempt_test, clear_violations, add_user

auth = Blueprint('auth', __name__)


@auth.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        regno = request.form.get('regno', '').strip()
        password = request.form.get('pw', '')
        test_code = request.form.get('code', '').strip()

        user = find_user_by_reg(regno)
        if not user:
            flash('Registration number not found. Please register first if you are a new student.', 'error')
            return redirect(url_for('auth.login'))

        canonical_regno = user['register_number']

        if not check_password(canonical_regno, password):
            flash('Incorrect password.', 'error')
            return redirect(url_for('auth.login'))

        # Teacher flow — role stored in DB, no hardcoded list
        if user.get('role') == 'teacher':
            session['regno'] = canonical_regno
            session['role'] = 'teacher'
            return redirect(url_for('teacher.dashboard'))

        # Student flow — test code is required
        if not test_code:
            flash('Please enter a test code.', 'error')
            return redirect(url_for('auth.login'))

        banned, ban_timer = is_user_banned(canonical_regno)
        if banned:
            flash(f'You are banned until {ban_timer.strftime("%I:%M %p")}.', 'warning')
            return redirect(url_for('auth.login'))

        question_set = current_app.mongo.db.question_sets.find_one({"code": test_code})
        if not question_set:
            flash('Invalid test code.', 'error')
            return redirect(url_for('auth.login'))

        if not can_attempt_test(canonical_regno, test_code, question_set.get('max_attempts', 1)):
            flash('You have already taken this test.', 'warning')
            return redirect(url_for('auth.login'))

        clear_violations(canonical_regno, test_code)

        session['regno'] = canonical_regno
        session['test_code'] = test_code
        session['test_in_progress'] = True
        session['role'] = 'student'
        return redirect(url_for('views.rules'))

    return render_template('login.html')


@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        regno = request.form.get('regno', '').strip()
        password = request.form.get('pw', '')
        confirm_password = request.form.get('cpw', '')

        # Basic validations
        if not name or not regno or not password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('register.html', name=name, regno=regno)

        if len(password) < 4:
            flash('Password must be at least 4 characters long.', 'error')
            return render_template('register.html', name=name, regno=regno)

        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'error')
            return render_template('register.html', name=name, regno=regno)

        # Check if already registered
        existing_user = find_user_by_reg(regno)
        if existing_user:
            flash(f'Register number "{regno}" is already registered. Please sign in.', 'warning')
            return redirect(url_for('auth.login'))

        # Register new student account
        add_user(regno, password, name=name)
        flash('Account registered successfully! Please sign in with your credentials.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth.route('/ban/<regno>', methods=['GET'])
def ban(regno):
    """
    Ban a student. Security rules:
    - A student can only ban THEMSELVES (triggered by proctoring engine auto-disqualification)
    - A teacher can ban any student
    - No unauthenticated access
    """
    if 'regno' not in session:
        flash('You need to log in first.', 'error')
        return redirect(url_for('auth.login'))

    # Teachers can ban any student
    if session.get('role') == 'teacher':
        user = find_user_by_reg(regno)
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('auth.login'))
        ban_user(regno)
        flash(f'User {regno} has been banned.', 'warning')
        return redirect(url_for('auth.login'))

    # Students can only ban themselves (auto-disqualification from proctoring engine)
    if session.get('role') == 'student':
        if session['regno'] != regno:
            flash('Unauthorized action.', 'error')
            return redirect(url_for('auth.login'))
        ban_user(regno)
        session.clear()
        flash('You have been banned due to proctoring violations.', 'warning')
        return redirect(url_for('auth.login'))

    flash('Unauthorized action.', 'error')
    return redirect(url_for('auth.login'))
