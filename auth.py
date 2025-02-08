from flask import Blueprint, request, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import check_password_hash

# Create Blueprint first
auth_bp = Blueprint('auth', __name__)

# User class
class User(UserMixin):
    def __init__(self, id):
        self.id = id

try:
    from config import USER_DATABASE
except ImportError:
    print("Warning: config.py not found. Please create it from config.template.py")
    USER_DATABASE = {}

def init_login_manager(login_manager):
    @login_manager.user_loader
    def load_user(user_id):
        if user_id in USER_DATABASE:
            user = User(user_id)
            return user
        return None

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USER_DATABASE and check_password_hash(USER_DATABASE[username]['password'], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
        
        return "Invalid credentials", 401
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))