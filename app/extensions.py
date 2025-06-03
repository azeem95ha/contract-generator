# contract_api/app/extensions.py
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bcrypt import Bcrypt # For password hashing

cors = CORS()
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
bcrypt = Bcrypt() # Initialize Bcrypt

# This tells Flask-Login where to redirect users if they try to access a protected page without being logged in.
login_manager.login_view = 'auth.login' # 'auth' is the blueprint name, 'login' is the route function name
login_manager.login_message_category = 'info' # For flash messages