"""
Application configuration for Airline No-Show Prediction System
"""

import os
from pathlib import Path
from urllib.parse import quote
from dotenv import dotenv_values, load_dotenv

# Load environment variables
dotenv_environment = dotenv_values().get('FLASK_ENV', 'development')
effective_environment = os.getenv('FLASK_ENV', str(dotenv_environment)).lower()
if effective_environment != 'production':
    load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Database configuration
SQLITE_DB_PATH = BASE_DIR / 'instance' / 'airline_noshow.db'
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
if not SQLALCHEMY_DATABASE_URI and os.getenv('DB_TYPE', 'sqlite').lower() == 'postgresql':
    if not os.getenv('DB_HOST'):
        raise ValueError('DB_HOST must be set when DB_TYPE=postgresql')
    db_user = quote(os.getenv('DB_USER', 'airline_user'), safe='')
    db_password = quote(os.getenv('DB_PASSWORD', ''), safe='')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = quote(os.getenv('DB_NAME', 'airline_noshow'), safe='')
    SQLALCHEMY_DATABASE_URI = (
        f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    )
if not SQLALCHEMY_DATABASE_URI:
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"
DB_TYPE = 'sqlite' if SQLALCHEMY_DATABASE_URI.startswith('sqlite:') else 'postgresql'
os.makedirs(BASE_DIR / 'instance', exist_ok=True)

# Application config
FLASK_ENV = os.getenv('FLASK_ENV', effective_environment).strip().lower()
if FLASK_ENV not in {'development', 'testing', 'production'}:
    raise ValueError('FLASK_ENV must be development, testing, or production')
IS_PRODUCTION = FLASK_ENV == 'production'
ALLOW_PUBLIC_REGISTRATION = os.getenv(
    'ALLOW_PUBLIC_REGISTRATION',
    'false' if IS_PRODUCTION else 'true'
).lower() == 'true'
AUTO_CREATE_SCHEMA = os.getenv(
    'AUTO_CREATE_SCHEMA',
    'false' if IS_PRODUCTION else 'true'
).lower() == 'true'


def _secret(name: str, development_default: str) -> str:
    value = os.getenv(name)
    if not value:
        if IS_PRODUCTION:
            raise ValueError(f'{name} must be set in production')
        return development_default
    if IS_PRODUCTION and (len(value) < 32 or value.lower().startswith('your_')):
        raise ValueError(f'{name} must be a non-placeholder value of at least 32 characters')
    return value


SECRET_KEY = _secret('SECRET_KEY', 'dev-secret-key-change-in-production-min-32-characters')
JWT_SECRET_KEY = _secret('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production-min-32-characters')
if IS_PRODUCTION and DB_TYPE != 'postgresql':
    raise ValueError('Production requires PostgreSQL')
JWT_ACCESS_TOKEN_EXPIRY_HOURS = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRY_HOURS', 24))
JWT_COOKIE_SECURE = IS_PRODUCTION
JWT_COOKIE_HTTPONLY = True
JWT_COOKIE_SAMESITE = 'Lax'
JWT_ACCESS_COOKIE_PATH = '/api'
JWT_ACCESS_CSRF_COOKIE_PATH = '/'
JWT_COOKIE_CSRF_PROTECT = True
CORS_ORIGINS = os.getenv(
    'CORS_ORIGINS',
    'http://localhost,http://localhost:3000,http://127.0.0.1:3000'
).split(',')
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Flask config
FLASK_APP = os.getenv('FLASK_APP', 'run.py')
DEBUG = os.getenv('DEBUG', 'false' if IS_PRODUCTION else 'true').lower() == 'true'

# Model config
MODEL_DIR = os.getenv('MODEL_DIR', str(BASE_DIR / 'models'))
os.makedirs(MODEL_DIR, exist_ok=True)

# Logging config
LOG_DIR = BASE_DIR / 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', str(LOG_DIR / 'app.log'))

# Overbooking config
DEFAULT_COMPENSATION_COST = float(os.getenv('DEFAULT_COMPENSATION_COST', 1500))
DEFAULT_EMPTY_SEAT_COST = float(os.getenv('DEFAULT_EMPTY_SEAT_COST', 300))
DEFAULT_DENIED_BOARDING_COST = float(os.getenv('DEFAULT_DENIED_BOARDING_COST', 5000))

# Drift config
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', 0.1))
SEVERE_DRIFT_THRESHOLD = float(os.getenv('SEVERE_DRIFT_THRESHOLD', 0.2))

# Rate limiting
RATELIMIT_DEFAULT = '200 per day'
RATELIMIT_STORAGE_URI = os.getenv('RATELIMIT_STORAGE_URI', 'memory://')
if IS_PRODUCTION and RATELIMIT_STORAGE_URI.startswith('memory://'):
    raise ValueError('Production requires shared rate-limit storage via RATELIMIT_STORAGE_URI')
TRUSTED_PROXY_HOPS = int(os.getenv('TRUSTED_PROXY_HOPS', '0'))
if IS_PRODUCTION and TRUSTED_PROXY_HOPS < 1:
    raise ValueError('Production requires TRUSTED_PROXY_HOPS for forwarded client addresses')

# SQLAlchemy settings
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ENGINE_OPTIONS = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}