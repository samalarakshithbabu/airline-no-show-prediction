"""
Flask Application Factory for Airline No-Show Prediction API
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Get absolute path to backend/app directory and add it to path for imports
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# Import configuration from the local config module
from config import (
    FLASK_ENV, DEBUG, SECRET_KEY, JWT_SECRET_KEY,
    SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS, CORS_ORIGINS,
    RATELIMIT_DEFAULT, RATELIMIT_STORAGE_URI, LOG_LEVEL, LOG_FILE, MODEL_DIR,
    ALLOW_PUBLIC_REGISTRATION, AUTO_CREATE_SCHEMA, JWT_ACCESS_TOKEN_EXPIRY_HOURS,
    JWT_COOKIE_SECURE, JWT_COOKIE_HTTPONLY, JWT_COOKIE_SAMESITE, JWT_ACCESS_COOKIE_PATH,
    JWT_ACCESS_CSRF_COOKIE_PATH, JWT_COOKIE_CSRF_PROTECT, TRUSTED_PROXY_HOPS
)

# Configure logging
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else '.', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE) if LOG_FILE else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize extensions
db = SQLAlchemy()
jwt_manager = JWTManager()
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=RATELIMIT_STORAGE_URI,
    default_limits=[RATELIMIT_DEFAULT]
)


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure app
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['FLASK_ENV'] = FLASK_ENV
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['JSON_SORT_KEYS'] = False
    app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 60 * 60 * JWT_ACCESS_TOKEN_EXPIRY_HOURS
    app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']
    app.config['JWT_COOKIE_SECURE'] = JWT_COOKIE_SECURE
    app.config['JWT_COOKIE_HTTPONLY'] = JWT_COOKIE_HTTPONLY
    app.config['JWT_COOKIE_SAMESITE'] = JWT_COOKIE_SAMESITE
    app.config['JWT_ACCESS_COOKIE_PATH'] = JWT_ACCESS_COOKIE_PATH
    app.config['JWT_ACCESS_CSRF_COOKIE_PATH'] = JWT_ACCESS_CSRF_COOKIE_PATH
    app.config['JWT_COOKIE_CSRF_PROTECT'] = JWT_COOKIE_CSRF_PROTECT
    app.config['JWT_CSRF_HEADER_NAME'] = 'X-CSRF-TOKEN'
    app.config['RATELIMIT_STORAGE_URI'] = RATELIMIT_STORAGE_URI
    app.config['TRUSTED_PROXY_HOPS'] = TRUSTED_PROXY_HOPS
    app.config['ALLOW_PUBLIC_REGISTRATION'] = ALLOW_PUBLIC_REGISTRATION
    app.config['AUTO_CREATE_SCHEMA'] = AUTO_CREATE_SCHEMA
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    if config:
        app.config.update(config)
    if (
        app.config.get('FLASK_ENV') == 'production'
        and app.config['RATELIMIT_STORAGE_URI'].startswith('memory://')
    ):
        raise ValueError('Production requires shared rate-limit storage')
    if (
        app.config.get('FLASK_ENV') == 'production'
        and app.config['TRUSTED_PROXY_HOPS'] < 1
    ):
        raise ValueError('Production requires trusted proxy configuration')
    
    # Initialize extensions with app
    if app.config['TRUSTED_PROXY_HOPS'] > 0:
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=app.config['TRUSTED_PROXY_HOPS'],
            x_proto=1,
        )
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}}, supports_credentials=True)
    db.init_app(app)
    jwt_manager.init_app(app)
    limiter.init_app(app)
    
    # Initialize database
    with app.app_context():
        try:
            from database.connection import db_manager
            connection_result = db_manager.test_connection()
            logger.info(f"Database status: {connection_result['status']}")
            
            if app.config['AUTO_CREATE_SCHEMA']:
                db_manager.init_schema()
                logger.info("Database schema initialized")
            else:
                logger.info("Automatic schema creation disabled")
        except Exception as e:
            logger.warning(f"Database initialization warning: {e}")
    
    # Initialize prediction service (optional, may fail if models not trained)
    try:
        from backend.app.services.prediction_service import PredictionService
        app.prediction_service = PredictionService(model_dir=MODEL_DIR)
        app.prediction_service.load_all_models()
        logger.info(f"Loaded {len(app.prediction_service.models)} models")
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
    
    # Register blueprints
    from routes.api import api
    app.register_blueprint(api, url_prefix='/api')
    
    # JWT error handlers
    @jwt_manager.token_in_blocklist_loader
    def check_if_token_in_blocklist(jwt_header, jwt_payload):
        return False
    
    @jwt_manager.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'Token has expired', 'message': 'Please log in again'}), 401
    
    @jwt_manager.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'Invalid token', 'message': str(error)}), 401
    
    @jwt_manager.unauthorized_loader
    def unauthorized_callback(error):
        return jsonify({'error': 'Authorization required', 'message': 'Please provide a valid token'}), 401
    
    # Rate limit response handler (Flask-Limiter 4.x style)
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': str(error.description) if hasattr(error, 'description') else 'Too many requests',
            'retry_after': getattr(error, 'reset_time', 60)
        }), 429
    
    # Root endpoint
    @app.route('/')
    def index():
        return {
            'name': 'Airline No-Show Prediction API',
            'version': '1.0.0',
            'status': 'operational',
            'flask_env': FLASK_ENV,
            'endpoints': {
                'health': '/api/health',
                'predict': '/api/predict',
                'analytics': '/api/analytics',
                'auth': '/api/auth',
                'documentation': '/api/docs'
            }
        }
    
    # API docs
    @app.route('/api/docs')
    def docs():
        return {
            'title': 'Airline No-Show Prediction API',
            'version': 'v1',
            'description': 'ML-powered airline no-show prediction system',
            'authentication': {
                'type': 'HttpOnly JWT cookie with CSRF protection; bearer headers are also accepted',
                'endpoint': '/api/auth/login',
            },
            'endpoints': {
                'POST /api/auth/register': {'request': ['username', 'email', 'password']},
                'POST /api/auth/login': {'request': ['username', 'password']},
                'POST /api/auth/logout': {'auth': True},
                'POST /api/predict': {'auth': True, 'request': ['booking_data']},
                'GET /api/analytics/overview': {'auth': True},
                'POST /api/overbooking/simulate': {'auth': True}
            }
        }
    
    # Health check
    @app.route('/api/health')
    def health_check():
        from database.connection import db_manager
        database = db_manager.test_connection()
        model_selected = bool(
            getattr(getattr(app, 'prediction_service', None), 'approved_model', None)
        )
        healthy = database['status'] == 'healthy'
        return jsonify({
            'status': 'healthy' if healthy else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'api_version': 'v1',
            'prediction_enabled': model_selected,
            'components': {
                'database': database['status'],
                'ml_models': 'approved model loaded' if model_selected else 'no approved model'
            }
        }), 200 if healthy else 503
    
    # Detailed health check
    @app.route('/api/health/detailed')
    def detailed_health_check():
        try:
            from database.connection import db_manager
            database = db_manager.test_connection()
            models_loaded = len(app.prediction_service.models) if hasattr(app, 'prediction_service') else 0
            model_selected = bool(
                getattr(getattr(app, 'prediction_service', None), 'approved_model', None)
            )
            api_healthy = database['status'] == 'healthy'
            return jsonify({
                'status': 'healthy' if api_healthy and model_selected else 'degraded',
                'timestamp': datetime.utcnow().isoformat(),
                'prediction_enabled': model_selected,
                'components': {
                    'api': 'healthy',
                    'database': database['status'],
                    'ml_models': 'healthy' if model_selected else 'no approved model selected'
                },
                'models': {'loaded_count': models_loaded}
            }), 200 if api_healthy else 503
        except Exception as e:
            return jsonify({'status': 'degraded', 'error': str(e)}), 503
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized', 'message': str(error)}), 401
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': 'The requested resource was not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({'error': 'Internal server error', 'message': 'An unexpected error occurred'}), 500
    
    logger.info(f"Flask application created in {FLASK_ENV} mode")
    return app


# Create app instance for direct execution
app = create_app()


if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = DEBUG
    
    print(f"\nStarting server on {host}:{port}")
    print(f"API Documentation: http://localhost:{port}/api/docs")
    print("Press Ctrl+C to stop\n")
    
    app.run(host=host, port=port, debug=debug)