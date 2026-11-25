"""
API Routes for the Airline No-Show Prediction System
"""

import os
import sys
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from functools import wraps
from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    jwt_required,
    verify_jwt_in_request,
    set_access_cookies,
    unset_jwt_cookies,
)
from datetime import datetime, timedelta
import logging
import math
from uuid import UUID

# Add backend/app to path for imports
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

# Create Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

logger = logging.getLogger(__name__)
from backend.app.services.prediction_service import (
    ModelNotApprovedError,
    ModelUnavailableError,
    SAFE_VERSION,
    SUPPORTED_MODELS,
    UNVERIFIED_HISTORY_FEATURES,
)


def admin_required(f):
    """Decorator for admin-only routes."""
    @wraps(f)
    @jwt_required()
    def decorated(*args, **kwargs):
        from database.connection import db_manager
        from services.auth_service import AuthService

        session = db_manager.get_session()
        try:
            user = AuthService(session).get_user_by_id(str(get_jwt_identity()))
        finally:
            session.close()

        if not user or not user['is_active']:
            return jsonify({'error': 'Active user required'}), 401
        if user['role'] != 'admin':
            return jsonify({
                'error': 'Admin access required',
                'message': 'This endpoint requires administrator privileges'
            }), 403
        return f(*args, **kwargs)
    return decorated


def token_required(f):
    """Decorator for JWT-protected routes."""
    @wraps(f)
    @jwt_required()
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated


# ============================================
# AUTHENTICATION ROUTES
# ============================================

@api.route('/auth/register', methods=['POST'])
def register():
    """
    Register a new user.
    
    Request body:
        username: str (required)
        email: str (required)
        password: str (required)
    
    Returns:
        Created user info
    """
    if not current_app.config.get('ALLOW_PUBLIC_REGISTRATION', True):
        return jsonify({
            'error': 'Registration disabled',
            'message': 'Public registration is disabled'
        }), 403

    data = request.get_json()
    
    if not isinstance(data, dict) or not data:
        return jsonify({
            'error': 'No data provided',
            'message': 'A JSON object is required'
        }), 400
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Validation
    if not isinstance(username, str) or len(username.strip()) < 3:
        return jsonify({
            'error': 'Invalid username',
            'message': 'Username must be at least 3 characters'
        }), 400
    
    if not isinstance(email, str) or '@' not in email:
        return jsonify({
            'error': 'Invalid email',
            'message': 'Please provide a valid email address'
        }), 400
    
    if not isinstance(password, str) or len(password) < 12 or len(password.encode('utf-8')) > 72:
        return jsonify({
            'error': 'Invalid password',
            'message': 'Password must be 12 to 72 UTF-8 bytes long'
        }), 400
    
    # Get database session
    from database.connection import db_manager
    session = db_manager.get_session()
    
    try:
        from services.auth_service import AuthService
        auth_service = AuthService(session)
        
        result = auth_service.register_user(
            username=username,
            email=email,
            password=password
        )
        
        if result.get('success'):
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({
            'error': 'Registration failed',
            'message': str(e)
        }), 500
    finally:
        session.close()


@api.route('/auth/login', methods=['POST'])
def login():
    """
    Authenticate user and return JWT token.
    
    Request body:
        username: str (required)
        password: str (required)
    
    Returns:
        JWT token and user info
    """
    data = request.get_json()
    
    if not isinstance(data, dict) or not data:
        return jsonify({
            'error': 'No data provided',
            'message': 'A JSON object is required'
        }), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not isinstance(username, str) or not username.strip() or not isinstance(password, str) or not password:
        return jsonify({
            'error': 'Missing credentials',
            'message': 'Username and password are required'
        }), 400
    
    # Get database session
    from database.connection import db_manager
    session = db_manager.get_session()
    
    try:
        from services.auth_service import AuthService
        auth_service = AuthService(session)
        
        result = auth_service.authenticate_user(username, password)
        
        if result.get('success'):
            response = jsonify({'user': result['user']})
            token = create_access_token(
                identity=result['user']['id'],
                additional_claims={
                    'username': result['user']['username'],
                    'role': result['user']['role'],
                },
            )
            set_access_cookies(response, token)
            return response
        else:
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'error': 'Authentication failed',
            'message': str(e)
        }), 500
    finally:
        session.close()


@api.route('/auth/logout', methods=['POST'])
@token_required
def logout():
    """Clear the authenticated browser session."""
    response = jsonify({'message': 'Logged out'})
    unset_jwt_cookies(response)
    return response


@api.route('/auth/profile', methods=['GET'])
@token_required
def get_profile():
    """Get current user's profile."""
    try:
        current_user = get_jwt_identity()
        # get_jwt_identity() returns the 'sub' claim (user_id as string)
        user_id = current_user if isinstance(current_user, str) else current_user.get('sub')
        
        from database.connection import db_manager
        session = db_manager.get_session()
        
        try:
            from services.auth_service import AuthService
            auth_service = AuthService(session)
            user = auth_service.get_user_by_id(user_id)

            if user:
                return jsonify({'user': user})
            return jsonify({'error': 'User not found'}), 404
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({
            'error': 'Failed to get profile',
            'message': str(e)
        }), 500


# ============================================
# PREDICTION ROUTES
# ============================================

@api.route('/predict', methods=['POST'])
@token_required
def predict():
    """Make a single prediction."""
    try:
        data = request.get_json()
        
        if not isinstance(data, dict) or not isinstance(data.get('booking_data'), dict) or not data['booking_data']:
            return jsonify({
                'error': 'Invalid request',
                'message': 'booking_data must be a non-empty object'
            }), 400
        
        booking_data = {
            key: value
            for key, value in data['booking_data'].items()
            if key not in UNVERIFIED_HISTORY_FEATURES
        }
        requested_model = data.get('model')
        requested_version = data.get('version')
        if requested_model is not None and (
            not isinstance(requested_model, str) or requested_model not in SUPPORTED_MODELS
        ):
            return jsonify({'error': 'Invalid model', 'message': 'Requested model is not supported'}), 400
        if requested_version is not None and (
            not isinstance(requested_version, str)
            or SAFE_VERSION.fullmatch(requested_version) is None
        ):
            return jsonify({'error': 'Invalid model version'}), 400
        
        if not hasattr(current_app, 'prediction_service'):
            return jsonify({
                'error': 'Prediction service unavailable',
                'message': 'Models not loaded. Please ensure models are trained and available.'
            }), 503
        
        pred_service = current_app.prediction_service
        
        result = pred_service.predict_single(
            booking_data,
            model_name=requested_model,
            version=requested_version,
            calibrate=data.get('calibrate', True),
            explain=data.get('explain', True)
        )

        booking_id = booking_data.get('booking_id')
        if booking_id is not None:
            try:
                booking_uuid = UUID(str(booking_id))
            except (TypeError, ValueError, AttributeError):
                return jsonify({'error': 'Invalid booking_id', 'message': 'booking_id must be a UUID'}), 400

            from database.connection import db_manager
            from models.schemas import Booking, Prediction
            session = db_manager.get_session()
            try:
                booking = session.get(Booking, booking_uuid)
                if booking is None:
                    return jsonify({'error': 'Booking not found'}), 404

                session.query(Prediction).filter(
                    Prediction.booking_id == booking_uuid
                ).update({'is_active': False})
                prediction = Prediction(
                    booking_id=booking_uuid,
                    model_name=result['model_used'],
                    model_version=result['model_version'],
                    noshow_probability=result['noshow_probability'],
                    prediction_class=result['prediction'],
                    risk_category=result['risk_category'],
                    feature_values=booking_data,
                    shap_explanation=result.get('explanation'),
                    model_run_time_ms=result['run_time_ms'],
                    is_active=True,
                )
                booking.noshow_probability = result['noshow_probability']
                session.add(prediction)
                session.commit()
                result['prediction_id'] = str(prediction.id)
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        
        return jsonify(result)
    
    except ModelUnavailableError as e:
        return jsonify({'error': 'Prediction service unavailable', 'message': str(e)}), 503
    except ModelNotApprovedError as e:
        return jsonify({'error': 'Model not approved', 'message': str(e)}), 400
    except ValueError as e:
        logger.warning(f"Prediction validation error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@api.route('/bulk-predict', methods=['POST'])
@token_required
def bulk_predict():
    """Make batch predictions."""
    try:
        data = request.get_json()
        
        if not isinstance(data, dict) or not isinstance(data.get('bookings'), list):
            return jsonify({
                'error': 'Invalid request',
                'message': 'bookings list is required'
            }), 400
        
        bookings = data['bookings']
        
        if not bookings:
            return jsonify({
                'error': 'Empty bookings list',
                'message': 'At least one booking is required'
            }), 400
        
        if len(bookings) > 1000:
            return jsonify({
                'error': 'Too many bookings',
                'message': 'Maximum 1000 bookings per request'
            }), 400
        if any(not isinstance(booking, dict) or not booking for booking in bookings):
            return jsonify({
                'error': 'Invalid booking data',
                'message': 'Each booking must be a non-empty object'
            }), 400
        requested_model = data.get('model')
        requested_version = data.get('version')
        if requested_model is not None and (
            not isinstance(requested_model, str) or requested_model not in SUPPORTED_MODELS
        ):
            return jsonify({'error': 'Invalid model', 'message': 'Requested model is not supported'}), 400
        if requested_version is not None and (
            not isinstance(requested_version, str)
            or SAFE_VERSION.fullmatch(requested_version) is None
        ):
            return jsonify({'error': 'Invalid model version'}), 400
        
        if not hasattr(current_app, 'prediction_service'):
            return jsonify({
                'error': 'Prediction service unavailable',
                'message': 'Models not loaded'
            }), 503
        
        pred_service = current_app.prediction_service
        
        result = pred_service.predict_batch(
            bookings,
            model_name=requested_model,
            version=requested_version,
            explain=data.get('explain', False)
        )
        
        return jsonify(result)
    
    except ModelUnavailableError as e:
        return jsonify({'error': 'Prediction service unavailable', 'message': str(e)}), 503
    except ModelNotApprovedError as e:
        return jsonify({'error': 'Model not approved', 'message': str(e)}), 400
    except ValueError as e:
        return jsonify({'error': 'Bulk prediction failed', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}")
        return jsonify({
            'error': 'Bulk prediction failed',
            'message': str(e)
        }), 500


@api.route('/models', methods=['GET'])
@token_required
def list_models():
    """List available models."""
    try:
        if not hasattr(current_app, 'prediction_service'):
            return jsonify({'error': 'Model service unavailable'}), 503
        
        pred_service = current_app.prediction_service
        models = pred_service.get_available_models()
        approved_model = getattr(pred_service, 'approved_model', None)
        return jsonify({
            'models': models,
            'prediction_enabled': approved_model is not None,
            'approved_model': approved_model,
        })
    
    except Exception as e:
        logger.error(f"List models error: {e}")
        return jsonify({
            'error': 'Failed to list models',
            'message': str(e)
        }), 500


# ============================================
# ANALYTICS ROUTES
# ============================================

@api.route('/analytics/overview', methods=['GET'])
@token_required
def get_analytics_overview():
    """Get executive dashboard analytics."""
    try:
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService(session)
        
        days = request.args.get('days', 30, type=int)
        if days is None or not 1 <= days <= 365:
            session.close()
            return jsonify({'error': 'Invalid days', 'message': 'days must be between 1 and 365'}), 400
        result = analytics_service.get_executive_dashboard(days)
        
        session.close()
        if result.get('status') == 'error':
            return jsonify(result), 503
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Analytics overview error: {e}")
        return jsonify({'error': 'Failed to get analytics overview', 'message': str(e)}), 503


@api.route('/analytics/routes', methods=['GET'])
@token_required
def get_route_analytics():
    """Get route-level analytics."""
    try:
        route = request.args.get('route')
        
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService(session)
        
        routes = analytics_service.get_route_analysis(route)
        
        session.close()
        return jsonify({'routes': routes})
    
    except Exception as e:
        logger.error(f"Route analytics error: {e}")
        return jsonify({
            'error': 'Failed to get route analytics',
            'message': str(e)
        }), 500


@api.route('/analytics/trends', methods=['GET'])
@token_required
def get_trends():
    """Get time-series trends."""
    try:
        group_by = request.args.get('group_by', 'day')
        days = request.args.get('days', 30, type=int)
        if group_by not in {'day', 'week', 'month'}:
            return jsonify({'error': 'Invalid group_by', 'message': 'group_by must be day, week, or month'}), 400
        if days is None or not 1 <= days <= 365:
            return jsonify({'error': 'Invalid days', 'message': 'days must be between 1 and 365'}), 400
        
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService(session)
        
        trends = analytics_service.get_time_series_trends(group_by, days)
        
        session.close()
        return jsonify({
            'trends': trends,
            'group_by': group_by,
            'days': days
        })
    
    except Exception as e:
        logger.error(f"Trends error: {e}")
        return jsonify({
            'error': 'Failed to get trends',
            'message': str(e)
        }), 500


@api.route('/analytics/channels', methods=['GET'])
@token_required
def get_channel_analytics():
    """Get booking channel performance."""
    try:
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService(session)
        
        channels = analytics_service.get_channel_performance()
        
        session.close()
        return jsonify({'channels': channels})
    
    except Exception as e:
        logger.error(f"Channel analytics error: {e}")
        return jsonify({
            'error': 'Failed to get channel analytics',
            'message': str(e)
        }), 500


@api.route('/analytics/high-risk', methods=['GET'])
@admin_required
def get_high_risk():
    """Get high-risk bookings."""
    try:
        min_prob = request.args.get('min_probability', 0.5, type=float)
        limit = request.args.get('limit', 100, type=int)
        if min_prob is None or not math.isfinite(min_prob) or not 0 <= min_prob <= 1:
            return jsonify({'error': 'Invalid min_probability', 'message': 'min_probability must be between 0 and 1'}), 400
        if limit is None or not 1 <= limit <= 1000:
            return jsonify({'error': 'Invalid limit', 'message': 'limit must be between 1 and 1000'}), 400
        
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService(session)
        
        bookings = analytics_service.get_high_risk_bookings(min_prob, limit)
        
        session.close()
        return jsonify({
            'bookings': bookings,
            'count': len(bookings),
            'min_probability': min_prob
        })
    
    except Exception as e:
        logger.error(f"High-risk error: {e}")
        return jsonify({
            'error': 'Failed to get high-risk bookings',
            'message': str(e)
        }), 500


# ============================================
# MODEL PERFORMANCE ROUTES
# ============================================

@api.route('/models/performance', methods=['GET'])
@token_required
def get_model_performance():
    """Get model performance comparison."""
    try:
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService(session)
        
        models = analytics_service.get_model_comparison()
        
        session.close()
        
        return jsonify({'models': models})
    
    except Exception as e:
        logger.error(f"Model performance error: {e}")
        return jsonify({
            'error': 'Failed to get model performance',
            'message': str(e)
        }), 500


# ============================================
# EXPLAINABILITY ROUTES
# ============================================

@api.route('/explainability/global', methods=['GET'])
@token_required
def get_global_explainability():
    """Get global feature importance."""
    try:
        if not hasattr(current_app, 'prediction_service'):
            return jsonify({'error': 'Prediction service unavailable'}), 503

        model_name = request.args.get('model', current_app.prediction_service.default_model_name)
        result = current_app.prediction_service.get_global_feature_importance(model_name)
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': 'Feature importance unavailable', 'message': str(e)}), 404
    except ModelUnavailableError as e:
        return jsonify({'error': 'Feature importance unavailable', 'message': str(e)}), 503
    except Exception as e:
        logger.error(f"Explainability error: {e}")
        return jsonify({
            'error': 'Failed to get explainability',
            'message': str(e)
        }), 500


# ============================================
# OVERBOOKING ROUTES
# ============================================

@api.route('/overbooking/simulate', methods=['POST'])
@token_required
def simulate_overbooking():
    """Simulate overbooking scenarios."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body is required'
            }), 400
        
        required = ['capacity', 'bookings', 'avg_noshow_prob', 'ticket_price']
        for field in required:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'message': f'{field} is required'
                }), 400

        numeric_fields = required + ['max_overbooking']
        for field in numeric_fields:
            value = data.get(field, 15 if field == 'max_overbooking' else None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return jsonify({'error': f'Invalid field: {field}', 'message': f'{field} must be a finite number'}), 400
        if data['capacity'] <= 0 or int(data['capacity']) != data['capacity']:
            return jsonify({'error': 'Invalid capacity', 'message': 'capacity must be a positive integer'}), 400
        if data['bookings'] < 0 or int(data['bookings']) != data['bookings']:
            return jsonify({'error': 'Invalid bookings', 'message': 'bookings must be a non-negative integer'}), 400
        if not 0 <= data['avg_noshow_prob'] <= 1:
            return jsonify({'error': 'Invalid probability', 'message': 'avg_noshow_prob must be between 0 and 1'}), 400
        if data['ticket_price'] < 0:
            return jsonify({'error': 'Invalid ticket_price', 'message': 'ticket_price must be non-negative'}), 400
        max_overbooking = data.get('max_overbooking', 15)
        if max_overbooking < 0 or int(max_overbooking) != max_overbooking or max_overbooking > 100:
            return jsonify({'error': 'Invalid max_overbooking', 'message': 'max_overbooking must be an integer between 0 and 100'}), 400
        cost_params = data.get('cost_params')
        if cost_params is not None and not isinstance(cost_params, dict):
            return jsonify({'error': 'Invalid cost_params', 'message': 'cost_params must be an object'}), 400
        if cost_params:
            for field in ('compensation_cost', 'empty_seat_cost', 'denied_boarding_cost'):
                value = cost_params.get(field)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    return jsonify({
                        'error': f'Invalid cost parameter: {field}',
                        'message': f'{field} must be a finite non-negative number'
                    }), 400
        
        if not hasattr(current_app, 'prediction_service'):
            return jsonify({
                'error': 'Prediction service unavailable',
                'message': 'Overbooking simulator not available'
            }), 503
        
        pred_service = current_app.prediction_service
        
        result = pred_service.simulate_overbooking(
            capacity=data['capacity'],
            bookings=data['bookings'],
            avg_noshow_prob=data['avg_noshow_prob'],
            ticket_price=data['ticket_price'],
            max_overbooking=data.get('max_overbooking', 15),
            cost_params=data.get('cost_params')
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Overbooking simulation error: {e}")
        return jsonify({
            'error': 'Overbooking simulation failed',
            'message': str(e)
        }), 500


# ============================================
# DRIFT MONITORING ROUTES
# ============================================

@api.route('/monitoring/drift', methods=['GET'])
@token_required
def get_drift_status():
    """
    Get model drift monitoring status with actual statistical analysis.
    
    Uses Population Stability Index (PSI), Kolmogorov-Smirnov test,
    and statistical comparison to detect data drift.
    """
    try:
        from database.connection import db_manager
        session = db_manager.get_session()
        
        from services.drift_service import create_drift_service
        drift_service = create_drift_service(session)
        
        report = drift_service.get_drift_status()
        
        session.close()
        
        return jsonify(report.to_dict())
    
    except Exception as e:
        logger.error(f"Drift monitoring error: {e}")
        return jsonify({
            'error': 'Failed to get drift status',
            'message': str(e),
            'status': 'error'
        }), 500


# ============================================
# ERROR HANDLERS
# ============================================

@api.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': str(error)
    }), 400


@api.errorhandler(401)
def unauthorized(error):
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Valid authentication token is required'
    }), 401


@api.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404


@api.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500
