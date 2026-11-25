"""
Unit Tests for the Airline No-Show Prediction System
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataProcessing:
    """Tests for data processing module."""
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicate rows."""
        from ml.data_processing import clean_data
        
        df = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [10, 10, 20, 30],
            'no_show': [0, 0, 1, 0]
        })
        
        cleaned = clean_data(df)
        assert len(cleaned) == 3
    
    def test_clean_data_handles_missing_values(self):
        """Test that clean_data fills missing values."""
        from ml.data_processing import clean_data
        
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [10, np.nan, 30, 40],
            'no_show': [0, 1, 0, 1]
        })
        
        cleaned = clean_data(df)
        assert cleaned['A'].isna().sum() == 0
        assert cleaned['B'].isna().sum() == 0
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        from ml.data_processing import encode_categorical
        
        df = pd.DataFrame({
            'A': ['cat', 'dog', 'cat'],
            'B': [1, 2, 3],
            'no_show': [0, 1, 0]
        })
        
        encoded, mappings = encode_categorical(df, ['A'])
        assert 'A' in mappings
        assert encoded['A'].dtype in [np.int64, np.int32]

    def test_preprocess_data_keeps_passengers_in_one_split_and_drops_unverified_history(self):
        from ml.data_processing import preprocess_data

        rows = []
        for passenger_id in range(60):
            for booking_number in range(2):
                rows.append({
                    'booking_id': f'b-{passenger_id}-{booking_number}',
                    'passenger_id': passenger_id,
                    'booking_date': f'2024-01-{booking_number + 1:02d}',
                    'flight_date': f'2024-02-{booking_number + 1:02d}',
                    'booking_channel': 'WEB',
                    'cabin_class': 'ECONOMY',
                    'ticket_price': 200 + passenger_id,
                    'departure_airport': 'JFK' if passenger_id % 2 else 'LAX',
                    'arrival_airport': 'LAX' if passenger_id % 2 else 'JFK',
                    'total_bookings': 10 + passenger_id,
                    'total_noshows': (passenger_id + booking_number) % 4,
                    'total_cancellations': passenger_id % 3,
                    'loyalty_tier': 'STANDARD',
                    'no_show': (passenger_id + booking_number) % 2,
                })
        df = pd.DataFrame(rows)

        X_train, X_test, y_train, y_test, preprocessing = preprocess_data(df)

        assert set(df.loc[X_train.index, 'passenger_id']).isdisjoint(
            set(df.loc[X_test.index, 'passenger_id'])
        )
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert preprocessing['split_method'] == 'stratified_group_by_passenger'
        unavailable_history = {
            'total_bookings',
            'total_noshows',
            'total_cancellations',
            'historical_noshow_rate',
            'has_noshow_history',
            'high_noshow_risk',
            'has_cancellation_history',
            'cancellation_rate',
            'is_new_passenger',
            'is_frequent_flyer',
        }
        assert unavailable_history.isdisjoint(preprocessing['feature_columns'])
        assert 'departure_airport_JFK' in preprocessing['feature_columns']
        assert 'arrival_airport_LAX' in preprocessing['feature_columns']


class TestFeatureEngineering:
    """Tests for feature engineering module."""
    
    def test_engineer_features_adds_columns(self):
        """Test that feature engineering adds new features."""
        from ml.feature_engineering import engineer_features
        
        df = pd.DataFrame({
            'booking_date': pd.date_range('2024-01-01', periods=10),
            'flight_date': pd.date_range('2024-01-15', periods=10),
            'departure_time': pd.date_range('2024-01-15 10:00', periods=10, freq='H'),
            'departure_airport': ['JFK', 'LAX'] * 5,
            'arrival_airport': ['LAX', 'JFK'] * 5,
            'no_show': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
        })
        
        engineered = engineer_features(df)
        
        # Check that new features were added
        assert 'booking_lead_time' in engineered.columns
        assert 'flight_dayofweek' in engineered.columns
        assert 'route' in engineered.columns
        assert 'is_weekend' in engineered.columns


class TestModelTraining:
    """Tests for model training module."""
    
    def test_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        from ml.training import ModelTrainer
        
        trainer = ModelTrainer(random_state=42)
        assert trainer.random_state == 42
        assert len(trainer.model_configs) == 5
    
    def test_compute_class_weights(self):
        """Test class weight computation."""
        from ml.training import ModelTrainer
        
        trainer = ModelTrainer()
        y = np.array([0, 0, 0, 0, 0, 1, 1])  # Imbalanced
        weights = trainer.compute_class_weights(y)
        
        assert weights[0] < weights[1]  # Minority class should have higher weight

    def test_grouped_cross_validation_returns_oof_metrics(self):
        from ml.training import ModelTrainer

        X = pd.DataFrame({
            'booking_lead_time': np.tile(np.arange(10), 12),
            'ticket_price': np.repeat(np.arange(12), 10),
        })
        groups = pd.Series(np.repeat(np.arange(60), 2))
        y = pd.Series(np.tile([0, 1], 60))

        result = ModelTrainer(random_state=7).cross_validate_grouped(
            X,
            y,
            groups,
            model_names=['LogisticRegression'],
            cv=5,
        )

        metrics = result['results']['LogisticRegression']
        assert not result['failures']
        assert len(metrics['out_of_fold_probabilities']) == len(y)
        assert len(metrics['cv_folds']) == 5
        assert 0.0 <= metrics['f1_threshold'] <= 1.0
        assert 0.0 <= metrics['pr_auc'] <= 1.0


class TestEvaluation:
    """Tests for model evaluation module."""
    
    def test_evaluator_metrics(self):
        """Test ModelEvaluator computes metrics correctly."""
        from ml.evaluation import ModelEvaluator
        
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])
        y_proba = np.array([0.3, 0.7, 0.8, 0.4, 0.9, 0.2])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba)
        metrics = evaluator.get_metrics()
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics

    def test_business_threshold_uses_explicit_false_positive_and_false_negative_costs(self):
        from ml.evaluation import evaluate_business_costs, select_business_threshold

        y_true = np.array([1, 0, 1, 0])
        probabilities = np.array([0.8, 0.6, 0.4, 0.1])
        observed = evaluate_business_costs(
            y_true,
            probabilities,
            threshold=0.5,
            missed_no_show_cost=300,
            false_positive_action_cost=50,
        )
        selected = select_business_threshold(
            y_true,
            probabilities,
            missed_no_show_cost=300,
            false_positive_action_cost=50,
        )

        assert observed['false_positives'] == 1
        assert observed['false_negatives'] == 1
        assert observed['total_cost'] == 350
        assert selected['threshold'] == 0.4
        assert selected['total_cost'] == 50
    
    def test_evaluator_confusion_matrix(self):
        """Test confusion matrix computation."""
        from ml.evaluation import ModelEvaluator
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.3, 0.7, 0.8, 0.4])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba)
        cm = evaluator.get_confusion_matrix()
        
        assert cm.shape == (2, 2)


class TestOverbookingSimulator:
    """Tests for overbooking simulation module."""
    
    def test_simulation_runs(self):
        """Test that simulation runs without errors."""
        from ml.overbooking_simulator import OverbookingSimulator, CostParameters
        
        params = CostParameters()
        simulator = OverbookingSimulator(params)
        
        results = simulator.simulate_scenario(
            capacity=180,
            bookings=195,
            avg_noshow_prob=0.15,
            ticket_price=350,
            max_overbooking=10
        )
        
        assert len(results) == 11  # 0 to 10 inclusive
        assert results[0].overbooking_level == 0
    
    def test_simulation_results_structure(self):
        """Test simulation result structure."""
        from ml.overbooking_simulator import OverbookingSimulator, CostParameters
        
        simulator = OverbookingSimulator()
        results = simulator.simulate_scenario(180, 195, 0.15, 350)
        
        first = results[0]
        assert hasattr(first, 'overbooking_level')
        assert hasattr(first, 'capacity')
        assert hasattr(first, 'expected_noshows')
        assert hasattr(first, 'expected_profit')
        assert hasattr(first, 'probability_denied')


class TestDriftDetection:
    """Tests for drift detection module."""
    
    def test_drift_detection(self):
        """Test drift detection computes metrics."""
        from ml.drift_detection import DriftDetector
        
        detector = DriftDetector(psi_threshold=0.1)
        
        # Create reference data and fit detector
        ref_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100)
        })
        detector.fit_reference(ref_data)
        
        # Current data similar to reference
        curr_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100)
        })
        drift_results = detector.detect_all_drift(curr_data)
        
        assert len(drift_results) >= 1
    
    def test_psi_calculation(self):
        """Test PSI calculation."""
        from ml.drift_detection import DriftDetector
        
        detector = DriftDetector()
        ref = np.random.normal(0, 1, 1000)
        curr = np.random.normal(0, 1, 1000)
        
        psi = detector.calculate_psi(ref, curr)
        
        assert psi >= 0
        assert psi < 1  # Should be small for similar distributions

    def test_backend_drift_result_exposes_psi_not_feature_mean(self):
        from backend.app.services.drift_service import DriftService

        service = DriftService()
        result = service.detect_feature_drift(
            'booking_lead_time',
            np.arange(100, dtype=float),
            np.arange(100, dtype=float) + 100,
        )

        assert result.drift_detected
        assert result.metric_name == 'psi'
        assert result.metric_value >= result.threshold
        assert result.ks_statistic > 0.9
        assert result.to_dict()['metric_value'] == round(result.metric_value, 4)

    def test_drift_service_scores_predictions_after_final_booking_labels(self):
        from datetime import datetime
        from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine
        from sqlalchemy.orm import Session
        from backend.app.services.drift_service import DriftService

        engine = create_engine('sqlite:///:memory:')
        metadata = MetaData()
        bookings = Table(
            'bookings',
            metadata,
            Column('booking_id', String, primary_key=True),
            Column('booking_status', String, nullable=False),
        )
        predictions = Table(
            'predictions',
            metadata,
            Column('prediction_id', Integer, primary_key=True),
            Column('booking_id', String, nullable=False),
            Column('model_name', String, nullable=False),
            Column('model_version', String, nullable=False),
            Column('prediction_class', Integer, nullable=False),
            Column('noshow_probability', Float, nullable=False),
            Column('prediction_timestamp', DateTime, nullable=False),
            Column('is_active', Boolean, nullable=False),
        )
        metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(bookings.insert(), [
                {'booking_id': 'b1', 'booking_status': 'NOSHOW'},
                {'booking_id': 'b2', 'booking_status': 'COMPLETED'},
                {'booking_id': 'b3', 'booking_status': 'NOSHOW'},
                {'booking_id': 'b4', 'booking_status': 'COMPLETED'},
            ])
            connection.execute(predictions.insert(), [
                {'prediction_id': 1, 'booking_id': 'b1', 'model_name': 'test', 'model_version': 'v1',
                 'prediction_class': 0, 'noshow_probability': 0.1,
                 'prediction_timestamp': datetime(2024, 1, 1), 'is_active': True},
                {'prediction_id': 2, 'booking_id': 'b1', 'model_name': 'test', 'model_version': 'v1',
                 'prediction_class': 1, 'noshow_probability': 0.8,
                 'prediction_timestamp': datetime(2024, 1, 2), 'is_active': True},
                {'prediction_id': 3, 'booking_id': 'b2', 'model_name': 'test', 'model_version': 'v1',
                 'prediction_class': 0, 'noshow_probability': 0.2,
                 'prediction_timestamp': datetime(2024, 1, 2), 'is_active': True},
                {'prediction_id': 4, 'booking_id': 'b3', 'model_name': 'test', 'model_version': 'v1',
                 'prediction_class': 1, 'noshow_probability': 0.7,
                 'prediction_timestamp': datetime(2024, 1, 2), 'is_active': True},
                {'prediction_id': 5, 'booking_id': 'b4', 'model_name': 'test', 'model_version': 'v1',
                 'prediction_class': 0, 'noshow_probability': 0.1,
                 'prediction_timestamp': datetime(2024, 1, 2), 'is_active': True},
            ])

        with Session(engine) as session:
            report = DriftService(session).get_labeled_performance()

        assert report['status'] == 'available'
        assert report['labeled_predictions'] == 4
        assert report['models'][0]['roc_auc'] == 1.0
        assert report['models'][0]['brier_score'] == pytest.approx(0.045)


class TestSHAPExplanations:
    def test_unapproved_production_pointer_disables_automatic_model_selection(self, tmp_path):
        import json
        from backend.app.services.prediction_service import PredictionService

        (tmp_path / 'production_model.json').write_text(json.dumps({
            'status': 'not_approved',
            'model_name': None,
            'version': None,
        }))

        service = PredictionService(str(tmp_path))
        service.load_all_models()

        assert service.default_model_name is None

    def test_approved_model_returns_individual_and_global_shap_values(self, tmp_path):
        from backend.app.services.prediction_service import PredictionService
        from tests.model_fixtures import write_approved_test_model

        model_name, version, _, _ = write_approved_test_model(tmp_path)
        service = PredictionService(str(tmp_path))
        assert service.load_all_models() == {'approved_model': True}
        prediction = service.predict_single(
            {'ticket_price': 350},
            model_name=model_name,
            version=version,
            explain=True,
        )
        explanation = prediction['explanation']
        assert explanation is not None
        assert explanation['explanation_scope'] == 'uncalibrated_base_estimator'
        assert explanation['all_contributions']
        global_result = service.get_global_feature_importance(model_name)
        assert len(global_result['features']) == 1
        assert abs(sum(item['importance'] for item in global_result['features']) - 1) < 1e-6


class TestOverbookingFinancialModel:
    def test_expected_financial_values_include_booked_revenue_and_both_denial_costs(self):
        from ml.overbooking_simulator import CostParameters, OverbookingSimulator

        simulator = OverbookingSimulator(
            CostParameters(
                compensation_cost=100,
                empty_seat_cost=10,
                denied_boarding_cost=50,
            ),
            random_seed=42,
        )
        simulator.simulate_shows = lambda bookings, show_prob, n_simulations: np.array([2, 0, 3, 1])

        result = simulator.calculate_expected_values(
            bookings=2,
            show_prob=0.5,
            capacity=2,
            ticket_price=100,
            n_simulations=4,
        )

        assert result['expected_revenue'] == 200
        assert result['expected_denied'] == 0.25
        assert result['expected_empty_seats'] == 0.75
        assert result['expected_cost'] == 45
        assert result['expected_profit'] == 155

    def test_denied_boarding_cost_and_overflow_probability_are_explicit(self):
        from ml.overbooking_simulator import CostParameters, OverbookingSimulator

        simulator = OverbookingSimulator(
            CostParameters(
                compensation_cost=100,
                empty_seat_cost=10,
                denied_boarding_cost=50,
            ),
            random_seed=42,
        )
        simulator.simulate_shows = lambda bookings, show_prob, n_simulations: np.array([3, 1])
        financials = simulator.calculate_expected_values(
            bookings=3,
            show_prob=0.5,
            capacity=2,
            ticket_price=100,
            n_simulations=2,
        )
        analysis = simulator.analyze_capacity(
            current_bookings=3,
            noshow_prob=0.5,
            capacity=2,
            ticket_price=100,
        )

        assert financials['expected_denied'] == 0.5
        assert financials['expected_cost'] == 80
        assert analysis['overflow_risk'] == 0.5
        guidance = simulator.get_recommendation()
        assert guidance['safe_overbooking_level'] is None
        assert not guidance['recommendation_available']

    def test_simulation_seed_and_cost_validation(self):
        from ml.overbooking_simulator import CostParameters, OverbookingSimulator

        params = dict(capacity=100, bookings=95, avg_noshow_prob=0.1, ticket_price=200, max_overbooking=3)
        first = OverbookingSimulator(random_seed=7).simulate_scenario(**params)
        second = OverbookingSimulator(random_seed=7).simulate_scenario(**params)
        assert [scenario.to_dict() for scenario in first] == [scenario.to_dict() for scenario in second]
        with pytest.raises(ValueError, match='non-negative'):
            CostParameters.from_dict({'empty_seat_cost': -1})


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from backend.app import create_app
        app = create_app({'TESTING': True})
        return app.test_client()

    @pytest.fixture
    def auth_headers(self, client):
        from backend.app.services.auth_service import AuthService
        from database.connection import db_manager
        from sqlalchemy import text
        from flask_jwt_extended import create_access_token

        username = f'api-test-{uuid.uuid4().hex}'
        session = db_manager.get_session()
        try:
            user = AuthService(session).register_user(
                username=username,
                email=f'{username}@example.test',
                password='ApiTest!Password2026',
            )
            if not user.get('success'):
                raise RuntimeError(f'Could not create API test user: {user.get("error")}')
            user_id = user['user']['id']
        finally:
            session.close()

        with client.application.app_context():
            token = create_access_token(identity=user_id)
        try:
            yield {'Authorization': f'Bearer {token}'}
        finally:
            session = db_manager.get_session()
            try:
                session.execute(text('DELETE FROM users WHERE id = :user_id'), {'user_id': user_id})
                session.commit()
            finally:
                session.close()
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['prediction_enabled'] is False
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        data = response.get_json()
        assert 'name' in data
        assert 'endpoints' in data

    def test_prediction_validation_and_overbooking_contract(self, client, auth_headers):
        invalid_prediction = client.post(
            '/api/predict',
            json={'booking_data': {}},
            headers=auth_headers,
        )
        assert invalid_prediction.status_code == 400

        invalid_simulation = client.post(
            '/api/overbooking/simulate',
            json={
                'capacity': 100,
                'bookings': 100,
                'avg_noshow_prob': 1.5,
                'ticket_price': 350,
            },
            headers=auth_headers,
        )
        assert invalid_simulation.status_code == 400

        response = client.post(
            '/api/overbooking/simulate',
            json={
                'capacity': 100,
                'bookings': 100,
                'avg_noshow_prob': 0.1,
                'ticket_price': 350,
                'max_overbooking': 2,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['scenarios']) == 3
        assert {'expected_denied', 'expected_cost', 'probability_denied'} <= set(data['scenarios'][0])
        assert 'best_overbooking_level' in data['recommendation']
        assert data['recommendation']['operational_recommendation_available'] is False
        assert 'not an operational recommendation' in data['recommendation']['disclaimer']

        invalid_cost = client.post(
            '/api/overbooking/simulate',
            json={
                'capacity': 100,
                'bookings': 100,
                'avg_noshow_prob': 0.1,
                'ticket_price': 350,
                'cost_params': {'denied_boarding_cost': -1},
            },
            headers=auth_headers,
        )
        assert invalid_cost.status_code == 400

    def test_registration_rejects_non_object_and_weak_password(self, client):
        assert client.post('/api/auth/register', json=[]).status_code == 400
        response = client.post('/api/auth/register', json={
            'username': 'valid-name',
            'email': 'valid@example.test',
            'password': 'short',
        })
        assert response.status_code == 400


class TestDatabaseSchema:
    """Tests that ORM identifiers match the SQL analytics schema."""

    def test_booking_tables_use_sql_primary_key_names(self):
        from sqlalchemy import create_engine, inspect
        from models.schemas import Base

        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(bind=engine)
        inspector = inspect(engine)

        assert 'booking_id' in {column['name'] for column in inspector.get_columns('bookings')}
        assert 'passenger_id' in {column['name'] for column in inspector.get_columns('passengers')}
        assert 'flight_id' in {column['name'] for column in inspector.get_columns('flights')}

    def test_high_risk_query_returns_latest_prediction_per_booking(self):
        from datetime import datetime, timedelta
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models.schemas import Base, Booking, Flight, Passenger, Prediction
        from services.analytics_service import AnalyticsService

        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()
        passenger = Passenger(passenger_code='PAX-1', first_name='A', last_name='Passenger')
        flight = Flight(
            flight_code='FL-1',
            departure_airport='JFK',
            arrival_airport='LAX',
            route='JFK-LAX',
            departure_time=datetime.utcnow() + timedelta(days=1),
            arrival_time=datetime.utcnow() + timedelta(days=1, hours=5),
            capacity=180,
            base_price=350,
        )
        booking = Booking(
            booking_code='BK-1',
            passenger=passenger,
            flight=flight,
            booking_date=datetime.utcnow(),
            ticket_price=350,
            booking_status='CONFIRMED',
        )
        session.add(booking)
        session.flush()
        session.add_all([
            Prediction(
                booking=booking,
                model_name='LightGBM',
                model_version='old',
                noshow_probability=0.75,
                prediction_class=1,
                risk_category='HIGH',
                feature_values={},
                prediction_timestamp=datetime.utcnow() - timedelta(minutes=1),
            ),
            Prediction(
                booking=booking,
                model_name='LightGBM',
                model_version='new',
                noshow_probability=0.2,
                prediction_class=0,
                risk_category='MODERATE',
                feature_values={},
                prediction_timestamp=datetime.utcnow(),
            ),
        ])
        session.commit()

        bookings = AnalyticsService(session).get_high_risk_bookings(min_probability=0.5)

        assert bookings == []
        from services.drift_service import DriftService
        current_data = DriftService(db=session).get_current_data()
        assert current_data is not None
        assert len(current_data) == 1
        session.close()


class TestPredictionService:
    """Tests for prediction service."""
    
    def test_risk_category(self):
        """Test risk category calculation."""
        from backend.app.services.prediction_service import PredictionService
        
        service = PredictionService()
        
        assert service._get_risk_category(0.1) == 'LOW'
        assert service._get_risk_category(0.3) == 'MODERATE'
        assert service._get_risk_category(0.5) == 'ELEVATED'
        assert service._get_risk_category(0.7) == 'HIGH'
        assert service._get_risk_category(0.9) == 'CRITICAL'

    def test_load_model_requires_the_exact_approved_model_version(self, tmp_path):
        from backend.app.services.prediction_service import PredictionService
        from tests.model_fixtures import write_approved_test_model

        model_name, version, _, _ = write_approved_test_model(tmp_path)
        service = PredictionService(str(tmp_path))

        assert service.load_model(model_name, version)
        assert not service.load_model('LightGBM')
        assert not service.load_model(model_name, 'unapproved-version')


class TestDatabaseConfiguration:
    def test_database_environment_configuration(self):
        import subprocess

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_loader = (
            "import importlib.util, os; "
            "spec = importlib.util.spec_from_file_location("
            "'test_config', os.path.join(os.getcwd(), 'backend', 'app', 'config.py')); "
            "config = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(config); "
        )
        env = os.environ.copy()
        env.update({
            'DATABASE_URL': '',
            'DB_TYPE': '',
            'DB_HOST': 'localhost',
        })
        local_check = subprocess.run(
            [
                sys.executable,
                '-c',
                config_loader +
                "assert config.DB_TYPE == 'sqlite'; "
                "assert config.SQLALCHEMY_DATABASE_URI.startswith('sqlite:')",
            ],
            env=env,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert local_check.returncode == 0, local_check.stderr

        env.update({
            'DB_TYPE': 'postgresql',
            'DB_USER': 'test user',
            'DB_PASSWORD': 'p@ss/word',
            'DB_NAME': 'airline_noshow',
        })
        postgres_check = subprocess.run(
            [
                sys.executable,
                '-c',
                config_loader +
                "from sqlalchemy.engine import make_url; "
                "url = make_url(config.SQLALCHEMY_DATABASE_URI); "
                "assert config.DB_TYPE == 'postgresql'; "
                "assert url.username == 'test user'; "
                "assert url.password == 'p@ss/word'; "
                "assert url.database == 'airline_noshow'",
            ],
            env=env,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert postgres_check.returncode == 0, postgres_check.stderr

    def test_production_configuration_rejects_missing_secrets(self):
        import subprocess

        env = os.environ.copy()
        env['PYTHON_DOTENV_DISABLED'] = '1'
        env['FLASK_ENV'] = 'production'
        env.pop('SECRET_KEY', None)
        env.pop('JWT_SECRET_KEY', None)
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                "import importlib.util, os; "
                "spec = importlib.util.spec_from_file_location("
                "'test_config', os.path.join(os.getcwd(), 'backend', 'app', 'config.py')); "
                "config = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(config)",
            ],
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert 'SECRET_KEY must be set in production' in result.stderr

    def test_dotenv_production_mode_does_not_fall_back_to_development(self, tmp_path):
        import subprocess

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'backend', 'app', 'config.py')
        (tmp_path / '.env').write_text('FLASK_ENV=production\n', encoding='utf-8')
        env = os.environ.copy()
        for name in ('FLASK_ENV', 'SECRET_KEY', 'JWT_SECRET_KEY'):
            env.pop(name, None)
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                "import importlib.util; "
                f"spec = importlib.util.spec_from_file_location('test_config', {config_path!r}); "
                "config = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(config)",
            ],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert 'SECRET_KEY must be set in production' in result.stderr


# Pytest configuration
@pytest.fixture(autouse=True)
def setup():
    """Setup before each test."""
    yield


if __name__ == '__main__':
    pytest.main([__file__, '-v'])