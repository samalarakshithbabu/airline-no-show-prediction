import uuid
from datetime import datetime, timedelta

from tests.runtime_support import auth_headers, client


def test_prediction_persists_latest_result_for_booking(client, auth_headers, tmp_path):
    from backend.app.services.prediction_service import PredictionService
    from database.connection import db_manager
    from models.schemas import Booking, Flight, Passenger, Prediction
    from tests.model_fixtures import write_approved_test_model

    original_service = client.application.prediction_service
    model_name, version, _, _ = write_approved_test_model(tmp_path)
    test_service = PredictionService(str(tmp_path))
    assert test_service.load_all_models() == {'approved_model': True}
    client.application.prediction_service = test_service
    db_manager.init_schema()
    session = db_manager.get_session()
    passenger = Passenger(
        passenger_code=f'RT-{uuid.uuid4().hex[:16]}',
        first_name='Runtime',
        last_name='Passenger',
    )
    flight = Flight(
        flight_code=f'RT-{uuid.uuid4().hex[:16]}',
        departure_airport='JFK',
        arrival_airport='LAX',
        route='JFK-LAX',
        departure_time=datetime.utcnow() + timedelta(days=3),
        arrival_time=datetime.utcnow() + timedelta(days=3, hours=5),
        capacity=180,
        base_price=350,
    )
    booking = Booking(
        booking_code=f'RT-{uuid.uuid4().hex[:16]}',
        passenger=passenger,
        flight=flight,
        booking_date=datetime.utcnow(),
        ticket_price=350,
        booking_status='CONFIRMED',
    )
    session.add(booking)
    session.commit()
    booking_id = str(booking.id)
    passenger_id = passenger.id
    flight_id = flight.id
    session.close()

    payload = {
        'booking_data': {
            'booking_id': booking_id,
            'booking_lead_time': 14,
            'flight_dayofweek': 1,
            'flight_hour': 10,
            'ticket_price': 350,
            'loyalty_tier_encoded': 1,
            'cabin_class_encoded': 0,
            'booking_channel_encoded': 0,
        },
        'model': model_name,
        'version': version,
        'explain': False,
    }

    try:
        unauthenticated = client.post('/api/predict', json=payload)
        assert unauthenticated.status_code == 401

        first = client.post('/api/predict', json=payload, headers=auth_headers)
        assert first.status_code == 200, first.get_json()
        first_result = first.get_json()
        assert 0 <= first_result['noshow_probability'] <= 1
        assert first_result['prediction_id']

        second = client.post('/api/predict', json=payload, headers=auth_headers)
        assert second.status_code == 200, second.get_json()

        session = db_manager.get_session()
        try:
            persisted_booking = session.get(Booking, uuid.UUID(booking_id))
            predictions = session.query(Prediction).filter(
                Prediction.booking_id == uuid.UUID(booking_id)
            ).order_by(Prediction.prediction_timestamp).all()
            assert persisted_booking.noshow_probability == second.get_json()['noshow_probability']
            assert len(predictions) == 2
            assert sum(prediction.is_active for prediction in predictions) == 1
        finally:
            session.close()
    finally:
        session = db_manager.get_session()
        try:
            session.query(Prediction).filter(
                Prediction.booking_id == uuid.UUID(booking_id)
            ).delete()
            session.query(Booking).filter(
                Booking.id == uuid.UUID(booking_id)
            ).delete()
            session.query(Passenger).filter(
                Passenger.id == passenger_id
            ).delete()
            session.query(Flight).filter(
                Flight.id == flight_id
            ).delete()
            session.commit()
        finally:
            session.close()
        client.application.prediction_service = original_service


def test_prediction_endpoints_fail_closed_without_approved_model(
    client, auth_headers, tmp_path
):
    from backend.app.services.prediction_service import PredictionService

    original_service = client.application.prediction_service
    unapproved_service = PredictionService(str(tmp_path))
    assert unapproved_service.load_all_models() == {'approved_model': False}
    client.application.prediction_service = unapproved_service
    try:
        single = client.post('/api/predict', json={
            'booking_data': {'ticket_price': 350},
        }, headers=auth_headers)
        assert single.status_code == 503
        assert single.get_json()['error'] == 'Prediction service unavailable'

        batch = client.post('/api/bulk-predict', json={
            'bookings': [{'ticket_price': 350}],
        }, headers=auth_headers)
        assert batch.status_code == 503
        assert batch.get_json()['error'] == 'Prediction service unavailable'

        arbitrary_model = client.post('/api/predict', json={
            'booking_data': {'ticket_price': 350},
            'model': '../../unapproved',
        }, headers=auth_headers)
        assert arbitrary_model.status_code == 400
    finally:
        client.application.prediction_service = original_service
