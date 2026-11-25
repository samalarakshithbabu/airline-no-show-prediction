import uuid
from datetime import datetime, timedelta

from tests.runtime_support import auth_headers, client


def test_analytics_reads_persisted_booking_and_reflects_update(client, auth_headers):
    from database.connection import db_manager
    from models.schemas import Booking, Flight, Passenger

    db_manager.init_schema()
    session = db_manager.get_session()
    passenger = Passenger(
        passenger_code=f'RT-{uuid.uuid4().hex[:16]}',
        first_name='Analytics',
        last_name='Passenger',
    )
    flight = Flight(
        flight_code=f'RT-{uuid.uuid4().hex[:16]}',
        departure_airport='SFO',
        arrival_airport='SEA',
        route='SFO-SEA',
        departure_time=datetime.utcnow() + timedelta(days=5),
        arrival_time=datetime.utcnow() + timedelta(days=5, hours=2),
        capacity=100,
        base_price=240,
    )
    booking = Booking(
        booking_code=f'RT-{uuid.uuid4().hex[:16]}',
        passenger=passenger,
        flight=flight,
        booking_date=datetime.utcnow(),
        booking_channel='MOBILE',
        booking_lead_time=5,
        ticket_price=240,
        booking_status='NOSHOW',
        noshow_probability=0.8,
    )
    session.add(booking)
    session.commit()
    booking_id = booking.id
    passenger_id = passenger.id
    flight_id = flight.id
    session.close()

    try:
        assert client.get('/api/analytics/overview').status_code == 401

        overview = client.get('/api/analytics/overview?days=30', headers=auth_headers)
        assert overview.status_code == 200, overview.get_json()
        assert overview.get_json()['total_bookings'] >= 1
        assert overview.get_json()['actual_noshows'] >= 1

        routes = client.get('/api/analytics/routes?route=SFO-SEA', headers=auth_headers)
        assert routes.status_code == 200
        assert any(row['route'] == 'SFO-SEA' for row in routes.get_json()['routes'])

        channels = client.get('/api/analytics/channels', headers=auth_headers)
        assert channels.status_code == 200
        assert any(row['channel'] == 'MOBILE' for row in channels.get_json()['channels'])

        trends = client.get('/api/analytics/trends?group_by=day&days=30', headers=auth_headers)
        assert trends.status_code == 200
        assert any(row['noshow_count'] >= 1 for row in trends.get_json()['trends'])

        session = db_manager.get_session()
        try:
            persisted_booking = session.get(Booking, booking_id)
            persisted_booking.booking_status = 'CONFIRMED'
            session.commit()
        finally:
            session.close()

        updated_overview = client.get('/api/analytics/overview?days=30', headers=auth_headers)
        assert updated_overview.status_code == 200
        assert updated_overview.get_json()['actual_noshows'] == 0
    finally:
        session = db_manager.get_session()
        try:
            session.query(Booking).filter(Booking.id == booking_id).delete()
            session.query(Passenger).filter(Passenger.id == passenger_id).delete()
            session.query(Flight).filter(Flight.id == flight_id).delete()
            session.commit()
        finally:
            session.close()


def test_high_risk_bookings_require_admin_role(client, auth_headers):
    from backend.app.services.auth_service import AuthService
    from database.connection import db_manager

    denied = client.get('/api/analytics/high-risk', headers=auth_headers)
    assert denied.status_code == 403

    admin_username = 'runtime-admin-' + uuid.uuid4().hex
    session = db_manager.get_session()
    try:
        admin = AuthService(session).register_user(
            username=admin_username,
            email=admin_username + '@example.test',
            password='RuntimeTest!Password2026',
            role='admin',
        )
        assert admin['success']
        admin_token = AuthService().create_access_token(
            user_id=admin['user']['id'],
            username=admin['user']['username'],
            role='admin',
        )
        authorized = client.get('/api/analytics/high-risk', headers={
            'Authorization': f'Bearer {admin_token}',
        })
        assert authorized.status_code == 200
    finally:
        from models.schemas import User

        cleanup = db_manager.get_session()
        try:
            cleanup.query(User).filter(User.username == admin_username).delete()
            cleanup.commit()
        finally:
            cleanup.close()
        session.close()
