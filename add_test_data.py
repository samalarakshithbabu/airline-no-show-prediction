"""Add test data and test drift monitoring"""
from backend.app.database.connection import db_manager
from sqlalchemy import text

session = db_manager.get_session()

# Add more bookings to meet minimum threshold
insert_sql = '''
    INSERT INTO bookings (id, booking_code, passenger_id, flight_id, booking_date, booking_channel, booking_lead_time, ticket_price, cabin_class, booking_status, noshow_probability, created_at)
    VALUES 
        ('b44444444-4444-4444-4444-444444444444', 'BK000000004', 'p11111111-1111-1111-1111-111111111111', 'f11111111-1111-1111-1111-111111111111', datetime('now', '-2 days'), 'WEB', 5, 320, 'ECONOMY', 'CONFIRMED', 0.38, datetime('now')),
        ('b55555555-5555-5555-5555-555555555555', 'BK000000005', 'p22222222-2222-2222-2222-222222222222', 'f11111111-1111-1111-1111-111111111111', datetime('now', '-4 days'), 'AGENT', 10, 450, 'BUSINESS', 'CONFIRMED', 0.25, datetime('now')),
        ('b66666666-6666-6666-6666-666666666666', 'BK000000006', 'p11111111-1111-1111-1111-111111111111', 'f22222222-2222-2222-2222-222222222222', datetime('now', '-1 days'), 'MOBILE', 2, 290, 'ECONOMY', 'CONFIRMED', 0.52, datetime('now')),
        ('b77777777-7777-7777-7777-777777777777', 'BK000000007', 'p22222222-2222-2222-2222-222222222222', 'f11111111-1111-1111-1111-111111111111', datetime('now', '-6 days'), 'WEB', 15, 380, 'PREMIUM_ECONOMY', 'CONFIRMED', 0.22, datetime('now')),
        ('b88888888-8888-8888-8888-888888888888', 'BK000000008', 'p11111111-1111-1111-1111-111111111111', 'f22222222-2222-2222-2222-222222222222', datetime('now', '-7 days'), 'CALL_CENTER', 20, 520, 'BUSINESS', 'CONFIRMED', 0.18, datetime('now')),
        ('b99999999-9999-9999-9999-999999999999', 'BK000000009', 'p22222222-2222-2222-2222-222222222222', 'f11111111-1111-1111-1111-111111111111', datetime('now', '-8 days'), 'WEB', 12, 340, 'ECONOMY', 'CONFIRMED', 0.32, datetime('now')),
        ('b00000000-0000-0000-0000-000000000000', 'BK000000010', 'p11111111-1111-1111-1111-111111111111', 'f22222222-2222-2222-2222-222222222222', datetime('now', '-9 days'), 'MOBILE', 8, 410, 'BUSINESS', 'CONFIRMED', 0.27, datetime('now'))
'''

session.execute(text(insert_sql))
session.commit()

# Check counts
result = session.execute(text('SELECT COUNT(*) FROM bookings')).fetchone()
print(f'Bookings after insert: {result[0]}')

# Test drift monitoring
from backend.app.services.drift_service import DriftService
service = DriftService(db=session)
report = service.get_drift_status()

print(f'\nDrift Status: {report.status}')
print(f'Features Analyzed: {len(report.feature_drift)}')
if report.feature_drift:
    print('\nFeature Drift Results:')
    for feat in report.feature_drift:
        print(f"  {feat['feature_name']}: PSI={feat['current_value']:.4f} [{feat['severity']}]")
print(f'Data Info: {report.data_info}')
print(f'Prediction Drift: {report.prediction_drift}')

session.close()