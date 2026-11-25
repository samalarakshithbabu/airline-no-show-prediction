from sqlalchemy import create_engine, inspect, text


def test_legacy_primary_key_migration_preserves_rows_and_foreign_keys():
    from backend.app.database.connection import db_manager

    test_engine = create_engine('sqlite:///:memory:')
    with test_engine.begin() as connection:
        connection.execute(text(
            'CREATE TABLE passengers (id TEXT PRIMARY KEY, passenger_code TEXT)'
        ))
        connection.execute(text(
            'CREATE TABLE flights (id TEXT PRIMARY KEY, flight_code TEXT)'
        ))
        connection.execute(text(
            'CREATE TABLE bookings (id TEXT PRIMARY KEY, passenger_id TEXT '
            'REFERENCES passengers(id), flight_id TEXT REFERENCES flights(id))'
        ))
        connection.execute(text(
            "INSERT INTO passengers (id, passenger_code) VALUES ('p-1', 'P-1')"
        ))
        connection.execute(text(
            "INSERT INTO flights (id, flight_code) VALUES ('f-1', 'F-1')"
        ))
        connection.execute(text(
            "INSERT INTO bookings (id, passenger_id, flight_id) "
            "VALUES ('b-1', 'p-1', 'f-1')"
        ))

    db_manager.migrate_legacy_primary_keys(test_engine)

    inspector = inspect(test_engine)
    assert 'passenger_id' in {column['name'] for column in inspector.get_columns('passengers')}
    assert 'flight_id' in {column['name'] for column in inspector.get_columns('flights')}
    assert 'booking_id' in {column['name'] for column in inspector.get_columns('bookings')}
    with test_engine.connect() as connection:
        assert connection.execute(text(
            'SELECT passenger_id FROM passengers'
        )).scalar_one() == 'p-1'
        assert connection.execute(text(
            'SELECT booking_id FROM bookings'
        )).scalar_one() == 'b-1'
        foreign_keys = connection.execute(
            text('PRAGMA foreign_key_list(bookings)')
        ).mappings().all()
    assert {row['to'] for row in foreign_keys} == {'passenger_id', 'flight_id'}
