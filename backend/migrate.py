#!/usr/bin/env python3
"""Apply PostgreSQL schema files using the dedicated migration database role."""

import os
from pathlib import Path

import psycopg2
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / 'sql'
MIGRATIONS = (
    ('0001_schema', 'schema.sql'),
    ('0002_views', 'views.sql'),
    ('0003_analytics', 'analytics.sql'),
)
REQUIRED_LEGACY_TABLES = {
    'bookings', 'flights', 'passengers', 'predictions',
    'model_performance', 'monitoring_logs', 'users',
}


def _connect():
    return psycopg2.connect(
        host=os.environ['DB_HOST'],
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.environ['DB_NAME'],
        user=os.getenv('DB_USER', 'airline_migrator'),
        password=os.environ['DB_PASSWORD'],
    )


def apply_migrations(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS schema_migrations '
            '(version VARCHAR(100) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())'
        )
        cursor.execute('SELECT version FROM schema_migrations')
        applied = {row[0] for row in cursor.fetchall()}

        for version, filename in MIGRATIONS:
            if version in applied:
                continue
            if version == '0001_schema':
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                    "AND table_name <> 'schema_migrations'"
                )
                existing_tables = {row[0] for row in cursor.fetchall()}
                if existing_tables:
                    missing = REQUIRED_LEGACY_TABLES - existing_tables
                    if missing:
                        raise RuntimeError(
                            'Existing database has an incomplete schema; missing tables: '
                            + ', '.join(sorted(missing))
                        )
                else:
                    cursor.execute((SQL_DIR / filename).read_text(encoding='utf-8'))
            else:
                cursor.execute((SQL_DIR / filename).read_text(encoding='utf-8'))
            cursor.execute(
                'INSERT INTO schema_migrations (version) VALUES (%s)',
                (version,),
            )

        runtime_role = sql.Identifier(os.getenv('DB_RUNTIME_USER', 'airline_runtime'))
        cursor.execute(
            sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(runtime_role)
        )
        cursor.execute(
            sql.SQL('GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}').format(runtime_role)
        )
        cursor.execute(
            sql.SQL(
                'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE '
                'passengers, flights, bookings, booking_history, predictions, '
                'model_performance, monitoring_logs, users TO {}'
            ).format(runtime_role)
        )
        cursor.execute(
            sql.SQL('GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO {}')
            .format(runtime_role)
        )


def main():
    with _connect() as connection:
        apply_migrations(connection)
    print('PostgreSQL schema migrations completed.')


if __name__ == '__main__':
    main()
