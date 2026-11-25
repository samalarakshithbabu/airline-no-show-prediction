#!/usr/bin/env python3
"""Create the first administrator for an empty application database."""

import getpass
import sys
from pathlib import Path

from sqlalchemy import inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend' / 'app'))

from database.connection import db_manager
from services.auth_service import AuthService
from config import AUTO_CREATE_SCHEMA


def main() -> None:
    username = input('Admin username: ').strip()
    email = input('Admin email: ').strip()
    password = getpass.getpass('Admin password (12-72 UTF-8 bytes): ')
    if len(username) < 3 or '@' not in email:
        raise SystemExit('Provide a username of at least 3 characters and a valid email.')
    if not 12 <= len(password.encode('utf-8')) <= 72:
        raise SystemExit('Password must be 12 to 72 UTF-8 bytes long.')

    if AUTO_CREATE_SCHEMA:
        db_manager.init_schema()
    elif not inspect(db_manager.engine).has_table('users'):
        raise SystemExit('Database schema is missing; run backend/migrate.py before creating an administrator.')
    session = db_manager.get_session()
    try:
        user_count = session.execute(text('SELECT COUNT(*) FROM users')).scalar_one()
        if user_count:
            raise SystemExit('Initial admin bootstrap is only allowed when no users exist.')

        result = AuthService(session).admin_create_user(username, email, password)
        if not result.get('success'):
            raise SystemExit(result.get('error', 'Admin creation failed.'))
        print('Initial administrator created.')
    finally:
        session.close()


if __name__ == '__main__':
    main()
