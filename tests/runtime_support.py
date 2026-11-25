import pytest
import uuid
from flask_jwt_extended import create_access_token


@pytest.fixture(scope='module')
def client():
    from backend.app import app

    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def auth_headers(client):
    from backend.app.services.auth_service import AuthService
    from database.connection import db_manager
    from sqlalchemy import text

    username = f'runtime-{uuid.uuid4().hex}'
    session = db_manager.get_session()
    try:
        user = AuthService(session).register_user(
            username=username,
            email=f'{username}@example.test',
            password='RuntimeTest!Password2026',
        )
        if not user.get('success'):
            raise RuntimeError(f'Could not create runtime test user: {user.get("error")}')
        user_id = user['user']['id']
    finally:
        session.close()

    with client.application.app_context():
        token = create_access_token(
            identity=user_id,
            additional_claims={'role': 'admin'},
        )
    try:
        yield {'Authorization': f'Bearer {token}'}
    finally:
        session = db_manager.get_session()
        try:
            session.execute(text('DELETE FROM users WHERE id = :user_id'), {'user_id': user_id})
            session.commit()
        finally:
            session.close()
