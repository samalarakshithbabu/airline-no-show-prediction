import uuid

from tests.runtime_support import client


def test_registration_login_profile_and_invalid_auth(client):
    from database.connection import db_manager
    from models.schemas import User

    suffix = uuid.uuid4().hex
    username = f'runtime-{suffix}'
    email = f'{suffix}@example.test'
    password = 'RuntimeTest!Password2026'

    try:
        rejected = client.post('/api/auth/register', json={
            'username': username,
            'email': email,
            'password': 'short',
        })
        assert rejected.status_code == 400

        registered = client.post('/api/auth/register', json={
            'username': username,
            'email': email,
            'password': password,
        })
        assert registered.status_code == 201
        user_id = registered.get_json()['user']['id']

        denied = client.get('/api/auth/profile')
        assert denied.status_code == 401

        login = client.post('/api/auth/login', json={
            'username': username,
            'password': password,
        })
        assert login.status_code == 200
        assert 'token' not in login.get_json()
        access_cookie = client.get_cookie('access_token_cookie', path='/api')
        csrf_cookie = client.get_cookie('csrf_access_token')
        assert access_cookie is not None and access_cookie.http_only
        assert csrf_cookie is not None

        profile = client.get('/api/auth/profile')
        assert profile.status_code == 200
        assert profile.get_json()['user']['id'] == user_id

        csrf_denied = client.post('/api/auth/logout')
        assert csrf_denied.status_code == 401
        logged_out = client.post('/api/auth/logout', headers={
            'X-CSRF-TOKEN': csrf_cookie.value,
        })
        assert logged_out.status_code == 200
        assert client.get('/api/auth/profile').status_code == 401

        session = db_manager.get_session()
        try:
            persisted = session.get(User, uuid.UUID(user_id))
            assert persisted is not None
            assert persisted.username == username
        finally:
            session.close()
    finally:
        session = db_manager.get_session()
        try:
            session.query(User).filter(User.username == username).delete()
            session.commit()
        finally:
            session.close()


def test_registration_can_be_disabled(client):
    previous_value = client.application.config['ALLOW_PUBLIC_REGISTRATION']
    client.application.config['ALLOW_PUBLIC_REGISTRATION'] = False
    try:
        response = client.post('/api/auth/register', json={
            'username': f'disabled-{uuid.uuid4().hex[:12]}',
            'email': f'{uuid.uuid4().hex}@example.test',
            'password': 'RuntimeTest!Password2026',
        })
        assert response.status_code == 403
        assert response.get_json()['error'] == 'Registration disabled'
    finally:
        client.application.config['ALLOW_PUBLIC_REGISTRATION'] = previous_value
