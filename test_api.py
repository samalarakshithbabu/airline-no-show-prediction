"""API smoke tests, collected by pytest."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def client():
    from backend.app import create_app

    app = create_app({'TESTING': True})
    return app.test_client()


def test_health_check(client):
    response = client.get('/api/health')

    assert response.status_code == 200
    assert response.get_json()['status'] == 'healthy'


def test_registration_requires_request_data(client):
    response = client.post('/api/auth/register', json={})

    assert response.status_code == 400
    assert response.get_json()['error'] == 'No data provided'


def test_prediction_requires_authentication(client):
    response = client.post('/api/predict', json={'booking_data': {}})

    assert response.status_code == 401
