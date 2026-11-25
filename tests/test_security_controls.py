import pytest


def test_production_app_rejects_process_local_rate_limit_storage():
    from backend.app import create_app

    with pytest.raises(ValueError, match='shared rate-limit storage'):
        create_app({
            'FLASK_ENV': 'production',
            'RATELIMIT_STORAGE_URI': 'memory://',
        })

    with pytest.raises(ValueError, match='trusted proxy'):
        create_app({
            'FLASK_ENV': 'production',
            'RATELIMIT_STORAGE_URI': 'redis://localhost:6379/0',
            'TRUSTED_PROXY_HOPS': 0,
        })
