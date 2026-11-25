import json
import pickle

import numpy as np
import pytest

from tests.model_fixtures import write_approved_test_model


def test_unapproved_models_never_load_or_predict(tmp_path, monkeypatch):
    from backend.app.services.prediction_service import (
        ModelUnavailableError,
        PredictionService,
    )

    (tmp_path / 'LightGBM_legacy.pkl').write_bytes(b'unapproved artifact')
    (tmp_path / 'LightGBM_legacy_scaler.pkl').write_bytes(b'unapproved scaler')
    service = PredictionService(str(tmp_path))

    def reject_pickle_load(_file):
        raise AssertionError('unapproved pickle must never be deserialized')

    monkeypatch.setattr(pickle, 'load', reject_pickle_load)
    assert service.load_all_models() == {'approved_model': False}
    assert not service.load_model('LightGBM')
    assert service.default_model_name is None
    with pytest.raises(ModelUnavailableError):
        service.predict_single({'ticket_price': 350}, explain=False)
    with pytest.raises(ModelUnavailableError):
        service.predict_batch([{'ticket_price': 350}])


def test_unapproved_pointer_does_not_deserialize_artifacts(tmp_path, monkeypatch):
    from backend.app.services.prediction_service import PredictionService

    (tmp_path / 'LightGBM_legacy.pkl').write_bytes(b'unapproved artifact')
    (tmp_path / 'production_model.json').write_text(json.dumps({
        'status': 'not_approved',
        'model_name': 'LightGBM',
        'version': 'legacy',
    }), encoding='utf-8')
    service = PredictionService(str(tmp_path))

    def reject_pickle_load(_file):
        raise AssertionError('unapproved pickle must never be deserialized')

    monkeypatch.setattr(pickle, 'load', reject_pickle_load)
    assert service.load_all_models() == {'approved_model': False}
    assert service.models == {}


def test_model_metadata_is_checked_before_deserialization(tmp_path, monkeypatch):
    from backend.app.services.prediction_service import PredictionService

    write_approved_test_model(tmp_path)
    info_path = tmp_path / 'RandomForest_test_2026_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    info['selected_for_serving'] = False
    info_path.write_text(json.dumps(info), encoding='utf-8')
    service = PredictionService(str(tmp_path))

    def reject_pickle_load(_file):
        raise AssertionError('invalid approval metadata must not deserialize pickle')

    monkeypatch.setattr(pickle, 'load', reject_pickle_load)
    assert service.load_all_models() == {'approved_model': False}
    assert service.models == {}


def test_unverified_history_features_cannot_be_approved_for_serving(tmp_path, monkeypatch):
    from backend.app.services.prediction_service import PredictionService

    write_approved_test_model(tmp_path, feature_names=['total_noshows'])
    service = PredictionService(str(tmp_path))

    def reject_pickle_load(_file):
        raise AssertionError('history-leaking metadata must not deserialize pickle')

    monkeypatch.setattr(pickle, 'load', reject_pickle_load)
    assert service.load_all_models() == {'approved_model': False}
    assert service.models == {}


def test_missing_prediction_time_features_are_not_silently_defaulted(tmp_path):
    from backend.app.services.prediction_service import PredictionService
    from tests.model_fixtures import write_approved_test_model

    model_name, version, _, _ = write_approved_test_model(
        tmp_path, feature_names=['flight_month']
    )
    service = PredictionService(str(tmp_path))
    assert service.load_all_models() == {'approved_model': True}

    with pytest.raises(ValueError, match='Missing required prediction inputs'):
        service.predict_single(
            {'ticket_price': 350},
            model_name=model_name,
            version=version,
            explain=False,
        )


def test_prediction_form_fields_cover_active_training_feature_contract(tmp_path):
    import pandas as pd
    from pathlib import Path
    from backend.app.services.prediction_service import PredictionService
    from ml.data_processing import preprocess_data

    training_data = pd.read_csv(
        Path(__file__).resolve().parents[1] / 'data' / 'training_data.csv'
    )
    _, _, _, _, preprocessing = preprocess_data(training_data)
    service = PredictionService(str(tmp_path))
    service.model_info['test_model'] = {'preprocessing': preprocessing}
    booking = {
        'booking_date': '2026-11-22',
        'flight_date': '2026-12-06',
        'departure_airport': 'JFK',
        'arrival_airport': 'LAX',
        'booking_channel_encoded': 0,
        'cabin_class_encoded': 0,
        'ticket_price': 350,
        'loyalty_tier_encoded': 1,
    }

    features = service._get_feature_vector(
        booking, preprocessing['feature_columns'], 'test_model'
    )

    assert len(features) == len(preprocessing['feature_columns'])
    assert all(np.isfinite(value) for value in features)


def test_arbitrary_model_identifier_in_pointer_is_rejected_before_deserialization(
    tmp_path, monkeypatch
):
    from backend.app.services.prediction_service import PredictionService

    write_approved_test_model(tmp_path)
    pointer_path = tmp_path / 'production_model.json'
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    pointer['model_name'] = '../../LightGBM'
    pointer_path.write_text(json.dumps(pointer), encoding='utf-8')
    service = PredictionService(str(tmp_path))

    def reject_pickle_load(_file):
        raise AssertionError('unsupported model identifiers must not deserialize pickle')

    monkeypatch.setattr(pickle, 'load', reject_pickle_load)
    assert service.load_all_models() == {'approved_model': False}
    assert service.models == {}


def test_approved_model_serving_matches_direct_inference_and_ignores_history(
    tmp_path,
):
    from backend.app.services.prediction_service import (
        ModelNotApprovedError,
        PredictionService,
    )

    model_name, version, model, scaler = write_approved_test_model(tmp_path)
    service = PredictionService(str(tmp_path))
    assert service.load_all_models() == {'approved_model': True}

    booking = {'ticket_price': 350, 'total_bookings': 100, 'total_noshows': 99}
    feature_vector = np.asarray(
        service._get_feature_vector(booking, ['ticket_price'], f'{model_name}_{version}')
    ).reshape(1, -1)
    result = service.predict_single(
        booking, model_name=model_name, version=version, explain=False
    )
    expected_probability = model.predict_proba(scaler.transform(feature_vector))[0][1]
    without_history = service.predict_single(
        {'ticket_price': 350}, model_name=model_name, version=version, explain=False
    )

    assert result['noshow_probability'] == round(float(expected_probability), 4)
    assert result['noshow_probability'] == without_history['noshow_probability']
    with pytest.raises(ModelNotApprovedError):
        service.predict_single({'ticket_price': 350}, model_name='../../LightGBM', explain=False)
    with pytest.raises(ModelNotApprovedError):
        service.predict_batch([{'ticket_price': 350}], version='other')
