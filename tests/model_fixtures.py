import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def write_approved_test_model(model_dir, feature_names=None):
    """Create an isolated synthetic artifact with an explicit test approval pointer."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = 'RandomForest'
    version = 'test_2026'
    experiment_id = 'synthetic-test-only'
    feature_names = feature_names or ['ticket_price']

    training_values = np.array([[100], [200], [300], [400], [500], [600], [700], [800]])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(training_values, labels)
    scaler = StandardScaler().fit(training_values)
    model_path = model_dir / f'{model_name}_{version}.pkl'
    scaler_path = model_dir / f'{model_name}_{version}_scaler.pkl'
    with model_path.open('wb') as artifact:
        pickle.dump(model, artifact)
    with scaler_path.open('wb') as artifact:
        pickle.dump(scaler, artifact)

    (model_dir / f'{model_name}_{version}_features.json').write_text(
        json.dumps(feature_names),
        encoding='utf-8',
    )
    (model_dir / f'{model_name}_{version}_info.json').write_text(
        json.dumps({
            'model_name': model_name,
            'version': version,
            'experiment_id': experiment_id,
            'selected_for_serving': True,
            'preprocessing': {},
            'metrics': {'accuracy': 0.75, 'f1_score': 0.75},
        }),
        encoding='utf-8',
    )
    gates = {
        name: {
            'required_strictly_greater_than': 0.35,
            'observed': 0.75,
            'passed': True,
        }
        for name in (
            'roc_auc',
            'f1_score',
            'precision',
            'recall',
            'pr_auc_above_test_prevalence',
        )
    }
    (model_dir / 'production_model.json').write_text(
        json.dumps({
            'status': 'approved',
            'model_name': model_name,
            'version': version,
            'experiment_id': experiment_id,
            'quality_gate': gates,
        }),
        encoding='utf-8',
    )
    return model_name, version, model, scaler
