"""
Prediction Service - Handles ML model operations with SHAP explainability
"""

import os
import sys
import json
import pickle
import time
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.overbooking_simulator import OverbookingSimulator, CostParameters

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = frozenset({
    'GradientBoosting',
    'LightGBM',
    'LogisticRegression',
    'RandomForest',
    'XGBoost',
})
SAFE_VERSION = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
REQUIRED_APPROVAL_GATES = frozenset({
    'roc_auc',
    'f1_score',
    'precision',
    'recall',
    'pr_auc_above_test_prevalence',
})
UNVERIFIED_HISTORY_FEATURES = frozenset({
    'total_bookings',
    'total_noshows',
    'total_cancellations',
    'total_flights',
    'historical_noshow_rate',
    'has_noshow_history',
    'high_noshow_risk',
    'very_high_noshow_risk',
    'has_cancellation_history',
    'cancellation_rate',
    'is_new_passenger',
    'is_frequent_flyer',
    'passenger_experience',
    'passenger_seniority',
    'simple_risk_score',
    'new_last_minute',
    'high_risk_new',
})


class ModelUnavailableError(RuntimeError):
    """Raised when no approved production model can serve predictions."""


class ModelNotApprovedError(ValueError):
    """Raised when a caller requests a model other than the approved model."""


class PredictionService:
    """
    Service for making and managing predictions with SHAP explainability.
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize prediction service.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        self.feature_names = None
        self.feature_names_by_model = {}
        self.explainers = {}
        self.approved_model = None
        self.default_model_name = None
    
    def load_model(self, model_name: str, version: str = None) -> bool:
        """Load only the artifact named by a valid production approval pointer."""
        if self.approved_model is None:
            self.load_all_models()
        if not self.approved_model:
            return False
        if model_name != self.approved_model['model_name']:
            return False
        if version is not None and version != self.approved_model['version']:
            return False
        return f"{self.approved_model['model_name']}_{self.approved_model['version']}" in self.models
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load the single explicitly approved artifact, or keep predictions disabled."""
        self.models.clear()
        self.scalers.clear()
        self.model_info.clear()
        self.feature_names_by_model.clear()
        self.feature_names = None
        self.approved_model = None
        self.default_model_name = None

        selection_path = self.model_dir / 'production_model.json'
        if not selection_path.is_file():
            logger.warning("No production model approval pointer exists; predictions are disabled")
            return {'approved_model': False}

        try:
            with selection_path.open('r', encoding='utf-8') as selection_file:
                selection = json.load(selection_file)
            if not self._is_approved_selection(selection):
                logger.warning("Production model pointer is missing valid approval gates")
                return {'approved_model': False}

            model_name = selection['model_name']
            version = selection['version']
            model_key = f'{model_name}_{version}'
            model_path = self.model_dir / f'{model_key}.pkl'
            scaler_path = self.model_dir / f'{model_key}_scaler.pkl'
            info_path = self.model_dir / f'{model_key}_info.json'
            features_path = self.model_dir / f'{model_key}_features.json'

            required_paths = (model_path, scaler_path, info_path, features_path)
            if any(not path.is_file() for path in required_paths):
                logger.error("Approved model %s is missing an artifact or metadata file", model_key)
                return {'approved_model': False}

            with info_path.open('r', encoding='utf-8') as info_file:
                info = json.load(info_file)
            with features_path.open('r', encoding='utf-8') as features_file:
                feature_names = json.load(features_file)
            if (
                not isinstance(info, dict)
                or info.get('model_name') != model_name
                or info.get('version') != version
                or info.get('experiment_id') != selection.get('experiment_id')
                or info.get('selected_for_serving') is not True
                or not isinstance(feature_names, list)
                or not feature_names
                or any(not isinstance(name, str) or not name for name in feature_names)
                or bool(UNVERIFIED_HISTORY_FEATURES.intersection(feature_names))
            ):
                logger.error("Approved model %s has mismatched approval metadata", model_key)
                return {'approved_model': False}

            with model_path.open('rb') as model_file:
                model = pickle.load(model_file)
            with scaler_path.open('rb') as scaler_file:
                scaler = pickle.load(scaler_file)
            if (
                not callable(getattr(model, 'predict', None))
                or not callable(getattr(model, 'predict_proba', None))
                or 1 not in list(getattr(model, 'classes_', []))
                or getattr(model, 'n_features_in_', len(feature_names)) != len(feature_names)
                or getattr(scaler, 'n_features_in_', len(feature_names)) != len(feature_names)
            ):
                logger.error("Approved model %s does not match its inference contract", model_key)
                return {'approved_model': False}

            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.model_info[model_key] = info
            self.feature_names_by_model[model_key] = feature_names
            self.feature_names = feature_names
            self.approved_model = {
                'model_name': model_name,
                'version': version,
            }
            self.default_model_name = model_name
            logger.info("Loaded approved model %s", model_key)
            return {'approved_model': True}
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.error("Could not validate production model approval: %s", error)
            return {'approved_model': False}
        except Exception:
            self.models.clear()
            self.scalers.clear()
            self.model_info.clear()
            self.feature_names_by_model.clear()
            self.feature_names = None
            self.approved_model = None
            self.default_model_name = None
            logger.exception("Could not load approved production model")
            return {'approved_model': False}

    @staticmethod
    def _is_approved_selection(selection: Any) -> bool:
        if not isinstance(selection, dict) or selection.get('status') != 'approved':
            return False
        model_name = selection.get('model_name')
        version = selection.get('version')
        quality_gate = selection.get('quality_gate')
        if (
            model_name not in SUPPORTED_MODELS
            or not isinstance(version, str)
            or SAFE_VERSION.fullmatch(version) is None
            or not isinstance(selection.get('experiment_id'), str)
            or not selection.get('experiment_id')
            or not isinstance(quality_gate, dict)
            or set(quality_gate) != REQUIRED_APPROVAL_GATES
        ):
            return False
        for check in quality_gate.values():
            if not isinstance(check, dict) or check.get('passed') is not True:
                return False
            observed = check.get('observed')
            required = check.get('required_strictly_greater_than')
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not np.isfinite(observed)
                or isinstance(required, bool)
                or not isinstance(required, (int, float))
                or not np.isfinite(required)
                or observed <= required
            ):
                return False
        return True

    def _get_approved_model_key(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        if not self.approved_model:
            raise ModelUnavailableError(
                'No approved production model is available; prediction is disabled'
            )
        approved_name = self.approved_model['model_name']
        approved_version = self.approved_model['version']
        if model_name is not None and model_name != approved_name:
            raise ModelNotApprovedError('Requested model is not approved for production')
        if version is not None and version != approved_version:
            raise ModelNotApprovedError('Requested model version is not approved for production')
        model_key = f'{approved_name}_{approved_version}'
        if model_key not in self.models:
            raise ModelUnavailableError('Approved production model artifact is not loaded')
        return model_key
    
    def _get_feature_vector(
        self,
        booking_data: Dict[str, Any],
        feature_names: List[str] = None,
        model_key: str = None,
    ) -> List[float]:
        """Create serving features with the same engineering used during training."""
        from ml.feature_engineering import engineer_features

        model_info = self.model_info.get(model_key, {}) if model_key else {}
        preprocessing = model_info.get('preprocessing', {})
        safe_booking_data = {
            key: value
            for key, value in booking_data.items()
            if key not in UNVERIFIED_HISTORY_FEATURES
        }
        engineered = engineer_features(
            pd.DataFrame([safe_booking_data]), preprocessing
        ).iloc[0]
        feature_names = feature_names or self.feature_names
        if not feature_names:
            raise ModelUnavailableError('Approved model has no serving feature contract')
        missing_features = [
            name for name in feature_names
            if name not in engineered.index and name not in safe_booking_data
        ]
        if missing_features:
            raise ValueError(
                'Missing required prediction inputs for model features: '
                + ', '.join(missing_features)
            )

        airport_categories = preprocessing.get('airport_categories', [])
        for column in ('departure_airport', 'arrival_airport'):
            if any(name.startswith(f'{column}_') for name in feature_names):
                airport = safe_booking_data.get(column)
                if airport not in airport_categories:
                    raise ValueError(
                        f'{column} must be one of the model training airport categories'
                    )

        features = []
        for fn in feature_names:
            value = engineered[fn] if fn in engineered.index else safe_booking_data[fn]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'Prediction feature {fn} must be numeric') from None
            if not np.isfinite(numeric_value):
                raise ValueError(f'Prediction feature {fn} must be finite')
            features.append(numeric_value)
        
        return features
    
    def _get_model_version(self, model_name: str, model_key: str = None) -> str:
        """Get the version for a loaded model."""
        if model_key and model_key.startswith(model_name + '_'):
            return model_key[len(model_name) + 1:]
        for key in self.models.keys():
            if key.startswith(model_name + '_'):
                return key.replace(model_name + '_', '')
        return 'unknown'
    
    def predict_single(self,
                       booking_data: Dict[str, Any],
                       model_name: str = None,
                       version: str = None,
                       calibrate: bool = True,
                       explain: bool = True) -> Dict[str, Any]:
        """
        Make a single prediction with optional SHAP explanation.
        
        Args:
            booking_data: Booking features dictionary
            model_name: Model to use
            version: Model version
            calibrate: Whether to use calibrated probabilities
            explain: Whether to include SHAP explanation
            
        Returns:
            Prediction result dictionary with explanation
        """
        start_time = time.time()
        model_name = self.default_model_name if model_name is None else model_name
        model_key = self._get_approved_model_key(model_name, version)
        model = self.models[model_key]
        
        # Prepare features
        if model_key not in self.feature_names_by_model:
            self._load_feature_names(model_key)
        
        feature_names = self.feature_names_by_model.get(model_key, self.feature_names)
        features = self._get_feature_vector(booking_data, feature_names, model_key)
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        scaler_key = model_key
        if scaler_key in self.scalers:
            X_scaled = self.scalers[scaler_key].transform(X)
        else:
            X_scaled = X
        
        # Make prediction
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # Run time
        run_time = int((time.time() - start_time) * 1000)
        
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Loaded model does not expose the no-show class (1)")
        noshow_prob = proba[classes.index(1)]
        
        # Risk category
        risk_category = self._get_risk_category(noshow_prob)
        
        # Model info
        model_info = self.model_info.get(model_key, {})
        
        result = {
            'booking_id': booking_data.get('booking_id'),
            'model_used': model_name,
            'model_version': self._get_model_version(model_name, model_key),
            'prediction': int(pred),
            'prediction_label': 'NO_SHOW' if pred == 1 else 'SHOW',
            'noshow_probability': round(float(noshow_prob), 4),
            'show_probability': round(float(1 - noshow_prob), 4),
            'risk_category': risk_category,
            'confidence': round(float(max(noshow_prob, 1 - noshow_prob)), 4),
            'run_time_ms': run_time,
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'model_accuracy': model_info.get('metrics', {}).get('accuracy') if model_info else None,
            'model_f1': model_info.get('metrics', {}).get('f1_score') if model_info else None
        }
        
        # Add SHAP explanation
        if explain and feature_names:
            try:
                explanation = self._get_shap_explanation(model, X_scaled[0], feature_names)
                result['explanation'] = explanation
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                result['explanation'] = None
        
        return result
    
    def _load_feature_names(self, model_key: str):
        """Read only the feature contract loaded with the approved artifact."""
        feature_names = self.feature_names_by_model.get(model_key)
        if not feature_names:
            raise ModelUnavailableError(
                f'Approved model {model_key} has no verified feature metadata'
            )
        self.feature_names = feature_names
    
    def _get_shap_explanation(self, model, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Get SHAP explanation for a prediction."""
        try:
            import shap
            
            # Get underlying model if wrapped
            underlying_model = self._get_underlying_model(model)
            
            # Determine model type for SHAP
            model_type = self._get_model_type(underlying_model)
            
            if model_type == 'tree':
                explainer = shap.TreeExplainer(underlying_model)
                shap_values = explainer.shap_values(features.reshape(1, -1))
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(underlying_model, features.reshape(1, -1))
                shap_values = explainer.shap_values(features.reshape(1, -1))
            else:
                background = shap.kmeans(pd.DataFrame([features]), 10)
                explainer = shap.KernelExplainer(underlying_model.predict_proba, background)
                shap_values = explainer.shap_values(features.reshape(1, -1))

            model_classes = list(getattr(underlying_model, 'classes_', [0, 1]))
            positive_class_index = model_classes.index(1) if 1 in model_classes else 0
            shap_values = self._positive_class_shap_values(shap_values, positive_class_index)
            contributions = shap_values[0].reshape(-1)
            shap_dict = dict(zip(feature_names, contributions))
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            
            positive_factors = [(k, v) for k, v in sorted_shap if v > 0][:5]
            negative_factors = [(k, v) for k, v in sorted_shap if v < 0][:5]
            
            expected_value = np.asarray(explainer.expected_value)
            base_value = float(expected_value.reshape(-1)[positive_class_index]) if expected_value.size > 1 else float(expected_value.reshape(-1)[0])
            return {
                'base_value': base_value,
                'positive_factors': [{'feature': k, 'value': round(float(v), 4)} for k, v in positive_factors],
                'negative_factors': [{'feature': k, 'value': round(float(v), 4)} for k, v in negative_factors],
                'all_contributions': [{'feature': k, 'shap_value': round(float(v), 4)} for k, v in sorted_shap[:15]],
                'model_type': model_type,
                'explanation_scope': 'uncalibrated_base_estimator'
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation error: {e}")
            return None

    @staticmethod
    def _positive_class_shap_values(shap_values, class_index: int) -> np.ndarray:
        """Normalize SHAP's supported binary-class output shapes."""
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[class_index])
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                values = values[:, :, class_index]
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f'Unexpected SHAP output shape: {values.shape}')
        return values
    
    def _get_model_type(self, model) -> str:
        """Determine model type for SHAP."""
        model_class = type(model).__name__.lower()
        
        # Handle CalibratedClassifierCV by accessing the underlying estimator
        if 'calibratedclassifiercv' in model_class:
            # Get the underlying estimator from the CalibratedClassifierCV wrapper
            if hasattr(model, 'estimator'):
                underlying = model.estimator
                underlying_class = type(underlying).__name__.lower()
                if any(x in underlying_class for x in ['forest', 'tree', 'xgb', 'lgb', 'gradient']):
                    return 'tree'
                elif any(x in underlying_class for x in ['logistic', 'linear', 'ridge', 'lasso']):
                    return 'linear'
            return 'tree'  # Default to tree if we can't determine
        
        if any(x in model_class for x in ['forest', 'tree', 'xgb', 'lgb', 'gradient']):
            return 'tree'
        elif any(x in model_class for x in ['logistic', 'linear', 'ridge', 'lasso']):
            return 'linear'
        else:
            return 'kernel'
    
    def _get_underlying_model(self, model):
        """
        Get the underlying model from wrapper classes like CalibratedClassifierCV.
        """
        if type(model).__name__ == 'ProbabilityCalibratedClassifier':
            return self._get_underlying_model(model.estimator)
        if type(model).__name__ == 'CalibratedClassifierCV':
            if hasattr(model, 'estimator'):
                return model.estimator
            calibrated_classifiers = getattr(model, 'calibrated_classifiers_', [])
            if calibrated_classifiers:
                calibrated = calibrated_classifiers[0]
                return getattr(calibrated, 'estimator', getattr(calibrated, 'base_estimator', model))
        return model
    
    def predict_batch(self,
                      bookings_data: List[Dict[str, Any]],
                      model_name: str = None,
                      version: str = None,
                      explain: bool = False) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            bookings_data: List of booking feature dictionaries
            model_name: Model to use
            version: Model version
            explain: Whether to include explanations (first 10 only)
            
        Returns:
            Batch prediction results
        """
        start_time = time.time()
        model_name = self.default_model_name if model_name is None else model_name
        model_key = self._get_approved_model_key(model_name, version)
        model = self.models[model_key]
        
        # Prepare batch features
        if model_key not in self.feature_names_by_model:
            self._load_feature_names(model_key)
        
        feature_names = self.feature_names_by_model.get(model_key, self.feature_names)
        features = np.array([
            self._get_feature_vector(b, feature_names, model_key)
            for b in bookings_data
        ])
        
        # Scale
        scaler_key = model_key
        if scaler_key in self.scalers:
            features = self.scalers[scaler_key].transform(features)
        
        # Make predictions
        preds = model.predict(features)
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Loaded model does not expose the no-show class (1)")
        probas = model.predict_proba(features)[:, classes.index(1)]
        
        run_time = time.time() - start_time
        
        results = []
        for i, (booking, pred, proba) in enumerate(zip(bookings_data, preds, probas)):
            result = {
                'booking_id': booking.get('booking_id', i),
                'booking_index': i,
                'prediction': int(pred),
                'prediction_label': 'NO_SHOW' if pred == 1 else 'SHOW',
                'noshow_probability': round(float(proba), 4),
                'risk_category': self._get_risk_category(proba)
            }
            
            if explain and i < 10:
                try:
                    result['explanation'] = self._get_shap_explanation(model, features[i], feature_names)
                except:
                    result['explanation'] = None
            
            results.append(result)
        
        return {
            'total_predictions': len(results),
            'predicted_noshows': int(sum(preds)),
            'predicted_shows': int(len(preds) - sum(preds)),
            'avg_noshow_probability': round(float(probas.mean()), 4),
            'high_risk_count': int((probas > 0.5).sum()),
            'predictions': results,
            'processing_time_seconds': round(run_time, 3)
        }
    
    def _get_risk_category(self, probability: float) -> str:
        """Convert probability to risk category."""
        if probability < 0.2:
            return 'LOW'
        elif probability < 0.4:
            return 'MODERATE'
        elif probability < 0.6:
            return 'ELEVATED'
        elif probability < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        models = []
        for key, info in self.model_info.items():
            parts = key.split('_')
            if len(parts) >= 2:
                model_name = parts[0]
                version = '_'.join(parts[1:])
            else:
                model_name = key
                version = 'unknown'
            models.append({
                'name': model_name,
                'version': version,
                'accuracy': info.get('metrics', {}).get('accuracy') if info else None,
                'f1_score': info.get('metrics', {}).get('f1_score') if info else None,
                'trained_at': info.get('trained_at') if info else None
            })
        return models

    def get_global_feature_importance(self, model_name: str = None) -> Dict[str, Any]:
        """Return mean absolute SHAP values on the checked-in training reference."""
        model_name = self.default_model_name if model_name is None else model_name
        model_key = self._get_approved_model_key(model_name)

        model = self._get_underlying_model(self.models[model_key])
        feature_names = self.feature_names_by_model.get(model_key) or self.feature_names
        if not feature_names:
            raise ValueError(f"Feature metadata does not match model {model_name}")

        reference_path = os.path.join(PROJECT_ROOT, 'data', 'training_data.csv')
        if not os.path.exists(reference_path):
            raise ValueError(f'Training reference data not found: {reference_path}')
        try:
            import shap
            from ml.feature_engineering import engineer_features
            from sklearn.preprocessing import LabelEncoder

            reference = pd.read_csv(reference_path)
            preprocessing = self.model_info.get(model_key, {}).get('preprocessing', {})
            reference = engineer_features(reference, preprocessing)
            for name in feature_names:
                if name not in reference.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(reference[name]):
                    encoder = LabelEncoder()
                    reference[name] = encoder.fit_transform(reference[name].fillna('UNKNOWN').astype(str))
            missing_features = [name for name in feature_names if name not in reference.columns]
            if missing_features:
                raise ValueError(f'Training reference is missing model features: {missing_features}')
            reference = reference[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)
            reference = reference.sample(n=min(128, len(reference)), random_state=42)
            if model_key in self.scalers:
                reference = self.scalers[model_key].transform(reference)
            if reference.shape[1] != len(feature_names):
                raise ValueError('Transformed training reference does not match model features')
            model_type = self._get_model_type(model)
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, reference)
            else:
                background = shap.kmeans(pd.DataFrame(reference), min(10, len(reference)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
            raw_values = explainer.shap_values(reference)
            classes = list(getattr(model, 'classes_', [0, 1]))
            class_index = classes.index(1) if 1 in classes else 0
            values = self._positive_class_shap_values(raw_values, class_index)
            importance = np.abs(values).mean(axis=0)
        except Exception as e:
            logger.warning(f'SHAP global importance failed for {model_name}: {e}')
            raise ValueError(f'SHAP global importance unavailable for {model_name}: {e}') from e

        total = float(np.sum(importance))
        if total > 0:
            importance = importance / total

        features = sorted(
            (
                {'feature': name, 'importance': float(value)}
                for name, value in zip(feature_names, importance)
            ),
            key=lambda feature: feature['importance'],
            reverse=True
        )
        return {
            'model': model_name,
            'version': self._get_model_version(model_name),
            'features': features
        }
    
    def simulate_overbooking(self,
                              capacity: int,
                              bookings: int,
                              avg_noshow_prob: float,
                              ticket_price: float,
                              max_overbooking: int = 15,
                              cost_params: Dict = None) -> Dict[str, Any]:
        """
        Run overbooking simulation.
        """
        params = CostParameters.from_dict(cost_params or {})
        simulator = OverbookingSimulator(params, random_seed=42)
        
        results = simulator.simulate_scenario(
            capacity=capacity,
            bookings=bookings,
            avg_noshow_prob=avg_noshow_prob,
            ticket_price=ticket_price,
            max_overbooking=max_overbooking
        )
        
        scenario_data = [r.to_dict() for r in results]
        recommendation = simulator.compare_scenarios(results, strategy='risk_adjusted')
        
        return {
            'scenarios': scenario_data,
            'recommendation': recommendation,
            'cost_parameters': params.to_dict(),
            'input_params': {
                'capacity': capacity,
                'bookings': bookings,
                'avg_noshow_prob': avg_noshow_prob,
                'ticket_price': ticket_price,
                'max_overbooking': max_overbooking
            }
        }


# Singleton instance
prediction_service = PredictionService()


if __name__ == "__main__":
    # Test prediction service
    service = PredictionService('models')
    
    print("Loading models...")
    service.load_all_models()
    print(f"Loaded {len(service.models)} models")
    
    # Test prediction
    test_data = {
        'booking_lead_time': 14,
        'flight_dayofweek': 1,
        'flight_hour': 10,
        'total_bookings': 5,
        'total_noshows': 1,
        'ticket_price': 350,
        'loyalty_tier_encoded': 0,
        'cabin_class_encoded': 0,
        'is_weekend': 0
    }
    
    result = service.predict_single(test_data, model_name='LightGBM', explain=True)
    print("\nPrediction Result:")
    print(f"  Prediction: {result['prediction_label']}")
    print(f"  No-Show Probability: {result['noshow_probability']:.2%}")
    print(f"  Risk Category: {result['risk_category']}")
    
    if result.get('explanation'):
        print(f"\n  Top Positive Factors:")
        for f in result['explanation']['positive_factors'][:3]:
            print(f"    - {f['feature']}: +{f['value']:.4f}")
        print(f"\n  Top Negative Factors:")
        for f in result['explanation']['negative_factors'][:3]:
            print(f"    - {f['feature']}: {f['value']:.4f}")