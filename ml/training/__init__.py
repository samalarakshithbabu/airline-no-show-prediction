"""
Model Training Module for Airline No-Show Prediction
Implements multiple ML algorithms with class imbalance handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import warnings
import logging

logger = logging.getLogger(__name__)

# Try to import imbalanced-learn, make it optional
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    SMOTE = None
    RandomUnderSampler = None


class ModelTrainer:
    """
    Main training class for no-show prediction models.
    Handles multiple algorithms, class imbalance, and hyperparameter tuning.
    """
    
    def __init__(self, 
                 config: Dict = None,
                 random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration
            random_state: Random seed for reproducibility
        """
        self.config = config or {}
        self.random_state = random_state
        self.models = {}
        self.failures = {}
        self.scaler = StandardScaler()
        self.class_weight = None
        
        # Default model configurations
        self.model_configs = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced',
                    'random_state': random_state,
                    'n_jobs': -1
                }
            },
            'XGBoost': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': None,
                    'random_state': random_state,
                    'eval_metric': 'logloss',
                    'verbosity': 0
                }
            },
            'LightGBM': {
                'class': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'class_weight': 'balanced',
                    'random_state': random_state,
                    'verbose': -1
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': random_state
                }
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'params': {
                    'max_iter': 1000,
                    'class_weight': 'balanced',
                    'random_state': random_state,
                    'solver': 'lbfgs'
                }
            }
        }
        
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        class_counts = np.bincount(y)
        
        weights = {i: n_samples / (n_classes * class_counts[i]) for i in range(n_classes)}
        return weights
    
    def handle_class_imbalance(self, 
                               X_train: pd.DataFrame, 
                               y_train: pd.Series,
                               method: str = 'weights') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using various techniques.
        """
        minority_ratio = y_train.value_counts(normalize=True).min()
        logger.info(f"Minority class ratio: {minority_ratio:.2%}")
        
        if method == 'smote' and SMOTE is not None:
            try:
                sampler = SMOTE(
                    sampling_strategy=0.5,
                    random_state=self.random_state,
                    k_neighbors=5
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                logger.info(f"SMOTE applied: {len(y_train)} -> {len(y_balanced)}")
                return X_balanced, y_balanced
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}, falling back to weights")
        
        elif method == 'undersampling' and RandomUnderSampler is not None:
            try:
                sampler = RandomUnderSampler(
                    sampling_strategy=0.5,
                    random_state=self.random_state
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                return X_balanced, y_balanced
            except Exception as e:
                logger.warning(f"Undersampling failed: {e}")
        
        # Default: use class weights (no resampling)
        logger.info("Using class weights for imbalance handling")
        return X_train, y_train
    
    def optimize_threshold(self, 
                           model, 
                           X_val: pd.DataFrame, 
                           y_val: pd.Series,
                           metric: str = 'f1') -> float:
        """
        Find optimal classification threshold.
        """
        proba = model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.3, 0.7, 0.01):
            pred = (proba >= threshold).astype(int)
            
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_val, pred, zero_division=0)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_val, pred, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_val, pred, zero_division=0)
            elif metric == 'youden':
                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} ({metric}: {best_score:.4f})")
        return best_threshold
    
    def create_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params_override: Dict[str, Any] = None,
    ):
        """
        Create and return a model instance.
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        params = config['params'].copy()
        if params_override:
            params.update(params_override)
        
        # Adjust for class imbalance if needed
        minority_count = (y_train == 1).sum()
        majority_count = (y_train == 0).sum()
        
        if 'scale_pos_weight' in params and minority_count > 0:
            params['scale_pos_weight'] = max(1, majority_count / minority_count)
        
        # Create model instance
        model = config['class'](**params)
        
        return model
    
    def train_model(self, 
                    model_name: str, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    balance_method: str = 'weights',
                    calibrate: bool = False) -> Dict[str, Any]:
        """
        Train a single model.
        """
        logger.info(f"Training {model_name}...")
        
        # Balance classes if needed
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train, y_train, method=balance_method
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        
        # Create and train model
        model = self.create_model(model_name, X_train_balanced, y_train_balanced)
        if model_name == 'GradientBoosting':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight('balanced', y_train_balanced)
        else:
            sample_weight = None
        
        calibration_method = None
        if calibrate and model_name == 'XGBoost':
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import train_test_split

            X_fit, X_calibration, y_fit, y_calibration = train_test_split(
                X_train_scaled,
                y_train_balanced,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train_balanced,
            )
            model.fit(X_fit, y_fit)
            raw_probabilities = model.predict_proba(X_calibration)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(raw_probabilities, y_calibration)
            model = ProbabilityCalibratedClassifier(model, calibrator)
            calibration_method = 'isotonic_holdout'
        else:
            if calibrate and hasattr(model, 'predict_proba'):
                model = CalibratedClassifierCV(model, method='isotonic', cv=5)
                calibration_method = 'isotonic_cv5'
            model.fit(X_train_scaled, y_train_balanced, sample_weight=sample_weight)
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            importance = None
        
        return {
            'name': model_name,
            'model': model,
            'feature_importance': importance,
            'trained': True,
            'calibrated': calibrate,
            'calibration_method': calibration_method,
            'balance_method': balance_method
        }
    
    def train_all(self, 
                  X_train: pd.DataFrame, 
                  y_train: pd.Series,
                  model_names: List[str] = None,
                  balance_method: str = 'weights',
                  calibrate: bool = True) -> Dict[str, Dict]:
        """
        Train all specified models.
        """
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        self.models = {}
        self.failures = {}
        
        for model_name in model_names:
            try:
                model_info = self.train_model(
                    model_name, X_train, y_train,
                    balance_method=balance_method,
                    calibrate=calibrate
                )
                self.models[model_name] = model_info
                logger.info(f"✓ {model_name} trained successfully")
            except Exception as e:
                self.failures[model_name] = str(e)
                logger.error(f"✗ {model_name} training failed: {e}")
        
        return self.models
    
    def get_predictions(self, 
                        model_name: str, 
                        X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from a trained model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]['model']
        X_scaled = self.scaler.transform(X)
        
        pred = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[:, 1]
        
        return pred, proba
    
    def cross_validate(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       model_name: str,
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.
        """
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
        
        model_info = self.create_model(model_name, X, y)
        
        # Define scorers
        scorers = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        
        cv_results = cross_validate(
            model_info, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scorers,
            return_train_score=True
        )
        
        return {
            'test_accuracy': cv_results['test_accuracy'].mean(),
            'test_precision': cv_results['test_precision'].mean(),
            'test_recall': cv_results['test_recall'].mean(),
            'test_f1': cv_results['test_f1'].mean()
        }

    def cross_validate_grouped(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        model_names: List[str] = None,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate candidates with out-of-fold predictions and passenger-safe folds."""
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler

        if model_names is None:
            model_names = list(self.model_configs)
        if len(X) != len(y) or len(y) != len(groups):
            raise ValueError('X, y, and groups must have the same number of rows')
        if groups.nunique() < cv:
            raise ValueError(f'At least {cv} unique passenger groups are required')

        splitter = StratifiedGroupKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state,
        )
        results = {}
        failures = {}

        for model_name in model_names:
            if model_name == 'LogisticRegression':
                parameter_candidates = [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}]
            elif model_name == 'RandomForest':
                parameter_candidates = [
                    {'n_estimators': 100, 'max_depth': 6},
                    {'n_estimators': 200, 'max_depth': 10},
                ]
            elif model_name == 'GradientBoosting':
                parameter_candidates = [
                    {'n_estimators': 100, 'max_depth': 3},
                    {'n_estimators': 150, 'max_depth': 5},
                ]
            elif model_name == 'XGBoost':
                parameter_candidates = [
                    {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05},
                    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
                ]
            elif model_name == 'LightGBM':
                parameter_candidates = [
                    {'n_estimators': 150, 'num_leaves': 15, 'learning_rate': 0.05},
                    {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05},
                ]
            else:
                parameter_candidates = [{}]

            candidate_results = []
            try:
                for params in parameter_candidates:
                    out_of_fold = np.full(len(y), np.nan, dtype=float)
                    fold_metrics = []
                    for train_indices, validation_indices in splitter.split(X, y, groups):
                        X_fold_train = X.iloc[train_indices]
                        y_fold_train = y.iloc[train_indices]
                        X_fold_validation = X.iloc[validation_indices]
                        y_fold_validation = y.iloc[validation_indices]

                        scaler = StandardScaler()
                        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                        X_fold_validation_scaled = scaler.transform(X_fold_validation)
                        model = self.create_model(
                            model_name,
                            X_fold_train,
                            y_fold_train,
                            params_override=params,
                        )
                        if model_name == 'GradientBoosting':
                            from sklearn.utils.class_weight import compute_sample_weight
                            sample_weight = compute_sample_weight('balanced', y_fold_train)
                            model.fit(
                                X_fold_train_scaled,
                                y_fold_train,
                                sample_weight=sample_weight,
                            )
                        else:
                            model.fit(X_fold_train_scaled, y_fold_train)

                        positive_index = list(model.classes_).index(1)
                        probabilities = model.predict_proba(X_fold_validation_scaled)[:, positive_index]
                        out_of_fold[validation_indices] = probabilities
                        fold_metrics.append({
                            'roc_auc': float(roc_auc_score(y_fold_validation, probabilities)),
                            'pr_auc': float(average_precision_score(y_fold_validation, probabilities)),
                            'brier_score': float(brier_score_loss(y_fold_validation, probabilities)),
                        })

                    if np.isnan(out_of_fold).any():
                        raise RuntimeError(f'{model_name} did not predict every row in grouped CV')
                    candidate_results.append({
                        'params': params,
                        'out_of_fold_probabilities': out_of_fold,
                        'fold_metrics': fold_metrics,
                        'roc_auc': float(roc_auc_score(y, out_of_fold)),
                        'pr_auc': float(average_precision_score(y, out_of_fold)),
                        'brier_score': float(brier_score_loss(y, out_of_fold)),
                    })

                if not candidate_results:
                    raise RuntimeError(f'{model_name} produced no valid hyperparameter candidates')

                selected = max(
                    candidate_results,
                    key=lambda candidate: (
                        candidate['pr_auc'],
                        candidate['roc_auc'],
                        -candidate['brier_score'],
                    ),
                )
                self.model_configs[model_name]['params'].update(selected['params'])
                out_of_fold = selected['out_of_fold_probabilities']
                thresholds = np.unique(np.concatenate(([0.0], out_of_fold, [1.0])))
                f1_values = [
                    f1_score(y, out_of_fold >= threshold, zero_division=0)
                    for threshold in thresholds
                ]
                best_index = int(np.argmax(f1_values))
                results[model_name] = {
                    'roc_auc': float(roc_auc_score(y, out_of_fold)),
                    'pr_auc': float(average_precision_score(y, out_of_fold)),
                    'brier_score': float(brier_score_loss(y, out_of_fold)),
                    'precision': float(precision_score(y, out_of_fold >= 0.5, zero_division=0)),
                    'recall': float(recall_score(y, out_of_fold >= 0.5, zero_division=0)),
                    'f1_score': float(f1_score(y, out_of_fold >= 0.5, zero_division=0)),
                    'f1_threshold': float(thresholds[best_index]),
                    'f1_at_threshold': float(f1_values[best_index]),
                    'selected_hyperparameters': selected['params'],
                    'hyperparameter_candidates': [
                        {
                            'params': candidate['params'],
                            'pr_auc': candidate['pr_auc'],
                            'roc_auc': candidate['roc_auc'],
                            'brier_score': candidate['brier_score'],
                        }
                        for candidate in candidate_results
                    ],
                    'cv_folds': selected['fold_metrics'],
                    'out_of_fold_probabilities': out_of_fold.tolist(),
                }
            except Exception as exc:
                failures[model_name] = str(exc)
                logger.exception('Grouped CV failed for %s', model_name)

        return {'results': results, 'failures': failures}
    
    def get_feature_importance_summary(self, 
                                        feature_names: List[str]) -> pd.DataFrame:
        """
        Get aggregated feature importance across all models.
        """
        importance_df = pd.DataFrame({
            'feature': feature_names
        })
        
        for model_name, model_info in self.models.items():
            importance = model_info.get('feature_importance')
            if importance is not None:
                importance_df[model_name] = importance
        
        # Compute mean importance
        model_cols = [col for col in importance_df.columns if col != 'feature']
        if model_cols:
            importance_df['mean_importance'] = importance_df[model_cols].mean(axis=1)
            importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        return importance_df


def train_models(X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 config: Dict = None) -> ModelTrainer:
    """
    Convenience function to train all models.
    """
    trainer = ModelTrainer(config)
    trainer.train_all(X_train, y_train)
    return trainer


class ProbabilityCalibratedClassifier:
    """Apply a fitted binary probability calibrator to an estimator."""

    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = estimator.classes_

    def predict_proba(self, X):
        positive_index = list(self.classes_).index(1)
        raw_probability = self.estimator.predict_proba(X)[:, positive_index]
        calibrated = np.clip(self.calibrator.predict(raw_probability), 0.0, 1.0)
        return np.column_stack((1.0 - calibrated, calibrated))

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return self.classes_[(probabilities >= 0.5).astype(int)]