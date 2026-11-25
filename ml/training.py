"""
Model Training Module - Train and compare multiple ML models
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss
)
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and compare multiple classification models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.feature_names = None
        
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model name -> model instance
        """
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1  # Handle class imbalance
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1,
                class_weight='balanced'
            ),
            'HistGradientBoosting': HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_model(self,
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    calibrate: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model with optional probability calibration.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        models = self.get_models()
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = models[model_name]
        
        logger.info(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Calibrate probabilities if requested
        if calibrate:
            # Use isotonic calibration for better probability estimates
            calibrated_model = CalibratedClassifierCV(
                model, 
                method='isotonic',
                cv=5,
                ensemble=True
            )
            calibrated_model.fit(X_train, y_train)
            model = calibrated_model
            logger.info(f"  Probability calibration applied")
        
        return model
    
    def evaluate_model(self,
                       model: Any,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       model_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
        
        try:
            pr_auc = average_precision_score(y_test, y_proba)
        except:
            pr_auc = 0.0
        
        # Brier score (probability calibration quality)
        try:
            brier = brier_score_loss(y_test, y_proba)
        except:
            brier = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'model_name': model_name,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4),
            'specificity': round(specificity, 4),
            'brier_score': round(brier, 4),
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return metrics
    
    def cross_validate_model(self,
                              model: Any,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to evaluate
            X_train: Training features
            y_train: Training labels
            cv: Number of folds
            
        Returns:
            Dictionary of CV scores
        """
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1')
        cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
        
        return {
            'cv_accuracy_mean': round(cv_accuracy.mean(), 4),
            'cv_accuracy_std': round(cv_accuracy.std(), 4),
            'cv_f1_mean': round(cv_f1.mean(), 4),
            'cv_f1_std': round(cv_f1.std(), 4),
            'cv_roc_auc_mean': round(cv_roc_auc.mean(), 4),
            'cv_roc_auc_std': round(cv_roc_auc.std(), 4)
        }
    
    def train_all(self,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  calibrate: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary of results for each model
        """
        self.feature_names = list(X_train.columns)
        models = self.get_models()
        results = {}
        
        for model_name in models:
            try:
                # Train model
                model = self.train_model(model_name, X_train, y_train, calibrate)
                
                # Evaluate on test set
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                
                # Cross-validation
                cv_metrics = self.cross_validate_model(model, X_train, y_train)
                metrics.update(cv_metrics)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                logger.info(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, "
                           f"F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
                
                # Track best model (using ROC-AUC as primary metric)
                if metrics['roc_auc'] > self.best_score:
                    self.best_score = metrics['roc_auc']
                    self.best_model = model_name
                
            except Exception as e:
                logger.error(f"  Error training {model_name}: {e}")
                results[model_name] = {
                    'model': None,
                    'error': str(e)
                }
        
        self.results = results
        logger.info(f"Best model: {self.best_model} (ROC-AUC={self.best_score:.4f})")
        
        return results
    
    def get_comparison_df(self) -> pd.DataFrame:
        """
        Get model comparison as DataFrame.
        
        Returns:
            DataFrame with all model metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for model_name, result in self.results.items():
            if 'metrics' in result:
                row = result['metrics'].copy()
                row['model_name'] = model_name
                comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('roc_auc', ascending=False)
        
        return df
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of feature -> importance
        """
        if model_name not in self.results:
            return {}
        
        result = self.results[model_name]
        if result.get('model') is None:
            return {}
        
        model = result['model']
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'best_iteration') and hasattr(model, 'feature_importances_'):
            # LightGBM specific
            importances = model.feature_importances_
        else:
            return {}
        
        # Create mapping
        if self.feature_names is not None and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances))
        
        return dict(zip(range(len(importances)), importances))


def train_models(X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 output_dir: str = 'models',
                 calibrate: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """
    Train all models and save the best one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save models
        calibrate: Whether to calibrate probabilities
        
    Returns:
        Tuple of (trainer results, comparison DataFrame)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all(X_train, y_train, X_test, y_test, calibrate)
    
    # Get comparison
    comparison_df = trainer.get_comparison_df()
    
    # Save all models and their artifacts
    timestamp = datetime.now().strftime('%Y%m%d')
    
    for model_name, result in results.items():
        if result.get('model') is None:
            continue
        
        model = result['model']
        metrics = result['metrics']
        
        # Model file
        model_file = os.path.join(output_dir, f'{model_name}_{timestamp}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names
        features_file = os.path.join(output_dir, f'{model_name}_{timestamp}_features.json')
        with open(features_file, 'w') as f:
            json.dump(trainer.feature_names, f)
        
        # Save model info/metrics
        info = {
            'model_name': model_name,
            'version': timestamp,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'class_distribution': {
                'train': {
                    '0': int((y_train == 0).sum()),
                    '1': int((y_train == 1).sum())
                },
                'minority_pct': round((y_train == 1).mean(), 3)
            }
        }
        
        info_file = os.path.join(output_dir, f'{model_name}_{timestamp}_info.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Saved {model_name} to {model_file}")
    
    return results, comparison_df


if __name__ == "__main__":
    # Demo training
    print("=" * 60)
    print("MODEL TRAINING DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Load data
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ml.data_processing import load_and_preprocess
    from ml.feature_engineering import create_features
    
    # Load and preprocess
    print("Loading data...")
    X_train, X_test, y_train, y_test, preprocessing = load_and_preprocess()
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Class distribution: {dict(y_train.value_counts())}")
    print()
    
    # Train models
    print("Training models...")
    results, comparison = train_models(X_train, y_train, X_test, y_test, output_dir='models')
    print()
    
    # Show results
    print("Model Comparison:")
    print(comparison[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))
    print()
    print(f"Best model: {results[0] if isinstance(results, list) else 'See comparison above'}")