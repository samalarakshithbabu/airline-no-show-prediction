"""
Main ML Pipeline - End-to-end machine learning pipeline for no-show prediction
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str = None,
                 output_dir: str = 'models',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 calibrate: bool = True) -> Dict[str, any]:
    """
    Run the complete ML pipeline.
    
    Args:
        data_path: Path to data directory
        output_dir: Directory to save models
        test_size: Proportion for test split
        random_state: Random seed
        calibrate: Whether to calibrate probabilities
        
    Returns:
        Dictionary with pipeline results
    """
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("AIRLINE NO-SHOW PREDICTION - ML PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    logger.info("STEP 1: Loading and validating data...")
    from ml.data_processing import DataLoader, preprocess_data
    from ml.training import ModelTrainer
    from ml.evaluation import (
        ModelEvaluator,
        create_evaluation_report,
        evaluate_business_costs,
        evaluate_models,
        select_business_threshold,
    )
    from ml.overbooking_simulator import CostParameters
    from ml.training import ProbabilityCalibratedClassifier
    import pickle

    loader = DataLoader(data_path)
    df = loader.load_training_data()
    validation = loader.validate_dataset(df)
    if not validation['is_valid']:
        raise ValueError(f"Invalid training dataset: {validation['issues']}")
    logger.info("  Loaded %s rows and %s columns", len(df), len(df.columns))

    X_train, X_test, y_train, y_test, preprocessing = preprocess_data(
        df, test_size=test_size, random_state=random_state
    )
    logger.info(
        "  Train: %s rows (%s no-shows); test: %s rows (%s no-shows); features: %s",
        len(X_train), int(y_train.sum()), len(X_test), int(y_test.sum()), X_train.shape[1]
    )

    if 'passenger_id' in df.columns:
        groups = df.loc[X_train.index, 'passenger_id']
    else:
        groups = pd.Series(X_train.index, index=X_train.index)
    cv_folds = min(5, int(groups.nunique()))
    if cv_folds < 2:
        raise ValueError('At least two independent passenger groups are required for model validation')

    logger.info("STEP 2: Comparing models with %s-fold passenger-grouped CV...", cv_folds)
    trainer = ModelTrainer(random_state=random_state)
    cross_validation = trainer.cross_validate_grouped(
        X_train,
        y_train,
        groups,
        cv=cv_folds,
    )
    cv_results = cross_validation['results']
    if not cv_results:
        raise RuntimeError(f"No models passed grouped cross-validation: {cross_validation['failures']}")

    logger.info("STEP 3: Fitting models and calibrating from out-of-fold predictions...")
    trained = trainer.train_all(
        X_train,
        y_train,
        model_names=list(cv_results),
        calibrate=False,
    )
    if not trained:
        raise RuntimeError("No models trained successfully")

    for model_name, result in trained.items():
        if model_name not in cv_results:
            continue
        if calibrate:
            from sklearn.isotonic import IsotonicRegression
            oof_probabilities = cv_results[model_name]['out_of_fold_probabilities']
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(oof_probabilities, y_train.to_numpy())
            result['model'] = ProbabilityCalibratedClassifier(result['model'], calibrator)
            result['calibrated'] = True
            result['calibration_method'] = 'isotonic_passenger_group_oof'

    logger.info("STEP 4: Evaluating held-out predictions...")
    evaluations = evaluate_models(trained, X_test, y_test, trainer.scaler)
    comparison_rows = []
    for name, evaluation in evaluations.items():
        if name not in cv_results:
            continue
        comparison_rows.append({
            'model_name': name,
            **evaluation.get_metrics(),
            'cv_roc_auc': cv_results[name]['roc_auc'],
            'cv_pr_auc': cv_results[name]['pr_auc'],
            'cv_brier_score': cv_results[name]['brier_score'],
            'cv_f1_threshold': cv_results[name]['f1_threshold'],
            'cv_f1_at_threshold': cv_results[name]['f1_at_threshold'],
        })
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ['cv_pr_auc', 'cv_roc_auc', 'cv_brier_score'],
        ascending=[False, False, True],
    )
    if comparison.empty:
        raise RuntimeError("No trained models produced evaluation results")
    best_model_name = str(comparison.iloc[0]['model_name'])
    best_oof_probabilities = np.asarray(
        cv_results[best_model_name]['out_of_fold_probabilities'],
        dtype=float,
    )
    if calibrate:
        best_oof_probabilities = trained[best_model_name]['model'].calibrator.predict(
            best_oof_probabilities
        )
    from sklearn.metrics import f1_score
    thresholds = np.unique(np.concatenate(([0.0], best_oof_probabilities, [1.0])))
    f1_values = [
        f1_score(y_train.to_numpy(), best_oof_probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    selected_threshold = float(thresholds[int(np.argmax(f1_values))])
    best_test_model = trained[best_model_name]['model']
    test_probabilities = best_test_model.predict_proba(
        trainer.scaler.transform(X_test)
    )[:, 1]
    best_test_at_cv_threshold = ModelEvaluator(
        y_test.to_numpy(),
        (test_probabilities >= selected_threshold).astype(int),
        test_probabilities,
    ).get_metrics()
    costs = CostParameters()
    false_positive_costs = sorted({
        0.0,
        costs.empty_seat_cost,
        costs.compensation_cost,
        costs.denied_boarding_cost,
    })
    business_cost_analysis = {
        'missed_no_show_cost_assumption': costs.empty_seat_cost,
        'false_positive_action_cost_assumptions': false_positive_costs,
        'source': 'configurable simulator defaults; action costs are sensitivity scenarios, not measured interventions',
        'operational_threshold_recommended': False,
        'scenarios': [],
    }
    for false_positive_cost in false_positive_costs:
        threshold_result = select_business_threshold(
            y_train.to_numpy(),
            best_oof_probabilities,
            costs.empty_seat_cost,
            false_positive_cost,
        )
        test_scenario_cost = evaluate_business_costs(
            y_test.to_numpy(),
            test_probabilities,
            threshold_result['threshold'],
            costs.empty_seat_cost,
            false_positive_cost,
        )
        test_default_threshold_cost = evaluate_business_costs(
            y_test.to_numpy(),
            test_probabilities,
            0.5,
            costs.empty_seat_cost,
            false_positive_cost,
        )
        business_cost_analysis['scenarios'].append({
            'false_positive_action_cost': false_positive_cost,
            'selected_oof_threshold': threshold_result['threshold'],
            'oof_cost': threshold_result['total_cost'],
            'oof_cost_per_booking': threshold_result['cost_per_booking'],
            'test_cost_at_oof_threshold': test_scenario_cost['total_cost'],
            'test_cost_per_booking_at_oof_threshold': test_scenario_cost['cost_per_booking'],
            'test_cost_at_0_5': test_default_threshold_cost['total_cost'],
            'test_cost_difference_vs_0_5': (
                test_default_threshold_cost['total_cost']
                - test_scenario_cost['total_cost']
            ),
            'test_false_positives': test_scenario_cost['false_positives'],
            'test_false_negatives': test_scenario_cost['false_negatives'],
        })

    acceptance_thresholds = {
        'roc_auc': 0.70,
        'f1_score': 0.50,
        'precision': 0.50,
        'recall': 0.50,
    }
    production_gate = {
        metric: {
            'required_strictly_greater_than': threshold,
            'observed': float(best_test_at_cv_threshold[metric]),
            'passed': float(best_test_at_cv_threshold[metric]) > threshold,
        }
        for metric, threshold in acceptance_thresholds.items()
    }
    production_gate['pr_auc_above_test_prevalence'] = {
        'required_strictly_greater_than': float(best_test_at_cv_threshold['prevalence']),
        'observed': float(best_test_at_cv_threshold['pr_auc']),
        'passed': bool(
            best_test_at_cv_threshold['pr_auc']
            > best_test_at_cv_threshold['prevalence']
        ),
    }
    production_approved = bool(all(
        check['passed'] for check in production_gate.values()
    ))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    experiment_id = version
    for model_name, result in trained.items():
        if model_name not in evaluations:
            continue
        model = result['model']
        metrics = evaluations[model_name].get_metrics()
        with (output_path / f'{model_name}_{version}.pkl').open('wb') as model_file:
            pickle.dump(model, model_file)
        with (output_path / f'{model_name}_{version}_scaler.pkl').open('wb') as scaler_file:
            pickle.dump(trainer.scaler, scaler_file)
        with (output_path / f'{model_name}_{version}_features.json').open('w') as features_file:
            json.dump(preprocessing['feature_columns'], features_file)
        info = {
            'model_name': model_name,
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'calibrated': bool(result.get('calibrated')),
            'calibration_method': result.get('calibration_method'),
            'class_balance_method': result.get('balance_method'),
            'metrics': {key: float(value) for key, value in metrics.items()},
            'cross_validation_metrics': {
                key: value for key, value in cv_results[model_name].items()
                if key not in ('out_of_fold_probabilities', 'cv_folds')
            },
            'threshold_selection': {
                'metric': 'f1',
                'threshold': (
                    selected_threshold
                    if model_name == best_model_name
                    else float(cv_results[model_name]['f1_threshold'])
                ),
                'source': (
                    'calibrated out-of-fold passenger-grouped training predictions'
                    if calibrate and model_name == best_model_name
                    else 'out-of-fold passenger-grouped training predictions'
                ),
                'for_operational_use': False,
            },
            'preprocessing': preprocessing,
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'training_class_counts': {str(key): int(value) for key, value in y_train.value_counts().items()},
            'experiment_id': experiment_id,
            'random_state': int(random_state),
            'selected_for_serving': production_approved and model_name == best_model_name,
            'model_selection_metric': 'passenger_grouped_cv_pr_auc',
            'data_sha256': hashlib.sha256(loader.data_path.read_bytes()).hexdigest(),
            'feature_importance': (
                {
                    feature: float(value)
                    for feature, value in zip(
                        preprocessing['feature_columns'],
                        result['feature_importance'],
                    )
                }
                if result.get('feature_importance') is not None
                else {}
            ),
        }
        with (output_path / f'{model_name}_{version}_info.json').open('w') as info_file:
            json.dump(info, info_file, indent=2)

    production_model = {
        'status': 'approved' if production_approved else 'not_approved',
        'model_name': best_model_name if production_approved else None,
        'version': version if production_approved else None,
        'experiment_id': experiment_id,
        'selection_metric': 'passenger_grouped_cv_pr_auc',
        'quality_gate': production_gate,
        'selected_threshold_is_advisory_only': selected_threshold,
    }
    (output_path / 'production_model.json').write_text(
        json.dumps(production_model, indent=2),
        encoding='utf-8',
    )

    comparison.to_csv(output_path / 'model_comparison.csv', index=False)
    comparison_history_path = output_path / 'model_comparison_history.csv'
    comparison.assign(experiment_id=experiment_id).to_csv(
        comparison_history_path,
        mode='a',
        header=not comparison_history_path.exists(),
        index=False,
    )
    experiment = {
        'experiment_id': experiment_id,
        'trained_at': datetime.now().isoformat(),
        'dataset_sha256': hashlib.sha256(loader.data_path.read_bytes()).hexdigest(),
        'random_state': int(random_state),
        'split_method': preprocessing['split_method'],
        'cv_folds': cv_folds,
        'training_rows': int(len(y_train)),
        'test_rows': int(len(y_test)),
        'best_model': best_model_name,
        'production_approved': production_approved,
        'production_quality_gate': production_gate,
        'cv_metrics': {
            key: value for key, value in cv_results[best_model_name].items()
            if key not in ('out_of_fold_probabilities', 'cv_folds')
        },
        'test_metrics_at_cv_f1_threshold': best_test_at_cv_threshold,
    }
    with (output_path / 'experiment_history.jsonl').open('a', encoding='utf-8') as history_file:
        history_file.write(json.dumps(experiment, default=str) + '\n')
    (output_path / 'business_cost_sensitivity.json').write_text(
        json.dumps(business_cost_analysis, indent=2),
        encoding='utf-8',
    )
    report = create_evaluation_report(evaluations)
    (output_path / 'evaluation_report.txt').write_text(report, encoding='utf-8')
    logger.info("Best candidate by grouped CV PR-AUC: %s", best_model_name)
    logger.info(
        "Production model approval: %s",
        'APPROVED' if production_approved else 'WITHHELD by held-out quality gates',
    )
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.2f} minutes")
    logger.info(f"Best model: {best_model_name}")
    
    best_row = comparison[comparison['model_name'] == best_model_name].iloc[0]
    logger.info(f"Best ROC-AUC: {best_row['roc_auc']:.4f}")
    logger.info(f"Best F1-Score: {best_row['f1_score']:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("")
    
    # Return summary
    return {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_minutes': duration,
        'best_model': best_model_name,
        'production_approved': production_approved,
        'production_quality_gate': production_gate,
        'best_roc_auc': float(best_row['roc_auc']),
        'best_f1': float(best_row['f1_score']),
        'selected_threshold_from_oof_f1': selected_threshold,
        'test_metrics_at_oof_f1_threshold': best_test_at_cv_threshold,
        'business_cost_sensitivity': business_cost_analysis,
        'cross_validation_failures': cross_validation['failures'],
        'cross_validation': {
            name: {
                key: value for key, value in metrics.items()
                if key not in ('out_of_fold_probabilities', 'cv_folds')
            }
            for name, metrics in cv_results.items()
        },
        'comparison': comparison.to_dict('records'),
        'models_trained': len(trained),
        'output_dir': output_dir
    }


def quick_predict(booking_data: Dict[str, any],
                  model_name: str = 'LightGBM',
                  model_dir: str = 'models') -> Dict[str, any]:
    """
    Make a quick prediction using saved models.
    
    Args:
        booking_data: Booking features dictionary
        model_name: Model to use
        model_dir: Directory with saved models
        
    Returns:
        Prediction result
    """
    import pickle
    import json
    from ml.feature_engineering import engineer_features
    
    # Find model file
    model_files = [f for f in os.listdir(model_dir) 
                   if f.startswith(model_name) and f.endswith('.pkl') 
                   and 'scaler' not in f and 'features' not in f]
    
    if not model_files:
        raise ValueError(f"No model found for {model_name}")
    
    # Get latest version
    model_files.sort()
    model_file = model_files[-1]
    
    version = model_file.replace('.pkl', '').replace(f'{model_name}_', '')
    
    # Load model
    with open(os.path.join(model_dir, model_file), 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    features_file = os.path.join(model_dir, f'{model_name}_{version}_features.json')
    with open(features_file, 'r') as f:
        feature_names = json.load(f)
    
    info_path = os.path.join(model_dir, f'{model_name}_{version}_info.json')
    with open(info_path, 'r') as info_file:
        info = json.load(info_file)
    features = engineer_features(pd.DataFrame([booking_data]), info.get('preprocessing', {}))
    X = features.reindex(columns=feature_names, fill_value=0).apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)
    scaler_path = os.path.join(model_dir, f'{model_name}_{version}_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as scaler_file:
            X = pickle.load(scaler_file).transform(X)
    
    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    # Determine risk
    noshow_prob = proba[1]
    if noshow_prob < 0.2:
        risk = 'LOW'
    elif noshow_prob < 0.4:
        risk = 'MODERATE'
    elif noshow_prob < 0.6:
        risk = 'ELEVATED'
    elif noshow_prob < 0.8:
        risk = 'HIGH'
    else:
        risk = 'CRITICAL'
    
    return {
        'prediction': int(pred),
        'prediction_label': 'NO_SHOW' if pred == 1 else 'SHOW',
        'noshow_probability': round(float(noshow_prob), 4),
        'show_probability': round(float(proba[0]), 4),
        'risk_category': risk,
        'model_used': model_name,
        'model_version': version
    }


def main():
    """Main entry point for command line."""
    parser = argparse.ArgumentParser(description='Airline No-Show Prediction Pipeline')
    parser.add_argument('--data', type=str, default=None, help='Data directory path')
    parser.add_argument('--output', type=str, default='models', help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--no-calibrate', action='store_true', help='Skip probability calibration')
    parser.add_argument('--predict', type=str, default=None, help='Make prediction (JSON string)')
    
    args = parser.parse_args()
    
    if args.predict:
        # Quick prediction mode
        booking_data = json.loads(args.predict)
        result = quick_predict(booking_data)
        print(json.dumps(result, indent=2))
    else:
        # Full pipeline
        result = run_pipeline(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            calibrate=not args.no_calibrate
        )
        
        # Save pipeline summary
        summary_file = os.path.join(args.output, 'pipeline_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nPipeline summary saved to {summary_file}")


if __name__ == "__main__":
    main()


@dataclass
class MLPipelineConfig:
    """Configuration for ML Pipeline."""
    data_path: Optional[str] = None
    output_dir: str = 'models'
    test_size: float = 0.2
    random_state: int = 42
    calibrate: bool = True
    model_names: Optional[List[str]] = None


class MLPipeline:
    """
    Main ML Pipeline class for end-to-end model management.
    
    Provides methods for:
    - Data loading and preprocessing
    - Model training and evaluation
    - Predictions and explanations
    - Model comparison and selection
    """
    
    def __init__(self, config: MLPipelineConfig = None):
        """
        Initialize ML Pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or MLPipelineConfig()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_groups = None
        self.preprocessing = None
        self.trainer = None
        self.models = {}
        self.comparison_df = None
        self.best_model = None
        self.best_model_name = None
        self.cross_validation = {}
        self.selected_threshold = None
    
    def load_and_preprocess(self, data_path: str = None) -> Dict[str, Any]:
        """Load and preprocess data."""
        from ml.data_processing import DataLoader, preprocess_data
        
        data_path = data_path or self.config.data_path
        loader = DataLoader(data_path)
        df = loader.load_training_data()
        validation = loader.validate_dataset(df)
        if not validation['is_valid']:
            raise ValueError(f"Invalid training dataset: {validation['issues']}")
        result = preprocess_data(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessing = result
        if 'passenger_id' in df.columns:
            self.train_groups = df.loc[self.X_train.index, 'passenger_id']
        else:
            self.train_groups = pd.Series(self.X_train.index, index=self.X_train.index)
        return result
    
    def train(self, calibrate: bool = None) -> pd.DataFrame:
        """Train all models."""
        from ml.training import ModelTrainer
        from ml.evaluation import evaluate_models
        
        if self.X_train is None:
            self.load_and_preprocess()
        
        calibrate = calibrate if calibrate is not None else self.config.calibrate
        
        self.trainer = ModelTrainer(random_state=self.config.random_state)
        cv_folds = min(5, int(self.train_groups.nunique()))
        if cv_folds < 2:
            raise ValueError('At least two independent passenger groups are required for model validation')
        self.cross_validation = self.trainer.cross_validate_grouped(
            self.X_train,
            self.y_train,
            self.train_groups,
            model_names=self.config.model_names,
            cv=cv_folds,
        )
        cv_results = self.cross_validation['results']
        if not cv_results:
            raise RuntimeError(
                f"No models passed grouped cross-validation: {self.cross_validation['failures']}"
            )
        self.models = self.trainer.train_all(
            self.X_train, self.y_train,
            model_names=list(cv_results),
            calibrate=False,
        )
        if calibrate:
            from sklearn.isotonic import IsotonicRegression
            from ml.training import ProbabilityCalibratedClassifier

            for name, result in self.models.items():
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(
                    cv_results[name]['out_of_fold_probabilities'],
                    self.y_train.to_numpy(),
                )
                result['model'] = ProbabilityCalibratedClassifier(result['model'], calibrator)
                result['calibrated'] = True
                result['calibration_method'] = 'isotonic_passenger_group_oof'
        evaluations = evaluate_models(
            self.models, self.X_test, self.y_test, self.trainer.scaler
        )
        self.comparison_df = pd.DataFrame([
            {
                'model_name': name,
                **evaluation.get_metrics(),
                'cv_roc_auc': cv_results[name]['roc_auc'],
                'cv_pr_auc': cv_results[name]['pr_auc'],
                'cv_brier_score': cv_results[name]['brier_score'],
                'cv_f1_threshold': cv_results[name]['f1_threshold'],
            }
            for name, evaluation in evaluations.items()
            if name in cv_results
        ]).sort_values(
            ['cv_pr_auc', 'cv_roc_auc', 'cv_brier_score'],
            ascending=[False, False, True],
        )
        
        if self.comparison_df is not None and len(self.comparison_df) > 0:
            self.best_model_name = self.comparison_df.iloc[0]['model_name']
            self.best_model = self.models[self.best_model_name]['model']
            self.selected_threshold = float(
                cv_results[self.best_model_name]['f1_threshold']
            )
        
        return self.comparison_df
    
    def predict(self,
                booking_data: Dict[str, Any],
                model_name: str = None) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self.models:
            raise ValueError("No models trained. Call train() first.")
        
        model_name = model_name or self.best_model_name
        model_info = self.models.get(model_name)
        
        if model_info is None:
            raise ValueError(f"Model {model_name} not found")
        model = model_info['model']
        
        from ml.feature_engineering import engineer_features

        features = engineer_features(
            pd.DataFrame([booking_data]), self.preprocessing or {}
        )
        feature_names = self.preprocessing['feature_columns']
        X = features.reindex(columns=feature_names, fill_value=0).apply(
            pd.to_numeric, errors='coerce'
        ).fillna(0)
        X_scaled = self.trainer.scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # Risk category
        noshow_prob = proba[1]
        if noshow_prob < 0.2:
            risk = 'LOW'
        elif noshow_prob < 0.4:
            risk = 'MODERATE'
        elif noshow_prob < 0.6:
            risk = 'ELEVATED'
        elif noshow_prob < 0.8:
            risk = 'HIGH'
        else:
            risk = 'CRITICAL'
        
        return {
            'prediction': int(pred),
            'prediction_label': 'NO_SHOW' if pred == 1 else 'SHOW',
            'noshow_probability': round(float(noshow_prob), 4),
            'show_probability': round(float(proba[0]), 4),
            'risk_category': risk,
            'model_used': model_name
        }
    
    def get_comparison(self) -> pd.DataFrame:
        """Get model comparison DataFrame."""
        return self.comparison_df
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get best model info."""
        if self.best_model_name is None:
            return None
        
        return {
            'name': self.best_model_name,
            'model': self.best_model,
            'metrics': self.comparison_df[
                self.comparison_df['model_name'] == self.best_model_name
            ].iloc[0].to_dict() if self.comparison_df is not None else None
        }
    
    def run_full_pipeline(self,
                          data_path: str = None,
                          output_dir: str = None,
                          test_size: float = None) -> Dict[str, Any]:
        """Run the complete ML pipeline."""
        data_path = data_path or self.config.data_path
        output_dir = output_dir or self.config.output_dir
        test_size = test_size or self.config.test_size
        
        # Load and preprocess
        self.load_and_preprocess(data_path)
        
        # Train models
        self.train()
        
        # Get comparison
        comparison = self.get_comparison()
        
        if comparison is not None and len(comparison) > 0:
            best = self.get_best_model()
            best_metrics = best['metrics'] if best and best['metrics'] else {}
            
            return {
                'status': 'success',
                'best_model': best['name'] if best else None,
                'best_roc_auc': best_metrics.get('roc_auc', 0),
                'best_f1': best_metrics.get('f1_score', 0),
                'models_trained': len(self.models),
                'training_samples': len(self.y_train),
                'test_samples': len(self.y_test),
                'comparison': comparison.to_dict('records')
            }
        
        return {'status': 'error', 'message': 'Training failed'}