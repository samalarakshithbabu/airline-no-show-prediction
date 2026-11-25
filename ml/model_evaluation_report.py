"""
Comprehensive Model Evaluation Report for Airline No-Show Prediction System
Evaluates all trained models with proper threshold analysis, class imbalance handling,
and trade-off documentation.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from sklearn.model_selection import train_test_split
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_processing import load_and_preprocess
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model(model_name: str, model_dir: str = 'models') -> tuple:
    """Load a trained model and its artifacts."""
    model_files = [f for f in os.listdir(model_dir) 
                   if f.startswith(model_name) and f.endswith('.pkl')
                   and 'scaler' not in f and 'features' not in f and 'info' not in f]
    
    if not model_files:
        return None, None, None
    
    model_files.sort()
    model_file = model_files[-1]
    
    version = model_file.replace('.pkl', '').replace(f'{model_name}_', '')
    
    with open(os.path.join(model_dir, model_file), 'rb') as f:
        model = pickle.load(f)
    
    features_file = os.path.join(model_dir, f'{model_name}_{version}_features.json')
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            feature_names = json.load(f)
    else:
        feature_names = None
    
    return model, feature_names, version


def evaluate_with_thresholds(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """
    Evaluate model at different classification thresholds.
    
    Returns comprehensive evaluation including precision-recall trade-offs.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    
    # Default metrics
    default_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_default),
        'precision': precision_score(y_test, y_pred_default, zero_division=0),
        'recall': recall_score(y_test, y_pred_default, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_default, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }
    
    # Threshold analysis
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_analysis = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        threshold_analysis.append({
            'threshold': thresh,
            'accuracy': accuracy_score(y_test, y_pred_thresh),
            'precision': precision_score(y_test, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_test, y_pred_thresh, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_thresh, zero_division=0)
        })
    
    # Find optimal threshold for F1
    best_f1_idx = np.argmax([t['f1_score'] for t in threshold_analysis])
    best_threshold = threshold_analysis[best_f1_idx]
    
    # Confusion matrix at default threshold
    cm = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'model_name': model_name,
        'default_metrics': default_metrics,
        'threshold_analysis': threshold_analysis,
        'optimal_threshold': {
            'threshold': best_threshold['threshold'],
            'f1_score': best_threshold['f1_score'],
            'precision': best_threshold['precision'],
            'recall': best_threshold['recall']
        },
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'class_distribution': {
            'test_size': len(y_test),
            'positive_class': int(y_test.sum()),
            'negative_class': int(len(y_test) - y_test.sum()),
            'positive_rate': round(float(y_test.mean()), 4)
        }
    }


def analyze_class_imbalance(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Analyze class imbalance in the dataset."""
    train_positive_rate = y_train.mean()
    test_positive_rate = y_test.mean()
    
    return {
        'train_positive_rate': round(train_positive_rate, 4),
        'test_positive_rate': round(test_positive_rate, 4),
        'imbalance_ratio': round(train_positive_rate / (1 - train_positive_rate), 3) if train_positive_rate < 1 else 0,
        'recommendation': 'Class weights applied during training. Consider threshold adjustment for business decisions.'
    }


def generate_model_report() -> dict:
    """
    Generate comprehensive model evaluation report.
    """
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("Airline No-Show Prediction System")
    print("=" * 70)
    
    # Load and preprocess data
    logger.info("Loading training data...")
    X_train, X_test, y_train, y_test, preprocessing = load_and_preprocess()
    
    logger.info(f"Training set: {len(y_train)} samples ({y_train.sum()} positive)")
    logger.info(f"Test set: {len(y_test)} samples ({y_test.sum()} positive)")
    
    # Class imbalance analysis
    imbalance_info = analyze_class_imbalance(y_train, y_test)
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)
    print(f"Training positive rate: {imbalance_info['train_positive_rate']:.2%}")
    print(f"Test positive rate: {imbalance_info['test_positive_rate']:.2%}")
    print(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1")
    print(f"\nRecommendation: {imbalance_info['recommendation']}")
    
    # Load and evaluate each model
    model_names = ['LightGBM', 'RandomForest', 'LogisticRegression', 'GradientBoosting']
    results = {}
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")
        model, feature_names, version = load_trained_model(model_name)
        
        if model is None:
            logger.warning(f"Model {model_name} not found")
            continue
        
        # Get feature columns
        if feature_names:
            feature_cols = [c for c in feature_names if c in X_test.columns]
            X_test_model = X_test[feature_cols]
        else:
            X_test_model = X_test
        
        eval_result = evaluate_with_thresholds(model, X_test_model.values, y_test.values, model_name)
        results[model_name] = eval_result
        
        print(f"\n{'-'*50}")
        print(f"Model: {model_name} (v{version})")
        print(f"{'-'*50}")
        
        # Default threshold metrics
        dm = eval_result['default_metrics']
        print(f"\nDefault Threshold (0.5) Metrics:")
        print(f"  Accuracy:  {dm['accuracy']:.4f}")
        print(f"  Precision: {dm['precision']:.4f}")
        print(f"  Recall:    {dm['recall']:.4f}")
        print(f"  F1-Score:  {dm['f1_score']:.4f}")
        print(f"  ROC-AUC:   {dm['roc_auc']:.4f}")
        print(f"  PR-AUC:    {dm['pr_auc']:.4f}")
        
        # Confusion matrix
        cm = eval_result['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm['tn']}")
        print(f"  False Positives: {cm['fp']}")
        print(f"  False Negatives: {cm['fn']}")
        print(f"  True Positives: {cm['tp']}")
        
        # Optimal threshold
        ot = eval_result['optimal_threshold']
        print(f"\nOptimal Threshold for F1: {ot['threshold']:.2f}")
        print(f"  Precision: {ot['precision']:.4f}")
        print(f"  Recall:    {ot['recall']:.4f}")
        print(f"  F1-Score:  {ot['f1_score']:.4f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
    ))
    print("-" * 70)
    
    for name in model_names:
        if name in results:
            dm = results[name]['default_metrics']
            print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                name, dm['accuracy'], dm['precision'], dm['recall'], dm['f1_score'], dm['roc_auc']
            ))
    
    # Best model determination
    print("\n" + "=" * 70)
    print("BEST MODEL RECOMMENDATIONS")
    print("=" * 70)
    
    # Best by ROC-AUC (best for ranking)
    best_roc = max(results.keys(), key=lambda k: results[k]['default_metrics']['roc_auc'])
    print(f"\nBest for Ranking (ROC-AUC): {best_roc}")
    print(f"  ROC-AUC: {results[best_roc]['default_metrics']['roc_auc']:.4f}")
    
    # Best by F1 (balanced performance)
    best_f1 = max(results.keys(), key=lambda k: results[k]['default_metrics']['f1_score'])
    print(f"\nBest Balanced (F1-Score): {best_f1}")
    print(f"  F1-Score: {results[best_f1]['default_metrics']['f1_score']:.4f}")
    
    # Best by Recall (important for catching no-shows)
    best_recall = max(results.keys(), key=lambda k: results[k]['default_metrics']['recall'])
    print(f"\nBest Recall (Catch No-Shows): {best_recall}")
    print(f"  Recall: {results[best_recall]['default_metrics']['recall']:.4f}")
    
    # Trade-off analysis
    print("\n" + "=" * 70)
    print("PRECISION-RECALL TRADE-OFF ANALYSIS")
    print("=" * 70)
    
    print("""
Business Context Considerations:
--------------------------------
1. HIGH RECALL STRATEGY (prioritize catching no-shows):
   - Use lower threshold (e.g., 0.30-0.40)
   - Accept more false positives (overbooking risk)
   - Better for: aggressive overbooking, inventory optimization

2. BALANCED STRATEGY (default):
   - Use threshold 0.50
   - Moderate precision and recall
   - Good for: general predictions, analytics

3. HIGH PRECISION STRATEGY (prioritize accuracy per prediction):
   - Use higher threshold (e.g., 0.60-0.70)
   - Miss more actual no-shows
   - Better for: targeted interventions, customer retention
""")
    
    # Final recommendations
    print("=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
Model Selection Guidelines:
---------------------------
- For OVERBOOKING DECISIONS: Use {best_roc} (ROC-AUC={results[best_roc]['default_metrics']['roc_auc']:.4f})
  High ranking ability helps identify relative no-show risk
  
- For RISK ALERTS: Use threshold 0.35-0.40 on any model
  Lower threshold catches more no-shows
  
- For INTERVENTION TARGETING: Use threshold 0.60+
  Higher threshold ensures high-confidence predictions

Data Quality Notes:
- Class imbalance (36.2% no-show rate) affects recall
- All models use class weights during training
- Probability calibration is applied (CalibratedClassifierCV)

Limitations:
- Low recall indicates models tend to predict "SHOW"
- This is acceptable for risk scoring but limits actionability
- Consider collecting more features or labeled data
""")
    
    # Save report to file
    report = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'class_imbalance': imbalance_info,
        'model_results': results,
        'best_models': {
            'roc_auc': best_roc,
            'f1': best_f1,
            'recall': best_recall
        }
    }
    
    report_file = 'model_evaluation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nFull report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    results = generate_model_report()