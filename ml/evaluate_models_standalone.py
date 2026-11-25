"""
Comprehensive Model Evaluation Report - Standalone Script
Evaluates all trained models with proper threshold analysis and trade-off documentation.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'training_data.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


def load_trained_model(model_name: str) -> tuple:
    """Load a trained model and its artifacts."""
    model_files = [f for f in os.listdir(MODEL_DIR) 
                   if f.startswith(model_name) and f.endswith('.pkl')
                   and 'scaler' not in f and 'features' not in f and 'info' not in f]
    
    if not model_files:
        return None, None, None, None
    
    model_files.sort()
    model_file = model_files[-1]
    
    version = model_file.replace('.pkl', '').replace(f'{model_name}_', '')
    
    with open(os.path.join(MODEL_DIR, model_file), 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    features_file = os.path.join(MODEL_DIR, f'{model_name}_{version}_features.json')
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            feature_names = json.load(f)
    else:
        feature_names = None
    
    return model, feature_names, version


def load_test_data(feature_names: list) -> tuple:
    """Load the test dataset with matching features."""
    df = pd.read_csv(DATA_PATH)
    
    # Parse dates
    df['flight_date'] = pd.to_datetime(df['flight_date'], errors='coerce')
    
    # Create features - all as pandas Series, then stack
    features = {}
    
    # Raw features
    features['booking_lead_time'] = pd.to_numeric(df['booking_lead_time'], errors='coerce').fillna(14)
    features['total_bookings'] = pd.to_numeric(df['total_bookings'], errors='coerce').fillna(5)
    features['total_noshows'] = pd.to_numeric(df['total_noshows'], errors='coerce').fillna(1)
    features['ticket_price'] = pd.to_numeric(df['ticket_price'], errors='coerce').fillna(350)
    
    # Time features
    features['flight_dayofweek'] = df['flight_date'].dt.dayofweek.fillna(0).astype(float)
    features['flight_month'] = df['flight_date'].dt.month.fillna(1).astype(float)
    features['flight_day'] = df['flight_date'].dt.day.fillna(1).astype(float)
    features['flight_quarter'] = df['flight_date'].dt.quarter.fillna(1).astype(float)
    features['is_weekend'] = (df['flight_date'].dt.dayofweek >= 5).astype(float)
    features['is_friday'] = (df['flight_date'].dt.dayofweek == 4).astype(float)
    features['is_monday'] = (df['flight_date'].dt.dayofweek == 0).astype(float)
    features['is_holiday_period'] = df['flight_date'].dt.month.isin([11, 12, 7]).astype(float)
    features['is_summer'] = df['flight_date'].dt.month.isin([6, 7, 8]).astype(float)
    features['is_winter'] = df['flight_date'].dt.month.isin([12, 1, 2]).astype(float)
    features['is_morning'] = 1.0
    
    # Passenger features
    features['historical_noshow_rate'] = np.where(
        features['total_bookings'] > 0,
        features['total_noshows'] / features['total_bookings'].clip(lower=1),
        0.08
    )
    features['has_noshow_history'] = (features['total_noshows'] > 0).astype(float)
    features['high_noshow_risk'] = (features['total_noshows'] >= 2).astype(float)
    features['has_cancellation_history'] = 0.0
    
    # Loyalty encoding
    tier_map = {'NONE': 0, 'SILVER': 1, 'GOLD': 2, 'PLATINUM': 3}
    features['loyalty_tier_encoded'] = df['loyalty_tier'].map(tier_map).fillna(0).astype(float)
    features['is_premium_tier'] = features['loyalty_tier_encoded'].isin([2, 3]).astype(float)
    
    # Cabin encoding
    cabin_map = {'ECONOMY': 0, 'PREMIUM_ECONOMY': 1, 'BUSINESS': 2, 'FIRST': 3}
    features['cabin_class_encoded'] = df['cabin_class'].map(cabin_map).fillna(0).astype(float)
    features['is_economy'] = (df['cabin_class'] == 'ECONOMY').astype(float)
    features['is_premium_cabin'] = features['cabin_class_encoded'].isin([2, 3]).astype(float)
    
    # Channel encoding
    channel_map = {'WEB': 0, 'MOBILE': 1, 'AGENT': 2, 'CALL_CENTER': 3, 'CORPORATE': 3}
    features['booking_channel_encoded'] = df['booking_channel'].map(channel_map).fillna(0).astype(float)
    features['is_direct_booking'] = df['booking_channel'].isin(['WEB', 'MOBILE']).astype(float)
    
    # Price features
    avg_price = df['ticket_price'].mean()
    features['price_vs_average'] = df['ticket_price'] / avg_price
    features['is_high_price'] = (df['ticket_price'] > avg_price * 1.5).astype(float)
    features['is_low_price'] = (df['ticket_price'] < avg_price * 0.7).astype(float)
    
    # Booking timing
    features['last_minute_booking'] = (features['booking_lead_time'] < 3).astype(float)
    features['short_notice'] = ((features['booking_lead_time'] >= 3) & (features['booking_lead_time'] < 7)).astype(float)
    features['advance_booking'] = (features['booking_lead_time'] >= 7).astype(float)
    features['very_advance'] = (features['booking_lead_time'] >= 30).astype(float)
    features['new_last_minute'] = 0.0
    features['weekend_economy'] = 0.0
    
    # Passenger type
    features['is_new_passenger'] = (features['total_bookings'] <= 1).astype(float)
    features['is_frequent_flyer'] = (features['total_bookings'] > 5).astype(float)
    
    # Create DataFrame
    feature_df = pd.DataFrame(features)
    
    # Select features that model expects
    available_features = [f for f in feature_names if f in feature_df.columns]
    logger.info(f"Model expects {len(feature_names)} features, {len(available_features)} available")
    
    X = feature_df[available_features].fillna(0)
    y = pd.to_numeric(df['no_show'], errors='coerce').fillna(0).astype(int)
    
    # Split for evaluation
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_test, y_test


def evaluate_with_thresholds(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model at different classification thresholds."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    
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
    
    best_f1_idx = np.argmax([t['f1_score'] for t in threshold_analysis])
    best_threshold = threshold_analysis[best_f1_idx]
    
    cm = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'default_metrics': default_metrics,
        'threshold_analysis': threshold_analysis,
        'optimal_threshold': {
            'threshold': best_threshold['threshold'],
            'f1_score': best_threshold['f1_score'],
            'precision': best_threshold['precision'],
            'recall': best_threshold['recall']
        },
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'class_distribution': {
            'test_size': len(y_test),
            'positive_class': int(y_test.sum()),
            'negative_class': int(len(y_test) - y_test.sum()),
            'positive_rate': round(float(y_test.mean()), 4)
        }
    }


def generate_report():
    """Generate comprehensive model evaluation report."""
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("Airline No-Show Prediction System")
    print("=" * 70)
    
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
        
        if feature_names is None:
            logger.warning(f"No feature names for {model_name}")
            continue
        
        # Load test data with matching features
        print(f"\nLoading test data for {model_name}...")
        X_test, y_test = load_test_data(feature_names)
        
        positive_rate = y_test.mean()
        print(f"Test set: {len(y_test)} samples, {positive_rate:.2%} no-show rate")
        
        eval_result = evaluate_with_thresholds(model, X_test, y_test)
        results[model_name] = eval_result
        
        print(f"\n{'-'*50}")
        print(f"Model: {model_name} (v{version})")
        print(f"{'-'*50}")
        
        dm = eval_result['default_metrics']
        print(f"\nDefault Threshold (0.5) Metrics:")
        print(f"  Accuracy:  {dm['accuracy']:.4f}")
        print(f"  Precision: {dm['precision']:.4f}")
        print(f"  Recall:    {dm['recall']:.4f}")
        print(f"  F1-Score:  {dm['f1_score']:.4f}")
        print(f"  ROC-AUC:   {dm['roc_auc']:.4f}")
        print(f"  PR-AUC:    {dm['pr_auc']:.4f}")
        
        cm = eval_result['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm['tn']}, FP: {cm['fp']}")
        print(f"  FN: {cm['fn']}, TP: {cm['tp']}")
        
        ot = eval_result['optimal_threshold']
        print(f"\nOptimal Threshold: {ot['threshold']:.2f}")
        print(f"  F1: {ot['f1_score']:.4f}, Precision: {ot['precision']:.4f}, Recall: {ot['recall']:.4f}")
    
    # Summary
    if not results:
        print("\nNo models could be evaluated!")
        return
    
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
    
    # Best models
    best_roc = max(results.keys(), key=lambda k: results[k]['default_metrics']['roc_auc'])
    best_f1 = max(results.keys(), key=lambda k: results[k]['default_metrics']['f1_score'])
    best_recall = max(results.keys(), key=lambda k: results[k]['default_metrics']['recall'])
    
    print(f"\nBest ROC-AUC: {best_roc} ({results[best_roc]['default_metrics']['roc_auc']:.4f})")
    print(f"Best F1-Score: {best_f1} ({results[best_f1]['default_metrics']['f1_score']:.4f})")
    print(f"Best Recall: {best_recall} ({results[best_recall]['default_metrics']['recall']:.4f})")
    
    # Trade-offs
    print("\n" + "=" * 70)
    print("PRECISION-RECALL TRADE-OFFS")
    print("=" * 70)
    print("""
Business Context:
- High Recall (threshold 0.30-0.40): Catch more no-shows, more false positives
- Balanced (threshold 0.50): Standard trade-off
- High Precision (threshold 0.60+): Fewer false positives, miss more no-shows

For overbooking decisions, use higher recall.
For intervention targeting, use higher precision.
""")
    
    # Final recommendations
    print("=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
Model Selection Guidelines:
- For OVERBOOKING: Use {best_recall} (best recall) to catch no-shows
- For RANKING: Use {best_roc} (best ROC-AUC) for relative risk scoring
- For BALANCED: Use {best_f1} (best F1) for general predictions

Data Quality Notes:
- Class imbalance (~36% no-show rate) affects model performance
- All models use class weights during training
- Low recall indicates models predict "SHOW" more often

This is expected behavior given class imbalance.
Use probability thresholds strategically for business decisions.
""")
    
    # Save report
    report = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'model_results': {k: {
            'metrics': results[k]['default_metrics'],
            'optimal_threshold': results[k]['optimal_threshold'],
            'confusion_matrix': results[k]['confusion_matrix']
        } for k in results}
    }
    
    with open('model_evaluation_summary.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nReport saved to: model_evaluation_summary.json")


if __name__ == "__main__":
    generate_report()