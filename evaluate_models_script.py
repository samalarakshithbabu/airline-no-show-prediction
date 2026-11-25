"""
Comprehensive ML Model Evaluation Script for Airline No-Show Prediction System
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.evaluation import ModelEvaluator
from ml.data_processing import preprocess_data

# Load the training data
print("=" * 80)
print("AIRLINE NO-SHOW PREDICTION SYSTEM - ML MODEL EVALUATION")
print("=" * 80)
print()

df = pd.read_csv('data/training_data.csv')
print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution: {df['no_show'].value_counts().to_dict()}")
print(f"No-show rate: {df['no_show'].mean()*100:.1f}%")
print()

# Load all models and evaluate
model_dir = 'models'
results = {}

for model_file in sorted(os.listdir(model_dir)):
    if model_file.endswith('.pkl') and 'scaler' not in model_file and 'features' not in model_file and 'info' not in model_file:
        # Extract model name and version
        parts = model_file.replace('.pkl', '').split('_')
        model_name = parts[0]
        version = '_'.join(parts[1:])
        
        print(f"Loading model: {model_name} (version: {version})")
        
        try:
            # Load model
            model_path = os.path.join(model_dir, model_file)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load features
            features_file = os.path.join(model_dir, f'{model_name}_{version}_features.json')
            with open(features_file, 'r') as f:
                feature_names = json.load(f)
            print(f"  Features: {len(feature_names)} features")
            
            # Prepare data
            X_train, X_test, y_train, y_test, preprocessing = preprocess_data(df)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_all(y_test.values, y_pred, y_proba, model_name)
            
            results[model_name] = {
                'version': version,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'specificity': metrics['specificity'],
                'confusion_matrix': metrics['confusion_matrix'],
                'tn': metrics['true_negatives'],
                'fp': metrics['false_positives'],
                'fn': metrics['false_negatives'],
                'tp': metrics['true_positives']
            }
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

# Print comparison table
print("\n" + "=" * 100)
print("MODEL EVALUATION RESULTS - COMPARISON TABLE")
print("=" * 100)
print()
print(f"{'Model':<25} {'Version':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10}")
print("-" * 100)

sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
for name, metrics in sorted_results:
    print(f"{name:<25} {metrics['version']:<15} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     "
          f"{metrics['recall']:.4f}     {metrics['f1_score']:.4f}     {metrics['roc_auc']:.4f}     {metrics['pr_auc']:.4f}")

print()
print("=" * 100)
print("DETAILED METRICS BY MODEL")
print("=" * 100)

for name, metrics in sorted_results:
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"  Core Metrics:")
    print(f"    Accuracy:    {metrics['accuracy']:.4f}")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC Metrics:")
    print(f"    ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  Additional:")
    print(f"    Specificity: {metrics['specificity']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {metrics['tn']:<6} FP: {metrics['fp']}")
    print(f"    FN: {metrics['fn']:<6} TP: {metrics['tp']}")

print()
print("=" * 100)
print("BEST MODEL ANALYSIS")
print("=" * 100)

best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = results[best_model_name]

print(f"\nBest Model (by ROC-AUC): {best_model_name}")
print(f"  ROC-AUC Score: {best_model['roc_auc']:.4f}")
print(f"  F1-Score:      {best_model['f1_score']:.4f}")
print(f"  Accuracy:      {best_model['accuracy']:.4f}")
print(f"  Precision:     {best_model['precision']:.4f}")
print(f"  Recall:        {best_model['recall']:.4f}")

# Rank all models
print("\nModel Rankings (by ROC-AUC):")
print("-" * 50)
for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"  {i}. {name:<20} ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")

print()
print("=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)