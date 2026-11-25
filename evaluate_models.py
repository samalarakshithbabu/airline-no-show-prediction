"""Comprehensive ML Model Evaluation Script"""
import pandas as pd
import numpy as np
import json
import os
import pickle
from ml.evaluation import ModelEvaluator
from ml.feature_engineering import engineer_features, select_features

# Load training data
df = pd.read_csv('data/training_data.csv')
print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution: Show={len(df[df['no_show']==0])}, No-Show={len(df[df['no_show']==1])}")
print(f"No-show rate: {df['no_show'].mean()*100:.1f}%")

# Engineer features
engineered_df = engineer_features(df)
print(f"After feature engineering: {engineered_df.shape[1]} columns")

# Use feature names from trained models (version 20261120)
model_dir = 'models'
feature_names = None

# Load the feature names from model metadata
for model_file in os.listdir(model_dir):
    if model_file.endswith('.json') and 'features' in model_file:
        with open(os.path.join(model_dir, model_file), 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} model features")
        break

if feature_names is None:
    print("Using selected features instead")
    feature_names = select_features(engineered_df)

print(f"Using {len(feature_names)} features for evaluation")

# Filter engineered_df to only include needed features (in same order)
available_features = [f for f in feature_names if f in engineered_df.columns]
print(f"Available features: {len(available_features)}")

# Prepare data
X = engineered_df[available_features].fillna(0)
y = engineered_df['no_show']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Load models and evaluate
results = {}

for model_file in os.listdir(model_dir):
    if model_file.endswith('.pkl') and 'scaler' not in model_file and 'features' not in model_file and 'info' not in model_file:
        model_name = model_file.split('_')[0]
        version = model_file.replace('.pkl', '').replace(f'{model_name}_', '')
        
        print(f"\nEvaluating {model_name}...")
        
        with open(os.path.join(model_dir, model_file), 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_test.values)
        y_proba = model.predict_proba(X_test.values)[:, 1]
        
        evaluator = ModelEvaluator(y_test.values, y_pred, y_proba)
        metrics = evaluator.evaluate_all(y_test.values, y_pred, y_proba, model_name)
        
        results[model_name] = metrics

print()
print("="*90)
print("MODEL EVALUATION RESULTS")
print("="*90)
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
print("-"*90)
for name, m in sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True):
    print(f"{name:<25} {m['accuracy']:.4f}      {m['precision']:.4f}      {m['recall']:.4f}      {m['f1_score']:.4f}      {m['roc_auc']:.4f}")

best = max(results.items(), key=lambda x: x[1]['roc_auc'])
print()
print(f"Best model by ROC-AUC: {best[0]} ({best[1]['roc_auc']:.4f})")

# Detailed metrics for best model
print()
print("="*90)
print(f"DETAILED METRICS FOR {best[0].upper()}")
print("="*90)
for key, value in best[1].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    elif isinstance(value, list):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value}")