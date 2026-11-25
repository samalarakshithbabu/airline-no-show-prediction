"""
Model Evaluation Module - Comprehensive model evaluation and comparison
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.figures_dir = 'figures'
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def evaluate_all(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_proba: np.ndarray,
                     model_name: str = 'Model') -> Dict[str, Any]:
        """
        Complete model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model_name: Name for plots
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Core metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Extract TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        metrics['false_discovery_rate'] = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        return metrics
    
    def generate_classification_report(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        model_name: str = 'Model') -> str:
        """
        Generate formatted classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
            
        Returns:
            Formatted report string
        """
        report = classification_report(y_true, y_pred, 
                                       target_names=['Show', 'No-Show'],
                                       digits=4)
        
        return f"Classification Report for {model_name}\n{report}"
    
    def plot_roc_curves(self,
                         results: Dict[str, Dict[str, Any]],
                         save_path: str = None) -> str:
        """
        Plot ROC curves for multiple models.
        
        Args:
            results: Dictionary of model_name -> {'y_true': ..., 'y_proba': ...}
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            y_true = result['y_true']
            y_proba = result['y_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = save_path or os.path.join(self.figures_dir, 'roc_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {save_path}")
        return save_path
    
    def plot_precision_recall_curves(self,
                                      results: Dict[str, Dict[str, Any]],
                                      save_path: str = None) -> str:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            results: Dictionary of model_name -> {'y_true': ..., 'y_proba': ...}
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            y_true = result['y_true']
            y_proba = result['y_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            auc = average_precision_score(y_true, y_proba)
            
            plt.plot(recall, precision, linewidth=2, 
                    label=f'{model_name} (AP={auc:.3f})')
        
        # Baseline (random classifier)
        baseline = y_true.sum() / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline ({baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = save_path or os.path.join(self.figures_dir, 'pr_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curves saved to {save_path}")
        return save_path
    
    def plot_confusion_matrix(self,
                               cm: np.ndarray,
                               model_name: str = 'Model',
                               save_path: str = None) -> str:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            model_name: Model name for title
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        plt.figure(figsize=(8, 6))
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.colorbar()
        
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Show', 'No-Show'])
        plt.yticks(tick_marks, ['Show', 'No-Show'])
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                        ha='center', va='center',
                        color='white' if cm_normalized[i, j] > thresh else 'black')
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        save_path = save_path or os.path.join(self.figures_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path
    
    def plot_feature_importance(self,
                                 feature_importance: Dict[str, float],
                                 model_name: str = 'Model',
                                 top_n: int = 15,
                                 save_path: str = None) -> str:
        """
        Plot feature importance bar chart.
        
        Args:
            feature_importance: Dictionary of feature -> importance
            model_name: Model name for title
            top_n: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        # Sort and take top N
        sorted_features = sorted(feature_importance.items(), 
                                 key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        plt.figure(figsize=(10, 8))
        colors = ['green' if i > 0 else 'red' for i in range(len(features))]
        bars = plt.barh(range(len(features)), importances, color='steelblue')
        
        # Color positive/negative importance
        for i, (feat, imp) in enumerate(sorted_features):
            if imp < 0:
                bars[i].set_color('indianred')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance (SHAP)', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        
        save_path = save_path or os.path.join(self.figures_dir, f'feature_importance_{model_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance saved to {save_path}")
        return save_path
    
    def plot_model_comparison(self,
                               comparison_df: pd.DataFrame,
                               metrics: List[str] = ['accuracy', 'f1_score', 'roc_auc'],
                               save_path: str = None) -> str:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison
            metrics: Metrics to compare
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = comparison_df[metric].values
            plt.bar(x + i * width, values, width, label=metric.upper())
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14)
        plt.xticks(x + width, comparison_df['model_name'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.ylim(0, 1)
        
        save_path = save_path or os.path.join(self.figures_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison saved to {save_path}")
        return save_path
    
    def create_evaluation_report(self,
                                  results: Dict[str, Dict[str, Any]],
                                  output_dir: str = 'evaluation') -> str:
        """
        Create comprehensive evaluation report.
        
        Args:
            results: Dictionary of model_name -> {'y_true': ..., 'y_pred': ..., 'y_proba': ...}
            output_dir: Output directory
            
        Returns:
            Path to saved report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Evaluate each model
        all_metrics = {}
        
        for model_name, data in results.items():
            report_lines.append(f"\n{'='*50}")
            report_lines.append(f"Model: {model_name}")
            report_lines.append("=" * 50)
            
            metrics = self.evaluate_all(
                data['y_true'], 
                data['y_pred'], 
                data['y_proba'],
                model_name
            )
            all_metrics[model_name] = metrics
            
            # Add to report
            report_lines.append("\nMetrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
            
            # Classification report
            report_lines.append(f"\n{self.generate_classification_report(data['y_true'], data['y_pred'], model_name)}")
            
            # Save individual plots
            cm_path = self.plot_confusion_matrix(
                np.array(metrics['confusion_matrix']), 
                model_name,
                os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
            )
            report_lines.append(f"\nConfusion Matrix: {cm_path}")
        
        # Summary comparison
        report_lines.append("\n" + "=" * 70)
        report_lines.append("SUMMARY COMPARISON")
        report_lines.append("=" * 70)
        
        summary_df = pd.DataFrame(all_metrics).T
        summary_df = summary_df.sort_values('roc_auc', ascending=False)
        report_lines.append("\n" + summary_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string())
        
        # Best model
        best_model = summary_df['roc_auc'].idxmax()
        report_lines.append(f"\nBest Model (by ROC-AUC): {best_model}")
        report_lines.append(f"  ROC-AUC: {summary_df.loc[best_model, 'roc_auc']:.4f}")
        report_lines.append(f"  F1-Score: {summary_df.loc[best_model, 'f1_score']:.4f}")
        
        # Save plots
        self.plot_roc_curves(results, os.path.join(output_dir, 'roc_curves.png'))
        self.plot_precision_recall_curves(results, os.path.join(output_dir, 'pr_curves.png'))
        self.plot_model_comparison(summary_df.reset_index().rename(columns={'index': 'model_name'}),
                                    os.path.join(output_dir, 'model_comparison.png'))
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report_path


def evaluate_models(models: Dict[str, Any],
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Evaluate multiple trained models.
    
    Args:
        models: Dictionary of model_name -> trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (results_dict, comparison_DataFrame)
    """
    evaluator = ModelEvaluator()
    results = {}
    
    for model_name, model in models.items():
        if model is None:
            continue
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results[model_name] = {
            'y_true': y_test.values,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    # Generate comparison
    comparison = []
    for model_name, result in results.items():
        metrics = evaluator.evaluate_all(
            result['y_true'], 
            result['y_pred'], 
            result['y_proba'],
            model_name
        )
        metrics['model_name'] = model_name
        comparison.append(metrics)
    
    comparison_df = pd.DataFrame(comparison).sort_values('roc_auc', ascending=False)
    
    return results, comparison_df


if __name__ == "__main__":
    # Demo evaluation
    print("=" * 60)
    print("MODEL EVALUATION DEMONSTRATION")
    print("=" * 60)
    print()
    
    from ml.training import train_models
    from ml.data_processing import load_and_preprocess
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    print(f"  Test set: {len(y_test)} samples")
    print(f"  No-show rate: {y_test.mean():.2%}")
    print()
    
    # Train models
    print("Training models...")
    trainer_results, _ = train_models(X_train, y_train, X_test, y_test, output_dir='models')
    print()
    
    # Evaluate
    print("Evaluating models...")
    results = {}
    for model_name, result in trainer_results.items():
        if result.get('model') is not None:
            model = result['model']
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            results[model_name] = {
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
    
    evaluator = ModelEvaluator()
    
    # Create report
    report_path = evaluator.create_evaluation_report(results, 'evaluation')
    print(f"\nEvaluation report: {report_path}")
    
    # Show comparison
    _, comparison_df = evaluate_models(
        {k: v['model'] for k, v in trainer_results.items() if v.get('model') is not None},
        X_test, y_test
    )
    
    print("\nModel Comparison:")
    print(comparison_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))