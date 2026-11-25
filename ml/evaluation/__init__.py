"""
Model Evaluation Module for Airline No-Show Prediction
Comprehensive metrics and evaluation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    log_loss, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import math

logger = logging.getLogger(__name__)


def evaluate_business_costs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    missed_no_show_cost: float,
    false_positive_action_cost: float,
) -> Dict[str, float]:
    """Translate a fixed threshold into explicit false-negative/false-positive costs."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be a finite value between 0 and 1')
    for name, value in (
        ('missed_no_show_cost', missed_no_show_cost),
        ('false_positive_action_cost', false_positive_action_cost),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')

    labels = np.asarray(y_true)
    probabilities = np.asarray(y_proba, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError('y_true and y_proba must be one-dimensional arrays of equal length')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('y_true must contain only binary 0/1 labels')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('y_proba must contain finite values between 0 and 1')

    predicted = probabilities >= threshold
    false_positives = int(((labels == 0) & predicted).sum())
    false_negatives = int(((labels == 1) & ~predicted).sum())
    total_cost = (
        false_negatives * missed_no_show_cost
        + false_positives * false_positive_action_cost
    )
    return {
        'threshold': float(threshold),
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'missed_no_show_cost': float(missed_no_show_cost),
        'false_positive_action_cost': float(false_positive_action_cost),
        'total_cost': float(total_cost),
        'cost_per_booking': float(total_cost / len(labels)) if len(labels) else 0.0,
    }


def select_business_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    missed_no_show_cost: float,
    false_positive_action_cost: float,
) -> Dict[str, float]:
    """Choose the minimum-cost threshold on validation/OOF predictions only."""
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError('y_proba must be a non-empty one-dimensional array')
    thresholds = np.unique(np.concatenate((
        [0.0],
        probabilities,
        [np.nextafter(float(probabilities.max()), np.inf)],
    )))
    candidates = [
        evaluate_business_costs(
            y_true,
            probabilities,
            float(candidate),
            missed_no_show_cost,
            false_positive_action_cost,
        )
        for candidate in thresholds
    ]
    return min(candidates, key=lambda candidate: (
        candidate['total_cost'],
        -candidate['threshold'],
    ))


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
        """
        Initialize the evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.metrics = {}
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute all evaluation metrics."""
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # AUC metrics
        if len(np.unique(self.y_true)) >= 2:
            self.metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba)
            self.metrics['pr_auc'] = average_precision_score(self.y_true, self.y_proba)
        else:
            self.metrics['roc_auc'] = 0.0
            self.metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = self.cm.ravel()
        
        # Derived metrics
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        self.metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        self.metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        self.metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Calibration
        self.metrics['log_loss'] = log_loss(self.y_true, self.y_proba)
        self.metrics['brier_score'] = brier_score_loss(self.y_true, self.y_proba)
        
        # Counts
        self.metrics['true_positives'] = int(tp)
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        
        # Prevalence
        self.metrics['prevalence'] = (self.y_true == 1).mean()
    
    def get_metrics(self) -> Dict[str, float]:
        """Return all computed metrics."""
        return self.metrics
    
    def get_classification_report(self) -> str:
        """Get sklearn classification report as string."""
        return classification_report(self.y_true, self.y_pred, zero_division=0)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix."""
        return self.cm
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data."""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        return fpr, tpr, thresholds
    
    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Precision-Recall curve data."""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_proba)
        return precision, recall, thresholds
    
    def get_optimal_threshold(self, metric: str = 'youden') -> float:
        """
        Find optimal classification threshold.
        
        Args:
            metric: Metric to optimize ('youden', 'f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold value
        """
        fpr, tpr, thresholds = self.get_roc_curve()
        
        if metric == 'youden':
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
        elif metric == 'f1':
            precision, recall, _ = self.get_pr_curve()
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        else:
            optimal_idx = len(thresholds) // 2
        
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def get_thresholds_at(self, target_tpr: float = 0.9) -> Dict[str, float]:
        """
        Get thresholds at specific sensitivity levels.
        
        Args:
            target_tpr: Target true positive rate
            
        Returns:
            Dictionary with thresholds and corresponding metrics
        """
        fpr, tpr, thresholds = self.get_roc_curve()
        
        # Find threshold for target TPR
        idx = np.argmin(np.abs(tpr - target_tpr))
        
        return {
            'threshold': thresholds[idx],
            'tpr': tpr[idx],
            'fpr': fpr[idx],
            'target_achieved': target_tpr
        }


class ModelComparator:
    """
    Compare multiple models side by side.
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.evaluations = {}
        
    def add_model(self, 
                  name: str, 
                  y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  y_proba: np.ndarray):
        """
        Add a model evaluation.
        
        Args:
            name: Model name
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
        """
        evaluator = ModelEvaluator(y_true, y_pred, y_proba)
        self.evaluations[name] = evaluator
        
    def compare_metrics(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare models on specified metrics.
        
        Args:
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'specificity']
        
        results = {}
        for name, eval_obj in self.evaluations.items():
            results[name] = {m: eval_obj.metrics.get(m, 0) for m in metrics}
        
        return pd.DataFrame(results).T
    
    def rank_models(self, 
                    primary_metric: str = 'f1_score', 
                    secondary_metric: str = 'roc_auc') -> pd.DataFrame:
        """
        Rank models by specified metrics.
        
        Args:
            primary_metric: Primary ranking metric
            secondary_metric: Secondary ranking metric
            
        Returns:
            Ranked DataFrame
        """
        comparison = self.compare_metrics()
        
        # Sort by primary metric
        comparison = comparison.sort_values(primary_metric, ascending=False)
        
        # Add ranks
        comparison['primary_rank'] = range(1, len(comparison) + 1)
        
        # Sort by secondary for tiebreaker
        comparison['secondary_rank'] = comparison[secondary_metric].rank(ascending=False)
        
        return comparison
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, ModelEvaluator]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model name, evaluator)
        """
        best_name = None
        best_score = -1
        
        for name, eval_obj in self.evaluations.items():
            score = eval_obj.metrics.get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name
        
        return best_name, self.evaluations[best_name]


def evaluate_models(models: Dict[str, Any],
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    scaler) -> Dict[str, ModelEvaluator]:
    """
    Evaluate multiple trained models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        scaler: Fitted StandardScaler
        
    Returns:
        Dictionary of evaluations
    """
    evaluations = {}
    X_test_scaled = scaler.transform(X_test)
    
    for name, model_info in models.items():
        model = model_info['model']
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        evaluations[name] = ModelEvaluator(y_test.values, y_pred, y_proba)
        logger.info(f"Evaluated {name}")
    
    return evaluations


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          title: str = 'Confusion Matrix',
                          save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Show', 'No-Show'],
                yticklabels=['Show', 'No-Show'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curves(evaluations: Dict[str, ModelEvaluator],
                    title: str = 'ROC Curve Comparison',
                    save_path: str = None):
    """
    Plot ROC curves for multiple models.
    
    Args:
        evaluations: Dictionary of model evaluations
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, eval_obj in evaluations.items():
        fpr, tpr, _ = eval_obj.get_roc_curve()
        auc = eval_obj.metrics['roc_auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_precision_recall_curves(evaluations: Dict[str, ModelEvaluator],
                                 title: str = 'Precision-Recall Curve Comparison',
                                 save_path: str = None):
    """
    Plot PR curves for multiple models.
    
    Args:
        evaluations: Dictionary of model evaluations
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, eval_obj in evaluations.items():
        precision, recall, _ = eval_obj.get_pr_curve()
        auc = eval_obj.metrics['pr_auc']
        plt.plot(recall, precision, label=f'{name} (AUC = {auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_calibration_curve(y_true: np.ndarray,
                           y_proba: np.ndarray,
                           n_bins: int = 10,
                           title: str = 'Calibration Curve',
                           save_path: str = None):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate mean predicted probability and actual fraction positive per bin
    mean_proba = []
    fraction_positive = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_proba.append(y_proba[mask].mean())
            fraction_positive.append(y_true[mask].mean())
        else:
            mean_proba.append(0)
            fraction_positive.append(0)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(mean_proba, fraction_positive, 'o-', label='Model', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_evaluation_report(evaluations: Dict[str, ModelEvaluator],
                             feature_names: List[str] = None,
                             feature_importance: Dict[str, np.ndarray] = None) -> str:
    """
    Create a text evaluation report.
    
    Args:
        evaluations: Dictionary of model evaluations
        feature_names: List of feature names
        feature_importance: Dictionary of feature importance arrays
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 60)
    
    # Compare metrics
    comparison = ModelComparator()
    for name, eval_obj in evaluations.items():
        comparison.add_model(
            name,
            eval_obj.y_true,
            eval_obj.y_pred,
            eval_obj.y_proba
        )
    
    metrics_df = comparison.compare_metrics()
    report.append("\n### MODEL COMPARISON ###\n")
    report.append(metrics_df.to_string())
    
    # Best model
    best_name, best_eval = comparison.get_best_model()
    report.append(f"\n### BEST MODEL: {best_name} ###\n")
    report.append(f"Accuracy: {best_eval.metrics['accuracy']:.4f}")
    report.append(f"Precision: {best_eval.metrics['precision']:.4f}")
    report.append(f"Recall: {best_eval.metrics['recall']:.4f}")
    report.append(f"F1-Score: {best_eval.metrics['f1_score']:.4f}")
    report.append(f"ROC-AUC: {best_eval.metrics['roc_auc']:.4f}")
    report.append(f"PR-AUC: {best_eval.metrics['pr_auc']:.4f}")
    
    # Confusion matrix breakdown
    report.append("\n### CONFUSION MATRIX ###\n")
    cm = best_eval.get_confusion_matrix()
    report.append(f"True Negatives: {cm[0,0]}")
    report.append(f"False Positives: {cm[0,1]}")
    report.append(f"False Negatives: {cm[1,0]}")
    report.append(f"True Positives: {cm[1,1]}")
    
    # Feature importance
    if feature_importance and feature_names:
        report.append("\n### FEATURE IMPORTANCE ###\n")
        for model_name, importance in feature_importance.items():
            sorted_idx = np.argsort(importance)[::-1]
            report.append(f"\n{model_name}:")
            for i in sorted_idx[:10]:
                report.append(f"  {feature_names[i]}: {importance[i]:.4f}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)