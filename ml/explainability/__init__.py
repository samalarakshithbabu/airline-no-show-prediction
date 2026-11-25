"""
SHAP Explainability Module for Airline No-Show Prediction
Provides local and global model explanations
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for no-show predictions.
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained model
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame):
        """
        Fit the SHAP explainer on training data.
        
        Args:
            X: Training features
        """
        self.feature_names = list(X.columns)
        
        if self.model_type == 'tree':
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            # Use LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(self.model, X.values)
        else:
            # Use KernelExplainer for other models (slower but general)
            # Use a sample of data as background
            background = shap.kmeans(X, min(100, len(X)))
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        
        logger.info(f"SHAP explainer fitted for {self.model_type} model")
        return self
    
    def explain_local(self, 
                      X: pd.DataFrame,
                      sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X: Feature DataFrame
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        sample = X.iloc[[sample_idx]]
        
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(sample.values)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (no-show) values
        
        # Create explanation dict
        explanation = {
            'sample_index': sample_idx,
            'prediction': self.model.predict(sample)[0],
            'probability': self.model.predict_proba(sample)[0][1],
            'feature_values': dict(zip(self.feature_names, sample.values[0])),
            'shap_values': dict(zip(self.feature_names, shap_values[0])),
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
        }
        
        # Categorize features by impact
        positive_impact = {k: v for k, v in explanation['shap_values'].items() if v > 0}
        negative_impact = {k: v for k, v in explanation['shap_values'].items() if v < 0}
        
        explanation['positive_factors'] = sorted(positive_impact.items(), 
                                                  key=lambda x: x[1], reverse=True)[:5]
        explanation['negative_factors'] = sorted(negative_impact.items(), 
                                                  key=lambda x: x[1])[:5]
        
        return explanation
    
    def explain_batch(self, X: pd.DataFrame, max_samples: int = 100) -> Dict[str, Any]:
        """
        Explain multiple predictions.
        
        Args:
            X: Feature DataFrame
            max_samples: Maximum samples to explain
            
        Returns:
            Dictionary with batch explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Use subset for efficiency
        X_sample = X.iloc[:min(max_samples, len(X))]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample.values)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Aggregate explanations
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        return {
            'n_samples': len(X_sample),
            'shap_values': shap_values,
            'mean_abs_shap': mean_abs_shap,
            'feature_names': self.feature_names,
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
        }
    
    def get_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get global feature importance from SHAP.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with feature importance rankings
        """
        batch_explanation = self.explain_batch(X)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': batch_explanation['mean_abs_shap'],
            'importance_rank': batch_explanation['mean_abs_shap'].argsort()[::-1].argsort() + 1
        })
        
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X: pd.DataFrame, save_path: str = None):
        """
        Create SHAP summary plot.
        
        Args:
            X: Feature DataFrame
            save_path: Path to save figure
        """
        batch_explanation = self.explain_batch(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            batch_explanation['shap_values'],
            X.iloc[:len(batch_explanation['shap_values'])].values,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_dependence(self, 
                        X: pd.DataFrame, 
                        feature: str,
                        interaction_feature: str = None,
                        save_path: str = None):
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            X: Feature DataFrame
            feature: Feature to plot
            interaction_feature: Feature for color interaction
            save_path: Path to save figure
        """
        batch_explanation = self.explain_batch(X)
        
        feature_idx = self.feature_names.index(feature)
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
            color_values = batch_explanation['shap_values'][:, interaction_idx]
        else:
            color_values = None
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            batch_explanation['shap_values'],
            X.iloc[:len(batch_explanation['shap_values'])].values,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_prediction_explanation(self,
                                        X: pd.DataFrame,
                                        booking_id: int = None,
                                        passenger_info: Dict = None) -> Dict[str, Any]:
        """
        Generate a human-readable prediction explanation.
        
        Args:
            X: Feature DataFrame
            booking_id: Optional booking ID
            passenger_info: Optional passenger information
            
        Returns:
            Dictionary with explanation text
        """
        # Get individual explanation
        explanation = self.explain_local(X, sample_idx=0)
        
        # Generate human-readable summary
        prediction_class = "HIGH RISK" if explanation['prediction'] == 1 else "LOW RISK"
        probability_pct = explanation['probability'] * 100
        
        factors = []
        
        # Positive factors (increases no-show probability)
        for feature, value in explanation['positive_factors']:
            factors.append(f"- **{feature}** contributes +{value:.3f} to no-show probability")
        
        # Negative factors (decreases no-show probability)
        for feature, value in explanation['negative_factors']:
            factors.append(f"- **{feature}** contributes {value:.3f} (reduces risk)")
        
        result = {
            'prediction': prediction_class,
            'probability': f"{probability_pct:.1f}%",
            'risk_level': self._get_risk_level(explanation['probability']),
            'key_factors': factors,
            'explanation_text': self._generate_text(explanation, passenger_info)
        }
        
        return result
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
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
    
    def _generate_text(self, 
                       explanation: Dict, 
                       passenger_info: Dict = None) -> str:
        """Generate natural language explanation."""
        lines = []
        
        # Prediction summary
        if explanation['prediction'] == 1:
            lines.append("This passenger is predicted to be a NO-SHOW.")
        else:
            lines.append("This passenger is predicted to ATTEND their flight.")
        
        lines.append(f"\nPredicted No-Show Probability: {explanation['probability']*100:.1f}%")
        
        # Top factors
        lines.append("\nKey factors influencing this prediction:")
        
        for feature, value in explanation['positive_factors'][:3]:
            lines.append(f"  ↑ {feature} increases no-show risk")
        
        for feature, value in explanation['negative_factors'][:3]:
            lines.append(f"  ↓ {feature} decreases no-show risk")
        
        # Confidence note
        base_value = explanation['base_value']
        lines.append(f"\nBase probability (population average): {base_value*100:.1f}%")
        
        return "\n".join(lines)


def get_tree_explainer(model, X: pd.DataFrame):
    """
    Convenience function to get a tree-based SHAP explainer.
    
    Args:
        model: Trained tree-based model
        X: Training data sample
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, 'tree')


def get_linear_explainer(model, X: pd.DataFrame):
    """
    Convenience function to get a linear SHAP explainer.
    
    Args:
        model: Trained linear model
        X: Training data sample
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, 'linear')


def get_kernel_explainer(model, X: pd.DataFrame):
    """
    Convenience function to get a kernel SHAP explainer.
    
    Args:
        model: Trained model
        X: Training data sample
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, 'kernel')


def waterfall_plot(explanation, save_path: str = None):
    """
    Create a waterfall plot for a single prediction.
    
    Args:
        explanation: SHAP explanation dictionary
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Sort SHAP values
    sorted_shap = sorted(
        explanation['shap_values'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    features = [x[0] for x in sorted_shap]
    values = [x[1] for x in sorted_shap]
    
    # Create waterfall-style bar chart
    base_value = explanation['base_value']
    
    cumulative = [base_value]
    for i, (feat, val) in enumerate(sorted_shap):
        cumulative.append(cumulative[-1] + val)
    
    # Plot
    plt.barh(range(len(features)), values, left=cumulative[:-1])
    plt.yticks(range(len(features)), features)
    plt.xlabel('SHAP Value (impact on prediction)')
    plt.title('Feature Contributions to No-Show Prediction')
    plt.axvline(x=0, color='black', linewidth=0.5)
    
    # Add value labels
    for i, (feat, val) in enumerate(sorted_shap[:10]):
        label_x = cumulative[i] + val/2
        plt.text(label_x, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()