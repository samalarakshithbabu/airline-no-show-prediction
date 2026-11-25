"""
Probability Calibration Module
Implements probability calibration using Platt Scaling and Isotonic Regression
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrate model probabilities using various methods.
    """
    
    def __init__(self, method: str = 'isotonic', cv_folds: int = 5):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', 'beta')
            cv_folds: Number of CV folds for out-of-fold predictions
        """
        self.method = method
        self.cv_folds = cv_folds
        self.calibrator = None
        self.is_fitted = False
        self.calibration_data = None
        
    def _get_calibrator(self):
        """Create the appropriate calibrator."""
        if self.method == 'platt':
            return CalibratedClassifierCV(
                method='sigmoid',
                cv='prefit'
            )
        elif self.method == 'isotonic':
            return CalibratedClassifierCV(
                method='isotonic',
                cv='prefit'
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, 
            model: Any, 
            X: np.ndarray, 
            y: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit the calibrator on model predictions.
        
        Args:
            model: Trained base model
            X: Features
            y: Target
            
        Returns:
            self
        """
        # Get out-of-fold predictions
        logger.info("Computing out-of-fold predictions for calibration...")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True)
        proba_oof = cross_val_predict(
            model, X, y, 
            cv=cv, 
            method='predict_proba'
        )[:, 1]
        
        self.calibration_data = {
            'proba': proba_oof,
            'y_true': y
        }
        
        # Fit calibrator
        self.calibrator = self._get_calibrator()
        self.calibrator.fit(proba_oof.reshape(-1, 1), y)
        self.is_fitted = True
        
        logger.info(f"Calibrator fitted using {self.method} method")
        return self
    
    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.
        
        Args:
            raw_proba: Raw uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        return self.calibrator.transform(raw_proba.reshape(-1, 1)).flatten()
    
    def get_calibration_curve(self, 
                               y_true: np.ndarray, 
                               y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate calibration curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Tuple of (fraction_positive, mean_predicted)
        """
        fraction_positive, mean_predicted = calibration_curve(
            y_true, y_proba, 
            n_bins=10,
            strategy='uniform'
        )
        return fraction_positive, mean_predicted


class EnsembleCalibrator:
    """
    Combine multiple calibration methods using ensemble approach.
    """
    
    def __init__(self, cv_folds: int = 5):
        """
        Initialize ensemble calibrator.
        
        Args:
            cv_folds: CV folds for OOF predictions
        """
        self.cv_folds = cv_folds
        self.platt_calibrator = ProbabilityCalibrator('platt', cv_folds)
        self.isotonic_calibrator = ProbabilityCalibrator('isotonic', cv_folds)
        self.is_fitted = False
        
    def fit(self, 
            model: Any, 
            X: np.ndarray, 
            y: np.ndarray) -> 'EnsembleCalibrator':
        """
        Fit all calibrators.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            
        Returns:
            self
        """
        self.platt_calibrator.fit(model, X, y)
        self.isotonic_calibrator.fit(model, X, y)
        self.is_fitted = True
        return self
    
    def calibrate(self, 
                  raw_proba: np.ndarray,
                  method: str = 'average') -> np.ndarray:
        """
        Calibrate using specified method.
        
        Args:
            raw_proba: Raw probabilities
            method: 'platt', 'isotonic', 'average', or 'weighted'
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrators not fitted. Call fit() first.")
        
        if method == 'platt':
            return self.platt_calibrator.calibrate(raw_proba)
        elif method == 'isotonic':
            return self.isotonic_calibrator.calibrate(raw_proba)
        elif method == 'average':
            platt_proba = self.platt_calibrator.calibrate(raw_proba)
            isotonic_proba = self.isotonic_calibrator.calibrate(raw_proba)
            return (platt_proba + isotonic_proba) / 2
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_calibration(self, 
                            y_true: np.ndarray, 
                            y_proba: np.ndarray) -> Dict[str, Dict]:
        """
        Compare different calibration methods.
        
        Args:
            y_true: True labels
            y_proba: Raw probabilities
            
        Returns:
            Dictionary with metrics for each method
        """
        from sklearn.metrics import brier_score_loss, log_loss
        
        results = {}
        
        for method in ['platt', 'isotonic', 'average']:
            if method == 'average':
                calibrated = self.calibrate(y_proba, method='average')
            else:
                calibrator = getattr(self, f'{method}_calibrator')
                calibrated = calibrator.calibrate(y_proba)
            
            results[method] = {
                'brier_score': brier_score_loss(y_true, calibrated),
                'log_loss': log_loss(y_true, calibrated),
                'ece': expected_calibration_error(y_true, calibrated)
            }
        
        return results


def expected_calibration_error(y_true: np.ndarray, 
                                y_proba: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            # Weight by proportion of samples in bin
            weight = mask.sum() / len(y_true)
            # Calculate accuracy in bin
            acc = y_true[mask].mean()
            # Calculate confidence (average probability in bin)
            conf = y_proba[mask].mean()
            # Add weighted absolute difference
            ece += weight * abs(acc - conf)
    
    return ece


def maximum_calibration_error(y_true: np.ndarray, 
                               y_proba: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        MCE value
    """
    bins = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0
    
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_proba[mask].mean()
            max_error = max(max_error, abs(acc - conf))
    
    return max_error


def plot_calibration_comparison(y_true: np.ndarray,
                                 raw_proba: np.ndarray,
                                 calibrated_proba: Dict[str, np.ndarray],
                                 save_path: str = None):
    """
    Plot calibration curves comparing raw vs calibrated.
    
    Args:
        y_true: True labels
        raw_proba: Raw uncalibrated probabilities
        calibrated_proba: Dict of method -> calibrated probabilities
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    
    # Raw predictions
    frac_pos, mean_pred = calibration_curve(y_true, raw_proba, n_bins=10)
    ax1.plot(mean_pred, frac_pos, 'o-', label='Raw Model', linewidth=2)
    
    # Calibrated predictions
    for method, proba in calibrated_proba.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax1.plot(mean_pred, frac_pos, 'o-', label=f'Calibrated ({method})', linewidth=2)
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reliability diagram (histogram)
    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    
    # Raw distribution
    ax2.hist(raw_proba, bins=bins, alpha=0.5, label='Raw', density=True)
    
    # Calibrated distributions
    for method, proba in calibrated_proba.items():
        ax2.hist(proba, bins=bins, alpha=0.5, label=f'Calibrated ({method})', density=True)
    
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def get_calibrated_model(model: Any, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          method: str = 'isotonic') -> CalibratedClassifierCV:
    """
    Get a fully calibrated model.
    
    Args:
        model: Base model
        X: Features
        y: Target
        method: Calibration method
        
    Returns:
        CalibratedClassifierCV wrapper
    """
    calibrator = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=5
    )
    calibrator.fit(X, y)
    return calibrator