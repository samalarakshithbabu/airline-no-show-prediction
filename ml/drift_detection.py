"""
Drift Detection Module - Monitoring for model and data drift
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from scipy import stats
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature_name: str
    drift_detected: bool
    metric_name: str
    reference_value: float
    current_value: float
    change_percent: float
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'drift_detected': self.drift_detected,
            'metric_name': self.metric_name,
            'reference_value': round(self.reference_value, 4),
            'current_value': round(self.current_value, 4),
            'change_percent': round(self.change_percent, 2),
            'threshold': self.threshold,
            'severity': self.severity
        }


@dataclass
class DriftReport:
    """Complete drift monitoring report."""
    timestamp: str
    status: str  # 'stable', 'warning', 'critical'
    feature_drift: List[Dict[str, Any]]
    prediction_drift: Dict[str, Any]
    overall_drift_score: float
    alerts: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'status': self.status,
            'feature_drift': self.feature_drift,
            'prediction_drift': self.prediction_drift,
            'overall_drift_score': round(self.overall_drift_score, 4),
            'alerts': self.alerts,
            'recommendations': self.recommendations
        }


class DriftDetector:
    """
    Drift detection using multiple statistical methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov Test
    - Chi-Square Test
    - Jensen-Shannon Divergence
    """
    
    def __init__(self, psi_threshold: float = 0.1, severe_psi_threshold: float = 0.2):
        """
        Initialize drift detector.
        
        Args:
            psi_threshold: PSI threshold for warning (default 0.1)
            severe_psi_threshold: PSI threshold for critical alert (default 0.2)
        """
        self.psi_threshold = psi_threshold
        self.severe_psi_threshold = severe_psi_threshold
        
        # Store reference distributions
        self.reference_distributions = {}
        self.reference_stats = {}
    
    def fit_reference(self, reference_data: pd.DataFrame, feature_names: List[str] = None):
        """
        Fit the detector on reference data (training data).
        
        Args:
            reference_data: Reference dataset
            feature_names: Features to monitor (defaults to all numeric)
        """
        if feature_names is None:
            feature_names = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in feature_names:
            values = reference_data[feature].dropna()
            
            if len(values) == 0:
                continue
            
            # Store distribution parameters
            self.reference_distributions[feature] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75),
                'n': len(values)
            }
            
            # Store statistics
            self.reference_stats[feature] = {
                'mean': values.mean(),
                'std': values.std(),
                'skewness': values.skew(),
                'kurtosis': values.kurtosis()
            }
        
        logger.info(f"Drift detector fitted on {len(feature_names)} features")
    
    def calculate_psi(self,
                      reference: np.ndarray,
                      current: np.ndarray,
                      buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Minor change (warning)
        PSI >= 0.2: Significant change (alert)
        
        Args:
            reference: Reference distribution
            current: Current distribution
            buckets: Number of bins
            
        Returns:
            PSI value
        """
        # Handle edge cases
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Calculate bin edges using reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if min_val == max_val:
            return 0.0
        
        edges = np.linspace(min_val, max_val, buckets + 1)
        
        # Calculate bin frequencies
        ref_counts = np.histogram(reference, bins=edges)[0]
        curr_counts = np.histogram(current, bins=edges)[0]
        
        # Avoid division by zero
        ref_pct = ref_counts / (len(reference) + 1e-10)
        curr_pct = curr_counts / (len(current) + 1e-10)
        
        # Ensure non-zero for log calculation
        ref_pct = np.clip(ref_pct, 1e-10, 1)
        curr_pct = np.clip(curr_pct, 1e-10, 1)
        
        # Calculate PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return psi
    
    def calculate_ks_statistic(self,
                                reference: np.ndarray,
                                current: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            Tuple of (ks_statistic, p_value)
        """
        if len(reference) < 2 or len(current) < 2:
            return 0.0, 1.0
        
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return ks_stat, p_value
    
    def calculate_chi_square_test(self,
                                   reference: np.ndarray,
                                   current: np.ndarray,
                                   bins: int = 10) -> Tuple[float, float]:
        """
        Chi-square test for distribution comparison.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            Tuple of (chi2_statistic, p_value)
        """
        # Create bins
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if min_val == max_val:
            return 0.0, 1.0
        
        edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_counts = np.histogram(reference, bins=edges)[0]
        curr_counts = np.histogram(current, bins=edges)[0]
        
        # Chi-square test
        chi2, p_value = stats.chi2_contingency([ref_counts, curr_counts])[:2]
        
        return chi2, p_value
    
    def calculate_js_divergence(self,
                                 reference: np.ndarray,
                                 current: np.ndarray,
                                 bins: int = 10) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            JS divergence (0-1, lower is better)
        """
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if min_val == max_val:
            return 0.0
        
        edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
        curr_pct = np.histogram(current, bins=edges)[0] / len(current)
        
        # Clip to avoid log(0)
        ref_pct = np.clip(ref_pct, 1e-10, 1)
        curr_pct = np.clip(curr_pct, 1e-10, 1)
        
        # Normalize
        ref_pct = ref_pct / ref_pct.sum()
        curr_pct = curr_pct / curr_pct.sum()
        
        # Calculate JS divergence
        m = 0.5 * (ref_pct + curr_pct)
        js_div = 0.5 * (
            np.sum(ref_pct * np.log(ref_pct / m)) +
            np.sum(curr_pct * np.log(curr_pct / m))
        )
        
        return js_div
    
    def detect_feature_drift(self,
                              feature_name: str,
                              current_values: np.ndarray,
                              method: str = 'psi') -> DriftResult:
        """
        Detect drift for a single feature.
        
        Args:
            feature_name: Name of the feature
            current_values: Current feature values
            method: Detection method ('psi', 'ks', 'chi2', 'js')
            
        Returns:
            DriftResult object
        """
        if feature_name not in self.reference_distributions:
            logger.warning(f"Feature {feature_name} not in reference data")
            return DriftResult(
                feature_name=feature_name,
                drift_detected=False,
                metric_name='unknown',
                reference_value=0,
                current_value=0,
                change_percent=0,
                threshold=self.psi_threshold,
                severity='low'
            )
        
        ref_dist = self.reference_distributions[feature_name]
        ref_values = np.random.normal(
            ref_dist['mean'],
            ref_dist['std'],
            ref_dist['n']
        )  # Approximate from stored stats
        
        # Calculate drift metric
        if method == 'psi':
            psi = self.calculate_psi(ref_values, current_values)
            metric_name = 'psi'
            threshold = self.psi_threshold
            reference_value = ref_dist['mean']
            current_value = current_values.mean()
            change_percent = (current_value - reference_value) / (abs(reference_value) + 1e-10) * 100
            
        elif method == 'ks':
            ks_stat, p_value = self.calculate_ks_statistic(ref_values, current_values)
            psi = ks_stat
            metric_name = 'ks_statistic'
            threshold = 0.2  # Common KS threshold
            reference_value = 0.0
            current_value = ks_stat
            change_percent = ks_stat * 100
            
        elif method == 'chi2':
            chi2, p_value = self.calculate_chi_square_test(ref_values, current_values)
            psi = chi2 / 1000  # Scale for comparison
            metric_name = 'chi2_statistic'
            threshold = 0.1
            reference_value = 0.0
            current_value = chi2
            change_percent = 0
            
        else:  # js
            psi = self.calculate_js_divergence(ref_values, current_values)
            metric_name = 'js_divergence'
            threshold = 0.1
            reference_value = 0.0
            current_value = psi
            change_percent = psi * 100
        
        # Determine severity
        if psi >= self.severe_psi_threshold:
            severity = 'critical'
            drift_detected = True
        elif psi >= self.psi_threshold:
            severity = 'high'
            drift_detected = True
        elif psi >= self.psi_threshold * 0.5:
            severity = 'medium'
            drift_detected = False
        else:
            severity = 'low'
            drift_detected = False
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            metric_name=metric_name,
            reference_value=reference_value,
            current_value=current_value,
            change_percent=change_percent,
            threshold=threshold,
            severity=severity
        )
    
    def detect_all_drift(self,
                          current_data: pd.DataFrame,
                          features: List[str] = None) -> List[DriftResult]:
        """
        Detect drift for all monitored features.
        
        Args:
            current_data: Current data
            features: Features to check (defaults to reference features)
            
        Returns:
            List of DriftResult objects
        """
        if features is None:
            features = list(self.reference_distributions.keys())
        
        results = []
        for feature in features:
            if feature in current_data.columns:
                values = current_data[feature].dropna().values
                if len(values) > 0:
                    result = self.detect_feature_drift(feature, values)
                    results.append(result)
        
        return results
    
    def detect_prediction_drift(self,
                                 reference_predictions: np.ndarray,
                                 current_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Reference prediction distribution
            current_predictions: Current prediction distribution
            
        Returns:
            Dictionary with prediction drift metrics
        """
        # Calculate PSI for predictions
        psi = self.calculate_psi(reference_predictions, current_predictions)
        
        # Calculate mean shift
        ref_mean = reference_predictions.mean()
        curr_mean = current_predictions.mean()
        mean_shift = curr_mean - ref_mean
        mean_shift_percent = (mean_shift / (abs(ref_mean) + 1e-10)) * 100 if ref_mean != 0 else 0
        
        # Calculate distribution shift
        ref_var = reference_predictions.var()
        curr_var = current_predictions.var()
        var_shift = (curr_var - ref_var) / (ref_var + 1e-10) * 100 if ref_var != 0 else 0
        
        drift_detected = psi >= self.psi_threshold
        
        return {
            'drift_detected': drift_detected,
            'psi': round(psi, 4),
            'mean_shift': round(mean_shift, 4),
            'mean_shift_percent': round(mean_shift_percent, 2),
            'variance_shift_percent': round(var_shift, 2),
            'severity': 'critical' if psi >= self.severe_psi_threshold else 'warning' if psi >= self.psi_threshold else 'stable',
            'reference_mean': round(ref_mean, 4),
            'current_mean': round(curr_mean, 4)
        }
    
    def generate_report(self,
                        feature_drift_results: List[DriftResult],
                        prediction_drift: Dict[str, Any],
                        model_name: str = 'unknown') -> DriftReport:
        """
        Generate comprehensive drift monitoring report.
        
        Args:
            feature_drift_results: Results from feature drift detection
            prediction_drift: Result from prediction drift detection
            model_name: Name of the monitored model
            
        Returns:
            DriftReport object
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Convert results to dicts
        feature_drift = [r.to_dict() for r in feature_drift_results]
        
        # Calculate overall drift score
        drift_scores = [r.current_value for r in feature_drift_results if r.drift_detected]
        overall_score = np.mean(drift_scores) if drift_scores else 0.0
        
        # Determine overall status
        if prediction_drift.get('severity') == 'critical' or any(r.severity == 'critical' for r in feature_drift_results):
            status = 'critical'
        elif prediction_drift.get('severity') == 'warning' or any(r.severity == 'high' for r in feature_drift_results):
            status = 'warning'
        else:
            status = 'stable'
        
        # Generate alerts
        alerts = []
        recommendations = []
        
        # Feature drift alerts
        for r in feature_drift_results:
            if r.drift_detected:
                alerts.append(f"Drift detected in '{r.feature_name}' ({r.metric_name}={r.current_value:.4f})")
        
        # Prediction drift alerts
        if prediction_drift.get('drift_detected'):
            alerts.append(f"Prediction drift detected (PSI={prediction_drift.get('psi', 0):.4f})")
        
        # Recommendations
        if status == 'critical':
            recommendations.append("URGENT: Model retraining recommended")
            recommendations.append("Investigate root cause of feature drift")
            recommendations.append("Consider feature engineering updates")
        elif status == 'warning':
            recommendations.append("Schedule model evaluation")
            recommendations.append("Monitor drift trend closely")
            recommendations.append("Prepare for potential retraining")
        else:
            recommendations.append("Model performance stable - continue monitoring")
            recommendations.append("Maintain current monitoring schedule")
        
        # Feature-specific recommendations
        drifted_features = [r.feature_name for r in feature_drift_results if r.drift_detected]
        if drifted_features:
            recommendations.append(f"Review data pipeline for drifted features: {', '.join(drifted_features)}")
        
        return DriftReport(
            timestamp=timestamp,
            status=status,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            overall_drift_score=overall_score,
            alerts=alerts,
            recommendations=recommendations
        )


class PerformanceMonitor:
    """
    Monitor model performance over time.
    Tracks accuracy, precision, recall, F1, and other metrics.
    """
    
    def __init__(self, decay_rate: float = 0.01):
        """
        Initialize performance monitor.
        
        Args:
            decay_rate: Weight decay for moving average
        """
        self.decay_rate = decay_rate
        self.metrics_history = []
        self.baseline_metrics = None
    
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics
    
    def record_metrics(self, metrics: Dict[str, float], timestamp: datetime = None):
        """
        Record metrics from a prediction batch.
        
        Args:
            metrics: Dictionary of metric values
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.metrics_history.append({
            'timestamp': timestamp.isoformat(),
            **metrics
        })
    
    def calculate_drift(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate performance drift from baseline.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary with drift values
        """
        if self.baseline_metrics is None:
            return {k: 0.0 for k in current_metrics.keys()}
        
        drift = {}
        for key, value in current_metrics.items():
            if key in self.baseline_metrics:
                baseline = self.baseline_metrics[key]
                drift[key] = value - baseline
        
        return drift
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent = self.metrics_history[-100:]  # Last 100 entries
        
        summary = {
            'sample_count': len(self.metrics_history),
            'recent_metrics': {
                'accuracy': np.mean([m.get('accuracy', 0) for m in recent]),
                'precision': np.mean([m.get('precision', 0) for m in recent]),
                'recall': np.mean([m.get('recall', 0) for m in recent]),
                'f1': np.mean([m.get('f1', 0) for m in recent])
            }
        }
        
        if self.baseline_metrics:
            summary['baseline'] = self.baseline_metrics
            summary['drift'] = self.calculate_drift(summary['recent_metrics'])
        
        return summary


def run_drift_demo():
    """Run demonstration of drift detection."""
    print("=" * 60)
    print("DRIFT DETECTION DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Create detector
    detector = DriftDetector(psi_threshold=0.1, severe_psi_threshold=0.2)
    
    # Create synthetic reference data
    np.random.seed(42)
    ref_data = pd.DataFrame({
        'booking_lead_time': np.random.exponential(14, 1000),
        'ticket_price': np.random.normal(350, 100, 1000).clip(100, 800),
        'total_bookings': np.random.poisson(5, 1000),
        'historical_noshow_rate': np.random.beta(2, 20, 1000)
    })
    
    # Fit detector
    print("Fitting detector on reference data...")
    detector.fit_reference(ref_data)
    print(f"  Fitted on {len(detector.reference_distributions)} features")
    print()
    
    # Create current data with drift
    curr_data = pd.DataFrame({
        'booking_lead_time': np.random.exponential(10, 500),  # Shorter lead times
        'ticket_price': np.random.normal(380, 120, 500).clip(100, 800),  # Higher prices
        'total_bookings': np.random.poisson(6, 500),  # More bookings
        'historical_noshow_rate': np.random.beta(3, 25, 500)  # Lower no-show rate
    })
    
    # Detect drift
    print("Detecting drift...")
    drift_results = detector.detect_all_drift(curr_data)
    
    for result in drift_results:
        status = "DRIFT" if result.drift_detected else "OK"
        print(f"  {result.feature_name}: {result.metric_name}={result.current_value:.4f} [{status}]")
    
    print()
    
    # Prediction drift
    ref_preds = np.random.beta(0.15, 0.85, 1000)  # 15% average no-show prob
    curr_preds = np.random.beta(0.10, 0.90, 500)  # 10% average no-show prob
    
    pred_drift = detector.detect_prediction_drift(ref_preds, curr_preds)
    print(f"Prediction Drift: PSI={pred_drift['psi']:.4f} [{pred_drift['severity'].upper()}]")
    print()
    
    # Generate report
    report = detector.generate_report(drift_results, pred_drift, 'XGBoost_v1')
    
    print("Drift Report:")
    print(f"  Status: {report.status.upper()}")
    print(f"  Alerts: {len(report.alerts)}")
    print(f"  Recommendations:")
    for rec in report.recommendations:
        print(f"    - {rec}")
    
    print()
    print("Full Report:")
    print(json.dumps(report.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    run_drift_demo()