"""
Drift Detection Service - Implements actual drift monitoring using PSI, KS, and Chi-square tests
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from sqlalchemy import text
from sqlalchemy.orm import Session
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""
    feature_name: str
    drift_detected: bool
    metric_name: str
    reference_value: float
    current_value: float
    change_percent: float
    threshold: float
    severity: str
    metric_value: float
    ks_statistic: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'drift_detected': self.drift_detected,
            'metric_name': self.metric_name,
            'reference_value': round(self.reference_value, 4),
            'current_value': round(self.current_value, 4),
            'change_percent': round(self.change_percent, 2),
            'threshold': self.threshold,
            'severity': self.severity,
            'metric_value': round(self.metric_value, 4),
            'ks_statistic': round(self.ks_statistic, 4)
        }


@dataclass
class DriftStatusReport:
    """Complete drift monitoring report."""
    timestamp: str
    status: str
    feature_drift: List[Dict[str, Any]]
    prediction_drift: Dict[str, Any]
    overall_drift_score: float
    alerts: List[str]
    recommendations: List[str]
    data_info: Dict[str, Any]
    performance_monitoring: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'status': self.status,
            'feature_drift': self.feature_drift,
            'prediction_drift': self.prediction_drift,
            'overall_drift_score': round(self.overall_drift_score, 4),
            'alerts': self.alerts,
            'recommendations': self.recommendations,
            'data_info': self.data_info,
            'performance_monitoring': self.performance_monitoring,
        }


class DriftService:
    """Service for monitoring model and data drift."""
    
    def __init__(self, db: Session = None, model_dir: str = 'models'):
        self.db = db
        self.model_dir = model_dir
        self.reference_data = None
        self.psi_threshold = 0.1
        self.severe_psi_threshold = 0.2
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load reference data from training dataset."""
        try:
            data_path = os.path.join(PROJECT_ROOT, 'data', 'training_data.csv')
            if os.path.exists(data_path):
                self.reference_data = pd.read_csv(data_path)
                logger.info(f"Loaded reference data: {len(self.reference_data)} records")
        except Exception as e:
            logger.warning(f"Could not load reference data: {e}")
            self.reference_data = None
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        all_values = np.concatenate([reference, current])
        min_val, max_val = np.min(all_values), np.max(all_values)
        
        if min_val == max_val:
            return 0.0
        
        edges = np.linspace(min_val, max_val, buckets + 1)
        ref_counts, _ = np.histogram(reference, bins=edges)
        curr_counts, _ = np.histogram(current, bins=edges)
        
        ref_pct = ref_counts / (len(reference) + 1e-10)
        curr_pct = curr_counts / (len(current) + 1e-10)
        
        ref_pct = np.clip(ref_pct, 1e-10, 1)
        curr_pct = np.clip(curr_pct, 1e-10, 1)
        
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)
    
    def _calculate_ks_statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        if len(reference) < 2 or len(current) < 2:
            return 0.0
        from scipy import stats
        ks_stat, _ = stats.ks_2samp(reference, current)
        return float(ks_stat)
    
    def detect_feature_drift(self, feature_name: str, ref_values: np.ndarray, curr_values: np.ndarray) -> FeatureDriftResult:
        """Detect drift for a single feature."""
        min_samples = 10
        if len(ref_values) < min_samples or len(curr_values) < min_samples:
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                metric_name='insufficient_data',
                reference_value=0,
                current_value=0,
                change_percent=0,
                threshold=self.psi_threshold,
                severity='low',
                metric_value=0,
                ks_statistic=0
            )
        
        psi = self._calculate_psi(ref_values, curr_values)
        ks_statistic = self._calculate_ks_statistic(ref_values, curr_values)
        ref_mean = float(np.mean(ref_values))
        curr_mean = float(np.mean(curr_values))
        
        if abs(ref_mean) > 1e-10:
            mean_change_pct = (curr_mean - ref_mean) / ref_mean * 100
        else:
            mean_change_pct = 0
        
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
        
        return FeatureDriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            metric_name='psi',
            reference_value=ref_mean,
            current_value=curr_mean,
            change_percent=round(mean_change_pct, 2),
            threshold=self.psi_threshold,
            severity=severity,
            metric_value=psi,
            ks_statistic=ks_statistic
        )
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """
        Get current booking data from database for drift comparison.
        Uses actual columns from the bookings and passengers tables.
        """
        if self.db is None:
            return None
        
        try:
            db_type = self.db.get_bind().dialect.name
            date_filter = (
                "b.booking_date >= datetime('now', '-30 days')"
                if db_type == 'sqlite'
                else "b.booking_date >= CURRENT_TIMESTAMP - INTERVAL '30 days'"
            )
            # Query with actual columns from bookings and passengers tables
            # Join passengers only for loyalty data available at booking time.
            query = text(f"""
                SELECT 
                    b.booking_lead_time,
                    b.ticket_price,
                    CASE 
                        WHEN b.cabin_class = 'ECONOMY' THEN 0
                        WHEN b.cabin_class = 'PREMIUM_ECONOMY' THEN 1
                        WHEN b.cabin_class = 'BUSINESS' THEN 2
                        WHEN b.cabin_class = 'FIRST' THEN 3
                        ELSE 0
                    END as cabin_class_encoded,
                    CASE 
                        WHEN b.booking_channel = 'WEB' THEN 0
                        WHEN b.booking_channel = 'MOBILE' THEN 1
                        WHEN b.booking_channel = 'AGENT' THEN 2
                        WHEN b.booking_channel = 'CALL_CENTER' THEN 3
                        ELSE 0
                    END as booking_channel_encoded,
                    CASE 
                        WHEN p.loyalty_tier = 'NONE' THEN 0
                        WHEN p.loyalty_tier = 'SILVER' THEN 1
                        WHEN p.loyalty_tier = 'GOLD' THEN 2
                        WHEN p.loyalty_tier = 'PLATINUM' THEN 3
                        ELSE 0
                    END as loyalty_tier_encoded,
                    b.noshow_probability
                FROM bookings b
                LEFT JOIN passengers p ON b.passenger_id = p.passenger_id
                WHERE {date_filter}
                LIMIT 1000
            """)
            result = self.db.execute(query).fetchall()
            
            if len(result) == 0:
                return None
            
            # Get column names from result
            columns = result[0].keys() if hasattr(result[0], 'keys') else [
                'booking_lead_time', 'ticket_price', 'cabin_class_encoded',
                'booking_channel_encoded', 'loyalty_tier_encoded', 'noshow_probability'
            ]
            df = pd.DataFrame([dict(zip(columns, row)) for row in result])
            
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            logger.warning(f"Could not fetch current data: {e}")
            return None

    def get_labeled_performance(self) -> Dict[str, Any]:
        """Evaluate stored predictions after bookings reach a final outcome."""
        if self.db is None:
            return {
                'status': 'unavailable',
                'reason': 'database_session_not_configured',
                'models': [],
            }

        query = text("""
            WITH ranked_labels AS (
                SELECT
                    p.booking_id,
                    p.model_name,
                    p.model_version,
                    p.prediction_class,
                    p.noshow_probability,
                    CASE WHEN UPPER(b.booking_status) = 'NOSHOW' THEN 1 ELSE 0 END AS actual_label,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.booking_id, p.model_name, p.model_version
                        ORDER BY p.prediction_timestamp DESC
                    ) AS prediction_rank
                FROM predictions p
                JOIN bookings b ON b.booking_id = p.booking_id
                WHERE p.is_active = TRUE
                  AND UPPER(b.booking_status) IN ('NOSHOW', 'COMPLETED')
            )
            SELECT model_name, model_version, prediction_class, noshow_probability, actual_label
            FROM ranked_labels
            WHERE prediction_rank = 1
            ORDER BY model_name, model_version
            LIMIT 5000
        """)
        try:
            rows = self.db.execute(query).mappings().all()
        except Exception as exc:
            logger.exception('Could not query labeled prediction outcomes')
            return {'status': 'error', 'reason': str(exc), 'models': []}

        if not rows:
            return {
                'status': 'awaiting_labeled_outcomes',
                'labeled_predictions': 0,
                'models': [],
            }

        frame = pd.DataFrame([dict(row) for row in rows])
        model_results = []
        for (model_name, model_version), group in frame.groupby(
            ['model_name', 'model_version'],
            dropna=False,
        ):
            actual = group['actual_label'].astype(int).to_numpy()
            predicted = group['prediction_class'].astype(int).to_numpy()
            probabilities = pd.to_numeric(
                group['noshow_probability'],
                errors='coerce',
            ).to_numpy(dtype=float)
            valid = (
                np.isfinite(probabilities)
                & (probabilities >= 0)
                & (probabilities <= 1)
                & np.isin(predicted, [0, 1])
            )
            actual = actual[valid]
            predicted = predicted[valid]
            probabilities = probabilities[valid]
            if not len(actual):
                continue
            both_classes = len(np.unique(actual)) == 2
            model_results.append({
                'model_name': model_name,
                'model_version': model_version,
                'labeled_predictions': int(len(actual)),
                'prevalence': float(actual.mean()),
                'accuracy': float(accuracy_score(actual, predicted)),
                'precision': float(precision_score(actual, predicted, zero_division=0)),
                'recall': float(recall_score(actual, predicted, zero_division=0)),
                'f1_score': float(f1_score(actual, predicted, zero_division=0)),
                'roc_auc': float(roc_auc_score(actual, probabilities)) if both_classes else None,
                'pr_auc': float(average_precision_score(actual, probabilities)) if both_classes else None,
                'brier_score': float(brier_score_loss(actual, probabilities)),
            })
        return {
            'status': 'available' if model_results else 'awaiting_labeled_outcomes',
            'labeled_predictions': int(sum(item['labeled_predictions'] for item in model_results)),
            'models': model_results,
        }
    
    def get_drift_status(self) -> DriftStatusReport:
        """Generate complete drift monitoring report."""
        timestamp = datetime.utcnow().isoformat()
        performance_monitoring = self.get_labeled_performance()
        current_data = self.get_current_data()
        
        if current_data is None or len(current_data) < 10:
            return DriftStatusReport(
                timestamp=timestamp,
                status='unknown',
                feature_drift=[],
                prediction_drift={'drift_detected': False, 'status': 'no_data'},
                overall_drift_score=0,
                alerts=['Insufficient current booking data for drift analysis'],
                recommendations=['Ensure recent booking data is available in the database'],
                performance_monitoring=performance_monitoring,
                data_info={
                    'reference_records': len(self.reference_data) if self.reference_data is not None else 0,
                    'current_records': len(current_data) if current_data is not None else 0,
                    'status': 'insufficient_data'
                }
            )
        
        if self.reference_data is None or len(self.reference_data) < 10:
            return DriftStatusReport(
                timestamp=timestamp,
                status='unknown',
                feature_drift=[],
                prediction_drift={'drift_detected': False, 'status': 'no_reference'},
                overall_drift_score=0,
                alerts=['Reference training data not available for comparison'],
                recommendations=['Upload training data (data/training_data.csv) to enable drift detection'],
                performance_monitoring=performance_monitoring,
                data_info={
                    'reference_records': 0,
                    'current_records': len(current_data),
                    'status': 'no_reference'
                }
            )
        
        # Map reference data columns to match current data
        # Reference data uses original column names
        reference_features = {
            'booking_lead_time': 'booking_lead_time',
            'ticket_price': 'ticket_price',
        }
        
        current_features = list(current_data.columns)
        
        feature_results = []
        alerts = []
        drift_scores = []
        
        for ref_col, curr_col in reference_features.items():
            if ref_col in self.reference_data.columns and curr_col in current_data.columns:
                ref_values = self.reference_data[ref_col].dropna().astype(float).values
                curr_values = current_data[curr_col].dropna().astype(float).values
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    result = self.detect_feature_drift(ref_col, ref_values, curr_values)
                    feature_results.append(result.to_dict())
                    
                    if result.drift_detected:
                        drift_scores.append(result.metric_value)
                        alerts.append(f"Data drift detected in '{ref_col}' (PSI={result.metric_value:.4f})")
        
        overall_score = float(np.mean(drift_scores)) if drift_scores else 0.0
        
        if alerts:
            critical_count = sum(1 for r in feature_results if r['severity'] == 'critical')
            if critical_count > 0:
                status = 'critical'
            elif any(r['severity'] == 'high' for r in feature_results):
                status = 'warning'
            else:
                status = 'warning'
        else:
            status = 'stable'
        
        recommendations = []
        if status == 'critical':
            recommendations = [
                'Investigate the cause of the significant feature-distribution shift',
                'Evaluate model performance against newly labeled outcomes before deciding whether to retrain',
                'Feature drift alone does not establish model-performance degradation'
            ]
        elif status == 'warning':
            recommendations = [
                'Evaluate model performance against labeled outcomes',
                'Monitor drift trends closely',
                'Do not infer model-performance degradation from feature drift alone'
            ]
        else:
            recommendations = [
                'No significant feature-distribution shift detected in the analyzed features',
                'Model performance still requires evaluation against labeled outcomes'
            ]
        
        prediction_drift = {
            'drift_detected': False,
            'status': 'unknown',
            'type': 'prediction_distribution',
            'note': (
                'Probability-distribution drift is not calculated. Labeled prediction '
                'performance is reported separately when outcomes are available.'
            )
        }
        
        return DriftStatusReport(
            timestamp=timestamp,
            status=status,
            feature_drift=feature_results,
            prediction_drift=prediction_drift,
            overall_drift_score=overall_score,
            alerts=alerts,
            recommendations=recommendations,
            performance_monitoring=performance_monitoring,
            data_info={
                'reference_records': len(self.reference_data),
                'current_records': len(current_data),
                'features_analyzed': len(feature_results),
                'status': 'analyzed'
            }
        )


def create_drift_service(db_session=None) -> DriftService:
    """Factory function to create drift service."""
    return DriftService(db=db_session, model_dir='models')


if __name__ == "__main__":
    print("=" * 60)
    print("DRIFT DETECTION SERVICE TEST")
    print("=" * 60)
    
    # Create a mock database session for testing
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine('sqlite:///instance/airline_noshow.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    service = DriftService(db=session)
    report = service.get_drift_status()
    
    session.close()
    
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Status: {report.status.upper()}")
    print(f"Features Analyzed: {len(report.feature_drift)}")
    print(f"Alerts: {len(report.alerts)}")
    
    if report.feature_drift:
        print("\nFeature Drift Results:")
        for feat in report.feature_drift:
            print(f"  {feat['feature_name']:25} PSI={feat['metric_value']:6.4f} [{feat['severity']:8}]")
    
    print(f"\nData Info: {report.data_info}")
    print(f"\nPrediction Drift: {report.prediction_drift}")