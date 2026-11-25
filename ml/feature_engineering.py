"""
Feature Engineering Module - Create and transform features for ML models
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for no-show prediction.
    
    Creates derived features from raw booking data.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names_ = []
        self.encoded_mappings_ = {}
        
        # Store means for numeric imputation
        self.numeric_means_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit feature engineer - compute statistics from training data."""
        # Compute means for numeric columns
        for col in X.select_dtypes(include=[np.number]).columns:
            self.numeric_means_[col] = X[col].mean()
        
        # Fit label encoders for categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            unique_vals = X[col].dropna().unique()
            self.encoded_mappings_[col] = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features into engineered features.
        
        Args:
            X: Raw DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        X = X.copy()
        
        # Ensure numeric types for key columns
        for col in ['total_bookings', 'total_noshows', 'total_cancellations']:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Parse dates if they're strings
        for date_col in ['booking_date', 'flight_date']:
            if date_col in X.columns:
                if X[date_col].dtype == 'object':
                    try:
                        X[date_col] = pd.to_datetime(X[date_col], errors='coerce')
                    except:
                        pass
        
        # Create time-based features from flight_date
        if 'flight_date' in X.columns and pd.api.types.is_datetime64_any_dtype(X['flight_date']):
            X['flight_dayofweek'] = X['flight_date'].dt.dayofweek
            X['flight_month'] = X['flight_date'].dt.month
            X['flight_day'] = X['flight_date'].dt.day
            X['flight_quarter'] = X['flight_date'].dt.quarter
            X['flight_hour'] = X['flight_date'].dt.hour if hasattr(X['flight_date'].dt, 'hour') else 12
        else:
            # Try to extract from raw data or use defaults
            X['flight_dayofweek'] = X.get('flight_dayofweek', 0)
            X['flight_month'] = X.get('flight_month', 1)
            X['flight_day'] = X.get('flight_day', 1)
            X['flight_quarter'] = X.get('flight_quarter', 1)
            X['flight_hour'] = X.get('flight_hour', 12)
        
        # Ensure numeric
        X['flight_dayofweek'] = pd.to_numeric(X['flight_dayofweek'], errors='coerce').fillna(0).astype(int)
        X['flight_month'] = pd.to_numeric(X['flight_month'], errors='coerce').fillna(1).astype(int)
        X['flight_day'] = pd.to_numeric(X['flight_day'], errors='coerce').fillna(1).astype(int)
        X['flight_quarter'] = pd.to_numeric(X['flight_quarter'], errors='coerce').fillna(1).astype(int)
        X['flight_hour'] = pd.to_numeric(X['flight_hour'], errors='coerce').fillna(12).astype(int)
        
        # Time-based binary features
        X['is_weekend'] = (X['flight_dayofweek'] >= 5).astype(int)
        X['is_friday'] = (X['flight_dayofweek'] == 4).astype(int)
        X['is_monday'] = (X['flight_dayofweek'] == 0).astype(int)
        
        # Seasonal features
        X['is_holiday_period'] = self._is_holiday_period(X)
        X['is_summer'] = X['flight_month'].isin([6, 7, 8]).astype(int)
        X['is_winter'] = X['flight_month'].isin([12, 1, 2]).astype(int)
        X['is_morning'] = ((X['flight_hour'] >= 6) & (X['flight_hour'] < 12)).astype(int)
        
        # Passenger history features
        X['has_noshow_history'] = (X['total_noshows'] > 0).astype(int)
        X['high_noshow_risk'] = (X['total_noshows'] >= 2).astype(int)
        X['has_cancellation_history'] = (X['total_cancellations'] > 0).astype(int)
        
        # Historical no-show rate
        X['historical_noshow_rate'] = np.where(
            X['total_bookings'] > 0,
            X['total_noshows'] / X['total_bookings'].clip(lower=1),
            0.08  # Default rate
        )
        
        # Loyalty tier encoding
        if 'loyalty_tier' in X.columns:
            tier_mapping = {'SILVER': 1, 'GOLD': 2, 'PLATINUM': 3, 'NONE': 0}
            X['loyalty_tier_encoded'] = X['loyalty_tier'].map(tier_mapping).fillna(0)
            X['is_premium_tier'] = X['loyalty_tier_encoded'].isin([2, 3]).astype(int)
        else:
            X['loyalty_tier_encoded'] = X.get('loyalty_tier_encoded', 0)
            X['is_premium_tier'] = X.get('is_premium_tier', 0)
        
        # Cabin class encoding
        if 'cabin_class' in X.columns:
            cabin_mapping = {'ECONOMY': 0, 'PREMIUM_ECONOMY': 1, 'BUSINESS': 2, 'FIRST': 3}
            X['cabin_class_encoded'] = X['cabin_class'].map(cabin_mapping).fillna(0)
            X['is_premium_cabin'] = X['cabin_class_encoded'].isin([2, 3]).astype(int)
            X['is_economy'] = (X['cabin_class'] == 'ECONOMY').astype(int)
        else:
            X['cabin_class_encoded'] = X.get('cabin_class_encoded', 0)
            X['is_premium_cabin'] = X.get('is_premium_cabin', 0)
            X['is_economy'] = X.get('is_economy', 1)
        
        # Booking channel encoding
        if 'booking_channel' in X.columns:
            channel_mapping = {'WEB': 0, 'MOBILE': 1, 'AGENT': 2, 'CORPORATE': 3}
            X['booking_channel_encoded'] = X['booking_channel'].map(channel_mapping).fillna(0)
            X['is_direct_booking'] = X['booking_channel'].isin(['WEB', 'MOBILE']).astype(int)
        else:
            X['booking_channel_encoded'] = X.get('booking_channel_encoded', 0)
            X['is_direct_booking'] = X.get('is_direct_booking', 1)
        
        # Price features
        avg_price = 350
        if 'ticket_price' in X.columns:
            avg_price = X['ticket_price'].mean()
            X['price_vs_average'] = X['ticket_price'] / avg_price
            X['is_high_price'] = (X['ticket_price'] > avg_price * 1.5).astype(int)
            X['is_low_price'] = (X['ticket_price'] < avg_price * 0.7).astype(int)
        else:
            X['price_vs_average'] = X.get('price_vs_average', 1.0)
            X['is_high_price'] = X.get('is_high_price', 0)
            X['is_low_price'] = X.get('is_low_price', 0)
        
        # Booking timing features
        lead_time = X.get('booking_lead_time', 14)
        if isinstance(lead_time, pd.Series):
            lead_time = lead_time.fillna(14)
        else:
            lead_time = pd.Series([lead_time]).fillna(14).iloc[0]
        
        X['last_minute_booking'] = (X.get('booking_lead_time', 14) < 3).astype(int)
        X['short_notice'] = ((X.get('booking_lead_time', 14) >= 3) & (X.get('booking_lead_time', 14) < 7)).astype(int)
        X['advance_booking'] = (X.get('booking_lead_time', 14) >= 7).astype(int)
        X['very_advance'] = (X.get('booking_lead_time', 14) >= 30).astype(int)
        X['new_last_minute'] = ((X.get('booking_lead_time', 14) < 7) & 
                                (X.get('is_new_passenger', 1) == 1)).astype(int)
        X['weekend_economy'] = (X['is_weekend'] & X['is_economy']).astype(int)
        
        # Passenger type features
        X['is_new_passenger'] = X.get('is_new_passenger', (X['total_bookings'] <= 1).astype(int))
        X['is_frequent_flyer'] = (X['total_bookings'] > 5).astype(int)
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        self.feature_names_ = list(X.columns)
        
        return X
    
    def _is_holiday_period(self, X: pd.DataFrame) -> pd.Series:
        """Identify holiday periods (simplified)."""
        month = X['flight_month']
        
        # Major US holidays (simplified)
        holiday_months = [11, 12, 7]  # Thanksgiving, Christmas, Summer
        
        return month.isin(holiday_months).astype(int)
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        return self.feature_names_
    
    def get_feature_importance(self, feature_importance: Dict[str, float]) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Args:
            feature_importance: Dictionary of feature -> importance
            
        Returns:
            DataFrame sorted by importance
        """
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in feature_importance.items()
        ])
        
        df = df.sort_values('importance', ascending=False)
        df['cumulative'] = df['importance'].cumsum()
        
        return df


def create_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create features from raw data.
    
    Args:
        raw_data: Raw booking data
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    engineer.fit(raw_data)
    return engineer.transform(raw_data)


# Feature list for the model
FEATURE_LIST = [
    # Core booking features
    'booking_lead_time',
    'flight_dayofweek',
    'flight_month',
    'flight_day',
    'flight_quarter',
    'flight_hour',
    
    # Time-based features
    'is_weekend',
    'is_friday',
    'is_monday',
    'is_holiday_period',
    'is_summer',
    'is_winter',
    'is_morning',
    
    # Loyalty features
    'loyalty_tier_encoded',
    'is_premium_tier',
    
    # Cabin features
    'cabin_class_encoded',
    'is_premium_cabin',
    'is_economy',
    
    # Channel features
    'booking_channel_encoded',
    'is_direct_booking',
    
    # Price features
    'ticket_price',
    'price_vs_average',
    'is_high_price',
    'is_low_price',
    
    # Booking timing
    'last_minute_booking',
    'short_notice',
    'advance_booking',
    'very_advance',
    'weekend_economy'
]


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering...")
    print()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'booking_lead_time': [14, 3, 45, 1, 7, 30, 2, 21],
        'flight_date': ['2024-12-15', '2024-12-20', '2024-07-10', '2024-11-28', 
                       '2024-03-15', '2024-08-01', '2024-06-15', '2024-04-20'],
        'total_bookings': [5, 2, 10, 1, 3, 15, 1, 8],
        'total_noshows': [1, 0, 2, 0, 0, 3, 0, 1],
        'total_cancellations': [0, 1, 1, 0, 0, 2, 0, 0],
        'loyalty_tier': ['SILVER', 'NONE', 'GOLD', 'NONE', 'SILVER', 'PLATINUM', 'NONE', 'GOLD'],
        'cabin_class': ['ECONOMY', 'ECONOMY', 'BUSINESS', 'ECONOMY', 'PREMIUM_ECONOMY', 'FIRST', 'ECONOMY', 'BUSINESS'],
        'booking_channel': ['WEB', 'MOBILE', 'AGENT', 'WEB', 'MOBILE', 'CORPORATE', 'WEB', 'WEB'],
        'ticket_price': [350, 280, 850, 320, 420, 2500, 300, 650]
    })
    
    # Parse dates
    sample_data['flight_date'] = pd.to_datetime(sample_data['flight_date'])
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.fit_transform(sample_data)
    
    print("Sample Features Created:")
    print(features[['booking_lead_time', 'is_weekend', 'historical_noshow_rate', 
                   'is_premium_tier', 'is_economy', 'is_direct_booking']].to_string())
    print()
    print(f"Total features: {len(engineer.get_feature_names())}")