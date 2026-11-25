"""
Feature Engineering Module for Airline No-Show Prediction
Creates meaningful features from raw booking data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, reference_stats: Dict = None) -> pd.DataFrame:
    """
    Create engineered features from raw booking data.
    
    Args:
        df: Raw DataFrame with booking data
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    reference_stats = reference_stats or {}
    
    # ========================================
    # TEMPORAL FEATURES
    # ========================================
    
    # Booking lead time (days between booking and flight)
    for date_column in ('booking_date', 'flight_date'):
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    if 'booking_date' in df.columns and 'flight_date' in df.columns:
        df['booking_lead_time'] = (df['flight_date'] - df['booking_date']).dt.days
        df['booking_lead_time'] = df['booking_lead_time'].clip(lower=0)
    
    # Flight temporal features
    if 'flight_date' in df.columns:
        df['flight_dayofweek'] = df['flight_date'].dt.dayofweek
        df['flight_month'] = df['flight_date'].dt.month
        df['flight_day'] = df['flight_date'].dt.day
        df['flight_quarter'] = df['flight_date'].dt.quarter
        df['flight_is_month_start'] = df['flight_date'].dt.is_month_start.astype(int)
        df['flight_is_month_end'] = df['flight_date'].dt.is_month_end.astype(int)
    
    # Weekend indicator
    if 'flight_dayofweek' in df.columns:
        df['is_weekend'] = (df['flight_dayofweek'] >= 5).astype(int)
        df['is_friday'] = (df['flight_dayofweek'] == 4).astype(int)
        df['is_monday'] = (df['flight_dayofweek'] == 0).astype(int)
    
    # Holiday periods
    if 'flight_month' in df.columns:
        df['is_holiday_period'] = df['flight_month'].isin([11, 12]).astype(int)  # Nov-Dec holidays
        df['is_summer'] = df['flight_month'].isin([6, 7, 8]).astype(int)  # Summer vacation
        df['is_winter'] = df['flight_month'].isin([12, 1, 2]).astype(int)  # Winter
        df['is_spring_break'] = df['flight_month'].isin([3, 4]).astype(int)  # Spring break
        df['is_leave_period'] = df['flight_month'].isin([7, 8, 12]).astype(int)
    
    # Time of day categories
    if 'flight_hour' in df.columns or 'departure_time' in df.columns:
        if 'flight_hour' not in df.columns and 'departure_time' in df.columns:
            df['flight_hour'] = pd.to_datetime(df['departure_time'], errors='coerce').dt.hour
        
        df['is_morning'] = ((df.get('flight_hour', 0) >= 6) & (df.get('flight_hour', 0) < 12)).astype(int)
        df['is_afternoon'] = ((df.get('flight_hour', 0) >= 12) & (df.get('flight_hour', 0) < 18)).astype(int)
        df['is_evening'] = ((df.get('flight_hour', 0) >= 18) & (df.get('flight_hour', 0) < 22)).astype(int)
        df['is_night'] = ((df.get('flight_hour', 0) >= 22) | (df.get('flight_hour', 0) < 6)).astype(int)
    
    # ========================================
    # PASSENGER HISTORY FEATURES
    # ========================================
    
    # Historical no-show rate
    if 'total_bookings' in df.columns and 'total_noshows' in df.columns:
        df['historical_noshow_rate'] = np.where(
            df['total_bookings'] > 0,
            df['total_noshows'] / df['total_bookings'].clip(lower=1),
            0.08  # Default for new passengers
        )
        df['has_noshow_history'] = (df['total_noshows'] > 0).astype(int)
        df['high_noshow_risk'] = (df['historical_noshow_rate'] > 0.15).astype(int)
        df['very_high_noshow_risk'] = (df['historical_noshow_rate'] > 0.25).astype(int)
    
    # Cancellation history
    if 'total_cancellations' in df.columns:
        df['has_cancellation_history'] = (df['total_cancellations'] > 0).astype(int)
        df['cancellation_rate'] = np.where(
            df['total_bookings'] > 0,
            df['total_cancellations'] / df['total_bookings'].clip(lower=1),
            0
        )
    
    # Passenger experience
    if 'total_bookings' in df.columns:
        df['is_new_passenger'] = (df['total_bookings'] <= 2).astype(int)
        df['is_frequent_flyer'] = (df['total_bookings'] >= 10).astype(int)
        df['passenger_experience'] = np.clip(df['total_bookings'] / 20, 0, 1)
        df['passenger_seniority'] = np.where(
            df['total_bookings'] < 5, 'NEW',
            np.where(df['total_bookings'] < 15, 'REGULAR', 'FREQUENT')
        )
    
    # ========================================
    # LOYALTY FEATURES
    # ========================================
    
    if 'loyalty_tier' in df.columns:
        tier_map = {'STANDARD': 0, 'SILVER': 1, 'GOLD': 2, 'PLATINUM': 3}
        df['loyalty_tier_encoded'] = df['loyalty_tier'].map(tier_map).fillna(0)
        df['is_premium_tier'] = df['loyalty_tier_encoded'].isin([2, 3]).astype(int)
        df['is_silver_plus'] = df['loyalty_tier_encoded'].isin([1, 2, 3]).astype(int)
    elif 'loyalty_tier_encoded' in df.columns:
        df['is_premium_tier'] = df['loyalty_tier_encoded'].isin([2, 3]).astype(int)
        df['is_silver_plus'] = df['loyalty_tier_encoded'].isin([1, 2, 3]).astype(int)
    
    # ========================================
    # CABIN CLASS FEATURES
    # ========================================
    
    if 'cabin_class' in df.columns:
        cabin_map = {'ECONOMY': 0, 'PREMIUM_ECONOMY': 1, 'BUSINESS': 2, 'FIRST': 3}
        df['cabin_class_encoded'] = df['cabin_class'].map(cabin_map).fillna(0)
        df['is_premium_cabin'] = df['cabin_class_encoded'].isin([2, 3]).astype(int)
        df['is_economy'] = (df['cabin_class_encoded'] == 0).astype(int)
        df['is_first_class'] = (df['cabin_class_encoded'] == 3).astype(int)
    elif 'cabin_class_encoded' in df.columns:
        df['is_premium_cabin'] = df['cabin_class_encoded'].isin([2, 3]).astype(int)
        df['is_economy'] = (df['cabin_class_encoded'] == 0).astype(int)
        df['is_first_class'] = (df['cabin_class_encoded'] == 3).astype(int)
    
    # ========================================
    # BOOKING CHANNEL FEATURES
    # ========================================
    
    if 'booking_channel' in df.columns:
        channel_map = {'WEB': 0, 'MOBILE': 1, 'AGENT': 2, 'CALL_CENTER': 3, 'ONLINE': 0, 'APP': 1}
        df['booking_channel_encoded'] = df['booking_channel'].map(channel_map).fillna(0)
        df['is_direct_booking'] = df['booking_channel_encoded'].isin([0, 1]).astype(int)
        df['is_agent_booking'] = df['booking_channel_encoded'].isin([2, 3]).astype(int)
        df['is_online_booking'] = df['booking_channel_encoded'].isin([0, 1]).astype(int)
    elif 'booking_channel_encoded' in df.columns:
        df['is_direct_booking'] = df['booking_channel_encoded'].isin([0, 1]).astype(int)
        df['is_agent_booking'] = df['booking_channel_encoded'].isin([2, 3]).astype(int)
        df['is_online_booking'] = df['booking_channel_encoded'].isin([0, 1]).astype(int)
    
    # ========================================
    # PRICE FEATURES
    # ========================================
    
    if 'ticket_price' in df.columns:
        avg_price = reference_stats.get('average_ticket_price', df['ticket_price'].mean())
        lower_price = reference_stats.get('ticket_price_q25', df['ticket_price'].quantile(0.25))
        upper_price = reference_stats.get('ticket_price_q75', df['ticket_price'].quantile(0.75))
        df['price_vs_average'] = df['ticket_price'] / avg_price if avg_price > 0 else 1
        df['is_high_price'] = (df['ticket_price'] > upper_price).astype(int)
        df['is_low_price'] = (df['ticket_price'] < lower_price).astype(int)
        df['is_mid_price'] = ((df['ticket_price'] >= lower_price) & 
                              (df['ticket_price'] <= upper_price)).astype(int)
        
        # Price tier
        df['price_tier'] = pd.cut(
            df['ticket_price'],
            bins=[0, 200, 400, 700, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(0)
        
        # Log price (often better for modeling)
        df['log_price'] = np.log1p(df['ticket_price'])
    
    # ========================================
    # BOOKING WINDOW FEATURES
    # ========================================
    
    if 'booking_lead_time' in df.columns:
        df['last_minute_booking'] = (df['booking_lead_time'] <= 3).astype(int)
        df['short_notice'] = (df['booking_lead_time'] <= 7).astype(int)
        df['advance_booking'] = (df['booking_lead_time'] >= 14).astype(int)
        df['very_advance'] = (df['booking_lead_time'] >= 30).astype(int)
        df['months_advance'] = (df['booking_lead_time'] / 30).clip(0, 12)
        
        # Booking window categories
        df['booking_window'] = pd.cut(
            df['booking_lead_time'],
            bins=[-1, 3, 7, 14, 30, 60, float('inf')],
            labels=['last_minute', 'short', 'week', 'half_month', 'month', 'advance']
        ).astype(str).fillna('unknown')
    
    # ========================================
    # ROUTE FEATURES
    # ========================================
    
    # Create route identifier
    if 'departure_airport' in df.columns and 'arrival_airport' in df.columns:
        df['route'] = df['departure_airport'].astype(str) + '-' + df['arrival_airport'].astype(str)

    airport_categories = reference_stats.get('airport_categories', [])
    for column in ('departure_airport', 'arrival_airport'):
        if column in df.columns:
            values = df[column].astype(str)
            for airport in airport_categories:
                df[f'{column}_{airport}'] = (values == airport).astype(int)
    
    # ========================================
    # INTERACTION FEATURES
    # ========================================
    
    # New passenger + last minute booking (high risk)
    if 'is_new_passenger' in df.columns and 'last_minute_booking' in df.columns:
        df['new_last_minute'] = (df['is_new_passenger'] & df['last_minute_booking']).astype(int)
    
    # Weekend + economy (common leisure travel)
    if 'is_weekend' in df.columns and 'is_economy' in df.columns:
        df['weekend_economy'] = (df['is_weekend'] & df['is_economy']).astype(int)
    
    # High no-show risk + new passenger
    if 'high_noshow_risk' in df.columns and 'is_new_passenger' in df.columns:
        df['high_risk_new'] = (df['high_noshow_risk'] & df['is_new_passenger']).astype(int)
    
    # Economy + short notice
    if 'is_economy' in df.columns and 'short_notice' in df.columns:
        df['economy_short_notice'] = (df['is_economy'] & df['short_notice']).astype(int)
    
    # Price sensitivity (low price + economy)
    if 'is_low_price' in df.columns and 'is_economy' in df.columns:
        df['budget_traveler'] = (df['is_low_price'] & df['is_economy']).astype(int)
    
    # Premium passenger
    if 'is_premium_tier' in df.columns and 'is_premium_cabin' in df.columns:
        df['premium_passenger'] = (df['is_premium_tier'] | df['is_premium_cabin']).astype(int)
    
    # ========================================
    # DERIVED RISK SCORE
    # ========================================
    
    # Create a simple heuristic risk score (for reference, not for modeling)
    if 'historical_noshow_rate' in df.columns:
        if 'is_premium_tier' not in df.columns:
            df['is_premium_tier'] = 0
        df['simple_risk_score'] = (
            df['historical_noshow_rate'] * 3 +
            df['is_new_passenger'] * 0.1 +
            df['last_minute_booking'] * 0.1 +
            (1 - df['is_premium_tier']) * 0.05
        )
    
    logger.info(f"Feature engineering complete: {df.shape[1]} columns")
    return df


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions for all engineered features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'booking_lead_time': 'Days between booking and flight date',
        'flight_dayofweek': 'Day of week (0=Monday, 6=Sunday)',
        'flight_month': 'Month of flight (1-12)',
        'is_weekend': '1 if flight is on weekend',
        'is_holiday_period': '1 if flight during holiday season (Nov-Dec)',
        'is_summer': '1 if flight during summer (Jun-Aug)',
        'is_morning': '1 if flight between 6am-12pm',
        'is_afternoon': '1 if flight between 12pm-6pm',
        'historical_noshow_rate': 'Historical no-shows / total bookings',
        'has_noshow_history': '1 if passenger has previous no-shows',
        'high_noshow_risk': '1 if historical no-show rate > 15%',
        'is_new_passenger': '1 if passenger has <=2 bookings',
        'is_frequent_flyer': '1 if passenger has 10+ bookings',
        'loyalty_tier_encoded': '0=STANDARD, 1=SILVER, 2=GOLD, 3=PLATINUM',
        'is_premium_tier': '1 if GOLD or PLATINUM tier',
        'cabin_class_encoded': '0=ECONOMY, 1=PREMIUM_ECONOMY, 2=BUSINESS, 3=FIRST',
        'is_premium_cabin': '1 if BUSINESS or FIRST class',
        'is_economy': '1 if ECONOMY class',
        'booking_channel_encoded': '0=WEB, 1=MOBILE, 2=AGENT, 3=CALL_CENTER',
        'is_direct_booking': '1 if booking via WEB or MOBILE',
        'price_vs_average': 'Ticket price / average price',
        'is_high_price': '1 if ticket price > 75th percentile',
        'is_low_price': '1 if ticket price < 25th percentile',
        'last_minute_booking': '1 if booking <=3 days before flight',
        'short_notice': '1 if booking <=7 days before flight',
        'advance_booking': '1 if booking >=14 days before flight',
        'new_last_minute': 'New passenger with last-minute booking',
        'weekend_economy': 'Weekend flight in economy class',
        'high_risk_new': 'High historical risk + new passenger'
    }


def select_features(df: pd.DataFrame, 
                    exclude_cols: List[str] = None) -> List[str]:
    """
    Select features suitable for modeling.
    
    Args:
        df: DataFrame with all features
        exclude_cols: Columns to exclude
        
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = [
            'booking_id', 'passenger_id', 'flight_id', 'booking_code',
            'passenger_code', 'flight_code', 'first_name', 'last_name',
            'email', 'phone', 'booking_date', 'flight_date', 'departure_time',
            'created_at', 'updated_at', 'no_show', 'route', 'booking_window',
            'passenger_seniority', 'total_bookings', 'total_noshows',
            'total_cancellations', 'total_flights', 'historical_noshow_rate',
            'has_noshow_history', 'high_noshow_risk', 'very_high_noshow_risk',
            'has_cancellation_history', 'cancellation_rate', 'is_new_passenger',
            'is_frequent_flyer', 'passenger_experience', 'simple_risk_score',
            'new_last_minute', 'high_risk_new'
        ]
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Remove columns with all same values
    features = [col for col in features if df[col].nunique() > 1]
    
    # Remove columns with too many missing values
    features = [col for col in features if df[col].isna().mean() < 0.5]
    
    return features


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    
    # Create sample data
    sample_data = {
        'booking_date': ['2024-01-15', '2024-02-20', '2024-03-10'],
        'flight_date': ['2024-02-01', '2024-03-01', '2024-03-20'],
        'departure_airport': ['JFK', 'LAX', 'ORD'],
        'arrival_airport': ['LAX', 'JFK', 'MIA'],
        'ticket_price': [350, 420, 280],
        'total_bookings': [5, 15, 2],
        'total_noshows': [1, 2, 0],
        'loyalty_tier': ['SILVER', 'PLATINUM', 'STANDARD'],
        'cabin_class': ['ECONOMY', 'BUSINESS', 'ECONOMY'],
        'booking_channel': ['WEB', 'MOBILE', 'AGENT'],
        'no_show': [0, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    
    engineered = engineer_features(df)
    print("Engineered features shape:", engineered.shape)
    print("Features:", list(engineered.columns))