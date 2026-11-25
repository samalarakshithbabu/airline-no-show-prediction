"""
Data Processing Module for Airline No-Show Prediction
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV or database connection.
    
    Args:
        filepath: Path to CSV file or database connection string
        
    Returns:
        Raw DataFrame
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            # Assume database connection string
            import sqlalchemy
            engine = sqlalchemy.create_engine(filepath)
            df = pd.read_sql("SELECT * FROM bookings", engine)
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


class DataLoader:
    """Load and validate the repository's training dataset."""

    def __init__(self, data_path: str = None):
        repository_root = Path(__file__).resolve().parents[2]
        path = Path(data_path) if data_path else repository_root / 'data' / 'training_data.csv'
        if path.is_dir():
            candidates = (path / 'training_data.csv', path / 'data' / 'training_data.csv')
            path = next((candidate for candidate in candidates if candidate.is_file()), path)
        self.data_path = path

    def load_training_data(self) -> pd.DataFrame:
        if not self.data_path.is_file():
            raise FileNotFoundError(f'Training data not found: {self.data_path}')
        return load_data(str(self.data_path))

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing_values = {
            column: {'percent': round(float(percent), 2)}
            for column, percent in (df.isna().mean() * 100).items()
            if percent > 0
        }
        issues = []
        if df.empty:
            issues.append('Dataset is empty')
        if 'no_show' not in df.columns:
            issues.append("Target column 'no_show' is missing")
        elif df['no_show'].nunique() < 2:
            issues.append("Target column 'no_show' must contain both classes")
        return {
            'is_valid': not issues,
            'issues': issues,
            'missing_values': missing_values,
            'rows': len(df),
            'columns': len(df.columns),
        }


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = 'no_show',
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split raw rows before fitting feature engineering on training rows only."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    from ml.feature_engineering import engineer_features, select_features

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found")
    if df[target_column].nunique() < 2:
        raise ValueError('Training requires examples from both target classes')
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1')

    target = df[target_column].astype(int)
    excluded = {target_column, 'booking_id', 'passenger_id', 'flight_id'}
    raw_features = df.drop(columns=[column for column in excluded if column in df.columns])
    passenger_groups = df['passenger_id'] if 'passenger_id' in df.columns else None
    if passenger_groups is not None and passenger_groups.nunique() >= 2:
        n_splits = min(max(round(1 / test_size), 2), passenger_groups.nunique())
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        candidates = list(splitter.split(raw_features, target, groups=passenger_groups))
        train_indices, test_indices = min(
            candidates,
            key=lambda split: (
                abs(len(split[1]) / len(df) - test_size)
                + abs(target.iloc[split[1]].mean() - target.mean())
            ),
        )
        split_method = 'stratified_group_by_passenger'
    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        split_method = 'stratified_random'

    X_train_raw = raw_features.iloc[train_indices]
    X_test_raw = raw_features.iloc[test_indices]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    ticket_prices = pd.to_numeric(X_train_raw.get('ticket_price', pd.Series(dtype=float)), errors='coerce')
    reference_stats = {
        'average_ticket_price': float(ticket_prices.mean()) if not ticket_prices.empty else 350.0,
        'ticket_price_q25': float(ticket_prices.quantile(0.25)) if not ticket_prices.empty else 0.0,
        'ticket_price_q75': float(ticket_prices.quantile(0.75)) if not ticket_prices.empty else float('inf'),
        'airport_categories': sorted({
            str(value)
            for column in ('departure_airport', 'arrival_airport')
            if column in X_train_raw
            for value in X_train_raw[column].dropna().unique()
        }),
    }
    if not np.isfinite(reference_stats['average_ticket_price']) or reference_stats['average_ticket_price'] <= 0:
        reference_stats['average_ticket_price'] = 350.0
    X_train_engineered = engineer_features(X_train_raw, reference_stats)
    X_test_engineered = engineer_features(X_test_raw, reference_stats)
    feature_columns = select_features(X_train_engineered)
    X_train = X_train_engineered.reindex(columns=feature_columns, fill_value=0).apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)
    X_test = X_test_engineered.reindex(columns=feature_columns, fill_value=0).apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)
    preprocessing = {
        'feature_columns': feature_columns,
        'split_method': split_method,
        **reference_stats,
    }
    return X_train, X_test, y_train, y_test, preprocessing


def load_and_preprocess(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    loader = DataLoader(data_path)
    df = loader.load_training_data()
    report = loader.validate_dataset(df)
    if not report['is_valid']:
        raise ValueError(f"Invalid training dataset: {report['issues']}")
    return preprocess_data(df, test_size=test_size, random_state=random_state)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, duplicates, and invalid entries.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'UNKNOWN')
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    # Remove invalid dates
    if 'booking_date' in df.columns:
        df = df[pd.to_datetime(df['booking_date'], errors='coerce').notna()]
    if 'flight_date' in df.columns:
        df = df[pd.to_datetime(df['flight_date'], errors='coerce').notna()]
    
    # Remove negative prices
    if 'ticket_price' in df.columns:
        df = df[df['ticket_price'] > 0]
    
    # Remove invalid lead times
    if 'booking_lead_time' in df.columns:
        df = df[df['booking_lead_time'] >= 0]
    
    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def handle_outliers(df: pd.DataFrame, 
                    columns: list = None, 
                    method: str = 'iqr',
                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: all numeric)
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold multiplier for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df[col].dtype in [np.int64, np.float64]:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
            else:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
            
            logger.info(f"Handled {outliers} outliers in {col}")
    
    return df


def encode_categorical(df: pd.DataFrame, 
                       cat_cols: list = None, 
                       method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        cat_cols: Columns to encode (default: all object columns)
        method: Encoding method ('label', 'onehot', 'target')
        
    Returns:
        Tuple of (encoded DataFrame, encoding mappings)
    """
    df = df.copy()
    encodings = {}
    
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        if method == 'label':
            # Label encoding
            df[col] = df[col].astype(str)
            unique_vals = df[col].unique()
            encodings[col] = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            df[col] = df[col].map(encodings[col])
            df[col] = df[col].fillna(0)
        elif method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
            encodings[col] = list(dummies.columns)
    
    logger.info(f"Encoded {len(cat_cols)} categorical columns")
    return df, encodings


def prepare_features(df: pd.DataFrame, 
                     target_col: str = 'no_show',
                     feature_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for model training.
    
    Args:
        df: Input DataFrame
        target_col: Name of target variable column
        feature_cols: Specific features to use (auto-detect if None)
        
    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    df = df.copy()
    
    # Identify target column
    possible_names = ['no_show', 'no show', 'noshow', 'no-show', 
                      'no_show_flag', 'show_status', 'noshow']
    
    actual_target = None
    for name in possible_names:
        if name in df.columns:
            actual_target = name
            break
    
    if actual_target is None:
        raise ValueError(f"Target column not found. Expected one of: {possible_names}")
    
    # Rename to standard
    if actual_target != target_col:
        df[target_col] = df[actual_target]
        df = df.drop(actual_target, axis=1)
    
    # Detect feature columns if not provided
    if feature_cols is None:
        # Exclude non-feature columns
        exclude_cols = ['passenger_id', 'booking_id', 'flight_id', 
                        'booking_code', 'passenger_code', 'flight_code',
                        'first_name', 'last_name', 'email', 'phone',
                        'booking_date', 'flight_date', 'departure_time',
                        'created_at', 'updated_at', target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Filter to numeric columns only for modeling
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                numeric_features.append(col)
            elif df[col].nunique() < 50:  # Categorical with few values
                df[col] = df[col].astype(str)
                numeric_features.append(col)
        
        feature_cols = numeric_features
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining missing values
    X = X.fillna(0)
    
    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    
    logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
    return X, y


def split_data(X: pd.DataFrame, 
               y: pd.Series, 
               test_size: float = 0.2, 
               random_state: int = 42,
               stratify: bool = True) -> Tuple:
    """
    Split data into train and test sets with stratification.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion for test set
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_col
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


class DataProcessor:
    """
    Main data processing pipeline class.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data processor.
        
        Args:
            config: Processing configuration dictionary
        """
        self.config = config or {}
        self.feature_mappings = {}
        
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the complete data processing pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Tuple of (processed features, target)
        """
        # Step 1: Clean data
        df = clean_data(df)
        
        # Step 2: Handle outliers
        outlier_cols = self.config.get('outlier_columns')
        df = handle_outliers(df, columns=outlier_cols)
        
        # Step 3: Encode categorical
        cat_cols = self.config.get('categorical_columns')
        df, self.feature_mappings['encodings'] = encode_categorical(df, cat_cols)
        
        # Step 4: Prepare features
        feature_cols = self.config.get('feature_columns')
        X, y = prepare_features(df, feature_cols=feature_cols)
        
        return X, y