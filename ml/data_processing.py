"""
Data Processing Module - Data loading, cleaning, and preprocessing
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate datasets."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV file.
        
        Args:
            filename: Name of CSV file
            
        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def load_training_data(self) -> pd.DataFrame:
        """Load the main training dataset."""
        return self.load_csv('training_data.csv')
    
    def validate_dataset(self, df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate dataset quality.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required columns
            
        Returns:
            Validation report
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': {},
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'is_valid': True,
            'issues': []
        }
        
        # Check required columns
        if required_columns:
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                report['is_valid'] = False
                report['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check missing values per column
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = {
                    'count': int(missing),
                    'percent': round(missing / len(df) * 100, 2)
                }
        
        # Overall missing percentage
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        report['total_missing_percent'] = round(total_missing / total_cells * 100, 2)
        
        if report['total_missing_percent'] > 10:
            report['issues'].append(f"High missing value rate: {report['total_missing_percent']}%")
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            report['issues'].append(f"Duplicate columns: {duplicate_cols}")
        
        return report


class DataCleaner:
    """Clean and preprocess data."""
    
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def identify_column_types(self, df: pd.DataFrame):
        """Identify numeric and categorical columns."""
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numeric columns: {len(self.numeric_columns)}")
        logger.info(f"Categorical columns: {len(self.categorical_columns)}")
    
    def handle_missing_values(self,
                               df: pd.DataFrame,
                               strategy: str = 'median',
                               columns: List[str] = None) -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'median', 'mean', 'mode', 'constant'
            columns: Specific columns (None for all)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        columns = columns or df.columns
        
        for col in columns:
            if col not in df.columns:
                continue
            
            missing = df[col].isnull()
            if missing.sum() == 0:
                continue
            
            if strategy == 'drop':
                df = df.dropna(subset=[col])
                
            elif strategy == 'median':
                if df[col].dtype in ['int64', 'float64']:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    
            elif strategy == 'mean':
                if df[col].dtype in ['int64', 'float64']:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    
            elif strategy == 'mode':
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                
            elif strategy == 'constant':
                df[col] = df[col].fillna('Unknown')
        
        logger.info(f"Handled missing values using '{strategy}' strategy")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider (None for all)
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = initial_rows - len(df)
        
        logger.info(f"Removed {removed} duplicate rows")
        return df
    
    def remove_outliers(self,
                        df: pd.DataFrame,
                        columns: List[str] = None,
                        method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None for all numeric)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            DataFrame without outliers
        """
        df = df.copy()
        columns = columns or self.numeric_columns
        
        initial_rows = len(df)
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    continue
                z_scores = (df[col] - mean) / std
                mask = np.abs(z_scores) < threshold
            
            df = df[mask]
        
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} outlier rows")
        return df
    
    def encode_categorical(self,
                           df: pd.DataFrame,
                           columns: List[str] = None,
                           encoding: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (None for all categorical)
            encoding: 'label' or 'onehot'
            
        Returns:
            DataFrame with encoded columns
        """
        df = df.copy()
        columns = columns or self.categorical_columns
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if encoding == 'label':
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
            elif encoding == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
        
        logger.info(f"Encoded {len(columns)} categorical columns using '{encoding}' encoding")
        return df
    
    def scale_numeric(self,
                       df: pd.DataFrame,
                       columns: List[str] = None,
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (None for all numeric)
            fit: Whether to fit scaler (False for test data)
            
        Returns:
            DataFrame with scaled columns
        """
        df = df.copy()
        columns = columns or [c for c in self.numeric_columns if c in df.columns]
        
        if fit:
            self.scaler.fit(df[columns])
        
        df[columns] = self.scaler.transform(df[columns])
        
        logger.info(f"Scaled {len(columns)} numeric columns")
        return df


def preprocess_data(df: pd.DataFrame,
                    target_column: str = 'no_show',
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple:
    """
    Complete data preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        target_column: Name of target column
        test_size: Proportion for test split
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessing_objects)
    """
    # Initialize processors
    cleaner = DataCleaner()
    
    # Identify column types
    cleaner.identify_column_types(df)
    
    # Validate target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Clean data
    df = cleaner.handle_missing_values(df, strategy='median')
    df = cleaner.remove_duplicates(df)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical columns
    X = cleaner.encode_categorical(X, encoding='label')
    
    # Select only numeric columns for modeling
    feature_columns = [c for c in X.columns if c in cleaner.numeric_columns or c.endswith('_encoded')]
    X = X[feature_columns]
    
    # Handle any remaining missing values
    X = X.fillna(0)
    
    # Scale features
    X = cleaner.scale_numeric(X, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Store preprocessing objects
    preprocessing = {
        'label_encoders': cleaner.label_encoders,
        'scaler': cleaner.scaler,
        'feature_columns': feature_columns,
        'numeric_columns': cleaner.numeric_columns,
        'categorical_columns': cleaner.categorical_columns
    }
    
    logger.info(f"Preprocessing complete: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Features: {len(feature_columns)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, preprocessing


def load_and_preprocess(data_path: str = None) -> Tuple:
    """
    Load and preprocess the training dataset.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessing)
    """
    loader = DataLoader(data_path)
    
    # Load data
    df = loader.load_training_data()
    
    # Validate
    report = loader.validate_dataset(df)
    logger.info(f"Dataset validation: {'PASSED' if report['is_valid'] else 'FAILED'}")
    if report['issues']:
        for issue in report['issues']:
            logger.warning(f"  Issue: {issue}")
    
    # Preprocess
    return preprocess_data(df)