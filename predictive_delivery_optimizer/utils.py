"""
Utility functions for the Predictive Delivery Optimizer.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame") -> bool:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the DataFrame for logging
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{df_name} missing required columns: {missing_columns}")
    logger.info(f"{df_name} validation passed")
    return True


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: DataFrame to process
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        df_copy = df_copy.dropna()
    elif strategy == 'mean':
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
    elif strategy == 'mode':
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else df_copy[col])
    
    logger.info(f"Missing values handled using strategy: {strategy}")
    return df_copy


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate common regression metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.2f}%"


def get_date_features(date_series: pd.Series) -> pd.DataFrame:
    """
    Extract date features from a datetime series.
    
    Args:
        date_series: Series containing datetime values
        
    Returns:
        DataFrame with extracted date features
    """
    date_series = pd.to_datetime(date_series)
    
    features = pd.DataFrame()
    features['year'] = date_series.dt.year
    features['month'] = date_series.dt.month
    features['day'] = date_series.dt.day
    features['dayofweek'] = date_series.dt.dayofweek
    features['quarter'] = date_series.dt.quarter
    features['is_weekend'] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
    
    return features


def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        df: DataFrame to normalize
        columns: List of columns to normalize (if None, normalize all numeric columns)
        
    Returns:
        DataFrame with normalized values
    """
    from sklearn.preprocessing import MinMaxScaler
    
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = MinMaxScaler()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    logger.info(f"Normalized {len(columns)} columns")
    return df_copy


def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk."""
    import joblib
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load a trained model from disk."""
    import joblib
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model
