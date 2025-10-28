"""
Shared utilities and configuration for Predictive Delivery Optimizer.

This module provides:
- Logging configuration
- Path management
- Reusable helper functions
- Constants and configuration
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


# ==================== Configuration ====================
class Config:
    """Centralized configuration for the application."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR.parent / "delivery data" / "Case study internship data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Classification thresholds
    HIGH_RISK_THRESHOLD = 0.7
    MODERATE_RISK_THRESHOLD = 0.4
    
    # Feature engineering
    TRAFFIC_DELAY_HIGH = 30  # minutes
    SHORT_DISTANCE = 15  # km
    
    # Dataset schema
    REQUIRED_COLUMNS = {
        'orders': ['order_id', 'customer_id', 'order_date', 'priority'],
        'delivery_performance': ['order_id', 'actual_delivery_date', 'expected_delivery_date'],
        'routes_distance': ['route_id', 'origin', 'destination', 'distance_km'],
        'vehicle_fleet': ['vehicle_id', 'vehicle_type', 'capacity'],
        'warehouse_inventory': ['warehouse_id', 'product_id', 'stock_level'],
        'customer_feedback': ['order_id', 'rating', 'feedback_text'],
        'cost_breakdown': ['order_id', 'fuel_cost', 'maintenance_cost']
    }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.OUTPUTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ==================== Logging Setup ====================
def setup_logging(name: str = "predictive_delivery", level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    Config.ensure_directories()
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    log_file = Config.LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==================== Helper Functions ====================
def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: JSON file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_currency(amount: float) -> str:
    """
    Format currency in Indian Rupees.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    return f"₹{amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def get_risk_category(probability: float) -> str:
    """
    Categorize delay probability into risk levels.
    
    Args:
        probability: Delay probability (0-1)
        
    Returns:
        Risk category: 'High', 'Moderate', or 'Low'
    """
    if probability >= Config.HIGH_RISK_THRESHOLD:
        return 'High'
    elif probability >= Config.MODERATE_RISK_THRESHOLD:
        return 'Moderate'
    else:
        return 'Low'


def get_timestamp() -> str:
    """
    Get current timestamp in standardized format.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def validate_dataframe(df, required_columns: list, df_name: str) -> None:
    """
    Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name for error messages
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{df_name} missing required columns: {missing_cols}")


# ==================== Constants ====================
PRIORITY_LEVELS = ['High', 'Medium', 'Low']
VEHICLE_TYPES = ['Truck', 'Van', 'Bike', 'Refrigerated']
CUSTOMER_SEGMENTS = ['Enterprise', 'SMB', 'Individual']
CARRIERS = ['BlueExpress', 'FastShip', 'QuickDeliver', 'PremiumLogistics']

# Color scheme for visualizations
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'info': '#17becf'
}

# Risk colors
RISK_COLORS = {
    'High': '#d62728',
    'Moderate': '#ff9800',
    'Low': '#2ca02c'
}
