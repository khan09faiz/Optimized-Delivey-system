"""
Feature engineering module for delivery delay prediction.

This module handles:
- Derived feature computation
- Categorical encoding
- Feature scaling and transformation
- Aggregations and carrier/warehouse metrics
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import streamlit as st

from utils import Config, setup_logging

logger = setup_logging("feature_engineering")


class FeatureEngineer:
    """Handles feature engineering and transformations."""
    
    def __init__(self):
        """Initialize FeatureEngineer with encoders."""
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        logger.info("FeatureEngineer initialized")
    
    @staticmethod
    def create_delay_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delay-related features.
        
        Args:
            df: Input DataFrame with delivery information
            
        Returns:
            DataFrame with delay features added
        """
        logger.info("Creating delay features...")
        df = df.copy()
        
        # Handle the actual data format: promised_delivery_days and actual_delivery_days
        if 'promised_delivery_days' in df.columns and 'actual_delivery_days' in df.columns:
            # Calculate delay in days
            df['delay_days'] = df['actual_delivery_days'] - df['promised_delivery_days']
            df['delay_days'] = df['delay_days'].fillna(0)
            
            # Create binary delay flag (1 if delayed, 0 otherwise)
            df['delay_flag'] = (df['delay_days'] > 0).astype(int)
            
            logger.info(f"Delay rate: {df['delay_flag'].mean():.2%}")
            logger.info(f"Average delay days: {df[df['delay_flag']==1]['delay_days'].mean():.2f}")
            
        # Also check for delivery_status column
        elif 'delivery_status' in df.columns:
            # Map delivery status to delay flag
            df['delay_flag'] = df['delivery_status'].apply(
                lambda x: 1 if 'delayed' in str(x).lower() else 0
            ).astype(int)
            
            # If we have promised and actual days, calculate delay_days
            if 'promised_delivery_days' in df.columns and 'actual_delivery_days' in df.columns:
                df['delay_days'] = df['actual_delivery_days'] - df['promised_delivery_days']
            else:
                df['delay_days'] = 0
                
            logger.info(f"Delay rate from status: {df['delay_flag'].mean():.2%}")
        else:
            logger.warning("Missing delivery day columns for delay calculation")
            df['delay_days'] = 0
            df['delay_flag'] = 0
        
        return df
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, date_col: str = 'order_date') -> pd.DataFrame:
        """
        Extract temporal features from date column.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with temporal features
        """
        logger.info(f"Creating temporal features from {date_col}...")
        df = df.copy()
        
        if date_col not in df.columns:
            logger.warning(f"Column {date_col} not found")
            return df
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        df['order_day_of_week'] = df[date_col].dt.dayofweek
        df['order_day'] = df[date_col].dt.day
        df['order_month'] = df[date_col].dt.month
        df['order_year'] = df[date_col].dt.year
        df['order_quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['order_day'] >= 28).astype(int)
        
        logger.info("Temporal features created successfully")
        return df
    
    @staticmethod
    def create_distance_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create distance-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with distance features
        """
        logger.info("Creating distance features...")
        df = df.copy()
        
        # Check for distance_km column (already in data)
        if 'distance_km' not in df.columns:
            logger.warning("distance_km column not found")
            df['distance_km'] = 100.0  # Default
        
        # Distance categories
        df['is_short_distance'] = (df['distance_km'] < Config.SHORT_DISTANCE).astype(int)
        df['is_long_distance'] = (df['distance_km'] > 200).astype(int)
        
        # Distance bins
        df['distance_category'] = pd.cut(
            df['distance_km'],
            bins=[0, 15, 50, 150, 500, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        )
        
        logger.info(f"Distance range: {df['distance_km'].min():.2f} - {df['distance_km'].max():.2f} km")
        
        return df
    
    @staticmethod
    def create_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create traffic-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with traffic features
        """
        logger.info("Creating traffic features...")
        df = df.copy()
        
        # Check for traffic_delay_minutes column (already in data)
        if 'traffic_delay_minutes' not in df.columns:
            # Try alternative column names
            if 'traffic_delay_mins' in df.columns:
                df['traffic_delay_minutes'] = df['traffic_delay_mins']
            else:
                logger.warning("traffic_delay column not found, creating default")
                df['traffic_delay_minutes'] = 0
        
        # Traffic categories
        df['high_traffic'] = (df['traffic_delay_minutes'] > Config.TRAFFIC_DELAY_HIGH).astype(int)
        df['traffic_category'] = pd.cut(
            df['traffic_delay_minutes'],
            bins=[0, 15, 30, 60, float('inf')],
            labels=['Low', 'Moderate', 'High', 'Severe']
        )
        
        logger.info(f"Avg traffic delay: {df['traffic_delay_minutes'].mean():.2f} minutes")
        
        return df
    
    @staticmethod
    def create_order_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create order-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with order features
        """
        logger.info("Creating order features...")
        df = df.copy()
        
        # Use existing order_value_inr or create from cost components
        if 'order_value_inr' not in df.columns:
            cost_cols = ['fuel_cost', 'labor_cost', 'vehicle_maintenance', 'insurance', 
                        'packaging_cost', 'technology_platform_fee', 'other_overhead']
            existing_cost_cols = [col for col in cost_cols if col in df.columns]
            
            if existing_cost_cols:
                df['order_value_inr'] = df[existing_cost_cols].sum(axis=1)
            else:
                df['order_value_inr'] = 1000  # Default
        
        # Order value categories
        df['order_value_category'] = pd.cut(
            df['order_value_inr'],
            bins=[0, 1000, 5000, 20000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        # Priority encoding
        if 'priority' in df.columns:
            # Map priority to numeric (Express > Standard > Economy)
            priority_map = {'Economy': 0, 'Standard': 1, 'Express': 2}
            df['priority_encoded'] = df['priority'].map(priority_map).fillna(1)
        
        # Total delivery cost
        if 'delivery_cost_inr' in df.columns:
            df['cost_per_km'] = df['delivery_cost_inr'] / (df['distance_km'] + 1)  # Avoid division by zero
        
        logger.info(f"Avg order value: ₹{df['order_value_inr'].mean():.2f}")
        
        return df
    
    @staticmethod
    def create_carrier_aggregates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create carrier performance aggregates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with carrier aggregate features
        """
        logger.info("Creating carrier aggregate features...")
        df = df.copy()
        
        if 'carrier' not in df.columns:
            logger.warning("carrier column not found, skipping carrier aggregates")
            return df
        
        # Ensure delay_flag exists
        if 'delay_flag' not in df.columns:
            logger.warning("delay_flag not found, creating from delivery_status")
            df['delay_flag'] = df.get('delivery_status', '').apply(
                lambda x: 1 if 'delayed' in str(x).lower() else 0
            )
        
        # Calculate carrier-level metrics
        carrier_stats = df.groupby('carrier').agg({
            'delay_flag': 'mean',  # Delay rate
            'delay_days': 'mean',  # Average delay
            'order_value_inr': 'mean'  # Average order value
        }).reset_index()
        
        carrier_stats.rename(columns={
            'delay_flag': 'carrier_delay_rate',
            'delay_days': 'carrier_avg_delay',
            'order_value_inr': 'carrier_avg_order_value'
        }, inplace=True)
        
        # Merge back to main dataframe
        df = df.merge(carrier_stats, on='carrier', how='left')
        
        logger.info(f"Carrier delay rates: {carrier_stats[['carrier', 'carrier_delay_rate']].to_dict('records')}")
        
        return df
    
    @staticmethod
    def create_warehouse_aggregates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create warehouse performance aggregates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with warehouse aggregate features
        """
        logger.info("Creating warehouse aggregate features...")
        df = df.copy()
        
        # Use origin as warehouse location
        if 'origin' not in df.columns:
            logger.warning("origin column not found, skipping warehouse aggregates")
            return df
        
        # If we have warehouse_stock from merge, use it
        if 'warehouse_stock' not in df.columns:
            df['warehouse_stock'] = 1000  # Default
        
        # Calculate warehouse-level metrics if we have enough data
        if 'delay_flag' in df.columns:
            warehouse_stats = df.groupby('origin').agg({
                'delay_flag': 'mean',
                'order_value_inr': 'sum' if 'order_value_inr' in df.columns else 'count'
            }).reset_index()
            
            warehouse_stats.rename(columns={
                'delay_flag': 'warehouse_delay_rate',
                'order_value_inr': 'warehouse_total_value'
            }, inplace=True)
            
            # Merge back
            df = df.merge(warehouse_stats, on='origin', how='left', suffixes=('', '_wh'))
        
        logger.info("Warehouse aggregates created")
        return df
    
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        onehot_cols: Optional[List[str]] = None,
        ordinal_cols: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            onehot_cols: Columns for one-hot encoding
            ordinal_cols: Columns for ordinal encoding
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        # Default columns based on actual data
        if onehot_cols is None:
            onehot_cols = ['carrier', 'customer_segment', 'product_category', 'origin', 'destination', 
                          'special_handling', 'quality_issue', 'weather_impact']
        if ordinal_cols is None:
            ordinal_cols = ['priority', 'distance_category', 'traffic_category', 'delivery_status']
        
        # Filter to existing columns
        onehot_cols = [col for col in onehot_cols if col in df.columns]
        ordinal_cols = [col for col in ordinal_cols if col in df.columns]
        
        # One-hot encoding
        if onehot_cols:
            # Convert to string type and fill NaN values
            onehot_data = df[onehot_cols].astype(str).replace('nan', 'Unknown')
            
            # Store original categorical columns for later use (visualization, filtering)
            preserve_cols = ['carrier', 'customer_segment', 'priority']
            preserved_data = {}
            for col in preserve_cols:
                if col in df.columns:
                    preserved_data[col] = df[col].copy()
            
            if fit:
                self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.onehot_encoder.fit_transform(onehot_data)
                feature_names = self.onehot_encoder.get_feature_names_out(onehot_cols)
            else:
                if self.onehot_encoder is None:
                    logger.error("Encoder not fitted. Call with fit=True first.")
                    raise ValueError("Encoder not fitted")
                encoded = self.onehot_encoder.transform(onehot_data)
                feature_names = self.onehot_encoder.get_feature_names_out(onehot_cols)
            
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=onehot_cols, inplace=True)
            
            # Restore preserved columns for visualization/filtering
            for col, data in preserved_data.items():
                df[col] = data
            
            logger.info(f"One-hot encoded {len(onehot_cols)} columns")
        
        # Ordinal encoding
        if ordinal_cols:
            # Convert to string type and fill NaN values
            ordinal_data = df[ordinal_cols].astype(str).replace('nan', 'Unknown')
            
            if fit:
                self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df[ordinal_cols] = self.ordinal_encoder.fit_transform(ordinal_data)
            else:
                if self.ordinal_encoder is None:
                    logger.error("Ordinal encoder not fitted")
                    raise ValueError("Ordinal encoder not fitted")
                df[ordinal_cols] = self.ordinal_encoder.transform(ordinal_data)
            
            logger.info(f"Ordinal encoded {len(ordinal_cols)} columns")
        
        return df
    
    @st.cache_data(ttl=3600)
    def engineer_features(_self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders
            
        Returns:
            Feature-engineered DataFrame
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Create derived features
        df = _self.create_delay_features(df)
        df = _self.create_temporal_features(df)
        df = _self.create_distance_features(df)
        df = _self.create_traffic_features(df)
        df = _self.create_order_features(df)
        df = _self.create_carrier_aggregates(df)
        df = _self.create_warehouse_aggregates(df)
        
        # Encode categorical features
        df = _self.encode_categorical_features(df, fit=fit)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, target_col: str = 'delay_flag') -> List[str]:
        """
        Get list of feature columns excluding target and identifiers.
        
        Args:
            df: DataFrame
            target_col: Target column name
            
        Returns:
            List of feature column names
        """
        # Exclude columns
        exclude_cols = [
            target_col, 'delay_days', 'order_id', 'customer_id', 'vehicle_id',
            'warehouse_id', 'route_id', 'order_date', 'expected_delivery_date',
            'actual_delivery_date', 'product_id', 'feedback_text', 'feedback_date',
            'route', 'origin', 'destination'
        ]
        
        # Get all columns excluding the explicit exclusions
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only numeric columns for modeling - exclude object and datetime types
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names = numeric_cols
        logger.info(f"Selected {len(numeric_cols)} feature columns")
        
        return numeric_cols


# ==================== Convenience Functions ====================
def get_feature_engineer() -> FeatureEngineer:
    """
    Create and return a FeatureEngineer instance.
    
    Returns:
        FeatureEngineer instance
    """
    return FeatureEngineer()


def engineer_features(df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience function to engineer features.
    
    Args:
        df: Input DataFrame
        fit: Whether to fit encoders
        
    Returns:
        Tuple of (engineered_df, feature_engineer_instance)
    """
    engineer = get_feature_engineer()
    engineered_df = engineer.engineer_features(df, fit=fit)
    return engineered_df, engineer
