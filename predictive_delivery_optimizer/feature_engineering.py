"""
Feature engineering module for the Predictive Delivery Optimizer.
Handles data preprocessing and feature extraction from multiple datasets.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from .utils import get_date_features, logger


class FeatureEngineer:
    """Handles feature engineering and data preprocessing."""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize FeatureEngineer.
        
        Args:
            datasets: Dictionary of DataFrames loaded by DataLoader
        """
        self.datasets = datasets
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_df = None
        
    def create_master_dataset(self) -> pd.DataFrame:
        """
        Create a master dataset by merging all relevant datasets.
        
        Returns:
            Merged DataFrame with all features
        """
        logger.info("Creating master dataset...")
        
        # Start with orders as the base
        master = self.datasets['orders'].copy()
        
        # Merge delivery performance
        if 'delivery_performance' in self.datasets:
            master = master.merge(
                self.datasets['delivery_performance'],
                on='order_id',
                how='left'
            )
        
        # Merge cost breakdown
        if 'cost_breakdown' in self.datasets:
            master = master.merge(
                self.datasets['cost_breakdown'],
                on='order_id',
                how='left'
            )
        
        # Merge customer feedback
        if 'customer_feedback' in self.datasets:
            master = master.merge(
                self.datasets['customer_feedback'][['order_id', 'rating', 'delivery_rating']],
                on='order_id',
                how='left'
            )
        
        # Add route information
        if 'routes_distance' in self.datasets:
            routes = self.datasets['routes_distance']
            master = master.merge(
                routes[['origin', 'destination', 'distance_km', 'estimated_time_hours', 'traffic_level']],
                on=['origin', 'destination'],
                how='left'
            )
        
        logger.info(f"Master dataset created with {len(master)} rows and {len(master.columns)} columns")
        return master
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer features from the master dataset.
        
        Returns:
            DataFrame with engineered features
        """
        master = self.create_master_dataset()
        
        # Extract date features
        if 'order_date' in master.columns:
            master['order_date'] = pd.to_datetime(master['order_date'])
            date_features = get_date_features(master['order_date'])
            for col in date_features.columns:
                master[f'order_{col}'] = date_features[col]
        
        if 'delivery_date' in master.columns:
            master['delivery_date'] = pd.to_datetime(master['delivery_date'])
            master['delivery_duration_days'] = (master['delivery_date'] - master['order_date']).dt.days
        
        # Create delivery delay feature
        if 'actual_delivery_time' in master.columns and 'scheduled_delivery_time' in master.columns:
            master['delivery_delay'] = master['actual_delivery_time'] - master['scheduled_delivery_time']
            master['is_delayed'] = (master['delivery_delay'] > 0).astype(int)
        
        # Priority encoding
        if 'priority' in master.columns:
            priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
            master['priority_score'] = master['priority'].map(priority_map)
        
        # Traffic level encoding
        if 'traffic_level' in master.columns:
            traffic_map = {'Low': 1, 'Medium': 2, 'High': 3}
            master['traffic_score'] = master['traffic_level'].map(traffic_map)
        
        # Weight categories
        if 'weight' in master.columns:
            master['weight_category'] = pd.cut(
                master['weight'],
                bins=[0, 50, 200, 500, np.inf],
                labels=['Light', 'Medium', 'Heavy', 'Extra Heavy']
            )
        
        # Cost efficiency metrics
        if 'total_cost' in master.columns and 'distance_km' in master.columns:
            master['cost_per_km'] = master['total_cost'] / (master['distance_km'] + 1)  # Add 1 to avoid division by zero
        
        if 'total_cost' in master.columns and 'weight' in master.columns:
            master['cost_per_kg'] = master['total_cost'] / (master['weight'] + 1)
        
        # Delivery performance metrics
        if 'delay_minutes' in master.columns:
            master['on_time_delivery'] = (master['delay_minutes'] <= 0).astype(int)
        
        # Customer satisfaction features
        if 'rating' in master.columns:
            master['high_rating'] = (master['rating'] >= 4).astype(int)
        
        self.feature_df = master
        logger.info(f"Feature engineering completed. Total features: {len(master.columns)}")
        
        return master
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: DataFrame to encode
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        return df_encoded
    
    def prepare_features_for_modeling(self, target_column: str = 'delivery_delay') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning modeling.
        
        Args:
            target_column: Name of the target variable
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.feature_df is None:
            self.feature_df = self.engineer_features()
        
        df = self.feature_df.copy()
        
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target from features
        id_columns = [col for col in numeric_features if 'id' in col.lower()]
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        numeric_features = [col for col in numeric_features if col not in id_columns]
        
        # Handle missing values
        X = df[numeric_features].fillna(df[numeric_features].median())
        
        # Get target variable
        y = df[target_column].fillna(df[target_column].median()) if target_column in df.columns else None
        
        logger.info(f"Prepared {len(numeric_features)} features for modeling")
        return X, y
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names used in modeling."""
        if self.feature_df is None:
            self.engineer_features()
        
        numeric_features = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()
        id_columns = [col for col in numeric_features if 'id' in col.lower()]
        return [col for col in numeric_features if col not in id_columns]
    
    def create_aggregated_features(self) -> Dict[str, pd.DataFrame]:
        """
        Create aggregated features from datasets.
        
        Returns:
            Dictionary of aggregated DataFrames
        """
        aggregations = {}
        
        # Customer-level aggregations
        if 'customer_feedback' in self.datasets:
            customer_agg = self.datasets['customer_feedback'].groupby('customer_id').agg({
                'rating': ['mean', 'count', 'std'],
                'delivery_rating': 'mean'
            }).reset_index()
            customer_agg.columns = ['customer_id', 'avg_rating', 'order_count', 'rating_std', 'avg_delivery_rating']
            aggregations['customer_stats'] = customer_agg
        
        # Route-level aggregations
        if 'routes_distance' in self.datasets:
            route_agg = self.datasets['routes_distance'].groupby(['origin', 'destination']).agg({
                'distance_km': 'mean',
                'estimated_time_hours': 'mean'
            }).reset_index()
            aggregations['route_stats'] = route_agg
        
        # Vehicle utilization
        if 'vehicle_fleet' in self.datasets:
            vehicle_agg = self.datasets['vehicle_fleet'].groupby('vehicle_type').agg({
                'capacity_kg': 'mean',
                'fuel_efficiency': 'mean'
            }).reset_index()
            aggregations['vehicle_stats'] = vehicle_agg
        
        logger.info(f"Created {len(aggregations)} aggregated feature sets")
        return aggregations
