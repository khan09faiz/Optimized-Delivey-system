import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import streamlit as st

from utils import Config, setup_logging, validate_dataframe

logger = setup_logging("data_loader")


class DataLoader:
    """Handles loading and preprocessing of all datasets."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CSV files (defaults to Config.DATA_DIR)
        """
        self.data_dir = data_dir or Config.DATA_DIR
        Config.ensure_directories()
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    @st.cache_data(ttl=3600)
    def load_csv(_self, filename: str, required_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Load and validate a CSV file.
        
        Args:
            filename: Name of CSV file
            required_cols: Optional list of required columns
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If validation fails
        """
        filepath = _self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Dataset not found: {filename}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            
            if required_cols:
                validate_dataframe(df, required_cols, filename)
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    @st.cache_data(ttl=3600)
    def load_all_datasets(_self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        logger.info("Loading all datasets...")
        
        datasets = {}
        
        try:
            datasets['orders'] = _self.load_csv('orders.csv')
            datasets['delivery_performance'] = _self.load_csv('delivery_performance.csv')
            datasets['routes_distance'] = _self.load_csv('routes_distance.csv')
            datasets['vehicle_fleet'] = _self.load_csv('vehicle_fleet.csv')
            datasets['warehouse_inventory'] = _self.load_csv('warehouse_inventory.csv')
            datasets['customer_feedback'] = _self.load_csv('customer_feedback.csv')
            datasets['cost_breakdown'] = _self.load_csv('cost_breakdown.csv')
            
            logger.info(f"Successfully loaded {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {str(e)}")
            raise
    
    @st.cache_data(ttl=3600)
    def merge_datasets(_self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all datasets into a single DataFrame.
        
        Args:
            datasets: Dictionary of DataFrames
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets...")
        
        try:
            # Start with orders as the base
            df = datasets['orders'].copy()
            
            # Standardize column names to lowercase with underscores
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Merge delivery performance on Order_ID
            delivery_perf = datasets['delivery_performance'].copy()
            delivery_perf.columns = delivery_perf.columns.str.lower().str.replace(' ', '_')
            df = df.merge(
                delivery_perf,
                on='order_id',
                how='left'
            )
            
            # Merge routes_distance on Order_ID
            routes = datasets['routes_distance'].copy()
            routes.columns = routes.columns.str.lower().str.replace(' ', '_')
            df = df.merge(
                routes,
                on='order_id',
                how='left'
            )
            
            # Merge cost breakdown on Order_ID
            costs = datasets['cost_breakdown'].copy()
            costs.columns = costs.columns.str.lower().str.replace(' ', '_')
            df = df.merge(
                costs,
                on='order_id',
                how='left'
            )
            
            # Merge customer feedback on Order_ID
            feedback = datasets['customer_feedback'].copy()
            feedback.columns = feedback.columns.str.lower().str.replace(' ', '_')
            df = df.merge(
                feedback,
                on='order_id',
                how='left'
            )
            
            # Add warehouse inventory aggregates by location (origin)
            warehouse = datasets['warehouse_inventory'].copy()
            warehouse.columns = warehouse.columns.str.lower().str.replace(' ', '_')
            
            # Aggregate warehouse data by location and product category
            warehouse_agg = warehouse.groupby(['location', 'product_category']).agg({
                'current_stock_units': 'sum',
                'storage_cost_per_unit': 'mean'
            }).reset_index()
            warehouse_agg.rename(columns={
                'current_stock_units': 'warehouse_stock',
                'storage_cost_per_unit': 'avg_storage_cost'
            }, inplace=True)
            
            df = df.merge(
                warehouse_agg,
                left_on=['origin', 'product_category'],
                right_on=['location', 'product_category'],
                how='left'
            )
            
            # Drop duplicate location column
            if 'location' in df.columns:
                df.drop(columns=['location'], inplace=True)
            
            # Add vehicle fleet information (sample random vehicle for orders without vehicle assignment)
            vehicles = datasets['vehicle_fleet'].copy()
            vehicles.columns = vehicles.columns.str.lower().str.replace(' ', '_')
            
            logger.info(f"Merged dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and impute missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df = df.copy()
        
        # Convert date columns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Numeric columns - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Filled {col} with median: {median_val}")
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in date_columns]
        
        for col in categorical_cols:
            if df[col].isnull().any():
                if df[col].mode().shape[0] > 0:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna('Unknown')
                logger.debug(f"Filled {col} with mode or 'Unknown'")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            
        Returns:
            Boolean Series indicating outlier rows
        """
        logger.info(f"Detecting outliers using {method} method...")
        outliers = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > 3
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outliers |= col_outliers
            logger.debug(f"Found {col_outliers.sum()} outliers in {col}")
        
        logger.info(f"Total outliers detected: {outliers.sum()}")
        return outliers
    
    @st.cache_data(ttl=3600)
    def load_and_prepare_data(_self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load, merge, and clean all data.
        
        Returns:
            Tuple of (merged_cleaned_df, raw_datasets_dict)
        """
        logger.info("Starting full data loading and preparation pipeline...")
        
        # Load all datasets
        datasets = _self.load_all_datasets()
        
        # Merge datasets
        merged_df = _self.merge_datasets(datasets)
        
        # Clean data
        cleaned_df = _self.clean_data(merged_df)
        
        logger.info("Data loading and preparation complete")
        return cleaned_df, datasets


# ==================== Convenience Functions ====================
def get_data_loader(data_dir: Optional[Path] = None) -> DataLoader:
    """
    Create and return a DataLoader instance.
    
    Args:
        data_dir: Optional custom data directory
        
    Returns:
        DataLoader instance
    """
    return DataLoader(data_dir)


def load_prepared_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function to load prepared data.
    
    Args:
        data_dir: Optional custom data directory
        
    Returns:
        Tuple of (merged_cleaned_df, raw_datasets_dict)
    """
    loader = get_data_loader(data_dir)
    return loader.load_and_prepare_data()
