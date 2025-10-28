"""
Data loading module for the Predictive Delivery Optimizer.
Handles loading and initial validation of all 7 datasets.
"""
import pandas as pd
import os
from typing import Dict, Tuple
import logging
from .utils import validate_dataframe, logger


class DataLoader:
    """Handles loading and validation of delivery system datasets."""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = data_dir
        self.datasets = {}
        
        # Define required columns for each dataset
        self.schema = {
            'orders': ['order_id', 'customer_id', 'order_date', 'delivery_date', 
                      'origin', 'destination', 'weight', 'priority'],
            'delivery_performance': ['delivery_id', 'order_id', 'actual_delivery_time', 
                                    'scheduled_delivery_time', 'status', 'delay_minutes'],
            'routes_distance': ['route_id', 'origin', 'destination', 'distance_km', 
                              'estimated_time_hours', 'traffic_level'],
            'vehicle_fleet': ['vehicle_id', 'vehicle_type', 'capacity_kg', 
                            'fuel_efficiency', 'maintenance_status', 'availability'],
            'warehouse_inventory': ['warehouse_id', 'location', 'capacity', 
                                   'current_stock', 'product_type'],
            'customer_feedback': ['feedback_id', 'order_id', 'customer_id', 
                                'rating', 'delivery_rating', 'comments'],
            'cost_breakdown': ['cost_id', 'order_id', 'fuel_cost', 'labor_cost', 
                             'maintenance_cost', 'total_cost']
        }
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets from CSV files.
        
        Returns:
            Dictionary of DataFrames with dataset names as keys
        """
        logger.info("Starting to load all datasets...")
        
        for dataset_name in self.schema.keys():
            filepath = os.path.join(self.data_dir, f'{dataset_name}.csv')
            
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}. Creating sample data.")
                self.datasets[dataset_name] = self._create_sample_data(dataset_name)
            else:
                self.datasets[dataset_name] = pd.read_csv(filepath)
                logger.info(f"Loaded {dataset_name}: {len(self.datasets[dataset_name])} rows")
        
        self._validate_all_datasets()
        return self.datasets
    
    def _validate_all_datasets(self) -> None:
        """Validate all loaded datasets against their schemas."""
        for dataset_name, df in self.datasets.items():
            required_cols = self.schema[dataset_name]
            validate_dataframe(df, required_cols, dataset_name)
    
    def _create_sample_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Create sample data for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with sample data
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        n_samples = 100
        
        if dataset_name == 'orders':
            return pd.DataFrame({
                'order_id': [f'ORD{i:04d}' for i in range(n_samples)],
                'customer_id': [f'CUST{i%50:03d}' for i in range(n_samples)],
                'order_date': [datetime.now() - timedelta(days=np.random.randint(1, 90)) for _ in range(n_samples)],
                'delivery_date': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_samples)],
                'origin': np.random.choice(['Warehouse_A', 'Warehouse_B', 'Warehouse_C'], n_samples),
                'destination': [f'Location_{i%30}' for i in range(n_samples)],
                'weight': np.random.uniform(5, 500, n_samples),
                'priority': np.random.choice(['High', 'Medium', 'Low'], n_samples)
            })
        
        elif dataset_name == 'delivery_performance':
            return pd.DataFrame({
                'delivery_id': [f'DEL{i:04d}' for i in range(n_samples)],
                'order_id': [f'ORD{i:04d}' for i in range(n_samples)],
                'actual_delivery_time': np.random.uniform(1, 48, n_samples),
                'scheduled_delivery_time': np.random.uniform(1, 36, n_samples),
                'status': np.random.choice(['Delivered', 'In Transit', 'Delayed'], n_samples),
                'delay_minutes': np.random.randint(-60, 300, n_samples)
            })
        
        elif dataset_name == 'routes_distance':
            origins = ['Warehouse_A', 'Warehouse_B', 'Warehouse_C']
            destinations = [f'Location_{i}' for i in range(30)]
            routes = []
            for i, origin in enumerate(origins):
                for j, dest in enumerate(destinations[:10]):
                    routes.append({
                        'route_id': f'ROUTE{len(routes):04d}',
                        'origin': origin,
                        'destination': dest,
                        'distance_km': np.random.uniform(10, 200),
                        'estimated_time_hours': np.random.uniform(0.5, 8),
                        'traffic_level': np.random.choice(['Low', 'Medium', 'High'])
                    })
            return pd.DataFrame(routes)
        
        elif dataset_name == 'vehicle_fleet':
            return pd.DataFrame({
                'vehicle_id': [f'VEH{i:03d}' for i in range(50)],
                'vehicle_type': np.random.choice(['Truck', 'Van', 'Motorcycle'], 50),
                'capacity_kg': np.random.choice([500, 1000, 2000, 5000], 50),
                'fuel_efficiency': np.random.uniform(5, 15, 50),
                'maintenance_status': np.random.choice(['Good', 'Fair', 'Needs Service'], 50),
                'availability': np.random.choice(['Available', 'In Use', 'Maintenance'], 50)
            })
        
        elif dataset_name == 'warehouse_inventory':
            return pd.DataFrame({
                'warehouse_id': [f'WH{i:02d}' for i in range(20)],
                'location': np.random.choice(['Warehouse_A', 'Warehouse_B', 'Warehouse_C'], 20),
                'capacity': np.random.randint(5000, 50000, 20),
                'current_stock': np.random.randint(1000, 40000, 20),
                'product_type': np.random.choice(['Electronics', 'Clothing', 'Food', 'Furniture'], 20)
            })
        
        elif dataset_name == 'customer_feedback':
            return pd.DataFrame({
                'feedback_id': [f'FB{i:04d}' for i in range(n_samples)],
                'order_id': [f'ORD{i:04d}' for i in range(n_samples)],
                'customer_id': [f'CUST{i%50:03d}' for i in range(n_samples)],
                'rating': np.random.randint(1, 6, n_samples),
                'delivery_rating': np.random.randint(1, 6, n_samples),
                'comments': ['Good service' if r > 3 else 'Needs improvement' for r in np.random.randint(1, 6, n_samples)]
            })
        
        elif dataset_name == 'cost_breakdown':
            return pd.DataFrame({
                'cost_id': [f'COST{i:04d}' for i in range(n_samples)],
                'order_id': [f'ORD{i:04d}' for i in range(n_samples)],
                'fuel_cost': np.random.uniform(20, 200, n_samples),
                'labor_cost': np.random.uniform(30, 150, n_samples),
                'maintenance_cost': np.random.uniform(10, 100, n_samples),
                'total_cost': np.random.uniform(100, 500, n_samples)
            })
        
        return pd.DataFrame()
    
    def get_dataset(self, name: str) -> pd.DataFrame:
        """
        Get a specific dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            The requested DataFrame
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")
        return self.datasets[name]
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about all loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {}
        for name, df in self.datasets.items():
            info[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            }
        return info
