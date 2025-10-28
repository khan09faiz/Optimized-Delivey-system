"""
Anomaly Detection Module
========================
Identifies unusual patterns and outliers in delivery operations.
Uses statistical methods and machine learning for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import setup_logging

logger = setup_logging("anomaly_detection")


class AnomalyDetector:
    """Detects anomalies in delivery operations."""
    
    def __init__(self, data: pd.DataFrame, contamination: float = 0.1):
        """
        Initialize AnomalyDetector.
        
        Args:
            data: Delivery data
            contamination: Expected proportion of outliers (0-0.5)
        """
        self.data = data
        self.contamination = contamination
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.anomaly_scores = None
        self.anomalies = None
        
        logger.info(f"AnomalyDetector initialized with {len(data)} records")
        logger.info(f"Expected contamination: {contamination*100}%")
    
    def detect_statistical_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using statistical methods.
        
        Args:
            columns: Columns to analyze (None = all numeric)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            DataFrame with outlier flags
        """
        logger.info(f"Detecting statistical outliers using {method} method...")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_flags = pd.DataFrame(index=self.data.index)
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            values = self.data[col].dropna()
            
            if method == 'iqr':
                # IQR method
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_flags[f'{col}_outlier'] = (
                    (self.data[col] < lower_bound) | 
                    (self.data[col] > upper_bound)
                )
                
            elif method == 'zscore':
                # Z-score method
                mean = values.mean()
                std = values.std()
                z_scores = np.abs((self.data[col] - mean) / std)
                outlier_flags[f'{col}_outlier'] = z_scores > threshold
        
        # Aggregate outlier flag
        outlier_flags['is_outlier'] = outlier_flags.any(axis=1)
        
        num_outliers = outlier_flags['is_outlier'].sum()
        logger.info(f"Detected {num_outliers} statistical outliers ({num_outliers/len(self.data)*100:.1f}%)")
        
        return outlier_flags
    
    def detect_ml_anomalies(
        self,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            feature_columns: Features for anomaly detection
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        logger.info("Detecting anomalies using Isolation Forest...")
        
        # Select features
        if feature_columns is None:
            feature_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target and ID columns
            feature_columns = [col for col in feature_columns if col not in ['order_id', 'is_delayed', 'delay_minutes']]
        
        # Prepare data
        X = self.data[feature_columns].fillna(self.data[feature_columns].median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.isolation_forest.fit_predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.isolation_forest.score_samples(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'is_anomaly': predictions == -1,
            'anomaly_score': scores,
            'anomaly_severity': self._calculate_severity(scores)
        }, index=self.data.index)
        
        self.anomaly_scores = scores
        self.anomalies = results
        
        num_anomalies = results['is_anomaly'].sum()
        logger.info(f"Detected {num_anomalies} ML-based anomalies ({num_anomalies/len(self.data)*100:.1f}%)")
        
        return results
    
    def _calculate_severity(self, scores: np.ndarray) -> List[str]:
        """Calculate anomaly severity based on scores."""
        percentiles = np.percentile(scores, [10, 30, 50])
        
        severity = []
        for score in scores:
            if score < percentiles[0]:
                severity.append('CRITICAL')
            elif score < percentiles[1]:
                severity.append('HIGH')
            elif score < percentiles[2]:
                severity.append('MEDIUM')
            else:
                severity.append('LOW')
        
        return severity
    
    def detect_delivery_time_anomalies(self) -> pd.DataFrame:
        """
        Detect anomalies specifically in delivery times.
        
        Returns:
            DataFrame with delivery time anomaly analysis
        """
        logger.info("Detecting delivery time anomalies...")
        
        anomalies = pd.DataFrame(index=self.data.index)
        
        # Check for unusually long delivery times
        if 'actual_delivery_days' in self.data.columns:
            delivery_times = self.data['actual_delivery_days']
            mean_time = delivery_times.mean()
            std_time = delivery_times.std()
            
            # Flag deliveries > 3 standard deviations
            anomalies['excessive_delivery_time'] = delivery_times > (mean_time + 3 * std_time)
            
            # Flag suspiciously fast deliveries
            anomalies['suspiciously_fast'] = delivery_times < max(1, mean_time - 2 * std_time)
        
        # Check for weekend/holiday patterns
        if 'order_date' in self.data.columns:
            # Convert to datetime if not already
            try:
                order_dates = pd.to_datetime(self.data['order_date'])
                # Check for weekend orders (potentially delayed)
                anomalies['weekend_order'] = order_dates.dt.dayofweek >= 5
            except:
                logger.warning("Could not parse order_date")
        
        # Aggregate
        anomalies['has_delivery_anomaly'] = anomalies.any(axis=1)
        
        num_anomalies = anomalies['has_delivery_anomaly'].sum()
        logger.info(f"Detected {num_anomalies} delivery time anomalies")
        
        return anomalies
    
    def detect_cost_anomalies(self) -> pd.DataFrame:
        """
        Detect anomalies in shipping costs.
        
        Returns:
            DataFrame with cost anomaly flags
        """
        logger.info("Detecting cost anomalies...")
        
        anomalies = pd.DataFrame(index=self.data.index)
        
        if 'shipping_cost' in self.data.columns:
            # Detect by carrier
            if 'carrier_name' in self.data.columns:
                for carrier in self.data['carrier_name'].unique():
                    carrier_mask = self.data['carrier_name'] == carrier
                    carrier_costs = self.data.loc[carrier_mask, 'shipping_cost']
                    
                    # Use IQR method
                    Q1 = carrier_costs.quantile(0.25)
                    Q3 = carrier_costs.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    
                    anomalies.loc[carrier_mask, 'unusual_cost'] = (
                        (carrier_costs < lower) | (carrier_costs > upper)
                    )
            else:
                # Global cost anomalies
                costs = self.data['shipping_cost']
                Q1 = costs.quantile(0.25)
                Q3 = costs.quantile(0.75)
                IQR = Q3 - Q1
                anomalies['unusual_cost'] = (
                    (costs < Q1 - 1.5 * IQR) | 
                    (costs > Q3 + 1.5 * IQR)
                )
        
        anomalies['has_cost_anomaly'] = anomalies.get('unusual_cost', False)
        
        num_anomalies = anomalies['has_cost_anomaly'].sum()
        logger.info(f"Detected {num_anomalies} cost anomalies")
        
        return anomalies
    
    def detect_pattern_anomalies(self) -> pd.DataFrame:
        """
        Detect anomalies in delivery patterns.
        
        Returns:
            DataFrame with pattern anomaly flags
        """
        logger.info("Detecting pattern anomalies...")
        
        anomalies = pd.DataFrame(index=self.data.index)
        
        # Unusual priority-delay combinations
        if 'priority' in self.data.columns and 'is_delayed' in self.data.columns:
            # High priority but delayed
            anomalies['high_priority_delayed'] = (
                (self.data['priority'] == 'High') & 
                (self.data['is_delayed'] == 1)
            )
            
            # Low priority but on-time
            anomalies['low_priority_ontime'] = (
                (self.data['priority'] == 'Low') & 
                (self.data['is_delayed'] == 0)
            )
        
        # Unusual distance-time combinations
        if 'distance_km' in self.data.columns and 'actual_delivery_days' in self.data.columns:
            # Short distance but long delivery
            distance_median = self.data['distance_km'].median()
            time_median = self.data['actual_delivery_days'].median()
            
            anomalies['short_distance_long_time'] = (
                (self.data['distance_km'] < distance_median) & 
                (self.data['actual_delivery_days'] > time_median * 1.5)
            )
        
        anomalies['has_pattern_anomaly'] = anomalies.any(axis=1)
        
        num_anomalies = anomalies['has_pattern_anomaly'].sum()
        logger.info(f"Detected {num_anomalies} pattern anomalies")
        
        return anomalies
    
    def get_comprehensive_anomalies(self) -> pd.DataFrame:
        """
        Run all anomaly detection methods and combine results.
        
        Returns:
            Comprehensive anomaly report
        """
        logger.info("Running comprehensive anomaly detection...")
        
        # Detect all types
        ml_anomalies = self.detect_ml_anomalies()
        delivery_anomalies = self.detect_delivery_time_anomalies()
        cost_anomalies = self.detect_cost_anomalies()
        pattern_anomalies = self.detect_pattern_anomalies()
        
        # Combine
        comprehensive = pd.concat([
            ml_anomalies,
            delivery_anomalies,
            cost_anomalies,
            pattern_anomalies
        ], axis=1)
        
        # Overall anomaly flag
        comprehensive['is_anomalous'] = (
            comprehensive['is_anomaly'] | 
            comprehensive.get('has_delivery_anomaly', False) |
            comprehensive.get('has_cost_anomaly', False) |
            comprehensive.get('has_pattern_anomaly', False)
        )
        
        # Count anomaly types
        comprehensive['anomaly_count'] = comprehensive[[
            'is_anomaly',
            'has_delivery_anomaly',
            'has_cost_anomaly',
            'has_pattern_anomaly'
        ]].sum(axis=1)
        
        total_anomalies = comprehensive['is_anomalous'].sum()
        logger.info(f"Total anomalies detected: {total_anomalies} ({total_anomalies/len(self.data)*100:.1f}%)")
        
        return comprehensive
    
    def plot_anomaly_overview(self, anomalies: pd.DataFrame) -> go.Figure:
        """
        Create overview visualization of anomalies.
        
        Args:
            anomalies: Anomaly detection results
            
        Returns:
            Plotly figure
        """
        logger.info("Creating anomaly overview chart...")
        
        # Count by type
        anomaly_types = {
            'ML-Based': anomalies.get('is_anomaly', pd.Series([False])).sum(),
            'Delivery Time': anomalies.get('has_delivery_anomaly', pd.Series([False])).sum(),
            'Cost': anomalies.get('has_cost_anomaly', pd.Series([False])).sum(),
            'Pattern': anomalies.get('has_pattern_anomaly', pd.Series([False])).sum()
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(anomaly_types.keys()),
                y=list(anomaly_types.values()),
                marker=dict(
                    color=['#e74c3c', '#f39c12', '#3498db', '#9b59b6'],
                    line=dict(color='white', width=2)
                ),
                text=list(anomaly_types.values()),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Anomalies: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Anomaly Detection Summary',
            xaxis_title='Anomaly Type',
            yaxis_title='Number of Anomalies',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def plot_anomaly_scores(self) -> go.Figure:
        """
        Plot anomaly score distribution.
        
        Returns:
            Plotly figure
        """
        logger.info("Creating anomaly score distribution...")
        
        if self.anomaly_scores is None:
            self.detect_ml_anomalies()
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=self.anomaly_scores,
            nbinsx=50,
            marker=dict(color='#3498db', line=dict(color='white', width=1)),
            name='Score Distribution',
            hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add threshold line
        threshold = np.percentile(self.anomaly_scores, self.contamination * 100)
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Anomaly Threshold: {threshold:.3f}"
        )
        
        fig.update_layout(
            title='Anomaly Score Distribution',
            xaxis_title='Anomaly Score (lower = more anomalous)',
            yaxis_title='Frequency',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def plot_anomalies_by_carrier(self, anomalies: pd.DataFrame) -> go.Figure:
        """
        Plot anomalies by carrier.
        
        Args:
            anomalies: Anomaly results
            
        Returns:
            Plotly figure
        """
        logger.info("Creating carrier anomaly analysis...")
        
        if 'carrier_name' not in self.data.columns:
            return go.Figure()
        
        # Combine data
        plot_data = pd.concat([self.data[['carrier_name']], anomalies[['is_anomalous']]], axis=1)
        
        # Calculate stats
        carrier_stats = plot_data.groupby('carrier_name').agg({
            'is_anomalous': ['sum', 'count']
        })
        carrier_stats.columns = ['anomalies', 'total']
        carrier_stats['anomaly_rate_%'] = (carrier_stats['anomalies'] / carrier_stats['total'] * 100)
        carrier_stats = carrier_stats.reset_index().sort_values('anomaly_rate_%', ascending=False)
        
        # Create chart
        fig = go.Figure(data=[
            go.Bar(
                x=carrier_stats['carrier_name'],
                y=carrier_stats['anomaly_rate_%'],
                marker=dict(
                    color=carrier_stats['anomaly_rate_%'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title='Anomaly Rate %')
                ),
                text=carrier_stats['anomaly_rate_%'].round(1),
                textposition='auto',
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    'Anomaly Rate: %{y:.1f}%<br>' +
                    'Total Orders: %{customdata}<br>' +
                    '<extra></extra>'
                ),
                customdata=carrier_stats['total']
            )
        ])
        
        fig.update_layout(
            title='Anomaly Rate by Carrier',
            xaxis_title='Carrier',
            yaxis_title='Anomaly Rate (%)',
            height=400,
            template='plotly_white'
        )
        
        return fig


# ==================== Convenience Functions ====================
def detect_all_anomalies(
    data: pd.DataFrame,
    contamination: float = 0.1
) -> pd.DataFrame:
    """
    Quick function to detect all anomalies.
    
    Args:
        data: Delivery data
        contamination: Expected outlier proportion
        
    Returns:
        Comprehensive anomaly results
    """
    detector = AnomalyDetector(data, contamination)
    return detector.get_comprehensive_anomalies()


def get_critical_anomalies(
    data: pd.DataFrame,
    contamination: float = 0.1
) -> pd.DataFrame:
    """
    Get only critical anomalies for immediate attention.
    
    Args:
        data: Delivery data
        contamination: Expected outlier proportion
        
    Returns:
        Critical anomalies only
    """
    detector = AnomalyDetector(data, contamination)
    anomalies = detector.get_comprehensive_anomalies()
    
    # Filter for critical
    critical = anomalies[
        (anomalies['anomaly_severity'] == 'CRITICAL') |
        (anomalies['anomaly_count'] >= 2)
    ]
    
    return critical
