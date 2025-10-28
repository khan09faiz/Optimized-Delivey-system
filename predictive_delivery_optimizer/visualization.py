"""
Visualization module for the Predictive Delivery Optimizer.
Handles data visualization using plotly and other libraries.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from .utils import logger


class Visualizer:
    """Handles data visualization for the delivery optimizer."""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize Visualizer.
        
        Args:
            datasets: Dictionary of DataFrames
        """
        self.datasets = datasets
        self.color_scheme = px.colors.qualitative.Set2
    
    def plot_delivery_performance(self) -> go.Figure:
        """
        Create delivery performance visualization.
        
        Returns:
            Plotly figure
        """
        if 'delivery_performance' not in self.datasets:
            return go.Figure()
        
        df = self.datasets['delivery_performance']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Delivery Status Distribution', 'Delay Distribution')
        )
        
        # Status distribution
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            fig.add_trace(
                go.Bar(x=status_counts.index, y=status_counts.values, 
                      name='Status', marker_color=self.color_scheme[0]),
                row=1, col=1
            )
        
        # Delay distribution
        if 'delay_minutes' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['delay_minutes'], name='Delay',
                           marker_color=self.color_scheme[1]),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Delivery Performance Overview",
            showlegend=False,
            height=400
        )
        
        logger.info("Delivery performance plot created")
        return fig
    
    def plot_cost_breakdown(self) -> go.Figure:
        """
        Create cost breakdown visualization.
        
        Returns:
            Plotly figure
        """
        if 'cost_breakdown' not in self.datasets:
            return go.Figure()
        
        df = self.datasets['cost_breakdown']
        
        # Calculate average costs
        cost_columns = ['fuel_cost', 'labor_cost', 'maintenance_cost']
        avg_costs = {}
        
        for col in cost_columns:
            if col in df.columns:
                avg_costs[col.replace('_', ' ').title()] = df[col].mean()
        
        if not avg_costs:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Pie(labels=list(avg_costs.keys()), 
                   values=list(avg_costs.values()),
                   hole=0.3)
        ])
        
        fig.update_layout(
            title_text="Average Cost Breakdown",
            height=400
        )
        
        logger.info("Cost breakdown plot created")
        return fig
    
    def plot_route_analysis(self) -> go.Figure:
        """
        Create route analysis visualization.
        
        Returns:
            Plotly figure
        """
        if 'routes_distance' not in self.datasets:
            return go.Figure()
        
        df = self.datasets['routes_distance']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distance Distribution', 'Traffic Level Distribution')
        )
        
        # Distance distribution
        if 'distance_km' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['distance_km'], name='Distance',
                           marker_color=self.color_scheme[2]),
                row=1, col=1
            )
        
        # Traffic level distribution
        if 'traffic_level' in df.columns:
            traffic_counts = df['traffic_level'].value_counts()
            fig.add_trace(
                go.Bar(x=traffic_counts.index, y=traffic_counts.values,
                      name='Traffic', marker_color=self.color_scheme[3]),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Route Analysis",
            showlegend=False,
            height=400
        )
        
        logger.info("Route analysis plot created")
        return fig
    
    def plot_customer_feedback(self) -> go.Figure:
        """
        Create customer feedback visualization.
        
        Returns:
            Plotly figure
        """
        if 'customer_feedback' not in self.datasets:
            return go.Figure()
        
        df = self.datasets['customer_feedback']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rating Distribution', 'Delivery Rating Distribution')
        )
        
        # Rating distribution
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=rating_counts.index, y=rating_counts.values,
                      name='Rating', marker_color=self.color_scheme[4]),
                row=1, col=1
            )
        
        # Delivery rating distribution
        if 'delivery_rating' in df.columns:
            delivery_rating_counts = df['delivery_rating'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=delivery_rating_counts.index, y=delivery_rating_counts.values,
                      name='Delivery Rating', marker_color=self.color_scheme[5]),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Customer Feedback Analysis",
            showlegend=False,
            height=400
        )
        
        logger.info("Customer feedback plot created")
        return fig
    
    def plot_fleet_status(self) -> go.Figure:
        """
        Create fleet status visualization.
        
        Returns:
            Plotly figure
        """
        if 'vehicle_fleet' not in self.datasets:
            return go.Figure()
        
        df = self.datasets['vehicle_fleet']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Vehicle Availability', 'Maintenance Status')
        )
        
        # Availability distribution
        if 'availability' in df.columns:
            avail_counts = df['availability'].value_counts()
            fig.add_trace(
                go.Pie(labels=avail_counts.index, values=avail_counts.values,
                      name='Availability'),
                row=1, col=1
            )
        
        # Maintenance status distribution
        if 'maintenance_status' in df.columns:
            maint_counts = df['maintenance_status'].value_counts()
            fig.add_trace(
                go.Pie(labels=maint_counts.index, values=maint_counts.values,
                      name='Maintenance'),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Fleet Status Overview",
            height=400
        )
        
        logger.info("Fleet status plot created")
        return fig
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame) -> go.Figure:
        """
        Create feature importance visualization.
        
        Args:
            feature_importance_df: DataFrame with feature importance
            
        Returns:
            Plotly figure
        """
        if feature_importance_df.empty:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_importance_df['importance'],
                y=feature_importance_df['feature'],
                orientation='h',
                marker_color=self.color_scheme[0]
            )
        ])
        
        fig.update_layout(
            title_text="Top Features by Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(feature_importance_df) * 30)
        )
        
        logger.info("Feature importance plot created")
        return fig
    
    def plot_predictions_vs_actual(self, actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
        """
        Create predictions vs actual values plot.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(color=self.color_scheme[0], size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title_text="Predictions vs Actual Values",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            height=500
        )
        
        logger.info("Predictions vs actual plot created")
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, date_column: str, 
                        value_column: str, title: str = "Time Series") -> go.Figure:
        """
        Create time series visualization.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            title: Plot title
            
        Returns:
            Plotly figure
        """
        df_sorted = df.sort_values(date_column)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted[date_column],
            y=df_sorted[value_column],
            mode='lines+markers',
            name=value_column,
            line=dict(color=self.color_scheme[0])
        ))
        
        fig.update_layout(
            title_text=title,
            xaxis_title=date_column,
            yaxis_title=value_column,
            height=400
        )
        
        logger.info(f"Time series plot created: {title}")
        return fig
    
    def create_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Create key metrics for dashboard display.
        
        Returns:
            Dictionary with key metrics
        """
        metrics = {}
        
        # Delivery metrics
        if 'delivery_performance' in self.datasets:
            df = self.datasets['delivery_performance']
            metrics['total_deliveries'] = len(df)
            if 'delay_minutes' in df.columns:
                metrics['on_time_rate'] = f"{(df['delay_minutes'] <= 0).mean() * 100:.1f}%"
                metrics['avg_delay'] = f"{df['delay_minutes'].mean():.1f} min"
        
        # Cost metrics
        if 'cost_breakdown' in self.datasets:
            df = self.datasets['cost_breakdown']
            if 'total_cost' in df.columns:
                metrics['avg_cost'] = f"${df['total_cost'].mean():.2f}"
                metrics['total_cost'] = f"${df['total_cost'].sum():.2f}"
        
        # Fleet metrics
        if 'vehicle_fleet' in self.datasets:
            df = self.datasets['vehicle_fleet']
            metrics['total_vehicles'] = len(df)
            if 'availability' in df.columns:
                metrics['available_vehicles'] = (df['availability'] == 'Available').sum()
        
        # Customer satisfaction
        if 'customer_feedback' in self.datasets:
            df = self.datasets['customer_feedback']
            if 'rating' in df.columns:
                metrics['avg_rating'] = f"{df['rating'].mean():.2f}/5"
        
        logger.info("Dashboard metrics created")
        return metrics
