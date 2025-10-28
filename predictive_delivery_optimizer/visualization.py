"""
Visualization module using Plotly for interactive charts.

This module provides:
- KPI metrics visualization
- Distribution plots
- Correlation heatmaps
- Model performance charts
- Risk analysis visualizations
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import setup_logging, COLOR_SCHEME, RISK_COLORS, format_currency, format_percentage

logger = setup_logging("visualization")


class Visualizer:
    """Creates interactive Plotly visualizations."""
    
    def __init__(self):
        """Initialize Visualizer."""
        self.color_scheme = COLOR_SCHEME
        self.risk_colors = RISK_COLORS
        logger.info("Visualizer initialized")
    
    def create_kpi_cards(self, metrics: Dict[str, Any]) -> List[go.Figure]:
        """
        Create KPI metric cards.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            List of Plotly figures
        """
        logger.info("Creating KPI cards...")
        
        figures = []
        
        # Total Orders
        fig1 = go.Figure(go.Indicator(
            mode="number",
            value=metrics.get('total_orders', 0),
            title={'text': "Total Orders"},
            number={'font': {'size': 40}}
        ))
        fig1.update_layout(height=200)
        figures.append(fig1)
        
        # Delay Rate
        fig2 = go.Figure(go.Indicator(
            mode="number+delta",
            value=metrics.get('delay_rate', 0) * 100,
            title={'text': "Delay Rate (%)"},
            number={'suffix': "%", 'font': {'size': 40}},
            delta={'reference': 20, 'relative': False}
        ))
        fig2.update_layout(height=200)
        figures.append(fig2)
        
        # Average Delay Days
        fig3 = go.Figure(go.Indicator(
            mode="number",
            value=metrics.get('avg_delay_days', 0),
            title={'text': "Avg Delay (Days)"},
            number={'font': {'size': 40}}
        ))
        fig3.update_layout(height=200)
        figures.append(fig3)
        
        # High Risk Orders
        fig4 = go.Figure(go.Indicator(
            mode="number",
            value=metrics.get('high_risk_count', 0),
            title={'text': "High Risk Orders"},
            number={'font': {'size': 40, 'color': 'red'}}
        ))
        fig4.update_layout(height=200)
        figures.append(fig4)
        
        return figures
    
    def plot_delay_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot delay distribution histogram.
        
        Args:
            df: DataFrame with delay_flag column
            
        Returns:
            Plotly figure
        """
        logger.info("Creating delay distribution plot...")
        
        delay_counts = df['delay_flag'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['On Time', 'Delayed'],
                y=[delay_counts.get(0, 0), delay_counts.get(1, 0)],
                marker_color=[self.color_scheme['success'], self.color_scheme['danger']],
                text=[delay_counts.get(0, 0), delay_counts.get(1, 0)],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Delivery Status Distribution",
            xaxis_title="Status",
            yaxis_title="Number of Orders",
            height=400
        )
        
        return fig
    
    def plot_risk_distribution(self, df: pd.DataFrame, prob_column: str = 'predicted_prob') -> go.Figure:
        """
        Plot risk category distribution.
        
        Args:
            df: DataFrame with risk_category column
            prob_column: Probability column name
            
        Returns:
            Plotly figure
        """
        logger.info("Creating risk distribution plot...")
        
        if 'risk_category' not in df.columns:
            # Create risk categories if not present
            from utils import get_risk_category
            df['risk_category'] = df[prob_column].apply(get_risk_category)
        
        risk_counts = df['risk_category'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[self.risk_colors.get(cat, 'gray') for cat in risk_counts.index],
                text=risk_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Risk Category Distribution",
            xaxis_title="Risk Category",
            yaxis_title="Number of Orders",
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        logger.info("Creating feature importance plot...")
        
        top_features = importance_df.head(top_n).sort_values('importance')
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color=self.color_scheme['primary'],
                text=top_features['importance'].round(4),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> go.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: 2x2 confusion matrix
            
        Returns:
            Plotly figure
        """
        logger.info("Creating confusion matrix plot...")
        
        labels = ['On Time', 'Delayed']
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            text=confusion_matrix,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        return fig
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            
        Returns:
            Plotly figure
        """
        logger.info("Creating ROC curve plot...")
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color=self.color_scheme['primary'], width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_priority_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot delay rate by priority level.
        
        Args:
            df: DataFrame with priority and delay_flag columns
            
        Returns:
            Plotly figure
        """
        logger.info("Creating priority analysis plot...")
        
        priority_delay = df.groupby('priority')['delay_flag'].agg(['mean', 'count']).reset_index()
        priority_delay['mean'] = priority_delay['mean'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=priority_delay['priority'],
            y=priority_delay['mean'],
            name='Delay Rate (%)',
            marker_color=self.color_scheme['warning'],
            text=priority_delay['mean'].round(1),
            texttemplate='%{text}%',
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Delay Rate by Priority Level",
            xaxis_title="Priority",
            yaxis_title="Delay Rate (%)",
            height=400
        )
        
        return fig
    
    def plot_carrier_performance(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot carrier performance comparison.
        
        Args:
            df: DataFrame with carrier and delay_flag columns
            
        Returns:
            Plotly figure
        """
        logger.info("Creating carrier performance plot...")
        
        if 'carrier' not in df.columns:
            logger.warning("carrier column not found")
            return go.Figure()
        
        carrier_stats = df.groupby('carrier').agg({
            'delay_flag': 'mean',
            'order_id': 'count'
        }).reset_index()
        carrier_stats.columns = ['carrier', 'delay_rate', 'total_orders']
        carrier_stats['delay_rate'] = carrier_stats['delay_rate'] * 100
        carrier_stats = carrier_stats.sort_values('delay_rate')
        
        fig = go.Figure(data=[
            go.Bar(
                x=carrier_stats['carrier'],
                y=carrier_stats['delay_rate'],
                marker_color=self.color_scheme['secondary'],
                text=carrier_stats['delay_rate'].round(1),
                texttemplate='%{text}%',
                textposition='auto',
                customdata=carrier_stats['total_orders'],
                hovertemplate='<b>%{x}</b><br>Delay Rate: %{y:.1f}%<br>Orders: %{customdata}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Carrier Performance Comparison",
            xaxis_title="Carrier",
            yaxis_title="Delay Rate (%)",
            height=400
        )
        
        return fig
    
    def plot_distance_vs_delay(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot scatter plot of distance vs delay.
        
        Args:
            df: DataFrame with distance_km and delay data
            
        Returns:
            Plotly figure
        """
        logger.info("Creating distance vs delay scatter plot...")
        
        # Use predicted_prob if available, otherwise use delay_flag
        if 'predicted_prob' in df.columns:
            y_col = 'predicted_prob'
            y_label = 'Delay Probability'
            color_col = 'risk_category' if 'risk_category' in df.columns else None
            color_map = self.risk_colors if color_col else None
        else:
            y_col = 'delay_flag'
            y_label = 'Delayed (1=Yes, 0=No)'
            color_col = 'delay_flag'
            color_map = {0: '#28a745', 1: '#dc3545'}  # Green for on-time, red for delayed
        
        fig = px.scatter(
            df,
            x='distance_km',
            y=y_col,
            color=color_col,
            color_discrete_map=color_map,
            title="Distance vs Delay",
            labels={
                'distance_km': 'Distance (km)',
                y_col: y_label,
                'delay_flag': 'Delayed',
                'risk_category': 'Risk Category'
            },
            opacity=0.6,
            height=500
        )
        
        return fig
    
    def plot_temporal_trends(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot delay trends over time.
        
        Args:
            df: DataFrame with order_date and delay_flag columns
            
        Returns:
            Plotly figure
        """
        logger.info("Creating temporal trends plot...")
        
        if 'order_date' not in df.columns:
            logger.warning("order_date column not found")
            return go.Figure()
        
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_month'] = df['order_date'].dt.to_period('M').astype(str)
        
        monthly_delay = df.groupby('order_month')['delay_flag'].agg(['mean', 'count']).reset_index()
        monthly_delay['mean'] = monthly_delay['mean'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_delay['order_month'],
            y=monthly_delay['mean'],
            mode='lines+markers',
            name='Delay Rate',
            line=dict(color=self.color_scheme['primary'], width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Delay Rate Trends Over Time",
            xaxis_title="Month",
            yaxis_title="Delay Rate (%)",
            height=400
        )
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str]) -> go.Figure:
        """
        Plot correlation heatmap for numeric features.
        
        Args:
            df: DataFrame
            features: List of features to include
            
        Returns:
            Plotly figure
        """
        logger.info("Creating correlation heatmap...")
        
        # Filter to numeric columns
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in [np.float64, np.int64]]
        
        if len(numeric_features) < 2:
            logger.warning("Not enough numeric features for correlation")
            return go.Figure()
        
        corr_matrix = df[numeric_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=600,
            width=800
        )
        
        return fig
    
    def plot_recommendation_breakdown(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot recommendation action breakdown.
        
        Args:
            df: DataFrame with action column
            
        Returns:
            Plotly figure
        """
        logger.info("Creating recommendation breakdown plot...")
        
        if 'action' not in df.columns:
            logger.warning("action column not found")
            return go.Figure()
        
        action_counts = df['action'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=action_counts.index,
                values=action_counts.values,
                hole=0.3,
                textinfo='label+percent'
            )
        ])
        
        fig.update_layout(
            title="Recommended Actions Distribution",
            height=500
        )
        
        return fig


# ==================== Convenience Functions ====================
def get_visualizer() -> Visualizer:
    """
    Create and return a Visualizer instance.
    
    Returns:
        Visualizer instance
    """
    return Visualizer()


def create_dashboard_plots(df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Create all dashboard plots.
    
    Args:
        df: DataFrame with all required columns
        metrics: Dictionary of KPI metrics
        
    Returns:
        Dictionary of plot names to figures
    """
    viz = get_visualizer()
    
    plots = {
        'delay_dist': viz.plot_delay_distribution(df),
        'risk_dist': viz.plot_risk_distribution(df),
        'priority_analysis': viz.plot_priority_analysis(df),
        'carrier_performance': viz.plot_carrier_performance(df),
        'distance_vs_delay': viz.plot_distance_vs_delay(df),
        'temporal_trends': viz.plot_temporal_trends(df)
    }
    
    return plots
