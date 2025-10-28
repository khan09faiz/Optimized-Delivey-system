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
    
    def plot_order_value_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot order value distribution with delay breakdown.
        
        Args:
            df: DataFrame with order_value and delay_flag columns
            
        Returns:
            Plotly figure
        """
        logger.info("Creating order value distribution plot...")
        
        if 'order_value_inr' not in df.columns:
            logger.warning("order_value_inr column not found")
            return go.Figure()
        
        # Create value bins
        df_copy = df.copy()
        df_copy['value_range'] = pd.cut(
            df_copy['order_value_inr'], 
            bins=[0, 1000, 2000, 3000, 4000, 5000],
            labels=['₹0-1K', '₹1K-2K', '₹2K-3K', '₹3K-4K', '₹4K+']
        )
        
        # Count by value range and delay status
        if 'delay_flag' in df_copy.columns:
            value_delay = df_copy.groupby(['value_range', 'delay_flag']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='On-Time',
                x=value_delay.index.astype(str),
                y=value_delay[0] if 0 in value_delay.columns else [],
                marker_color='#28a745'
            ))
            fig.add_trace(go.Bar(
                name='Delayed',
                x=value_delay.index.astype(str),
                y=value_delay[1] if 1 in value_delay.columns else [],
                marker_color='#dc3545'
            ))
            
            fig.update_layout(
                title='Order Value Distribution by Delivery Status',
                xaxis_title='Order Value Range',
                yaxis_title='Number of Orders',
                barmode='stack',
                height=400
            )
        else:
            value_counts = df_copy['value_range'].value_counts().sort_index()
            fig = go.Figure(data=[
                go.Bar(x=value_counts.index.astype(str), y=value_counts.values, marker_color='#4ECDC4')
            ])
            fig.update_layout(
                title='Order Value Distribution',
                xaxis_title='Order Value Range',
                yaxis_title='Number of Orders',
                height=400
            )
        
        return fig
    
    def plot_delivery_time_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot actual vs promised delivery time analysis.
        
        Args:
            df: DataFrame with delivery time columns
            
        Returns:
            Plotly figure
        """
        logger.info("Creating delivery time analysis plot...")
        
        if 'promised_delivery_days' not in df.columns or 'actual_delivery_days' not in df.columns:
            logger.warning("Delivery time columns not found")
            return go.Figure()
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=df['promised_delivery_days'],
            y=df['actual_delivery_days'],
            mode='markers',
            marker=dict(
                size=8,
                color=df.get('delay_flag', 0),
                colorscale=[[0, '#28a745'], [1, '#dc3545']],
                showscale=True,
                colorbar=dict(title='Delayed')
            ),
            text=[f"Order: {i}<br>Promised: {p}d<br>Actual: {a}d" 
                  for i, p, a in zip(df.index, df['promised_delivery_days'], df['actual_delivery_days'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add perfect delivery line (y=x)
        max_days = max(df['promised_delivery_days'].max(), df['actual_delivery_days'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_days],
            y=[0, max_days],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Perfect Delivery',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Promised vs Actual Delivery Time',
            xaxis_title='Promised Delivery (days)',
            yaxis_title='Actual Delivery (days)',
            height=450
        )
        
        return fig
    
    def plot_traffic_impact(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot traffic delay impact on delivery.
        
        Args:
            df: DataFrame with traffic_delay_minutes column
            
        Returns:
            Plotly figure
        """
        logger.info("Creating traffic impact plot...")
        
        if 'traffic_delay_minutes' not in df.columns:
            logger.warning("traffic_delay_minutes column not found")
            return go.Figure()
        
        # Create traffic delay bins
        df_copy = df.copy()
        df_copy['traffic_category'] = pd.cut(
            df_copy['traffic_delay_minutes'],
            bins=[-1, 0, 15, 30, 60, float('inf')],
            labels=['No Delay', 'Low (1-15min)', 'Medium (15-30min)', 'High (30-60min)', 'Severe (60min+)']
        )
        
        if 'delay_flag' in df_copy.columns:
            traffic_delay = df_copy.groupby(['traffic_category', 'delay_flag']).size().unstack(fill_value=0)
            
            # Calculate delay rate
            delay_rate = (traffic_delay[1] / (traffic_delay[0] + traffic_delay[1]) * 100).round(1)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Stacked bars
            fig.add_trace(
                go.Bar(name='On-Time', x=traffic_delay.index.astype(str), 
                       y=traffic_delay[0] if 0 in traffic_delay.columns else [],
                       marker_color='#28a745'),
                secondary_y=False
            )
            fig.add_trace(
                go.Bar(name='Delayed', x=traffic_delay.index.astype(str),
                       y=traffic_delay[1] if 1 in traffic_delay.columns else [],
                       marker_color='#dc3545'),
                secondary_y=False
            )
            
            # Delay rate line
            fig.add_trace(
                go.Scatter(name='Delay Rate', x=delay_rate.index.astype(str), y=delay_rate.values,
                          mode='lines+markers', line=dict(color='#FF6B6B', width=3),
                          marker=dict(size=10)),
                secondary_y=True
            )
            
            fig.update_layout(
                title='Traffic Delay Impact on Delivery Performance',
                xaxis_title='Traffic Delay Category',
                barmode='stack',
                height=400
            )
            fig.update_yaxes(title_text="Number of Orders", secondary_y=False)
            fig.update_yaxes(title_text="Delay Rate (%)", secondary_y=True)
        else:
            traffic_counts = df_copy['traffic_category'].value_counts().sort_index()
            fig = go.Figure(data=[
                go.Bar(x=traffic_counts.index.astype(str), y=traffic_counts.values, marker_color='#FF9F43')
            ])
            fig.update_layout(
                title='Traffic Delay Distribution',
                xaxis_title='Traffic Delay Category',
                yaxis_title='Number of Orders',
                height=400
            )
        
        return fig
    
    def plot_customer_segment_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot customer segment performance.
        
        Args:
            df: DataFrame with customer_segment column
            
        Returns:
            Plotly figure
        """
        logger.info("Creating customer segment analysis plot...")
        
        if 'customer_segment' not in df.columns:
            logger.warning("customer_segment column not found")
            return go.Figure()
        
        segment_stats = df.groupby('customer_segment').agg({
            'order_id': 'count',
            'delay_flag': 'mean' if 'delay_flag' in df.columns else 'count',
            'order_value_inr': 'mean' if 'order_value_inr' in df.columns else 'count'
        }).round(2)
        
        segment_stats.columns = ['Orders', 'Delay_Rate', 'Avg_Value']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Order Volume', 'Delay Rate (%)', 'Avg Order Value (₹)'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Order volume
        fig.add_trace(
            go.Bar(x=segment_stats.index, y=segment_stats['Orders'], 
                   marker_color='#4ECDC4', name='Orders'),
            row=1, col=1
        )
        
        # Delay rate
        fig.add_trace(
            go.Bar(x=segment_stats.index, y=segment_stats['Delay_Rate']*100, 
                   marker_color='#FF6B6B', name='Delay %'),
            row=1, col=2
        )
        
        # Average value
        fig.add_trace(
            go.Bar(x=segment_stats.index, y=segment_stats['Avg_Value'], 
                   marker_color='#95E1D3', name='Avg Value'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text='Customer Segment Performance Analysis',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_weather_impact(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot weather impact on deliveries.
        
        Args:
            df: DataFrame with weather_impact column
            
        Returns:
            Plotly figure
        """
        logger.info("Creating weather impact plot...")
        
        if 'weather_impact' not in df.columns:
            logger.warning("weather_impact column not found")
            return go.Figure()
        
        weather_stats = df.groupby('weather_impact').agg({
            'order_id': 'count',
            'delay_flag': 'mean' if 'delay_flag' in df.columns else 'count'
        }).round(3)
        
        weather_stats.columns = ['Count', 'Delay_Rate']
        weather_stats = weather_stats.sort_values('Delay_Rate', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=weather_stats.index,
            y=weather_stats['Count'],
            name='Orders',
            marker_color='#74B9FF',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=weather_stats.index,
            y=weather_stats['Delay_Rate'] * 100,
            name='Delay Rate (%)',
            mode='lines+markers',
            line=dict(color='#FD79A8', width=3),
            marker=dict(size=10),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Weather Impact on Delivery Performance',
            xaxis=dict(title='Weather Condition'),
            yaxis=dict(title='Number of Orders', side='left'),
            yaxis2=dict(title='Delay Rate (%)', side='right', overlaying='y'),
            height=400,
            hovermode='x unified'
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
