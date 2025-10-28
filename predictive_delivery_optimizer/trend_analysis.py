"""
Historical Trend Analysis Module
Analyzes delivery performance trends over time
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyze historical trends in delivery performance"""
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize TrendAnalyzer
        
        Args:
            data: Optional DataFrame with delivery data
        """
        self.data = data
        logger.info("TrendAnalyzer initialized")
        
    def analyze_time_trends(self, df: pd.DataFrame = None, period: str = 'daily') -> Dict:
        """
        Analyze trends over different time periods
        
        Args:
            df: DataFrame with order_date and delivery date columns
            period: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Dictionary containing trend metrics
        """
        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No data provided")
        
        logger.info(f"Analyzing {period} trends...")
        
        df = df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Calculate delay metrics if not present
        if 'delay_flag' not in df.columns and 'actual_delivery_date' in df.columns and 'expected_delivery_date' in df.columns:
            df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
            df['expected_delivery_date'] = pd.to_datetime(df['expected_delivery_date'])
            df['delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
            df['delay_flag'] = (df['delay_days'] > 0).astype(int)
        
        # If still no delay columns, use dummy data
        if 'delay_flag' not in df.columns:
            logger.warning("No delay information available, using order count only")
            df['delay_flag'] = 0
            df['delay_days'] = 0
        
        # Daily trends
        daily_trends = df.groupby(df['order_date'].dt.date).agg({
            'delay_flag': ['mean', 'count'],
            'delay_days': 'mean'
        }).reset_index()
        daily_trends.columns = ['date', 'delay_rate', 'total_orders', 'avg_delay_days']
        
        # Weekly trends
        df['week'] = df['order_date'].dt.isocalendar().week
        weekly_trends = df.groupby('week').agg({
            'delay_flag': ['mean', 'count'],
            'delay_days': 'mean'
        }).reset_index()
        weekly_trends.columns = ['week', 'delay_rate', 'total_orders', 'avg_delay_days']
        
        # Monthly trends
        df['month'] = df['order_date'].dt.month
        monthly_trends = df.groupby('month').agg({
            'delay_flag': ['mean', 'count'],
            'delay_days': 'mean'
        }).reset_index()
        monthly_trends.columns = ['month', 'delay_rate', 'total_orders', 'avg_delay_days']
        
        # Calculate trend direction
        if len(daily_trends) > 1:
            recent_trend = np.polyfit(range(len(daily_trends)), daily_trends['delay_rate'], 1)[0]
            trend_direction = "improving" if recent_trend < 0 else "worsening"
        else:
            trend_direction = "insufficient data"
        
        logger.info(f"Trend analysis complete. Direction: {trend_direction}")
        
        return {
            'daily': daily_trends,
            'weekly': weekly_trends,
            'monthly': monthly_trends,
            'trend_direction': trend_direction,
            'trend_slope': recent_trend if len(daily_trends) > 1 else 0
        }
    
    def analyze_carrier_trends(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analyze carrier performance trends over time
        
        Args:
            df: DataFrame with carrier and delay data
            
        Returns:
            DataFrame with carrier trend metrics
        """
        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No data provided")
            
        logger.info("Analyzing carrier trends...")
        
        df = df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['month'] = df['order_date'].dt.month
        
        # Calculate delay metrics if not present
        if 'delay_flag' not in df.columns and 'actual_delivery_date' in df.columns and 'expected_delivery_date' in df.columns:
            df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
            df['expected_delivery_date'] = pd.to_datetime(df['expected_delivery_date'])
            df['delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
            df['delay_flag'] = (df['delay_days'] > 0).astype(int)
        
        # If still no delay columns, use dummy data
        if 'delay_flag' not in df.columns:
            logger.warning("No delay information available")
            df['delay_flag'] = 0
        
        carrier_trends = df.groupby(['carrier', 'month']).agg({
            'delay_flag': 'mean',
            'order_id': 'count'
        }).reset_index()
        carrier_trends.columns = ['carrier', 'month', 'delay_rate', 'order_count']
        
        return carrier_trends
    
    def plot_delay_trend(self, trends: Dict) -> go.Figure:
        """
        Create delay rate trend visualization
        
        Args:
            trends: Dictionary from analyze_time_trends
            
        Returns:
            Plotly figure
        """
        logger.info("Creating delay trend plot...")
        
        daily = trends['daily']
        
        fig = go.Figure()
        
        # Daily delay rate line
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['delay_rate'] * 100,
            mode='lines+markers',
            name='Daily Delay Rate',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=6)
        ))
        
        # Moving average (7-day)
        if len(daily) >= 7:
            daily['ma7'] = daily['delay_rate'].rolling(window=7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=daily['date'],
                y=daily['ma7'] * 100,
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='#4ECDC4', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title='Delivery Delay Rate Over Time',
            xaxis_title='Date',
            yaxis_title='Delay Rate (%)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_carrier_performance_trends(self, carrier_trends: pd.DataFrame) -> go.Figure:
        """
        Create carrier performance trend visualization
        
        Args:
            carrier_trends: DataFrame from analyze_carrier_trends
            
        Returns:
            Plotly figure
        """
        logger.info("Creating carrier performance trend plot...")
        
        fig = go.Figure()
        
        for carrier in carrier_trends['carrier'].unique():
            data = carrier_trends[carrier_trends['carrier'] == carrier]
            fig.add_trace(go.Scatter(
                x=data['month'],
                y=data['delay_rate'] * 100,
                mode='lines+markers',
                name=carrier,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Carrier Performance Trends by Month',
            xaxis_title='Month',
            yaxis_title='Delay Rate (%)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def get_trend_summary(self, trends: Dict) -> Dict:
        """
        Get summary statistics for trends
        
        Args:
            trends: Dictionary from analyze_time_trends
            
        Returns:
            Summary statistics
        """
        daily = trends['daily']
        
        summary = {
            'current_delay_rate': daily['delay_rate'].iloc[-1] if len(daily) > 0 else 0,
            'avg_delay_rate': daily['delay_rate'].mean(),
            'best_day': daily.loc[daily['delay_rate'].idxmin()]['date'] if len(daily) > 0 else None,
            'worst_day': daily.loc[daily['delay_rate'].idxmax()]['date'] if len(daily) > 0 else None,
            'trend_direction': trends['trend_direction'],
            'improvement_rate': abs(trends['trend_slope']) * 100 if trends['trend_slope'] < 0 else 0
        }
        
        return summary
