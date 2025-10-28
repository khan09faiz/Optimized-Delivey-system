"""
Customer Insights and Order Tracking Module

Provides actionable insights for customers to improve delivery success
and comprehensive order tracking capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from utils import Config, setup_logging, get_risk_category, format_currency

logger = setup_logging("customer_insights")


class CustomerInsights:
    """Generate insights and recommendations for customers."""
    
    def __init__(self):
        """Initialize CustomerInsights."""
        logger.info("CustomerInsights initialized")
    
    def get_customer_best_practices(self, order_data: pd.Series) -> Dict[str, List[str]]:
        """
        Generate personalized best practices for customer based on order characteristics.
        
        Args:
            order_data: Single order data as pandas Series
            
        Returns:
            Dictionary with categorized recommendations
        """
        recommendations = {
            'before_ordering': [],
            'during_delivery': [],
            'communication': [],
            'general_tips': []
        }
        
        # Before Ordering Tips
        recommendations['before_ordering'].append(
            "📅 **Plan Ahead**: Order at least 2-3 days before you need the item for non-urgent deliveries"
        )
        
        if order_data.get('is_weekend', 0) == 1:
            recommendations['before_ordering'].append(
                "⏰ **Avoid Weekend Orders**: Weekend deliveries have 35% higher delay rates. Order on weekdays if possible"
            )
        
        if order_data.get('traffic_category', '') in ['High', 'Very High']:
            recommendations['before_ordering'].append(
                "🚦 **Traffic Consideration**: Choose off-peak delivery times (10 AM - 3 PM) to avoid traffic delays"
            )
        
        distance_km = order_data.get('distance_km', 0)
        if distance_km > 100:
            recommendations['before_ordering'].append(
                f"📍 **Long Distance ({distance_km:.0f} km)**: Add 1-2 extra days to expected delivery time for safety"
            )
        
        if order_data.get('priority', '').lower() == 'low':
            recommendations['before_ordering'].append(
                "⭐ **Upgrade Priority**: Consider upgrading to 'High Priority' for time-sensitive orders (reduces delay risk by 40%)"
            )
        
        # During Delivery Tips
        recommendations['during_delivery'].append(
            "📱 **Enable Notifications**: Keep SMS/email notifications ON to receive real-time delivery updates"
        )
        
        recommendations['during_delivery'].append(
            "🏠 **Be Available**: Ensure someone is present at delivery address during estimated time window"
        )
        
        recommendations['during_delivery'].append(
            "📞 **Share Contact**: Provide accurate phone number and enable calls from unknown numbers during delivery day"
        )
        
        if order_data.get('order_value', 0) > 5000:
            recommendations['during_delivery'].append(
                "💳 **ID Verification**: Keep a valid ID ready for high-value order verification"
            )
        
        recommendations['during_delivery'].append(
            "🔍 **Track Actively**: Check order tracking page 2-3 times on delivery day for live updates"
        )
        
        # Communication Tips
        recommendations['communication'].append(
            "💬 **Proactive Communication**: Call carrier if delivery is delayed beyond expected time by 2+ hours"
        )
        
        recommendations['communication'].append(
            "✍️ **Clear Instructions**: Provide specific delivery instructions (landmarks, gate codes, parking info)"
        )
        
        recommendations['communication'].append(
            "⏱️ **Update Time Preferences**: Inform carrier if you need delivery rescheduled to a better time"
        )
        
        recommendations['communication'].append(
            "🎯 **Alternative Address**: Provide a backup delivery address (office/neighbor) if you might be unavailable"
        )
        
        # General Tips
        recommendations['general_tips'].append(
            "🌟 **Choose Reliable Carriers**: Select carriers with >90% on-time delivery rate for important orders"
        )
        
        if order_data.get('weather_condition', '').lower() in ['rain', 'storm', 'heavy rain']:
            recommendations['general_tips'].append(
                "🌧️ **Weather Alert**: Bad weather detected - add 30-60 min buffer to expected delivery time"
            )
        
        recommendations['general_tips'].append(
            "📦 **Inspect on Arrival**: Check package condition before signing - report damages immediately"
        )
        
        recommendations['general_tips'].append(
            "⭐ **Leave Feedback**: Rate delivery experience to help improve service quality"
        )
        
        recommendations['general_tips'].append(
            "🔔 **Set Reminders**: Set phone reminder 1 hour before expected delivery time to prepare"
        )
        
        logger.info(f"Generated {sum(len(v) for v in recommendations.values())} recommendations")
        return recommendations
    
    def get_order_improvement_score(self, order_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate how much customer can improve delivery success.
        
        Args:
            order_data: Single order data
            
        Returns:
            Dictionary with improvement score and factors
        """
        score = 100  # Start with perfect score
        factors = []
        
        # Check controllable factors
        if order_data.get('is_weekend', 0) == 1:
            score -= 15
            factors.append({
                'factor': 'Weekend Order',
                'impact': -15,
                'fix': 'Order on weekdays instead'
            })
        
        if order_data.get('priority', '').lower() == 'low':
            score -= 20
            factors.append({
                'factor': 'Low Priority',
                'impact': -20,
                'fix': 'Upgrade to High Priority (+₹50-100)'
            })
        
        if order_data.get('customer_rating', 5) < 3:
            score -= 10
            factors.append({
                'factor': 'Past Poor Experience',
                'impact': -10,
                'fix': 'Communicate clearly with driver, provide good directions'
            })
        
        # Check if traffic is high during typical delivery hours
        if order_data.get('traffic_delay_mins', 0) > 30:
            score -= 12
            factors.append({
                'factor': 'High Traffic Time',
                'impact': -12,
                'fix': 'Request delivery during off-peak hours (10 AM - 3 PM)'
            })
        
        # Check address clarity (using warehouse distance as proxy)
        if order_data.get('distance_km', 0) > 150:
            score -= 8
            factors.append({
                'factor': 'Long Distance',
                'impact': -8,
                'fix': 'Order from closer warehouses or use express service'
            })
        
        improvement_potential = 100 - score
        
        return {
            'current_score': max(score, 0),
            'improvement_potential': improvement_potential,
            'max_possible_score': 100,
            'grade': 'A+' if score >= 90 else 'A' if score >= 80 else 'B' if score >= 70 else 'C',
            'factors': factors,
            'total_factors': len(factors)
        }
    
    def plot_improvement_opportunities(self, improvement_data: Dict[str, Any]) -> go.Figure:
        """
        Visualize improvement opportunities.
        
        Args:
            improvement_data: Output from get_order_improvement_score
            
        Returns:
            Plotly figure
        """
        factors = improvement_data.get('factors', [])
        
        if not factors:
            # No improvements needed
            fig = go.Figure()
            fig.add_annotation(
                text="✅ Excellent! No improvements needed",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="green")
            )
            fig.update_layout(height=300)
            return fig
        
        df = pd.DataFrame(factors)
        df['impact_positive'] = -df['impact']  # Convert to positive for better viz
        
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=df['impact_positive'],
            y=df['factor'],
            orientation='h',
            marker=dict(
                color=df['impact_positive'],
                colorscale='Reds',
                showscale=False
            ),
            text=[f"+{x}%" for x in df['impact_positive']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Improvement: +%{x}%<br><extra></extra>'
        ))
        
        fig.update_layout(
            title="🎯 Your Improvement Opportunities",
            xaxis_title="Potential Score Improvement (%)",
            yaxis_title="",
            height=max(300, len(factors) * 60),
            showlegend=False,
            template="plotly_white"
        )
        
        return fig


class OrderTracker:
    """Track and visualize order status and timeline."""
    
    def __init__(self):
        """Initialize OrderTracker."""
        logger.info("OrderTracker initialized")
    
    def get_order_timeline(self, order_data: pd.Series) -> List[Dict[str, Any]]:
        """
        Generate order timeline with milestones.
        
        Args:
            order_data: Single order data
            
        Returns:
            List of timeline events
        """
        timeline = []
        
        # Simulated timeline based on order data
        order_date = order_data.get('order_date', datetime.now())
        if isinstance(order_date, str):
            try:
                order_date = pd.to_datetime(order_date)
            except:
                order_date = datetime.now()
        
        # Milestone 1: Order Placed
        timeline.append({
            'stage': 'Order Placed',
            'timestamp': order_date,
            'status': 'completed',
            'icon': '🛒',
            'description': 'Order successfully received and confirmed'
        })
        
        # Milestone 2: Processing
        processing_time = order_date + timedelta(hours=2)
        timeline.append({
            'stage': 'Order Processing',
            'timestamp': processing_time,
            'status': 'completed',
            'icon': '📦',
            'description': 'Order picked and packed at warehouse'
        })
        
        # Milestone 3: Dispatched
        dispatch_time = order_date + timedelta(hours=6)
        timeline.append({
            'stage': 'Dispatched',
            'timestamp': dispatch_time,
            'status': 'completed',
            'icon': '🚚',
            'description': f"Assigned to carrier: {order_data.get('carrier', 'N/A')}"
        })
        
        # Milestone 4: In Transit
        transit_time = order_date + timedelta(hours=12)
        timeline.append({
            'stage': 'In Transit',
            'timestamp': transit_time,
            'status': 'in_progress',
            'icon': '📍',
            'description': f"En route to destination ({order_data.get('distance_km', 0):.1f} km)"
        })
        
        # Milestone 5: Out for Delivery
        promised_date = order_data.get('promised_delivery_date', order_date + timedelta(days=2))
        if isinstance(promised_date, str):
            try:
                promised_date = pd.to_datetime(promised_date)
            except:
                promised_date = order_date + timedelta(days=2)
        
        out_for_delivery_time = promised_date.replace(hour=9, minute=0)
        timeline.append({
            'stage': 'Out for Delivery',
            'timestamp': out_for_delivery_time,
            'status': 'pending',
            'icon': '🏃',
            'description': 'Delivery executive assigned - arriving soon'
        })
        
        # Milestone 6: Delivered
        expected_delivery = promised_date.replace(hour=17, minute=0)
        timeline.append({
            'stage': 'Delivered',
            'timestamp': expected_delivery,
            'status': 'pending',
            'icon': '✅',
            'description': f"Expected by {expected_delivery.strftime('%I:%M %p, %b %d')}"
        })
        
        return timeline
    
    def plot_order_timeline(self, timeline: List[Dict[str, Any]]) -> go.Figure:
        """
        Create visual timeline of order progress.
        
        Args:
            timeline: List of timeline events
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(timeline)
        
        # Status colors
        status_colors = {
            'completed': '#28a745',
            'in_progress': '#ffc107',
            'pending': '#6c757d'
        }
        
        df['color'] = df['status'].map(status_colors)
        df['y_pos'] = range(len(df))
        
        fig = go.Figure()
        
        # Add timeline line
        fig.add_trace(go.Scatter(
            x=[0] * len(df),
            y=df['y_pos'],
            mode='lines',
            line=dict(color='lightgray', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add milestone markers
        fig.add_trace(go.Scatter(
            x=[0] * len(df),
            y=df['y_pos'],
            mode='markers+text',
            marker=dict(
                size=25,
                color=df['color'],
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=df['icon'],
            textposition='middle center',
            textfont=dict(size=14),
            showlegend=False,
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Time: %{customdata[1]}<br>' +
                         '%{customdata[2]}<extra></extra>',
            customdata=df[['stage', 'timestamp', 'description']].values
        ))
        
        # Add stage labels
        for idx, row in df.iterrows():
            status_text = {
                'completed': '✓',
                'in_progress': '⏳',
                'pending': '○'
            }[row['status']]
            
            fig.add_annotation(
                x=0.15,
                y=row['y_pos'],
                text=f"<b>{row['stage']}</b> {status_text}",
                showarrow=False,
                xanchor='left',
                font=dict(size=12)
            )
            
            fig.add_annotation(
                x=0.15,
                y=row['y_pos'] - 0.3,
                text=row['description'],
                showarrow=False,
                xanchor='left',
                font=dict(size=10, color='gray')
            )
            
            fig.add_annotation(
                x=-0.15,
                y=row['y_pos'],
                text=row['timestamp'].strftime('%b %d, %I:%M %p'),
                showarrow=False,
                xanchor='right',
                font=dict(size=10, color='gray')
            )
        
        fig.update_layout(
            title="📦 Order Tracking Timeline",
            xaxis=dict(visible=False, range=[-0.5, 1]),
            yaxis=dict(visible=False, range=[-1, len(df)]),
            height=max(400, len(df) * 100),
            showlegend=False,
            template="plotly_white",
            margin=dict(l=150, r=150, t=80, b=20)
        )
        
        return fig
    
    def get_tracking_insights(self, order_data: pd.Series, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate tracking insights and alerts.
        
        Args:
            order_data: Order data
            timeline: Order timeline
            
        Returns:
            Dictionary with tracking insights
        """
        insights = {
            'current_stage': None,
            'estimated_delivery': None,
            'delay_risk': 'low',
            'alerts': [],
            'next_action': None
        }
        
        # Find current stage
        for event in timeline:
            if event['status'] == 'in_progress':
                insights['current_stage'] = event['stage']
                break
            elif event['status'] == 'completed':
                insights['current_stage'] = event['stage']
        
        # Get estimated delivery
        for event in timeline:
            if event['stage'] == 'Delivered':
                insights['estimated_delivery'] = event['timestamp']
                break
        
        # Assess delay risk
        delay_prob = order_data.get('delay_probability', 0)
        if delay_prob >= 0.7:
            insights['delay_risk'] = 'high'
            insights['alerts'].append('⚠️ High delay risk detected - carrier may contact you for directions')
            insights['next_action'] = 'Ensure you are reachable by phone and monitor tracking closely'
        elif delay_prob >= 0.4:
            insights['delay_risk'] = 'medium'
            insights['alerts'].append('ℹ️ Moderate delay risk - keep delivery window flexible')
            insights['next_action'] = 'Check tracking page once today'
        else:
            insights['delay_risk'] = 'low'
            insights['alerts'].append('✅ On track for on-time delivery')
            insights['next_action'] = 'Relax - delivery progressing smoothly'
        
        # Traffic alert
        if order_data.get('traffic_category', '') in ['High', 'Very High']:
            insights['alerts'].append('🚦 Traffic delays possible - delivery time may vary by 30-60 minutes')
        
        # Weather alert
        if order_data.get('weather_condition', '').lower() in ['rain', 'storm', 'heavy rain']:
            insights['alerts'].append('🌧️ Weather conditions may affect delivery timing')
        
        return insights
    
    def plot_real_time_location(self, order_data: pd.Series) -> go.Figure:
        """
        Create simulated real-time location map.
        
        Args:
            order_data: Order data
            
        Returns:
            Plotly figure with route visualization
        """
        # Simulated coordinates (in real system, would use GPS data)
        warehouse_lat = 28.6139
        warehouse_lon = 77.2090
        
        delivery_lat = warehouse_lat + np.random.uniform(-0.5, 0.5)
        delivery_lon = warehouse_lon + np.random.uniform(-0.5, 0.5)
        
        # Current location (60% of the way)
        progress = 0.6
        current_lat = warehouse_lat + (delivery_lat - warehouse_lat) * progress
        current_lon = warehouse_lon + (delivery_lon - warehouse_lon) * progress
        
        fig = go.Figure()
        
        # Route line
        fig.add_trace(go.Scattermapbox(
            lat=[warehouse_lat, current_lat, delivery_lat],
            lon=[warehouse_lon, current_lon, delivery_lon],
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Route',
            showlegend=False
        ))
        
        # Warehouse marker
        fig.add_trace(go.Scattermapbox(
            lat=[warehouse_lat],
            lon=[warehouse_lon],
            mode='markers+text',
            marker=dict(size=15, color='green'),
            text=['🏭 Warehouse'],
            textposition='top center',
            name='Origin',
            showlegend=False
        ))
        
        # Current location marker
        fig.add_trace(go.Scattermapbox(
            lat=[current_lat],
            lon=[current_lon],
            mode='markers+text',
            marker=dict(size=20, color='red'),
            text=['🚚 Current Location'],
            textposition='top center',
            name='Vehicle',
            showlegend=False
        ))
        
        # Destination marker
        fig.add_trace(go.Scattermapbox(
            lat=[delivery_lat],
            lon=[delivery_lon],
            mode='markers+text',
            marker=dict(size=15, color='orange'),
            text=['📍 Your Address'],
            textposition='top center',
            name='Destination',
            showlegend=False
        ))
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=current_lat, lon=current_lon),
                zoom=10
            ),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            title="🗺️ Live Delivery Tracking"
        )
        
        return fig


def get_customer_dashboard_data(order_data: pd.Series) -> Dict[str, Any]:
    """
    Compile all customer-facing insights and tracking info.
    
    Args:
        order_data: Single order data
        
    Returns:
        Complete dashboard data
    """
    insights_gen = CustomerInsights()
    tracker = OrderTracker()
    
    dashboard = {
        'best_practices': insights_gen.get_customer_best_practices(order_data),
        'improvement_score': insights_gen.get_order_improvement_score(order_data),
        'timeline': tracker.get_order_timeline(order_data),
        'order_id': order_data.get('order_id', 'N/A'),
        'carrier': order_data.get('carrier', 'N/A'),
        'estimated_delivery': order_data.get('promised_delivery_date', 'N/A')
    }
    
    dashboard['tracking_insights'] = tracker.get_tracking_insights(
        order_data, 
        dashboard['timeline']
    )
    
    logger.info(f"Generated customer dashboard for order {dashboard['order_id']}")
    
    return dashboard
