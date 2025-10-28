"""
Cost Savings Calculator Module
==============================
Calculates ROI, cost savings, and financial impact of delivery optimization.
Provides carrier optimization recommendations with cost projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import setup_logging

logger = setup_logging("cost_calculator")


class CostCalculator:
    """Calculates costs and savings for delivery operations."""
    
    # Default cost parameters (can be customized)
    DEFAULT_COSTS = {
        'delay_penalty_per_day': 50.0,  # Cost per day of delay
        'customer_compensation': 25.0,  # Compensation per delayed order
        'redelivery_cost': 75.0,  # Cost of redelivery
        'lost_customer_value': 500.0,  # Lifetime value of lost customer
        'churn_rate_per_delay': 0.05,  # 5% chance of losing customer per delay
        'carrier_switching_cost': 100.0,  # One-time cost to switch carrier
        'route_optimization_savings': 15.0  # Savings per optimized route
    }
    
    def __init__(
        self,
        data: pd.DataFrame,
        cost_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CostCalculator.
        
        Args:
            data: Delivery data with delay information
            cost_params: Optional custom cost parameters
        """
        self.data = data
        self.costs = {**self.DEFAULT_COSTS, **(cost_params or {})}
        
        logger.info(f"CostCalculator initialized with {len(data)} records")
        logger.info(f"Cost parameters: {self.costs}")
    
    def calculate_delay_costs(self) -> Dict[str, float]:
        """
        Calculate total costs incurred due to delays.
        
        Returns:
            Dictionary with cost breakdown
        """
        logger.info("Calculating delay costs...")
        
        # Check for delay columns and calculate if needed
        if 'is_delayed' not in self.data.columns:
            if 'actual_delivery_date' in self.data.columns and 'expected_delivery_date' in self.data.columns:
                self.data['actual_delivery_date'] = pd.to_datetime(self.data['actual_delivery_date'])
                self.data['expected_delivery_date'] = pd.to_datetime(self.data['expected_delivery_date'])
                self.data['delay_days'] = (self.data['actual_delivery_date'] - self.data['expected_delivery_date']).dt.days
                self.data['is_delayed'] = (self.data['delay_days'] > 0).astype(int)
            else:
                logger.warning("No delay information available")
                return {
                    'total_delayed_orders': 0,
                    'delay_penalties': 0.0,
                    'customer_compensation': 0.0,
                    'redelivery_costs': 0.0,
                    'customer_churn_cost': 0.0,
                    'total_delay_cost': 0.0
                }
        
        delayed_orders = self.data[self.data['is_delayed'] == 1]
        num_delayed = len(delayed_orders)
        
        costs = {
            'total_delayed_orders': num_delayed,
            'delay_penalties': 0.0,
            'customer_compensation': 0.0,
            'redelivery_costs': 0.0,
            'customer_churn_cost': 0.0,
            'total_delay_cost': 0.0
        }
        
        # Delay penalties (based on delivery days)
        if 'actual_delivery_days' in self.data.columns and 'expected_delivery_days' in self.data.columns:
            delay_days = delayed_orders['actual_delivery_days'] - delayed_orders['expected_delivery_days']
            costs['delay_penalties'] = (delay_days * self.costs['delay_penalty_per_day']).sum()
        
        # Customer compensation
        costs['customer_compensation'] = num_delayed * self.costs['customer_compensation']
        
        # Redelivery costs (assume 20% require redelivery)
        costs['redelivery_costs'] = num_delayed * 0.2 * self.costs['redelivery_cost']
        
        # Customer churn cost
        expected_churn = num_delayed * self.costs['churn_rate_per_delay']
        costs['customer_churn_cost'] = expected_churn * self.costs['lost_customer_value']
        
        # Total
        costs['total_delay_cost'] = sum([
            costs['delay_penalties'],
            costs['customer_compensation'],
            costs['redelivery_costs'],
            costs['customer_churn_cost']
        ])
        
        logger.info(f"Total delay cost: ${costs['total_delay_cost']:,.2f}")
        
        return costs
    
    def calculate_carrier_costs(self) -> pd.DataFrame:
        """
        Calculate costs by carrier.
        
        Returns:
            DataFrame with carrier cost analysis
        """
        logger.info("Calculating carrier costs...")
        
        # Check for carrier column (could be 'carrier' or 'carrier_name')
        carrier_col = None
        if 'carrier' in self.data.columns:
            carrier_col = 'carrier'
        elif 'carrier_name' in self.data.columns:
            carrier_col = 'carrier_name'
        else:
            logger.warning("No carrier information available")
            return pd.DataFrame()
        
        carrier_stats = []
        
        for carrier in self.data[carrier_col].unique():
            carrier_data = self.data[self.data[carrier_col] == carrier]
            
            # Calculate carrier-specific costs
            carrier_calculator = CostCalculator(carrier_data, self.costs)
            carrier_costs = carrier_calculator.calculate_delay_costs()
            
            stats = {
                'carrier_name': carrier,
                'total_orders': len(carrier_data),
                'delayed_orders': carrier_costs.get('total_delayed_orders', 0),
                'delay_rate_%': (carrier_costs.get('total_delayed_orders', 0) / len(carrier_data) * 100),
                'total_cost': carrier_costs.get('total_delay_cost', 0),
                'cost_per_order': carrier_costs.get('total_delay_cost', 0) / len(carrier_data)
            }
            
            # Add base shipping costs if available
            if 'shipping_cost' in carrier_data.columns:
                stats['shipping_cost'] = carrier_data['shipping_cost'].sum()
                stats['total_cost_with_shipping'] = stats['total_cost'] + stats['shipping_cost']
            
            carrier_stats.append(stats)
        
        carrier_df = pd.DataFrame(carrier_stats).sort_values('total_cost', ascending=False)
        
        logger.info(f"Analyzed {len(carrier_df)} carriers")
        
        return carrier_df
    
    def calculate_optimization_savings(
        self,
        target_delay_reduction: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate potential savings from optimization.
        
        Args:
            target_delay_reduction: Target reduction in delays (0-1, e.g., 0.5 = 50% reduction)
            
        Returns:
            Dictionary with savings projections
        """
        logger.info(f"Calculating savings for {target_delay_reduction*100}% delay reduction...")
        
        current_costs = self.calculate_delay_costs()
        
        # Project savings
        savings = {
            'current_delay_cost': current_costs.get('total_delay_cost', 0),
            'target_delay_reduction_%': target_delay_reduction * 100,
            'projected_delay_cost': current_costs.get('total_delay_cost', 0) * (1 - target_delay_reduction),
            'annual_savings': current_costs.get('total_delay_cost', 0) * target_delay_reduction,
            'monthly_savings': (current_costs.get('total_delay_cost', 0) * target_delay_reduction) / 12,
            'roi_%': 0.0
        }
        
        # Calculate ROI (assuming implementation cost is 10% of annual savings)
        implementation_cost = savings['annual_savings'] * 0.10
        savings['implementation_cost'] = implementation_cost
        savings['roi_%'] = (savings['annual_savings'] / implementation_cost * 100) if implementation_cost > 0 else 0
        savings['payback_period_months'] = (implementation_cost / savings['monthly_savings']) if savings['monthly_savings'] > 0 else 0
        
        logger.info(f"Projected annual savings: ${savings['annual_savings']:,.2f}")
        logger.info(f"ROI: {savings['roi_%']:.1f}%")
        
        return savings
    
    def recommend_carrier_optimization(self) -> pd.DataFrame:
        """
        Recommend carrier changes for cost optimization.
        
        Returns:
            DataFrame with optimization recommendations
        """
        logger.info("Generating carrier optimization recommendations...")
        
        carrier_costs = self.calculate_carrier_costs()
        
        if carrier_costs.empty:
            return pd.DataFrame()
        
        # Find best and worst carriers
        best_carrier = carrier_costs.loc[carrier_costs['cost_per_order'].idxmin()]
        
        recommendations = []
        
        for _, carrier in carrier_costs.iterrows():
            if carrier['carrier_name'] == best_carrier['carrier_name']:
                continue
            
            # Calculate potential savings by switching
            potential_savings = (carrier['cost_per_order'] - best_carrier['cost_per_order']) * carrier['total_orders']
            switching_cost = carrier['total_orders'] * self.costs['carrier_switching_cost']
            net_savings = potential_savings - switching_cost
            
            if net_savings > 0:
                recommendations.append({
                    'from_carrier': carrier['carrier_name'],
                    'to_carrier': best_carrier['carrier_name'],
                    'orders_to_switch': carrier['total_orders'],
                    'current_cost_per_order': carrier['cost_per_order'],
                    'optimized_cost_per_order': best_carrier['cost_per_order'],
                    'potential_savings': potential_savings,
                    'switching_cost': switching_cost,
                    'net_annual_savings': net_savings,
                    'recommendation': 'SWITCH' if net_savings > switching_cost else 'MONITOR'
                })
        
        rec_df = pd.DataFrame(recommendations).sort_values('net_annual_savings', ascending=False)
        
        logger.info(f"Generated {len(rec_df)} optimization recommendations")
        
        return rec_df
    
    def calculate_route_optimization_savings(self) -> Dict[str, float]:
        """
        Calculate potential savings from route optimization.
        
        Returns:
            Dictionary with route optimization savings
        """
        logger.info("Calculating route optimization savings...")
        
        total_routes = len(self.data)
        
        savings = {
            'total_routes': total_routes,
            'optimization_rate': 0.3,  # Assume 30% of routes can be optimized
            'optimizable_routes': int(total_routes * 0.3),
            'savings_per_route': self.costs['route_optimization_savings'],
            'total_route_savings': int(total_routes * 0.3) * self.costs['route_optimization_savings'],
            'annual_route_savings': int(total_routes * 0.3) * self.costs['route_optimization_savings'] * 12
        }
        
        logger.info(f"Annual route optimization savings: ${savings['annual_route_savings']:,.2f}")
        
        return savings
    
    def generate_cost_breakdown_chart(self) -> go.Figure:
        """
        Create visualization of cost breakdown.
        
        Returns:
            Plotly figure
        """
        logger.info("Creating cost breakdown chart...")
        
        costs = self.calculate_delay_costs()
        
        # Prepare data
        cost_categories = [
            'Delay Penalties',
            'Customer Compensation',
            'Redelivery Costs',
            'Customer Churn'
        ]
        
        cost_values = [
            costs.get('delay_penalties', 0),
            costs.get('customer_compensation', 0),
            costs.get('redelivery_costs', 0),
            costs.get('customer_churn_cost', 0)
        ]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=cost_categories,
            values=cost_values,
            hole=0.4,
            marker=dict(colors=['#e74c3c', '#f39c12', '#3498db', '#9b59b6']),
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'Delay Cost Breakdown (Total: ${costs.get("total_delay_cost", 0):,.2f})',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def generate_carrier_comparison_chart(self) -> go.Figure:
        """
        Create carrier cost comparison chart.
        
        Returns:
            Plotly figure
        """
        logger.info("Creating carrier comparison chart...")
        
        carrier_costs = self.calculate_carrier_costs()
        
        if carrier_costs.empty:
            return go.Figure()
        
        # Create grouped bar chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Cost by Carrier', 'Cost per Order by Carrier'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Total cost
        fig.add_trace(
            go.Bar(
                x=carrier_costs['carrier_name'],
                y=carrier_costs['total_cost'],
                name='Total Cost',
                marker_color='#e74c3c',
                hovertemplate='<b>%{x}</b><br>Total Cost: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Cost per order
        fig.add_trace(
            go.Bar(
                x=carrier_costs['carrier_name'],
                y=carrier_costs['cost_per_order'],
                name='Cost per Order',
                marker_color='#3498db',
                hovertemplate='<b>%{x}</b><br>Cost per Order: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Carrier Cost Comparison',
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text='Carrier', row=1, col=1)
        fig.update_xaxes(title_text='Carrier', row=1, col=2)
        fig.update_yaxes(title_text='Total Cost ($)', row=1, col=1)
        fig.update_yaxes(title_text='Cost per Order ($)', row=1, col=2)
        
        return fig
    
    def generate_savings_projection_chart(
        self,
        delay_reductions: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ) -> go.Figure:
        """
        Create savings projection chart for different optimization levels.
        
        Args:
            delay_reductions: List of delay reduction percentages
            
        Returns:
            Plotly figure
        """
        logger.info("Creating savings projection chart...")
        
        projections = []
        
        for reduction in delay_reductions:
            savings = self.calculate_optimization_savings(reduction)
            projections.append({
                'delay_reduction_%': reduction * 100,
                'annual_savings': savings['annual_savings'],
                'monthly_savings': savings['monthly_savings'],
                'roi_%': savings['roi_%']
            })
        
        proj_df = pd.DataFrame(projections)
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Annual savings
        fig.add_trace(
            go.Scatter(
                x=proj_df['delay_reduction_%'],
                y=proj_df['annual_savings'],
                name='Annual Savings',
                mode='lines+markers',
                line=dict(color='#27ae60', width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}% Delay Reduction</b><br>Annual Savings: $%{y:,.2f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # ROI
        fig.add_trace(
            go.Scatter(
                x=proj_df['delay_reduction_%'],
                y=proj_df['roi_%'],
                name='ROI %',
                mode='lines+markers',
                line=dict(color='#f39c12', width=3, dash='dash'),
                marker=dict(size=8),
                hovertemplate='<b>%{x}% Delay Reduction</b><br>ROI: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Savings Projection by Optimization Level',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Delay Reduction Target (%)')
        fig.update_yaxes(title_text='Annual Savings ($)', secondary_y=False)
        fig.update_yaxes(title_text='ROI (%)', secondary_y=True)
        
        return fig


# ==================== Convenience Functions ====================
def calculate_total_costs(
    data: pd.DataFrame,
    cost_params: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Quick function to calculate total delay costs.
    
    Args:
        data: Delivery data
        cost_params: Optional cost parameters
        
    Returns:
        Cost breakdown dictionary
    """
    calculator = CostCalculator(data, cost_params)
    return calculator.calculate_delay_costs()


def get_optimization_roi(
    data: pd.DataFrame,
    target_reduction: float = 0.5
) -> Dict[str, float]:
    """
    Quick function to get ROI projection.
    
    Args:
        data: Delivery data
        target_reduction: Target delay reduction (0-1)
        
    Returns:
        ROI and savings dictionary
    """
    calculator = CostCalculator(data)
    return calculator.calculate_optimization_savings(target_reduction)
