"""
What-If Scenario Analysis Module
================================
Enables strategic planning through scenario simulation.
Allows testing different operational changes and predicting outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

from utils import setup_logging

logger = setup_logging("scenario_analysis")


class ScenarioAnalyzer:
    """Performs what-if analysis for delivery optimization."""
    
    def __init__(self, data: pd.DataFrame, model: Optional[Any] = None):
        """
        Initialize ScenarioAnalyzer.
        
        Args:
            data: Base delivery data
            model: Optional trained prediction model
        """
        self.base_data = data.copy()
        self.model = model
        self.scenarios = {}
        
        logger.info(f"ScenarioAnalyzer initialized with {len(data)} records")
    
    def create_carrier_switch_scenario(
        self,
        from_carrier: str,
        to_carrier: str,
        percentage: float = 1.0
    ) -> pd.DataFrame:
        """
        Simulate switching orders from one carrier to another.
        
        Args:
            from_carrier: Current carrier
            to_carrier: Target carrier
            percentage: Percentage of orders to switch (0-1)
            
        Returns:
            Modified data with carrier changes
        """
        logger.info(f"Creating scenario: Switch {percentage*100}% from {from_carrier} to {to_carrier}")
        
        scenario_data = self.base_data.copy()
        
        # Find orders to switch
        from_mask = scenario_data['carrier_name'] == from_carrier
        num_to_switch = int(from_mask.sum() * percentage)
        
        # Randomly select orders to switch
        switch_indices = scenario_data[from_mask].sample(n=num_to_switch, random_state=42).index
        
        # Get target carrier's average performance
        to_carrier_data = self.base_data[self.base_data['carrier_name'] == to_carrier]
        
        if len(to_carrier_data) > 0:
            avg_delay_rate = to_carrier_data['is_delayed'].mean() if 'is_delayed' in to_carrier_data.columns else 0
            avg_delivery_days = to_carrier_data['actual_delivery_days'].mean() if 'actual_delivery_days' in to_carrier_data.columns else 0
            
            # Update switched orders
            scenario_data.loc[switch_indices, 'carrier_name'] = to_carrier
            
            # Adjust expected performance based on new carrier
            if 'is_delayed' in scenario_data.columns:
                # Probabilistic delay based on carrier's history
                scenario_data.loc[switch_indices, 'is_delayed'] = np.random.binomial(1, avg_delay_rate, num_to_switch)
            
            if 'actual_delivery_days' in scenario_data.columns:
                # Adjust delivery days with some randomness
                scenario_data.loc[switch_indices, 'actual_delivery_days'] = (
                    avg_delivery_days + np.random.normal(0, avg_delivery_days * 0.1, num_to_switch)
                ).clip(1, None)
        
        scenario_name = f"carrier_switch_{from_carrier}_to_{to_carrier}_{int(percentage*100)}"
        self.scenarios[scenario_name] = scenario_data
        
        logger.info(f"Switched {num_to_switch} orders")
        
        return scenario_data
    
    def create_priority_rebalance_scenario(
        self,
        new_priority_distribution: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Simulate rebalancing priority distribution.
        
        Args:
            new_priority_distribution: Target distribution {priority: percentage}
            
        Returns:
            Modified data with new priorities
        """
        logger.info("Creating priority rebalance scenario...")
        
        scenario_data = self.base_data.copy()
        
        if 'priority' not in scenario_data.columns:
            logger.warning("No priority column found")
            return scenario_data
        
        total_orders = len(scenario_data)
        
        # Calculate how many orders per priority
        target_counts = {
            priority: int(total_orders * percentage)
            for priority, percentage in new_priority_distribution.items()
        }
        
        # Rebalance
        new_priorities = []
        for priority, count in target_counts.items():
            new_priorities.extend([priority] * count)
        
        # Handle rounding differences
        while len(new_priorities) < total_orders:
            new_priorities.append(list(target_counts.keys())[0])
        
        # Randomly assign new priorities
        np.random.shuffle(new_priorities)
        scenario_data['priority'] = new_priorities[:total_orders]
        
        scenario_name = "priority_rebalance"
        self.scenarios[scenario_name] = scenario_data
        
        logger.info(f"Rebalanced priorities: {target_counts}")
        
        return scenario_data
    
    def create_route_optimization_scenario(
        self,
        distance_reduction: float = 0.15,
        affected_percentage: float = 0.3
    ) -> pd.DataFrame:
        """
        Simulate route optimization impact.
        
        Args:
            distance_reduction: Average distance reduction (0-1)
            affected_percentage: Percentage of routes optimized
            
        Returns:
            Modified data with optimized routes
        """
        logger.info(f"Creating route optimization scenario: {distance_reduction*100}% reduction on {affected_percentage*100}% of routes")
        
        scenario_data = self.base_data.copy()
        
        if 'distance_km' not in scenario_data.columns:
            logger.warning("No distance column found")
            return scenario_data
        
        # Select routes to optimize
        num_to_optimize = int(len(scenario_data) * affected_percentage)
        optimize_indices = scenario_data.sample(n=num_to_optimize, random_state=42).index
        
        # Reduce distance
        scenario_data.loc[optimize_indices, 'distance_km'] *= (1 - distance_reduction)
        
        # Adjust delivery time (assume 10% improvement in time for 15% distance reduction)
        if 'actual_delivery_days' in scenario_data.columns:
            time_improvement = distance_reduction * 0.67  # 2/3 of distance improvement
            scenario_data.loc[optimize_indices, 'actual_delivery_days'] *= (1 - time_improvement)
            scenario_data.loc[optimize_indices, 'actual_delivery_days'] = scenario_data.loc[optimize_indices, 'actual_delivery_days'].clip(1, None)
            
            # Update delay status based on new time
            if 'expected_delivery_days' in scenario_data.columns and 'is_delayed' in scenario_data.columns:
                scenario_data.loc[optimize_indices, 'is_delayed'] = (
                    scenario_data.loc[optimize_indices, 'actual_delivery_days'] > 
                    scenario_data.loc[optimize_indices, 'expected_delivery_days']
                ).astype(int)
        
        scenario_name = f"route_optimization_{int(distance_reduction*100)}pct"
        self.scenarios[scenario_name] = scenario_data
        
        logger.info(f"Optimized {num_to_optimize} routes")
        
        return scenario_data
    
    def create_capacity_increase_scenario(
        self,
        carrier: str,
        capacity_increase: float = 0.25
    ) -> pd.DataFrame:
        """
        Simulate impact of increasing carrier capacity.
        
        Args:
            carrier: Carrier to increase capacity
            capacity_increase: Percentage increase in capacity
            
        Returns:
            Modified data
        """
        logger.info(f"Creating capacity increase scenario for {carrier}: +{capacity_increase*100}%")
        
        scenario_data = self.base_data.copy()
        
        if 'carrier_name' not in scenario_data.columns:
            return scenario_data
        
        # Find carrier's orders
        carrier_mask = scenario_data['carrier_name'] == carrier
        carrier_orders = scenario_data[carrier_mask]
        
        # With increased capacity, assume delay rate decreases
        if 'is_delayed' in scenario_data.columns:
            current_delay_rate = carrier_orders['is_delayed'].mean()
            # Assume 50% of capacity increase translates to delay reduction
            new_delay_rate = current_delay_rate * (1 - capacity_increase * 0.5)
            
            # Probabilistically update delays
            num_carrier_orders = carrier_mask.sum()
            scenario_data.loc[carrier_mask, 'is_delayed'] = np.random.binomial(1, new_delay_rate, num_carrier_orders)
        
        # Improve delivery times
        if 'actual_delivery_days' in scenario_data.columns:
            time_improvement = capacity_increase * 0.3  # 30% of capacity increase
            scenario_data.loc[carrier_mask, 'actual_delivery_days'] *= (1 - time_improvement)
            scenario_data.loc[carrier_mask, 'actual_delivery_days'] = scenario_data.loc[carrier_mask, 'actual_delivery_days'].clip(1, None)
        
        scenario_name = f"capacity_increase_{carrier}_{int(capacity_increase*100)}pct"
        self.scenarios[scenario_name] = scenario_data
        
        logger.info(f"Increased capacity for {carrier}")
        
        return scenario_data
    
    def create_custom_scenario(
        self,
        modifications: Dict[str, Any],
        scenario_name: str
    ) -> pd.DataFrame:
        """
        Create custom scenario with specified modifications.
        
        Args:
            modifications: Dictionary of column modifications
            scenario_name: Name for the scenario
            
        Returns:
            Modified data
        """
        logger.info(f"Creating custom scenario: {scenario_name}")
        
        scenario_data = self.base_data.copy()
        
        for column, change_spec in modifications.items():
            if column not in scenario_data.columns:
                logger.warning(f"Column {column} not found")
                continue
            
            if isinstance(change_spec, dict):
                # Conditional modification
                if 'condition' in change_spec and 'value' in change_spec:
                    mask = eval(change_spec['condition'])
                    scenario_data.loc[mask, column] = change_spec['value']
            else:
                # Direct value assignment
                scenario_data[column] = change_spec
        
        self.scenarios[scenario_name] = scenario_data
        
        return scenario_data
    
    def compare_scenarios(
        self,
        scenario_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across different scenarios.
        
        Args:
            scenario_names: List of scenarios to compare (None = all)
            
        Returns:
            Comparison DataFrame
        """
        logger.info("Comparing scenarios...")
        
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        comparisons = []
        
        # Base scenario
        comparisons.append(self._calculate_scenario_metrics(self.base_data, "baseline"))
        
        # All scenarios
        for name in scenario_names:
            if name in self.scenarios:
                comparisons.append(self._calculate_scenario_metrics(self.scenarios[name], name))
        
        comparison_df = pd.DataFrame(comparisons).set_index('scenario')
        
        logger.info(f"Compared {len(comparison_df)} scenarios")
        
        return comparison_df
    
    def _calculate_scenario_metrics(
        self,
        data: pd.DataFrame,
        scenario_name: str
    ) -> Dict[str, Any]:
        """Calculate key metrics for a scenario."""
        metrics = {'scenario': scenario_name}
        
        if 'is_delayed' in data.columns:
            metrics['total_orders'] = len(data)
            metrics['delayed_orders'] = data['is_delayed'].sum()
            metrics['delay_rate_%'] = data['is_delayed'].mean() * 100
            metrics['on_time_rate_%'] = (1 - data['is_delayed'].mean()) * 100
        
        if 'actual_delivery_days' in data.columns:
            metrics['avg_delivery_days'] = data['actual_delivery_days'].mean()
            metrics['median_delivery_days'] = data['actual_delivery_days'].median()
        
        if 'distance_km' in data.columns:
            metrics['avg_distance_km'] = data['distance_km'].mean()
            metrics['total_distance_km'] = data['distance_km'].sum()
        
        # Carrier distribution
        if 'carrier_name' in data.columns:
            top_carrier = data['carrier_name'].value_counts().index[0]
            metrics['primary_carrier'] = top_carrier
            metrics['primary_carrier_share_%'] = (data['carrier_name'] == top_carrier).mean() * 100
        
        return metrics
    
    def predict_scenario_outcomes(
        self,
        scenario_data: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use ML model to predict outcomes for a scenario.
        
        Args:
            scenario_data: Scenario data
            feature_columns: Features for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            logger.warning("No model provided for predictions")
            return np.array([]), np.array([])
        
        logger.info("Predicting scenario outcomes...")
        
        X = scenario_data[feature_columns].fillna(0)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        logger.info(f"Predicted delay rate: {predictions.mean()*100:.1f}%")
        
        return predictions, probabilities
    
    def plot_scenario_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['delay_rate_%', 'avg_delivery_days']
    ) -> go.Figure:
        """
        Visualize scenario comparison.
        
        Args:
            comparison_df: Comparison results
            metrics: Metrics to visualize
            
        Returns:
            Plotly figure
        """
        logger.info("Creating scenario comparison chart...")
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[m.replace('_', ' ').title() for m in metrics]
        )
        
        for i, metric in enumerate(metrics, 1):
            if metric not in comparison_df.columns:
                continue
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df.index,
                    y=comparison_df[metric],
                    name=metric,
                    marker=dict(
                        color=comparison_df[metric],
                        colorscale='RdYlGn_r' if 'delay' in metric else 'RdYlGn',
                        showscale=False
                    ),
                    text=comparison_df[metric].round(2),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title='Scenario Comparison',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_impact_analysis(
        self,
        comparison_df: pd.DataFrame,
        baseline_name: str = 'baseline'
    ) -> go.Figure:
        """
        Plot impact of scenarios vs baseline.
        
        Args:
            comparison_df: Comparison results
            baseline_name: Name of baseline scenario
            
        Returns:
            Plotly figure
        """
        logger.info("Creating impact analysis chart...")
        
        if baseline_name not in comparison_df.index:
            logger.warning(f"Baseline '{baseline_name}' not found")
            return go.Figure()
        
        baseline = comparison_df.loc[baseline_name]
        
        # Calculate percentage changes
        impact_data = []
        
        for scenario in comparison_df.index:
            if scenario == baseline_name:
                continue
            
            row = comparison_df.loc[scenario]
            
            # Delay rate change
            delay_change = row.get('delay_rate_%', 0) - baseline.get('delay_rate_%', 0)
            
            # Delivery time change
            time_change = row.get('avg_delivery_days', 0) - baseline.get('avg_delivery_days', 0)
            
            # Distance change
            distance_change_pct = 0
            if 'total_distance_km' in row and 'total_distance_km' in baseline:
                distance_change_pct = ((row['total_distance_km'] - baseline['total_distance_km']) / 
                                       baseline['total_distance_km'] * 100)
            
            impact_data.append({
                'scenario': scenario,
                'delay_rate_change_%': delay_change,
                'delivery_time_change_days': time_change,
                'distance_change_%': distance_change_pct
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Create waterfall-style chart
        fig = go.Figure()
        
        # Delay rate impact
        fig.add_trace(go.Bar(
            name='Delay Rate Change',
            x=impact_df['scenario'],
            y=impact_df['delay_rate_change_%'],
            marker=dict(
                color=impact_df['delay_rate_change_%'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title='Change (%)', x=1.15)
            ),
            text=impact_df['delay_rate_change_%'].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Delay Rate Change: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Scenario Impact Analysis (vs {baseline_name})',
            xaxis_title='Scenario',
            yaxis_title='Delay Rate Change (%)',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def generate_scenario_report(
        self,
        scenario_name: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report for a scenario.
        
        Args:
            scenario_name: Name of scenario
            
        Returns:
            Dictionary with report data
        """
        logger.info(f"Generating report for scenario: {scenario_name}")
        
        if scenario_name not in self.scenarios:
            logger.error(f"Scenario '{scenario_name}' not found")
            return {}
        
        scenario_data = self.scenarios[scenario_name]
        baseline_metrics = self._calculate_scenario_metrics(self.base_data, 'baseline')
        scenario_metrics = self._calculate_scenario_metrics(scenario_data, scenario_name)
        
        report = {
            'scenario_name': scenario_name,
            'baseline_metrics': baseline_metrics,
            'scenario_metrics': scenario_metrics,
            'improvements': {},
            'summary': ""
        }
        
        # Calculate improvements
        if 'delay_rate_%' in baseline_metrics and 'delay_rate_%' in scenario_metrics:
            delay_improvement = baseline_metrics['delay_rate_%'] - scenario_metrics['delay_rate_%']
            report['improvements']['delay_rate_reduction_%'] = delay_improvement
        
        if 'avg_delivery_days' in baseline_metrics and 'avg_delivery_days' in scenario_metrics:
            time_improvement = baseline_metrics['avg_delivery_days'] - scenario_metrics['avg_delivery_days']
            report['improvements']['delivery_time_reduction_days'] = time_improvement
        
        # Generate summary
        summary_parts = []
        if delay_improvement > 0:
            summary_parts.append(f"Reduces delay rate by {delay_improvement:.1f}%")
        if time_improvement > 0:
            summary_parts.append(f"Improves delivery time by {time_improvement:.2f} days")
        
        report['summary'] = "; ".join(summary_parts) if summary_parts else "No significant improvement"
        
        return report


# ==================== Convenience Functions ====================
def quick_carrier_analysis(
    data: pd.DataFrame,
    from_carrier: str,
    to_carrier: str
) -> Dict[str, Any]:
    """
    Quick analysis of carrier switch impact.
    
    Args:
        data: Base data
        from_carrier: Current carrier
        to_carrier: Target carrier
        
    Returns:
        Impact analysis
    """
    analyzer = ScenarioAnalyzer(data)
    scenario_data = analyzer.create_carrier_switch_scenario(from_carrier, to_carrier)
    comparison = analyzer.compare_scenarios()
    
    return {
        'comparison': comparison,
        'scenario_data': scenario_data
    }


def simulate_optimization(
    data: pd.DataFrame,
    distance_reduction: float = 0.15
) -> pd.DataFrame:
    """
    Quick simulation of route optimization.
    
    Args:
        data: Base data
        distance_reduction: Target distance reduction
        
    Returns:
        Comparison results
    """
    analyzer = ScenarioAnalyzer(data)
    analyzer.create_route_optimization_scenario(distance_reduction)
    return analyzer.compare_scenarios()
