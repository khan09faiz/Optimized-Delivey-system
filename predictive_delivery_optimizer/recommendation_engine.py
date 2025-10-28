"""
Recommendation engine for corrective actions.

This module generates actionable recommendations based on:
- Delay risk probability
- Order characteristics
- Business rules
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from utils import Config, setup_logging, get_risk_category

logger = setup_logging("recommendation_engine")


class RecommendationEngine:
    """Generates corrective action recommendations."""
    
    def __init__(self):
        """Initialize RecommendationEngine."""
        self.rules: List[Dict[str, Any]] = []
        self._setup_rules()
        logger.info("RecommendationEngine initialized")
    
    def _setup_rules(self) -> None:
        """Define recommendation rules."""
        self.rules = [
            {
                'name': 'high_priority_high_risk',
                'condition': lambda row: (
                    row.get('priority', '').lower() == 'high' and
                    row.get('predicted_prob', 0) >= Config.HIGH_RISK_THRESHOLD
                ),
                'action': 'Expedite via premium carrier',
                'priority': 'Critical',
                'estimated_cost_increase': 0.25
            },
            {
                'name': 'severe_traffic_delay',
                'condition': lambda row: (
                    row.get('traffic_delay_mins', 0) > Config.TRAFFIC_DELAY_HIGH
                ),
                'action': 'Reroute to alternative path',
                'priority': 'High',
                'estimated_cost_increase': 0.15
            },
            {
                'name': 'perishable_goods_risk',
                'condition': lambda row: (
                    row.get('product_category', '').lower() in ['perishable', 'food', 'pharmaceuticals'] and
                    row.get('predicted_prob', 0) >= Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Assign refrigerated vehicle with priority routing',
                'priority': 'Critical',
                'estimated_cost_increase': 0.30
            },
            {
                'name': 'short_distance_express',
                'condition': lambda row: (
                    row.get('distance_km', 999) < Config.SHORT_DISTANCE and
                    row.get('predicted_prob', 0) >= Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Use express bike delivery',
                'priority': 'Medium',
                'estimated_cost_increase': -0.10
            },
            {
                'name': 'high_value_order_risk',
                'condition': lambda row: (
                    row.get('order_value', 0) > 20000 and
                    row.get('predicted_prob', 0) >= Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Assign dedicated vehicle with real-time tracking',
                'priority': 'High',
                'estimated_cost_increase': 0.20
            },
            {
                'name': 'warehouse_stock_issue',
                'condition': lambda row: (
                    row.get('avg_warehouse_stock', 100) < 20 and
                    row.get('predicted_prob', 0) >= Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Coordinate with warehouse for priority dispatch',
                'priority': 'High',
                'estimated_cost_increase': 0.05
            },
            {
                'name': 'poor_carrier_performance',
                'condition': lambda row: (
                    row.get('carrier_delay_rate', 0) > 0.3 and
                    row.get('predicted_prob', 0) >= Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Switch to more reliable carrier',
                'priority': 'Medium',
                'estimated_cost_increase': 0.12
            },
            {
                'name': 'weekend_delivery_risk',
                'condition': lambda row: (
                    row.get('is_weekend', 0) == 1 and
                    row.get('predicted_prob', 0) >= Config.HIGH_RISK_THRESHOLD
                ),
                'action': 'Schedule for early week delivery or add weekend premium service',
                'priority': 'Medium',
                'estimated_cost_increase': 0.18
            },
            {
                'name': 'long_distance_high_risk',
                'condition': lambda row: (
                    row.get('distance_km', 0) > 200 and
                    row.get('predicted_prob', 0) >= Config.HIGH_RISK_THRESHOLD
                ),
                'action': 'Use hub-and-spoke model with intermediate waypoints',
                'priority': 'High',
                'estimated_cost_increase': 0.10
            },
            {
                'name': 'moderate_risk_monitoring',
                'condition': lambda row: (
                    Config.MODERATE_RISK_THRESHOLD <= row.get('predicted_prob', 0) < Config.HIGH_RISK_THRESHOLD
                ),
                'action': 'Enable real-time monitoring and proactive customer notification',
                'priority': 'Medium',
                'estimated_cost_increase': 0.03
            },
            {
                'name': 'customer_feedback_poor',
                'condition': lambda row: (
                    row.get('rating', 5) < 3 and
                    row.get('predicted_prob', 0) >= Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Assign to top-rated driver with customer service follow-up',
                'priority': 'High',
                'estimated_cost_increase': 0.08
            },
            {
                'name': 'low_risk_standard',
                'condition': lambda row: (
                    row.get('predicted_prob', 0) < Config.MODERATE_RISK_THRESHOLD
                ),
                'action': 'Proceed with standard delivery protocol',
                'priority': 'Low',
                'estimated_cost_increase': 0.00
            }
        ]
        
        logger.info(f"Configured {len(self.rules)} recommendation rules")
    
    def get_recommendation_for_order(self, order: pd.Series) -> Dict[str, Any]:
        """
        Get recommendation for a single order.
        
        Args:
            order: Order data as pandas Series
            
        Returns:
            Dictionary containing recommendation details
        """
        # Evaluate rules in priority order
        for rule in self.rules:
            try:
                if rule['condition'](order):
                    risk_category = get_risk_category(order.get('predicted_prob', 0))
                    
                    return {
                        'rule_name': rule['name'],
                        'action': rule['action'],
                        'priority': rule['priority'],
                        'risk_category': risk_category,
                        'predicted_probability': order.get('predicted_prob', 0),
                        'estimated_cost_increase': rule['estimated_cost_increase'],
                        'order_id': order.get('order_id', 'N/A')
                    }
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule['name']}: {e}")
                continue
        
        # Default recommendation if no rule matches
        return {
            'rule_name': 'default',
            'action': 'Review manually',
            'priority': 'Low',
            'risk_category': 'Unknown',
            'predicted_probability': order.get('predicted_prob', 0),
            'estimated_cost_increase': 0.0,
            'order_id': order.get('order_id', 'N/A')
        }
    
    def get_recommendations_for_dataset(
        self,
        df: pd.DataFrame,
        prob_column: str = 'predicted_prob'
    ) -> pd.DataFrame:
        """
        Generate recommendations for all orders in dataset.
        
        Args:
            df: DataFrame with predictions
            prob_column: Column name containing prediction probabilities
            
        Returns:
            DataFrame with recommendations added
        """
        logger.info(f"Generating recommendations for {len(df)} orders...")
        
        recommendations = []
        
        for idx, row in df.iterrows():
            rec = self.get_recommendation_for_order(row)
            recommendations.append(rec)
        
        # Convert to DataFrame
        rec_df = pd.DataFrame(recommendations)
        
        # Merge with original data
        result = pd.concat([df.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)
        
        # Summary statistics
        action_counts = rec_df['action'].value_counts()
        priority_counts = rec_df['priority'].value_counts()
        
        logger.info(f"Recommendations generated: {len(rec_df)}")
        logger.info(f"Priority breakdown: {priority_counts.to_dict()}")
        logger.info(f"Top action: {action_counts.index[0]} ({action_counts.iloc[0]} orders)")
        
        return result
    
    def get_high_priority_actions(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get high-priority recommendations requiring immediate action.
        
        Args:
            df: DataFrame with recommendations
            top_n: Number of top priority items to return
            
        Returns:
            DataFrame of high-priority actions
        """
        logger.info(f"Extracting top {top_n} high-priority actions...")
        
        # Filter for Critical and High priority
        high_priority = df[df['priority'].isin(['Critical', 'High'])].copy()
        
        # Sort by probability (descending) and priority
        priority_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
        high_priority['priority_score'] = high_priority['priority'].map(priority_map)
        
        high_priority_sorted = high_priority.sort_values(
            ['priority_score', 'predicted_probability'],
            ascending=[False, False]
        ).head(top_n)
        
        high_priority_sorted.drop(columns=['priority_score'], inplace=True)
        
        logger.info(f"Found {len(high_priority_sorted)} high-priority actions")
        
        return high_priority_sorted
    
    def estimate_cost_impact(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate total cost impact of recommendations.
        
        Args:
            df: DataFrame with recommendations and order values
            
        Returns:
            Dictionary with cost impact metrics
        """
        logger.info("Calculating cost impact of recommendations...")
        
        # Calculate additional costs
        if 'order_value' in df.columns and 'estimated_cost_increase' in df.columns:
            df['additional_cost'] = df['order_value'] * df['estimated_cost_increase']
            
            total_current_cost = df['order_value'].sum()
            total_additional_cost = df['additional_cost'].sum()
            total_new_cost = total_current_cost + total_additional_cost
            
            cost_impact = {
                'current_total_cost': total_current_cost,
                'additional_cost': total_additional_cost,
                'new_total_cost': total_new_cost,
                'percentage_increase': (total_additional_cost / total_current_cost * 100) if total_current_cost > 0 else 0,
                'avg_cost_per_order': total_new_cost / len(df) if len(df) > 0 else 0
            }
            
            logger.info(f"Cost impact: +{cost_impact['percentage_increase']:.2f}% (₹{cost_impact['additional_cost']:,.2f})")
            
            return cost_impact
        else:
            logger.warning("Missing columns for cost calculation")
            return {}
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary report of recommendations.
        
        Args:
            df: DataFrame with recommendations
            
        Returns:
            Dictionary containing summary statistics
        """
        logger.info("Generating recommendation summary report...")
        
        report = {
            'total_orders': len(df),
            'high_risk_orders': len(df[df['risk_category'] == 'High']),
            'moderate_risk_orders': len(df[df['risk_category'] == 'Moderate']),
            'low_risk_orders': len(df[df['risk_category'] == 'Low']),
            'critical_actions': len(df[df['priority'] == 'Critical']),
            'high_priority_actions': len(df[df['priority'] == 'High']),
            'medium_priority_actions': len(df[df['priority'] == 'Medium']),
            'action_distribution': df['action'].value_counts().to_dict(),
            'avg_delay_probability': df['predicted_probability'].mean(),
            'max_delay_probability': df['predicted_probability'].max()
        }
        
        # Add cost impact if available
        cost_impact = self.estimate_cost_impact(df)
        if cost_impact:
            report['cost_impact'] = cost_impact
        
        logger.info(f"Report generated: {report['high_risk_orders']} high-risk orders")
        
        return report


# ==================== Convenience Functions ====================
def get_recommendation_engine() -> RecommendationEngine:
    """
    Create and return a RecommendationEngine instance.
    
    Returns:
        RecommendationEngine instance
    """
    return RecommendationEngine()


def generate_recommendations(
    df: pd.DataFrame,
    prob_column: str = 'predicted_prob'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to generate recommendations and summary.
    
    Args:
        df: DataFrame with predictions
        prob_column: Probability column name
        
    Returns:
        Tuple of (recommendations_df, summary_report)
    """
    engine = get_recommendation_engine()
    recommendations = engine.get_recommendations_for_dataset(df, prob_column)
    summary = engine.generate_summary_report(recommendations)
    
    return recommendations, summary
