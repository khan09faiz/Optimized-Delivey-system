"""
Recommendation engine module for the Predictive Delivery Optimizer.
Provides optimization recommendations based on model predictions and data analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .utils import logger


class RecommendationEngine:
    """Generates optimization recommendations for delivery operations."""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize RecommendationEngine.
        
        Args:
            datasets: Dictionary of DataFrames
        """
        self.datasets = datasets
        self.recommendations = []
    
    def analyze_delivery_performance(self) -> Dict[str, Any]:
        """
        Analyze overall delivery performance and identify issues.
        
        Returns:
            Dictionary with performance analysis
        """
        if 'delivery_performance' not in self.datasets:
            return {}
        
        df = self.datasets['delivery_performance']
        
        analysis = {
            'total_deliveries': len(df),
            'on_time_rate': (df['delay_minutes'] <= 0).mean() * 100 if 'delay_minutes' in df.columns else 0,
            'average_delay': df['delay_minutes'].mean() if 'delay_minutes' in df.columns else 0,
            'delayed_deliveries': (df['delay_minutes'] > 0).sum() if 'delay_minutes' in df.columns else 0,
            'delivery_status_distribution': df['status'].value_counts().to_dict() if 'status' in df.columns else {}
        }
        
        logger.info("Delivery performance analyzed")
        return analysis
    
    def recommend_route_optimizations(self) -> List[Dict[str, Any]]:
        """
        Generate route optimization recommendations.
        
        Returns:
            List of route recommendations
        """
        recommendations = []
        
        if 'routes_distance' not in self.datasets:
            return recommendations
        
        routes = self.datasets['routes_distance']
        
        # Identify high-traffic routes
        if 'traffic_level' in routes.columns:
            high_traffic = routes[routes['traffic_level'] == 'High']
            if len(high_traffic) > 0:
                recommendations.append({
                    'type': 'route_optimization',
                    'priority': 'High',
                    'issue': f'{len(high_traffic)} routes with high traffic',
                    'recommendation': 'Consider alternative routes or adjust delivery schedules to avoid peak traffic',
                    'affected_routes': high_traffic['route_id'].tolist()[:5] if 'route_id' in high_traffic.columns else []
                })
        
        # Identify long-distance routes
        if 'distance_km' in routes.columns:
            long_routes = routes[routes['distance_km'] > routes['distance_km'].quantile(0.9)]
            if len(long_routes) > 0:
                recommendations.append({
                    'type': 'route_optimization',
                    'priority': 'Medium',
                    'issue': f'{len(long_routes)} routes exceed 90th percentile distance ({routes["distance_km"].quantile(0.9):.1f} km)',
                    'recommendation': 'Consider establishing intermediate distribution centers or consolidating shipments',
                    'avg_distance': long_routes['distance_km'].mean()
                })
        
        logger.info(f"Generated {len(recommendations)} route recommendations")
        return recommendations
    
    def recommend_fleet_optimizations(self) -> List[Dict[str, Any]]:
        """
        Generate fleet optimization recommendations.
        
        Returns:
            List of fleet recommendations
        """
        recommendations = []
        
        if 'vehicle_fleet' not in self.datasets:
            return recommendations
        
        fleet = self.datasets['vehicle_fleet']
        
        # Analyze vehicle availability
        if 'availability' in fleet.columns:
            unavailable = fleet[fleet['availability'] != 'Available']
            availability_rate = (fleet['availability'] == 'Available').mean() * 100
            
            if availability_rate < 70:
                recommendations.append({
                    'type': 'fleet_management',
                    'priority': 'High',
                    'issue': f'Only {availability_rate:.1f}% of fleet is available',
                    'recommendation': 'Increase fleet size or improve maintenance scheduling to ensure higher availability',
                    'current_available': (fleet['availability'] == 'Available').sum(),
                    'total_fleet': len(fleet)
                })
        
        # Analyze maintenance status
        if 'maintenance_status' in fleet.columns:
            needs_service = fleet[fleet['maintenance_status'] == 'Needs Service']
            if len(needs_service) > 0:
                recommendations.append({
                    'type': 'fleet_maintenance',
                    'priority': 'High',
                    'issue': f'{len(needs_service)} vehicles need service',
                    'recommendation': 'Schedule immediate maintenance for vehicles to prevent breakdowns and delays',
                    'vehicles': needs_service['vehicle_id'].tolist() if 'vehicle_id' in needs_service.columns else []
                })
        
        # Analyze fuel efficiency
        if 'fuel_efficiency' in fleet.columns:
            low_efficiency = fleet[fleet['fuel_efficiency'] < fleet['fuel_efficiency'].quantile(0.25)]
            if len(low_efficiency) > 0:
                recommendations.append({
                    'type': 'fleet_optimization',
                    'priority': 'Medium',
                    'issue': f'{len(low_efficiency)} vehicles have below-average fuel efficiency',
                    'recommendation': 'Consider replacing or servicing low-efficiency vehicles to reduce operating costs',
                    'potential_savings': 'Up to 15-25% reduction in fuel costs'
                })
        
        logger.info(f"Generated {len(recommendations)} fleet recommendations")
        return recommendations
    
    def recommend_cost_optimizations(self) -> List[Dict[str, Any]]:
        """
        Generate cost optimization recommendations.
        
        Returns:
            List of cost recommendations
        """
        recommendations = []
        
        if 'cost_breakdown' not in self.datasets:
            return recommendations
        
        costs = self.datasets['cost_breakdown']
        
        # Analyze cost components
        if all(col in costs.columns for col in ['fuel_cost', 'labor_cost', 'maintenance_cost', 'total_cost']):
            avg_fuel_pct = (costs['fuel_cost'] / costs['total_cost']).mean() * 100
            avg_labor_pct = (costs['labor_cost'] / costs['total_cost']).mean() * 100
            avg_maintenance_pct = (costs['maintenance_cost'] / costs['total_cost']).mean() * 100
            
            if avg_fuel_pct > 40:
                recommendations.append({
                    'type': 'cost_optimization',
                    'priority': 'High',
                    'issue': f'Fuel costs represent {avg_fuel_pct:.1f}% of total costs',
                    'recommendation': 'Implement route optimization and consider fuel-efficient vehicles to reduce fuel costs',
                    'potential_impact': f'Reduce fuel costs by 10-15%'
                })
            
            if avg_maintenance_pct > 25:
                recommendations.append({
                    'type': 'cost_optimization',
                    'priority': 'Medium',
                    'issue': f'Maintenance costs are {avg_maintenance_pct:.1f}% of total costs',
                    'recommendation': 'Implement preventive maintenance program to reduce unexpected repairs',
                    'potential_impact': 'Reduce maintenance costs by 15-20%'
                })
        
        # Identify high-cost orders
        if 'total_cost' in costs.columns:
            high_cost_threshold = costs['total_cost'].quantile(0.9)
            high_cost_orders = costs[costs['total_cost'] > high_cost_threshold]
            
            if len(high_cost_orders) > 0:
                recommendations.append({
                    'type': 'cost_optimization',
                    'priority': 'Medium',
                    'issue': f'{len(high_cost_orders)} orders have unusually high costs (>${high_cost_threshold:.2f})',
                    'recommendation': 'Investigate high-cost orders for inefficiencies in routing or resource allocation',
                    'avg_high_cost': high_cost_orders['total_cost'].mean()
                })
        
        logger.info(f"Generated {len(recommendations)} cost recommendations")
        return recommendations
    
    def recommend_customer_satisfaction_improvements(self) -> List[Dict[str, Any]]:
        """
        Generate customer satisfaction improvement recommendations.
        
        Returns:
            List of customer satisfaction recommendations
        """
        recommendations = []
        
        if 'customer_feedback' not in self.datasets:
            return recommendations
        
        feedback = self.datasets['customer_feedback']
        
        # Analyze ratings
        if 'rating' in feedback.columns:
            avg_rating = feedback['rating'].mean()
            low_ratings = feedback[feedback['rating'] < 3]
            
            if avg_rating < 4.0:
                recommendations.append({
                    'type': 'customer_satisfaction',
                    'priority': 'High',
                    'issue': f'Average customer rating is {avg_rating:.2f} (below target of 4.0)',
                    'recommendation': 'Focus on improving delivery reliability and customer communication',
                    'low_rating_count': len(low_ratings)
                })
        
        if 'delivery_rating' in feedback.columns:
            avg_delivery_rating = feedback['delivery_rating'].mean()
            
            if avg_delivery_rating < 4.0:
                recommendations.append({
                    'type': 'customer_satisfaction',
                    'priority': 'High',
                    'issue': f'Delivery rating is {avg_delivery_rating:.2f} (below target)',
                    'recommendation': 'Improve delivery time accuracy and driver professionalism training',
                    'target_improvement': '15-20% increase in delivery ratings'
                })
        
        logger.info(f"Generated {len(recommendations)} customer satisfaction recommendations")
        return recommendations
    
    def generate_all_recommendations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate all types of recommendations.
        
        Returns:
            Dictionary with all recommendations by category
        """
        all_recommendations = {
            'route_optimizations': self.recommend_route_optimizations(),
            'fleet_optimizations': self.recommend_fleet_optimizations(),
            'cost_optimizations': self.recommend_cost_optimizations(),
            'customer_satisfaction': self.recommend_customer_satisfaction_improvements()
        }
        
        # Count total recommendations
        total = sum(len(recs) for recs in all_recommendations.values())
        logger.info(f"Generated {total} total recommendations across all categories")
        
        return all_recommendations
    
    def prioritize_recommendations(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Prioritize all recommendations by priority level.
        
        Args:
            recommendations: Dictionary of recommendations by category
            
        Returns:
            List of prioritized recommendations
        """
        all_recs = []
        for category, recs in recommendations.items():
            for rec in recs:
                rec['category'] = category
                all_recs.append(rec)
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        all_recs.sort(key=lambda x: priority_order.get(x.get('priority', 'Low'), 3))
        
        return all_recs
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all recommendations.
        
        Returns:
            Dictionary with recommendation summary
        """
        all_recs = self.generate_all_recommendations()
        prioritized = self.prioritize_recommendations(all_recs)
        
        summary = {
            'total_recommendations': len(prioritized),
            'high_priority': len([r for r in prioritized if r.get('priority') == 'High']),
            'medium_priority': len([r for r in prioritized if r.get('priority') == 'Medium']),
            'low_priority': len([r for r in prioritized if r.get('priority') == 'Low']),
            'by_category': {k: len(v) for k, v in all_recs.items()},
            'top_recommendations': prioritized[:5]
        }
        
        return summary
