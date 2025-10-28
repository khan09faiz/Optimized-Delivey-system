"""
Explainability module using SHAP for model interpretability.

This module provides:
- SHAP value computation
- Global feature importance
- Local explanations for individual predictions
- Visualization functions
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import shap
import streamlit as st

from utils import setup_logging

logger = setup_logging("explainability")


class ExplainabilityAnalyzer:
    """Handles model explainability using SHAP."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize ExplainabilityAnalyzer.
        
        Args:
            model: Trained model
            X_train: Training data for background distribution
        """
        self.model = model
        self.X_train = X_train
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.expected_value: Optional[float] = None
        
        logger.info("ExplainabilityAnalyzer initialized")
    
    def create_explainer(self, max_samples: int = 100) -> shap.Explainer:
        """
        Create SHAP explainer for the model.
        
        Args:
            max_samples: Maximum samples for background data
            
        Returns:
            SHAP Explainer instance
        """
        logger.info("Creating SHAP explainer...")
        
        # Sample background data if dataset is large
        if len(self.X_train) > max_samples:
            background = shap.sample(self.X_train, max_samples, random_state=42)
        else:
            background = self.X_train
        
        try:
            # Try TreeExplainer for tree-based models (faster)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background,
                link="logit"
            )
        
        logger.info("SHAP explainer created successfully")
        return self.explainer
    
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity property
            
        Returns:
            SHAP values array
        """
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        if self.explainer is None:
            self.create_explainer()
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
        
        self.shap_values = shap_values
        
        # Get expected value
        if hasattr(self.explainer, 'expected_value'):
            expected = self.explainer.expected_value
            self.expected_value = expected[1] if isinstance(expected, list) else expected
        
        logger.info("SHAP values computed successfully")
        return shap_values
    
    def get_global_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get global feature importance using SHAP.
        
        Args:
            X: Data to analyze
            top_n: Number of top features
            
        Returns:
            DataFrame of feature importances
        """
        logger.info("Computing global feature importance...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False).head(top_n)
        
        logger.info(f"Top feature: {feature_importance.iloc[0]['feature']}")
        
        return feature_importance
    
    def get_local_explanation(
        self,
        X: pd.DataFrame,
        instance_idx: int
    ) -> pd.DataFrame:
        """
        Get SHAP explanation for a single instance.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance to explain
            
        Returns:
            DataFrame of feature contributions
        """
        logger.info(f"Getting local explanation for instance {instance_idx}...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get SHAP values for the instance
        instance_shap = self.shap_values[instance_idx]
        
        # Create explanation DataFrame
        explanation = pd.DataFrame({
            'feature': X.columns,
            'value': X.iloc[instance_idx].values,
            'shap_value': instance_shap
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return explanation
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = 'bar',
        max_display: int = 20
    ) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to visualize
            plot_type: 'bar' or 'dot'
            max_display: Maximum features to display
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating SHAP summary plot ({plot_type})...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.summary_plot(
            self.shap_values,
            X,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        
        return fig
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        max_display: int = 15
    ) -> plt.Figure:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance
            max_display: Maximum features to display
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating waterfall plot for instance {instance_idx}...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create Explanation object for waterfall plot
        if self.expected_value is not None:
            explanation = shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.expected_value,
                data=X.iloc[instance_idx].values,
                feature_names=X.columns.tolist()
            )
            
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
        else:
            logger.warning("Expected value not available for waterfall plot")
        
        plt.tight_layout()
        
        return fig
    
    def plot_force(
        self,
        X: pd.DataFrame,
        instance_idx: int
    ) -> Any:
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance
            
        Returns:
            SHAP force plot object
        """
        logger.info(f"Creating force plot for instance {instance_idx}...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if self.expected_value is None:
            logger.warning("Expected value not available")
            return None
        
        force_plot = shap.force_plot(
            self.expected_value,
            self.shap_values[instance_idx],
            X.iloc[instance_idx],
            matplotlib=True
        )
        
        return force_plot
    
    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP dependence plot showing feature effect.
        
        Args:
            X: Data to visualize
            feature: Feature to plot
            interaction_feature: Optional feature for interaction coloring
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating dependence plot for {feature}...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.tight_layout()
        
        return fig
    
    def get_top_features_for_instance(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top contributing features for a single instance.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance
            top_n: Number of top features
            
        Returns:
            DataFrame of top features and their contributions
        """
        explanation = self.get_local_explanation(X, instance_idx)
        return explanation.head(top_n)
    
    def explain_predictions(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """
        Add SHAP explanations to predictions DataFrame.
        
        Args:
            X: Features
            predictions: Binary predictions
            probabilities: Prediction probabilities
            
        Returns:
            DataFrame with predictions and explanations
        """
        logger.info("Adding SHAP explanations to predictions...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Create base DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        }, index=X.index)
        
        # Add top contributing features for each instance
        for idx in range(len(X)):
            explanation = self.get_local_explanation(X, idx)
            
            # Store top 3 features
            for rank in range(min(3, len(explanation))):
                feature = explanation.iloc[rank]['feature']
                contribution = explanation.iloc[rank]['shap_value']
                
                results.loc[X.index[idx], f'top_{rank+1}_feature'] = feature
                results.loc[X.index[idx], f'top_{rank+1}_contribution'] = contribution
        
        logger.info("Explanations added successfully")
        
        return results


# ==================== Convenience Functions ====================
def create_explainer(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[ExplainabilityAnalyzer, np.ndarray]:
    """
    Create explainer and compute SHAP values.
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        
    Returns:
        Tuple of (analyzer, shap_values)
    """
    analyzer = ExplainabilityAnalyzer(model, X_train)
    analyzer.create_explainer()
    shap_values = analyzer.compute_shap_values(X_test)
    
    return analyzer, shap_values


def get_feature_contributions(
    analyzer: ExplainabilityAnalyzer,
    X: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get global feature importance from SHAP values.
    
    Args:
        analyzer: ExplainabilityAnalyzer instance
        X: Data to analyze
        top_n: Number of top features
        
    Returns:
        DataFrame of feature importances
    """
    return analyzer.get_global_importance(X, top_n)
