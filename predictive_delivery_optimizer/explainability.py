

import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
        # Handle both 2D and 3D SHAP values arrays
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]  # Take first class for multi-class
        
        if len(shap_vals.shape) > 2:
            shap_vals = shap_vals[:, :, 0]  # Take first output for multi-output
        
        importance_values = np.abs(shap_vals).mean(axis=0)
        
        # Ensure importance_values is 1D
        if len(importance_values.shape) > 1:
            importance_values = importance_values.flatten()
        
        feature_importance = pd.DataFrame({
            'feature': [str(col) for col in X.columns],  # Convert to list of strings
            'importance': importance_values
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
        
        # Ensure all arrays are 1D
        features = [str(col) for col in X.columns]
        values = np.array(X.iloc[instance_idx].values).flatten()
        shap_vals = np.array(instance_shap).flatten()
        
        # Create explanation DataFrame
        explanation = pd.DataFrame({
            'feature': features,
            'value': values,
            'shap_value': shap_vals
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
    
    # ==================== Enhanced Plotly Visualizations ====================
    
    def plot_shap_summary_interactive(
        self,
        X: pd.DataFrame,
        max_display: int = 20
    ) -> go.Figure:
        """
        Create interactive Plotly SHAP summary plot (beeswarm style).
        
        Args:
            X: Data to visualize
            max_display: Maximum features to display
            
        Returns:
            Plotly figure
        """
        logger.info("Creating interactive SHAP summary plot...")
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Calculate feature importance
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-max_display:][::-1]
        
        # Create plot data
        fig = go.Figure()
        
        for idx in top_indices:
            feature_name = str(X.columns[idx])  # Convert to string to avoid pandas Index
            shap_vals = self.shap_values[:, idx]
            feature_vals = X.iloc[:, idx].values
            
            # Normalize feature values for color
            if feature_vals.std() > 0:
                normalized_vals = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
            else:
                normalized_vals = np.zeros_like(feature_vals)
            
            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=[feature_name] * len(shap_vals),
                mode='markers',
                marker=dict(
                    color=normalized_vals,
                    colorscale='RdBu',
                    size=6,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate=(
                    f'<b>{feature_name}</b><br>' +
                    'SHAP Value: %{x:.4f}<br>' +
                    'Feature Value: %{customdata:.2f}<br>' +
                    '<extra></extra>'
                ),
                customdata=feature_vals,
                name=feature_name,
                showlegend=False
            ))
        
        fig.update_layout(
            title='SHAP Feature Importance (Interactive Summary)',
            xaxis_title='SHAP Value (impact on model output)',
            yaxis_title='Features',
            height=max(500, max_display * 30),
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def plot_shap_bar_interactive(
        self,
        X: pd.DataFrame,
        top_n: int = 20
    ) -> go.Figure:
        """
        Create interactive Plotly bar chart of mean absolute SHAP values.
        
        Args:
            X: Data to visualize
            top_n: Number of top features
            
        Returns:
            Plotly figure
        """
        logger.info("Creating interactive SHAP bar chart...")
        
        importance_df = self.get_global_importance(X, top_n)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(
                color=importance_df['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Mean |SHAP|')
            ),
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Mean |SHAP|: %{x:.4f}<br>' +
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Features by SHAP Importance',
            xaxis_title='Mean Absolute SHAP Value',
            yaxis_title='Features',
            height=max(400, top_n * 25),
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_shap_waterfall_interactive(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        max_display: int = 15
    ) -> go.Figure:
        """
        Create interactive Plotly waterfall plot for single prediction.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance
            max_display: Maximum features to display
            
        Returns:
            Plotly figure
        """
        logger.info(f"Creating interactive waterfall plot for instance {instance_idx}...")
        
        explanation = self.get_local_explanation(X, instance_idx)
        top_explanation = explanation.head(max_display)
        
        # Prepare waterfall data
        features = top_explanation['feature'].tolist()
        shap_values = top_explanation['shap_value'].tolist()
        
        # Calculate cumulative values
        base_value = self.expected_value if self.expected_value is not None else 0
        cumulative = [base_value]
        
        for val in shap_values:
            cumulative.append(cumulative[-1] + val)
        
        # Create waterfall
        fig = go.Figure()
        
        # Add bars
        colors = ['red' if v < 0 else 'green' for v in shap_values]
        
        for i, (feature, shap_val) in enumerate(zip(features, shap_values)):
            fig.add_trace(go.Bar(
                name=feature,
                x=[feature],
                y=[abs(shap_val)],
                base=cumulative[i] if shap_val > 0 else cumulative[i] - abs(shap_val),
                marker_color=colors[i],
                hovertemplate=(
                    f'<b>{feature}</b><br>' +
                    f'SHAP: {shap_val:.4f}<br>' +
                    f'Cumulative: {cumulative[i+1]:.4f}<br>' +
                    '<extra></extra>'
                ),
                showlegend=False
            ))
        
        # Add base value and prediction lines
        fig.add_hline(
            y=base_value,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Base: {base_value:.4f}"
        )
        
        fig.update_layout(
            title=f'SHAP Waterfall - Instance {instance_idx}',
            xaxis_title='Features',
            yaxis_title='Model Output Value',
            height=500,
            template='plotly_white',
            barmode='relative'
        )
        
        return fig
    
    def plot_shap_force_interactive(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        max_display: int = 10
    ) -> go.Figure:
        """
        Create interactive Plotly force plot alternative.
        
        Args:
            X: Data containing the instance
            instance_idx: Index of instance
            max_display: Maximum features to display
            
        Returns:
            Plotly figure
        """
        logger.info(f"Creating interactive force plot for instance {instance_idx}...")
        
        explanation = self.get_local_explanation(X, instance_idx)
        top_explanation = explanation.head(max_display)
        
        # Separate positive and negative contributions
        positive = top_explanation[top_explanation['shap_value'] > 0].sort_values('shap_value', ascending=False)
        negative = top_explanation[top_explanation['shap_value'] < 0].sort_values('shap_value', ascending=True)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Positive Contributions', 'Negative Contributions'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Positive contributions
        if len(positive) > 0:
            fig.add_trace(
                go.Bar(
                    y=positive['feature'],
                    x=positive['shap_value'],
                    orientation='h',
                    marker_color='green',
                    name='Positive',
                    hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<br><extra></extra>'
                ),
                row=1, col=1
            )
        
        # Negative contributions
        if len(negative) > 0:
            fig.add_trace(
                go.Bar(
                    y=negative['feature'],
                    x=negative['shap_value'],
                    orientation='h',
                    marker_color='red',
                    name='Negative',
                    hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<br><extra></extra>'
                ),
                row=1, col=2
            )
        
        base_value = self.expected_value if self.expected_value is not None else 0
        
        fig.update_layout(
            title=f'SHAP Force Plot - Instance {instance_idx} (Base: {base_value:.4f})',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text='SHAP Value', row=1, col=1)
        fig.update_xaxes(title_text='SHAP Value', row=1, col=2)
        
        return fig
    
    def plot_shap_dependence_interactive(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        sample_size: int = 1000
    ) -> go.Figure:
        """
        Create interactive Plotly dependence plot.
        
        Args:
            X: Data to visualize
            feature: Feature to plot
            interaction_feature: Optional feature for color interaction
            sample_size: Maximum samples to plot
            
        Returns:
            Plotly figure
        """
        logger.info(f"Creating interactive dependence plot for {feature}...")
        
        # Ensure feature is a string, not an Index or other type
        feature = str(feature)
        if interaction_feature is not None:
            interaction_feature = str(interaction_feature)
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[indices]
            shap_sample = self.shap_values[indices]
        else:
            X_sample = X
            shap_sample = self.shap_values
        
        feature_idx = X.columns.get_loc(feature)
        feature_values = X_sample[feature].values
        shap_feature = shap_sample[:, feature_idx]
        
        # Determine color variable
        if interaction_feature and interaction_feature in X.columns:
            color_values = X_sample[interaction_feature].values
            color_label = str(interaction_feature)
        else:
            # Auto-select interaction feature (highest correlation with SHAP values)
            correlations = []
            for col_idx, col in enumerate(X.columns):
                if col != feature:
                    corr = np.abs(np.corrcoef(shap_sample[:, col_idx], shap_feature)[0, 1])
                    correlations.append((col, corr))
            
            if correlations:
                interaction_feature = max(correlations, key=lambda x: x[1])[0]
                # Ensure interaction_feature is a string
                interaction_feature = str(interaction_feature)
                color_values = X_sample[interaction_feature].values
                color_label = f"{interaction_feature} (auto)"
            else:
                color_values = None
                color_label = None
        
        # Create scatter plot
        fig = go.Figure()
        
        if color_values is not None:
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=shap_feature,
                mode='markers',
                marker=dict(
                    color=color_values,
                    colorscale='Viridis',
                    size=6,
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title=color_label)
                ),
                hovertemplate=(
                    f'<b>{feature}</b>: %{{x:.2f}}<br>' +
                    f'SHAP Value: %{{y:.4f}}<br>' +
                    f'{color_label}: %{{marker.color:.2f}}<br>' +
                    '<extra></extra>'
                )
            ))
        else:
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=shap_feature,
                mode='markers',
                marker=dict(
                    color='blue',
                    size=6,
                    opacity=0.6
                ),
                hovertemplate=(
                    f'<b>{feature}</b>: %{{x:.2f}}<br>' +
                    f'SHAP Value: %{{y:.4f}}<br>' +
                    '<extra></extra>'
                )
            ))
        
        fig.update_layout(
            title=f'SHAP Dependence Plot: {feature}',
            xaxis_title=feature,
            yaxis_title=f'SHAP Value for {feature}',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_feature_impact_simple(
        self,
        X: pd.DataFrame,
        top_n: int = 15
    ) -> go.Figure:
        """
        Create a simple, clean feature impact visualization.
        
        Args:
            X: Data to analyze
            top_n: Number of top features to display
            
        Returns:
            Plotly figure
        """
        logger.info("Creating simple feature impact chart...")
        
        importance_df = self.get_global_importance(X, top_n)
        
        # Create a clean horizontal bar chart
        fig = go.Figure()
        
        colors = ['#FF6B6B' if i < 5 else '#4ECDC4' if i < 10 else '#95E1D3' 
                  for i in range(len(importance_df))]
        
        fig.add_trace(go.Bar(
            y=importance_df['feature'],
            x=importance_df['importance'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.4f}" for val in importance_df['importance']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'🎯 Top {top_n} Most Important Features',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50'}
            },
            xaxis=dict(
                title='Feature Importance (Mean |SHAP Value|)',
                showgrid=True,
                gridcolor='#E8E8E8'
            ),
            yaxis=dict(
                title='',
                categoryorder='total ascending'
            ),
            height=max(400, top_n * 35),
            template='plotly_white',
            showlegend=False,
            margin=dict(l=150, r=100, t=80, b=60)
        )
        
        return fig
    
    def create_shap_dashboard(
        self,
        X: pd.DataFrame,
        instance_idx: Optional[int] = None
    ) -> Dict[str, go.Figure]:
        """
        Create a comprehensive dashboard of SHAP visualizations.
        
        Args:
            X: Data to visualize
            instance_idx: Optional specific instance to analyze
            
        Returns:
            Dictionary of figure names to Plotly figures
        """
        logger.info("Creating comprehensive SHAP dashboard...")
        
        dashboard = {}
        
        # Global importance
        dashboard['bar_chart'] = self.plot_shap_bar_interactive(X, top_n=20)
        dashboard['summary_plot'] = self.plot_shap_summary_interactive(X, max_display=20)
        
        # Instance-level analysis
        if instance_idx is not None:
            dashboard['waterfall'] = self.plot_shap_waterfall_interactive(X, instance_idx)
            dashboard['force_plot'] = self.plot_shap_force_interactive(X, instance_idx)
        
        # Top feature dependence plots
        importance_df = self.get_global_importance(X, top_n=5)
        for i, feature in enumerate(importance_df['feature'].head(3)):
            dashboard[f'dependence_{i+1}'] = self.plot_shap_dependence_interactive(X, feature)
        
        logger.info(f"Dashboard created with {len(dashboard)} visualizations")
        
        return dashboard


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
