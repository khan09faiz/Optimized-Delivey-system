"""
Explainability module for the Predictive Delivery Optimizer.
Provides model interpretation and explanation capabilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .utils import logger


class ModelExplainer:
    """Handles model explainability and interpretation."""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize ModelExplainer.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explanations = {}
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top {top_n} features extracted")
        return importance_df.head(top_n)
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """
        Explain a specific prediction.
        
        Args:
            X: Feature DataFrame
            index: Index of the sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        if index >= len(X):
            raise ValueError(f"Index {index} out of range for dataset of size {len(X)}")
        
        sample = X.iloc[index:index+1]
        prediction = self.model.predict(sample)[0]
        
        # Get feature contributions (simplified version)
        feature_values = sample.iloc[0].to_dict()
        
        # For tree-based models, we can use feature importance as proxy for contribution
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_,
                'value': sample.iloc[0].values
            })
            importance['contribution'] = importance['importance'] * importance['value']
            importance = importance.sort_values('contribution', ascending=False)
            top_contributors = importance.head(5)
        else:
            top_contributors = pd.DataFrame()
        
        explanation = {
            'prediction': prediction,
            'feature_values': feature_values,
            'top_contributors': top_contributors.to_dict('records') if not top_contributors.empty else []
        }
        
        logger.info(f"Explained prediction for sample {index}: {prediction:.2f}")
        return explanation
    
    def get_global_explanations(self) -> Dict[str, Any]:
        """
        Get global model explanations.
        
        Returns:
            Dictionary with global explanations
        """
        explanations = {
            'feature_importance': self.get_feature_importance(top_n=15).to_dict('records'),
            'model_type': type(self.model).__name__,
            'num_features': len(self.feature_names)
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'n_estimators'):
            explanations['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            explanations['max_depth'] = self.model.max_depth
        
        return explanations
    
    def analyze_feature_relationships(self, X: pd.DataFrame, y: pd.Series, 
                                     top_n: int = 5) -> pd.DataFrame:
        """
        Analyze relationships between top features and target.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            top_n: Number of top features to analyze
            
        Returns:
            DataFrame with correlation analysis
        """
        # Get top features
        top_features = self.get_feature_importance(top_n)
        
        # Calculate correlations
        correlations = []
        for feature in top_features['feature']:
            if feature in X.columns:
                corr = X[feature].corr(y)
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        
        logger.info("Feature relationship analysis completed")
        return corr_df
    
    def get_prediction_distribution(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Analyze the distribution of predictions.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            Dictionary with distribution statistics
        """
        distribution = {
            'mean': float(np.mean(predictions)),
            'median': float(np.median(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75))
        }
        
        logger.info("Prediction distribution analyzed")
        return distribution
    
    def explain_delivery_factors(self, feature_importance_df: pd.DataFrame) -> List[str]:
        """
        Generate human-readable explanations for delivery factors.
        
        Args:
            feature_importance_df: DataFrame with feature importance
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        for _, row in feature_importance_df.head(5).iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if 'distance' in feature.lower():
                explanations.append(
                    f"Distance is a key factor ({importance*100:.1f}% importance) - longer routes tend to have different delivery patterns"
                )
            elif 'weight' in feature.lower():
                explanations.append(
                    f"Package weight significantly impacts delivery ({importance*100:.1f}% importance)"
                )
            elif 'traffic' in feature.lower():
                explanations.append(
                    f"Traffic conditions are crucial ({importance*100:.1f}% importance) for delivery timing"
                )
            elif 'priority' in feature.lower():
                explanations.append(
                    f"Order priority affects delivery performance ({importance*100:.1f}% importance)"
                )
            elif 'cost' in feature.lower():
                explanations.append(
                    f"Cost factors influence delivery efficiency ({importance*100:.1f}% importance)"
                )
            else:
                explanations.append(
                    f"{feature} is an important factor ({importance*100:.1f}% importance)"
                )
        
        return explanations
    
    def create_explanation_report(self, X: pd.DataFrame, y: pd.Series, 
                                  predictions: np.ndarray) -> Dict[str, Any]:
        """
        Create a comprehensive explanation report.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            predictions: Model predictions
            
        Returns:
            Dictionary with comprehensive explanations
        """
        feature_importance = self.get_feature_importance(top_n=10)
        
        report = {
            'model_type': type(self.model).__name__,
            'feature_importance': feature_importance.to_dict('records'),
            'feature_relationships': self.analyze_feature_relationships(X, y).to_dict('records'),
            'prediction_distribution': self.get_prediction_distribution(predictions),
            'human_readable_factors': self.explain_delivery_factors(feature_importance),
            'num_samples_analyzed': len(X)
        }
        
        logger.info("Comprehensive explanation report created")
        return report
