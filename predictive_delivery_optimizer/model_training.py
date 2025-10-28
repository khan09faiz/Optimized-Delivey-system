"""
Model training module for the Predictive Delivery Optimizer.
Handles training and evaluation of predictive models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from .utils import calculate_metrics, logger


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        """Initialize ModelTrainer."""
        self.models = {}
        self.trained_models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Initialize model candidates
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all model candidates.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = calculate_metrics(y_train, y_pred_train)
            test_metrics = calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            # Store results
            results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_mae': cv_mae,
                'model': model
            }
            
            # Store trained model
            self.trained_models[model_name] = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Store metrics
            self.performance_metrics[model_name] = test_metrics
            
            logger.info(f"{model_name} - Test MAE: {test_metrics['MAE']:.2f}, R2: {test_metrics['R2']:.4f}")
        
        return results
    
    def get_best_model(self, metric: str = 'MAE') -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for selection ('MAE', 'RMSE', 'R2')
            
        Returns:
            Tuple of (model_name, model)
        """
        if not self.performance_metrics:
            raise ValueError("No models have been trained yet")
        
        if metric == 'R2':
            # For R2, higher is better
            best_model_name = max(self.performance_metrics.items(), 
                                 key=lambda x: x[1][metric])[0]
        else:
            # For MAE, RMSE, lower is better
            best_model_name = min(self.performance_metrics.items(), 
                                 key=lambda x: x[1][metric])[0]
        
        logger.info(f"Best model by {metric}: {best_model_name}")
        return best_model_name, self.trained_models[best_model_name]
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Feature DataFrame
            model_name: Name of the model to use (uses best if None)
            
        Returns:
            Array of predictions
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained")
        
        predictions = self.trained_models[model_name].predict(X)
        return predictions
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from a model.
        
        Args:
            model_name: Name of the model (uses best if None)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance not available for '{model_name}'")
        
        return self.feature_importance[model_name].head(top_n)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str = None) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            model_name: Name of the model to evaluate
            
        Returns:
            Dictionary of metrics
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        predictions = self.predict(X_test, model_name)
        metrics = calculate_metrics(y_test, predictions)
        
        logger.info(f"Evaluation metrics for {model_name}: {metrics}")
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            DataFrame comparing model performance
        """
        if not self.performance_metrics:
            raise ValueError("No models have been trained yet")
        
        comparison = pd.DataFrame(self.performance_metrics).T
        comparison = comparison.sort_values('MAE')
        
        return comparison
    
    def train_delivery_delay_predictor(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train models specifically for delivery delay prediction.
        
        Args:
            X: Feature DataFrame
            y: Delivery delay target
            
        Returns:
            Training results
        """
        logger.info("Training delivery delay prediction models...")
        return self.train_models(X, y)
    
    def train_cost_predictor(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train models specifically for cost prediction.
        
        Args:
            X: Feature DataFrame
            y: Cost target
            
        Returns:
            Training results
        """
        logger.info("Training cost prediction models...")
        return self.train_models(X, y)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models.
        
        Returns:
            Dictionary with model summaries
        """
        summary = {
            'num_models_trained': len(self.trained_models),
            'models': list(self.trained_models.keys()),
            'performance': self.performance_metrics,
            'best_model': self.get_best_model()[0] if self.trained_models else None
        }
        return summary
