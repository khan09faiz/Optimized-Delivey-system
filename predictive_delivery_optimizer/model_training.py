"""
Model training and evaluation module.

This module handles:
- Train/test splitting
- Model training (RandomForest, XGBoost)
- Hyperparameter tuning
- Model evaluation and metrics
- Model persistence
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve,
    precision_recall_curve, average_precision_score, make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import streamlit as st

from utils import Config, setup_logging, get_timestamp, save_json

logger = setup_logging("model_training")


class ModelTrainer:
    """Handles model training, evaluation, and persistence."""
    
    def __init__(self, random_state: int = Config.RANDOM_STATE):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        
        Config.ensure_directories()
        logger.info(f"ModelTrainer initialized with random_state={random_state}")
    
    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'delay_flag',
        test_size: float = Config.TEST_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing train/test split...")
        
        # Validate columns exist
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining missing values
        X.fillna(X.median(), inplace=True)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.feature_names = feature_cols
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained RandomForestClassifier
        """
        logger.info("Training Random Forest classifier...")
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['RandomForest'] = model
        logger.info("Random Forest training complete")
        
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Trained XGBClassifier
        """
        logger.info("Training XGBoost classifier...")
        
        # Calculate scale_pos_weight for imbalanced classes
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss'
        }
        params.update(kwargs)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['XGBoost'] = model
        logger.info("XGBoost training complete")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'model_name': model_name,
            'accuracy': model.score(X_test, y_test),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Feature importance - use X_test columns for feature names
        if hasattr(model, 'feature_importances_'):
            feature_names = list(X_test.columns)
            if len(feature_names) == len(model.feature_importances_):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                metrics['feature_importance'] = feature_importance.to_dict('records')
        
        logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = Config.CV_FOLDS
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of CV scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
            'f1': cross_val_score(model, X, y, cv=skf, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        }
        
        cv_results = {
            'accuracy_mean': scores['accuracy'].mean(),
            'accuracy_std': scores['accuracy'].std(),
            'f1_mean': scores['f1'].mean(),
            'f1_std': scores['f1'].std(),
            'roc_auc_mean': scores['roc_auc'].mean(),
            'roc_auc_std': scores['roc_auc'].std()
        }
        
        logger.info(f"CV Results - ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        
        return cv_results
    
    def check_class_imbalance(self, y: pd.Series) -> Dict[str, Any]:
        """
        Check for class imbalance in the target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary with imbalance metrics
        """
        logger.info("Checking class imbalance...")
        
        class_counts = y.value_counts()
        total = len(y)
        
        imbalance_info = {
            'class_counts': class_counts.to_dict(),
            'class_percentages': (class_counts / total * 100).round(2).to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'is_imbalanced': (class_counts.max() / class_counts.min()) > 1.5
        }
        
        logger.info(f"Class distribution: {imbalance_info['class_percentages']}")
        logger.info(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}")
        
        if imbalance_info['is_imbalanced']:
            logger.warning("⚠️ Dataset is imbalanced! Consider using SMOTE or class weights.")
        
        return imbalance_info
    
    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sampling_strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance classes.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: SMOTE sampling strategy
            
        Returns:
            Balanced X_train, y_train
        """
        logger.info("Applying SMOTE for class balancing...")
        
        original_counts = y_train.value_counts().to_dict()
        logger.info(f"Original distribution: {original_counts}")
        
        smote = SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        new_counts = pd.Series(y_resampled).value_counts().to_dict()
        logger.info(f"After SMOTE: {new_counts}")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
    
    def train_with_cross_validation_detailed(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'RandomForest',
        n_splits: int = 5,
        use_smote: bool = True
    ) -> Dict[str, Any]:
        """
        Train model with detailed cross-validation and handle imbalance.
        
        Args:
            X: Features
            y: Target
            model_type: 'RandomForest' or 'XGBoost'
            n_splits: Number of CV folds
            use_smote: Whether to apply SMOTE
            
        Returns:
            Dictionary with detailed CV results
        """
        logger.info(f"Training {model_type} with {n_splits}-fold cross-validation...")
        
        # Check class imbalance
        imbalance_info = self.check_class_imbalance(y)
        
        # Setup stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'fold_scores': [],
            'fold_predictions': [],
            'fold_metrics': [],
            'imbalance_info': imbalance_info
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"Training fold {fold}/{n_splits}...")
            
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply SMOTE if requested
            if use_smote and imbalance_info['is_imbalanced']:
                X_fold_train, y_fold_train = self.apply_smote(X_fold_train, y_fold_train)
            
            # Train model
            if model_type == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:
                neg_count = (y_fold_train == 0).sum()
                pos_count = (y_fold_train == 1).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
                
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                )
            
            model.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation fold
            y_pred = model.predict(X_fold_val)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            
            fold_metrics = {
                'fold': fold,
                'accuracy': (y_pred == y_fold_val).mean(),
                'precision': precision_score(y_fold_val, y_pred, zero_division=0),
                'recall': recall_score(y_fold_val, y_pred, zero_division=0),
                'f1': f1_score(y_fold_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_fold_val, y_pred_proba)
            }
            
            cv_results['fold_scores'].append(fold_metrics['roc_auc'])
            cv_results['fold_metrics'].append(fold_metrics)
            
            logger.info(f"Fold {fold} - ROC-AUC: {fold_metrics['roc_auc']:.4f}, F1: {fold_metrics['f1']:.4f}")
        
        # Calculate mean and std
        scores_array = np.array(cv_results['fold_scores'])
        cv_results['mean_score'] = scores_array.mean()
        cv_results['std_score'] = scores_array.std()
        cv_results['cv_scores'] = scores_array.tolist()
        
        logger.info(f"Cross-Validation Results:")
        logger.info(f"  Mean ROC-AUC: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        logger.info(f"  Scores: {[f'{s:.4f}' for s in cv_results['fold_scores']]}")
        
        # Check for overfitting
        if cv_results['std_score'] > 0.1:
            logger.warning("⚠️ High variance in CV scores - possible overfitting!")
        
        return cv_results
    
    def train_and_evaluate_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary of all model metrics
        """
        logger.info("Training and evaluating all models...")
        
        all_metrics = {}
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, 'RandomForest')
        all_metrics['RandomForest'] = rf_metrics
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
        all_metrics['XGBoost'] = xgb_metrics
        
        # Select best model based on ROC-AUC
        best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['roc_auc'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.metrics = all_metrics
        
        logger.info(f"Best model: {best_model_name} with ROC-AUC: {all_metrics[best_model_name]['roc_auc']:.4f}")
        
        return all_metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model.
        
        Args:
            X: Features
            model_name: Model to use (defaults to best model)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_model(
        self,
        model_name: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save trained model to disk.
        
        Args:
            model_name: Model to save (defaults to best model)
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved model
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        if filename is None:
            timestamp = get_timestamp()
            filename = f"{model_name}_{timestamp}.pkl"
        
        filepath = Config.MODELS_DIR / filename
        
        # Save model
        joblib.dump(self.models[model_name], filepath)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'feature_names': self.feature_names,
            'timestamp': get_timestamp(),
            'metrics': self.metrics.get(model_name, {})
        }
        
        metadata_file = filepath.with_suffix('.json')
        save_json(metadata, metadata_file)
        
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: Path) -> Any:
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        
        # Load metadata if available
        metadata_file = filepath.with_suffix('.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
        
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def get_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Model to analyze (defaults to best model)
            top_n: Number of top features to return
            
        Returns:
            DataFrame of feature importances
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


# ==================== Convenience Functions ====================
@st.cache_resource
def get_model_trainer() -> ModelTrainer:
    """
    Create and return a ModelTrainer instance.
    
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer()


def train_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'delay_flag'
) -> Tuple[ModelTrainer, Dict[str, Any]]:
    """
    Convenience function to train all models.
    
    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Tuple of (trainer, metrics)
    """
    trainer = get_model_trainer()
    
    X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
        df, feature_cols, target_col
    )
    
    metrics = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    return trainer, metrics
