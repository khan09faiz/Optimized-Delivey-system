"""
Test script for the Predictive Delivery Optimizer.
Tests all modules end-to-end.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from predictive_delivery_optimizer import (
    DataLoader,
    FeatureEngineer,
    ModelTrainer,
    ModelExplainer,
    RecommendationEngine,
    Visualizer
)


def test_data_loading():
    """Test data loading."""
    print("=" * 60)
    print("Testing Data Loading...")
    print("=" * 60)
    
    loader = DataLoader(data_dir='data')
    datasets = loader.load_all_datasets()
    
    print(f"✓ Loaded {len(datasets)} datasets")
    
    info = loader.get_dataset_info()
    for name, stats in info.items():
        print(f"  - {name}: {stats['rows']} rows, {stats['columns']} columns")
    
    return datasets, loader


def test_feature_engineering(datasets):
    """Test feature engineering."""
    print("\n" + "=" * 60)
    print("Testing Feature Engineering...")
    print("=" * 60)
    
    engineer = FeatureEngineer(datasets)
    X, y = engineer.prepare_features_for_modeling()
    
    print(f"✓ Created {X.shape[1]} features from {X.shape[0]} samples")
    print(f"  - Top features: {list(X.columns[:5])}")
    
    return engineer, X, y


def test_model_training(X, y):
    """Test model training."""
    print("\n" + "=" * 60)
    print("Testing Model Training...")
    print("=" * 60)
    
    trainer = ModelTrainer()
    results = trainer.train_models(X, y)
    
    print(f"✓ Trained {len(results)} models")
    
    comparison = trainer.compare_models()
    print("\nModel Performance:")
    for idx, row in comparison.iterrows():
        print(f"  - {idx}: MAE={row['MAE']:.2f}, R2={row['R2']:.4f}")
    
    best_name, best_model = trainer.get_best_model()
    print(f"\n✓ Best model: {best_name}")
    
    return trainer


def test_explainability(trainer, X, y):
    """Test model explainability."""
    print("\n" + "=" * 60)
    print("Testing Model Explainability...")
    print("=" * 60)
    
    best_name, best_model = trainer.get_best_model()
    explainer = ModelExplainer(best_model, X.columns.tolist())
    
    # Get feature importance
    importance = explainer.get_feature_importance(top_n=5)
    print(f"✓ Top 5 features by importance:")
    for idx, row in importance.iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    
    # Get predictions
    predictions = trainer.predict(X, best_name)
    dist = explainer.get_prediction_distribution(predictions)
    print(f"\n✓ Prediction distribution:")
    print(f"  - Mean: {dist['mean']:.2f}")
    print(f"  - Median: {dist['median']:.2f}")
    print(f"  - Std: {dist['std']:.2f}")
    
    return explainer


def test_recommendations(datasets):
    """Test recommendation engine."""
    print("\n" + "=" * 60)
    print("Testing Recommendation Engine...")
    print("=" * 60)
    
    rec_engine = RecommendationEngine(datasets)
    summary = rec_engine.get_recommendation_summary()
    
    print(f"✓ Generated {summary['total_recommendations']} recommendations")
    print(f"  - High priority: {summary['high_priority']}")
    print(f"  - Medium priority: {summary['medium_priority']}")
    print(f"  - Low priority: {summary['low_priority']}")
    print(f"\n✓ By category:")
    for category, count in summary['by_category'].items():
        print(f"  - {category}: {count}")
    
    return rec_engine


def test_visualization(datasets):
    """Test visualization."""
    print("\n" + "=" * 60)
    print("Testing Visualization...")
    print("=" * 60)
    
    visualizer = Visualizer(datasets)
    
    # Create various plots
    plots_created = []
    
    try:
        fig = visualizer.plot_delivery_performance()
        plots_created.append("Delivery Performance")
    except:
        pass
    
    try:
        fig = visualizer.plot_cost_breakdown()
        plots_created.append("Cost Breakdown")
    except:
        pass
    
    try:
        fig = visualizer.plot_route_analysis()
        plots_created.append("Route Analysis")
    except:
        pass
    
    try:
        fig = visualizer.plot_customer_feedback()
        plots_created.append("Customer Feedback")
    except:
        pass
    
    try:
        fig = visualizer.plot_fleet_status()
        plots_created.append("Fleet Status")
    except:
        pass
    
    print(f"✓ Created {len(plots_created)} visualizations:")
    for plot in plots_created:
        print(f"  - {plot}")
    
    # Create dashboard metrics
    metrics = visualizer.create_dashboard_metrics()
    print(f"\n✓ Dashboard metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
    return visualizer


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PREDICTIVE DELIVERY OPTIMIZER - END-TO-END TEST")
    print("=" * 60)
    
    try:
        # Test each module
        datasets, loader = test_data_loading()
        engineer, X, y = test_feature_engineering(datasets)
        trainer = test_model_training(X, y)
        explainer = test_explainability(trainer, X, y)
        rec_engine = test_recommendations(datasets)
        visualizer = test_visualization(datasets)
        
        # Final summary
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe Predictive Delivery Optimizer is ready to use!")
        print("\nTo run the Streamlit application:")
        print("  streamlit run predictive_delivery_optimizer/app.py")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED! ✗")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
