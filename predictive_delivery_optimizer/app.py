"""
Streamlit application for the Predictive Delivery Optimizer.
Main UI interface for the delivery optimization system.
"""
import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_delivery_optimizer.data_loader import DataLoader
from predictive_delivery_optimizer.feature_engineering import FeatureEngineer
from predictive_delivery_optimizer.model_training import ModelTrainer
from predictive_delivery_optimizer.explainability import ModelExplainer
from predictive_delivery_optimizer.recommendation_engine import RecommendationEngine
from predictive_delivery_optimizer.visualization import Visualizer


# Page configuration
st.set_page_config(
    page_title="Predictive Delivery Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load all datasets."""
    data_loader = DataLoader(data_dir='data')
    datasets = data_loader.load_all_datasets()
    return datasets, data_loader


@st.cache_resource
def train_model(_datasets):
    """Train the predictive model."""
    # Feature engineering
    feature_engineer = FeatureEngineer(_datasets)
    X, y = feature_engineer.prepare_features_for_modeling(target_column='delay_minutes')
    
    # Model training
    trainer = ModelTrainer()
    trainer.train_models(X, y)
    
    return trainer, feature_engineer, X, y


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚚 Predictive Delivery Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Data Analysis", "Model Performance", "Recommendations", "Dataset Info"]
    )
    
    # Load data
    try:
        datasets, data_loader = load_data()
        st.sidebar.success(f"✅ Loaded {len(datasets)} datasets")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Initialize components
    visualizer = Visualizer(datasets)
    
    # Page routing
    if page == "Dashboard":
        show_dashboard(datasets, visualizer)
    elif page == "Data Analysis":
        show_data_analysis(datasets, visualizer)
    elif page == "Model Performance":
        show_model_performance(datasets)
    elif page == "Recommendations":
        show_recommendations(datasets)
    elif page == "Dataset Info":
        show_dataset_info(datasets, data_loader)


def show_dashboard(datasets, visualizer):
    """Display the main dashboard."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    metrics = visualizer.create_dashboard_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Deliveries", metrics.get('total_deliveries', 'N/A'))
    with col2:
        st.metric("On-Time Rate", metrics.get('on_time_rate', 'N/A'))
    with col3:
        st.metric("Avg Delay", metrics.get('avg_delay', 'N/A'))
    with col4:
        st.metric("Avg Rating", metrics.get('avg_rating', 'N/A'))
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(visualizer.plot_delivery_performance(), use_container_width=True)
    
    with col2:
        st.plotly_chart(visualizer.plot_cost_breakdown(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(visualizer.plot_customer_feedback(), use_container_width=True)
    
    with col2:
        st.plotly_chart(visualizer.plot_fleet_status(), use_container_width=True)


def show_data_analysis(datasets, visualizer):
    """Display data analysis page."""
    st.header("🔍 Data Analysis")
    
    # Dataset selector
    dataset_name = st.selectbox("Select Dataset", list(datasets.keys()))
    
    if dataset_name:
        df = datasets[dataset_name]
        
        # Dataset overview
        st.subheader(f"📋 {dataset_name.replace('_', ' ').title()} Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations based on dataset
        st.subheader("Visualizations")
        if dataset_name == 'routes_distance':
            st.plotly_chart(visualizer.plot_route_analysis(), use_container_width=True)
        elif dataset_name == 'customer_feedback':
            st.plotly_chart(visualizer.plot_customer_feedback(), use_container_width=True)
        elif dataset_name == 'vehicle_fleet':
            st.plotly_chart(visualizer.plot_fleet_status(), use_container_width=True)
        elif dataset_name == 'cost_breakdown':
            st.plotly_chart(visualizer.plot_cost_breakdown(), use_container_width=True)


def show_model_performance(datasets):
    """Display model performance page."""
    st.header("🤖 Model Performance")
    
    with st.spinner("Training models..."):
        try:
            trainer, feature_engineer, X, y = train_model(datasets)
            
            # Model comparison
            st.subheader("Model Comparison")
            comparison_df = trainer.compare_models()
            st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE'])
                        .highlight_max(axis=0, subset=['R2']), use_container_width=True)
            
            # Best model
            best_model_name, best_model = trainer.get_best_model()
            st.success(f"🏆 Best Model: **{best_model_name}** (MAE: {trainer.performance_metrics[best_model_name]['MAE']:.2f})")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = trainer.get_feature_importance(best_model_name, top_n=15)
            
            if not feature_importance.empty:
                visualizer = Visualizer(datasets)
                st.plotly_chart(visualizer.plot_feature_importance(feature_importance), use_container_width=True)
            
            # Model explainability
            st.subheader("Model Explainability")
            explainer = ModelExplainer(best_model, X.columns.tolist())
            
            # Get predictions for visualization
            predictions = trainer.predict(X, best_model_name)
            
            # Predictions vs Actual
            visualizer = Visualizer(datasets)
            st.plotly_chart(visualizer.plot_predictions_vs_actual(y.values, predictions), use_container_width=True)
            
            # Global explanations
            global_explanations = explainer.get_global_explanations()
            st.json(global_explanations)
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_recommendations(datasets):
    """Display recommendations page."""
    st.header("💡 Optimization Recommendations")
    
    # Generate recommendations
    rec_engine = RecommendationEngine(datasets)
    all_recommendations = rec_engine.generate_all_recommendations()
    
    # Summary
    summary = rec_engine.get_recommendation_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recommendations", summary['total_recommendations'])
    with col2:
        st.metric("High Priority", summary['high_priority'])
    with col3:
        st.metric("Medium Priority", summary['medium_priority'])
    
    st.markdown("---")
    
    # Display recommendations by category
    tabs = st.tabs(["All", "Route", "Fleet", "Cost", "Customer Satisfaction"])
    
    with tabs[0]:
        st.subheader("All Recommendations")
        prioritized = rec_engine.prioritize_recommendations(all_recommendations)
        for i, rec in enumerate(prioritized, 1):
            with st.expander(f"#{i} [{rec.get('priority', 'N/A')}] {rec.get('issue', 'N/A')}", expanded=i<=3):
                st.write(f"**Category:** {rec.get('category', 'N/A')}")
                st.write(f"**Recommendation:** {rec.get('recommendation', 'N/A')}")
                if 'potential_impact' in rec:
                    st.write(f"**Potential Impact:** {rec['potential_impact']}")
    
    with tabs[1]:
        display_category_recommendations(all_recommendations.get('route_optimizations', []))
    
    with tabs[2]:
        display_category_recommendations(all_recommendations.get('fleet_optimizations', []))
    
    with tabs[3]:
        display_category_recommendations(all_recommendations.get('cost_optimizations', []))
    
    with tabs[4]:
        display_category_recommendations(all_recommendations.get('customer_satisfaction', []))


def display_category_recommendations(recommendations):
    """Display recommendations for a specific category."""
    if not recommendations:
        st.info("No recommendations in this category")
        return
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"#{i} [{rec.get('priority', 'N/A')}] {rec.get('issue', 'N/A')}", expanded=True):
            st.write(f"**Recommendation:** {rec.get('recommendation', 'N/A')}")
            for key, value in rec.items():
                if key not in ['type', 'priority', 'issue', 'recommendation', 'category']:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")


def show_dataset_info(datasets, data_loader):
    """Display dataset information page."""
    st.header("📚 Dataset Information")
    
    # Dataset info
    dataset_info = data_loader.get_dataset_info()
    
    # Create DataFrame for display
    info_df = pd.DataFrame(dataset_info).T
    st.dataframe(info_df, use_container_width=True)
    
    st.markdown("---")
    
    # Dataset schemas
    st.subheader("Dataset Schemas")
    for dataset_name, columns in data_loader.schema.items():
        with st.expander(f"{dataset_name.replace('_', ' ').title()} Schema"):
            st.write("**Required Columns:**")
            for col in columns:
                st.write(f"- {col}")


if __name__ == "__main__":
    main()
