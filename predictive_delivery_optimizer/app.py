"""
Streamlit Dashboard for Predictive Delivery Optimizer - Simplified Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from visualization import Visualizer
from recommendation_engine import RecommendationEngine
from utils import Config, setup_logging, get_risk_category, format_currency, format_percentage

logger = setup_logging("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="Delivery Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; padding: 1rem 0;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🚚 Predictive Delivery Optimizer</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Load Data
    if not st.session_state.data_loaded or st.sidebar.button("🔄 Reload Data"):
        with st.spinner("Loading data..."):
            try:
                loader = DataLoader()
                raw_df, _ = loader.load_and_prepare_data()
                
                # Engineer features
                engineer = FeatureEngineer()
                engineered_df = engineer.engineer_features(raw_df, fit=True)
                
                # Store in session state
                st.session_state.raw_df = raw_df
                st.session_state.engineered_df = engineered_df
                st.session_state.engineer = engineer
                st.session_state.data_loaded = True
                
                st.sidebar.success(f"✅ Loaded {len(raw_df)} orders")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Data loading error: {str(e)}", exc_info=True)
                return
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first")
        return
    
    raw_df = st.session_state.raw_df
    engineered_df = st.session_state.engineered_df
    engineer = st.session_state.engineer
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    
    selected_carriers = st.sidebar.multiselect(
        "Carrier",
        options=sorted(raw_df['carrier'].unique()) if 'carrier' in raw_df.columns else [],
        default=None
    )
    
    selected_segments = st.sidebar.multiselect(
        "Customer Segment",
        options=sorted(raw_df['customer_segment'].unique()) if 'customer_segment' in raw_df.columns else [],
        default=None
    )
    
    selected_priorities = st.sidebar.multiselect(
        "Priority",
        options=sorted(raw_df['priority'].unique()) if 'priority' in raw_df.columns else [],
        default=None
    )
    
    # Apply filters to engineered data (not raw data)
    # This ensures feature consistency
    filtered_raw_df = raw_df.copy()
    filtered_df = engineered_df.copy()
    
    if selected_carriers:
        mask = filtered_raw_df['carrier'].isin(selected_carriers)
        filtered_raw_df = filtered_raw_df[mask]
        filtered_df = filtered_df[mask]
    
    if selected_segments:
        mask = filtered_raw_df['customer_segment'].isin(selected_segments)
        filtered_raw_df = filtered_raw_df[mask]
        filtered_df = filtered_df[mask]
    
    if selected_priorities:
        mask = filtered_raw_df['priority'].isin(selected_priorities)
        filtered_raw_df = filtered_raw_df[mask]
        filtered_df = filtered_df[mask]
    
    st.sidebar.info(f"📊 {len(filtered_df)} orders displayed")
    
    # Model Training
    st.sidebar.markdown("### 🤖 Model Training")
    if st.sidebar.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            try:
                # Get feature columns (only numeric)
                feature_cols = engineer.get_feature_columns(filtered_df, target_col='delay_flag')
                
                # Prepare data
                X = filtered_df[feature_cols].fillna(0)
                y = filtered_df['delay_flag']
                
                # Split data for training
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train
                trainer = ModelTrainer()
                results = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
                
                # Store in session
                st.session_state.trainer = trainer
                st.session_state.feature_cols = feature_cols
                
                st.sidebar.success("✅ Models trained!")
            except Exception as e:
                st.sidebar.error(f"Training error: {str(e)}")
                logger.error(f"Training error: {str(e)}", exc_info=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Predictions", "💡 Recommendations"])
    
    # ==================== Tab 1: Dashboard ====================
    with tab1:
        st.header("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        delay_rate = filtered_df['delay_flag'].mean() if 'delay_flag' in filtered_df.columns else 0
        total_orders = len(filtered_df)
        delayed = filtered_df['delay_flag'].sum() if 'delay_flag' in filtered_df.columns else 0
        on_time = total_orders - delayed
        
        col1.metric("Total Orders", f"{total_orders:,}")
        col2.metric("Delay Rate", f"{delay_rate:.1%}")
        col3.metric("On-Time Orders", f"{on_time:,}")
        col4.metric("Delayed Orders", f"{delayed:,}")
        
        st.markdown("---")
        
        # Visualizations
        viz = Visualizer()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Delay Distribution")
            fig = viz.plot_delay_distribution(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Carrier Performance")
            fig = viz.plot_carrier_performance(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Priority Analysis")
            fig = viz.plot_priority_analysis(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("Distance vs Delay")
            fig = viz.plot_distance_vs_delay(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== Tab 2: Predictions ====================
    with tab2:
        st.header("Delay Predictions")
        
        if 'trainer' not in st.session_state:
            st.warning("⚠️ Please train models first using the sidebar button")
        else:
            trainer = st.session_state.trainer
            feature_cols = st.session_state.feature_cols
            
            # Prepare data for prediction
            X = filtered_df[feature_cols].fillna(0)
            
            # Make predictions
            predictions, probabilities = trainer.predict(X)
            
            # Add to dataframe
            result_df = filtered_raw_df.copy()
            result_df['predicted_delay'] = predictions
            result_df['delay_probability'] = probabilities
            result_df['risk_category'] = pd.Series(probabilities).apply(get_risk_category).values
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Delays", f"{predictions.sum():,}")
            col2.metric("High Risk Orders", f"{(probabilities >= Config.HIGH_RISK_THRESHOLD).sum():,}")
            col3.metric("Avg Risk", f"{probabilities.mean():.1%}")
            
            # Show results
            st.subheader("Prediction Results")
            
            display_cols = ['order_id', 'carrier', 'customer_segment', 'priority', 
                           'distance_km', 'delay_probability', 'risk_category']
            display_cols = [c for c in display_cols if c in result_df.columns]
            
            st.dataframe(
                result_df[display_cols].sort_values('delay_probability', ascending=False).head(50),
                use_container_width=True
            )
            
            # Download
            csv = result_df.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
    
    # ==================== Tab 3: Recommendations ====================
    with tab3:
        st.header("Corrective Action Recommendations")
        
        if 'trainer' not in st.session_state:
            st.warning("⚠️ Please train models and generate predictions first")
        else:
            # Get high-risk orders
            X = filtered_df[st.session_state.feature_cols].fillna(0)
            _, probabilities = st.session_state.trainer.predict(X)
            
            high_risk_mask = probabilities >= Config.HIGH_RISK_THRESHOLD
            high_risk_df = filtered_raw_df[high_risk_mask].copy()
            high_risk_df['delay_probability'] = probabilities[high_risk_mask]
            
            if len(high_risk_df) == 0:
                st.success("✅ No high-risk orders found!")
            else:
                st.warning(f"⚠️ {len(high_risk_df)} high-risk orders need attention")
                
                # Generate recommendations
                recommender = RecommendationEngine()
                
                for idx, row in high_risk_df.head(10).iterrows():
                    with st.expander(f"Order {row.get('order_id', idx)} - Risk: {row['delay_probability']:.1%}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Order Details:**")
                            st.write(f"Carrier: {row.get('carrier', 'N/A')}")
                            st.write(f"Priority: {row.get('priority', 'N/A')}")
                            st.write(f"Distance: {row.get('distance_km', 0):.1f} km")
                            st.write(f"Segment: {row.get('customer_segment', 'N/A')}")
                        
                        with col2:
                            st.write("**Recommended Actions:**")
                            
                            # Simple rule-based recommendations
                            if row['delay_probability'] > 0.8:
                                st.error("🚨 URGENT: Immediate action required")
                                st.write("- Contact carrier immediately")
                                st.write("- Notify customer of potential delay")
                                st.write("- Prepare alternative delivery options")
                            elif row['delay_probability'] > 0.6:
                                st.warning("⚠️ HIGH RISK: Monitor closely")
                                st.write("- Check carrier status")
                                st.write("- Alert logistics team")
                                st.write("- Prepare contingency plan")
                            else:
                                st.info("ℹ️ MODERATE RISK: Track progress")
                                st.write("- Monitor delivery status")
                                st.write("- Schedule follow-up check")


if __name__ == "__main__":
    main()
