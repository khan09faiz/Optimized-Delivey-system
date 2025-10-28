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
from explainability import ExplainabilityAnalyzer
from trend_analysis import TrendAnalyzer
from advanced_reporting import ReportGenerator
from anomaly_detection import AnomalyDetector
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🎯 Predictions", 
        "💡 Recommendations",
        "📈 Trends & Analysis",
        "🔍 Anomaly Detection"
    ])
    
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
            st.plotly_chart(fig, use_container_width=True, key="dashboard_delay_dist")
        
        with col2:
            st.subheader("Carrier Performance")
            fig = viz.plot_carrier_performance(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_carrier_perf")
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Priority Analysis")
            fig = viz.plot_priority_analysis(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_priority")
        
        with col4:
            st.subheader("Distance vs Delay")
            fig = viz.plot_distance_vs_delay(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_distance")
        
        # Report Export Section
        st.markdown("---")
        st.subheader("📊 Advanced Reporting & Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export to Excel"):
                try:
                    # Create report generator
                    report_gen = ReportGenerator(filtered_raw_df)
                    excel_file = report_gen.export_to_excel()
                    
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_file,
                        file_name=f"delivery_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("✅ Excel report generated!")
                except Exception as e:
                    st.error(f"Error generating Excel: {str(e)}")
        
        with col2:
            if st.button("📄 Generate HTML Report"):
                try:
                    report_gen = ReportGenerator(filtered_raw_df)
                    html_report = report_gen.create_html_report()
                    
                    st.download_button(
                        label="Download HTML Report",
                        data=html_report,
                        file_name=f"delivery_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html"
                    )
                    st.success("✅ HTML report generated!")
                except Exception as e:
                    st.error(f"Error generating HTML: {str(e)}")
        
        with col3:
            if st.button("📋 View Executive Summary"):
                try:
                    report_gen = ReportGenerator(filtered_raw_df)
                    exec_summary = report_gen.create_executive_summary()
                    
                    st.json(exec_summary)
                except Exception as e:
                    st.error(f"Error creating summary: {str(e)}")
    
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
            
            # Enhanced SHAP Explainability
            st.markdown("---")
            st.subheader("🎯 Model Explainability (SHAP)")
            
            if st.checkbox("Show SHAP Analysis"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        # Get train/test split
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, _, _ = train_test_split(
                            X, filtered_df['delay_flag'], 
                            test_size=0.2, random_state=42, 
                            stratify=filtered_df['delay_flag']
                        )
                        
                        # Create explainer
                        explainer = ExplainabilityAnalyzer(
                            trainer.best_model, 
                            X_train
                        )
                        explainer.create_explainer()
                        explainer.compute_shap_values(X_test)
                        
                        # Store in session
                        st.session_state.explainer = explainer
                        st.session_state.X_test = X_test
                        
                        # Global importance
                        st.subheader("Global Feature Importance")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Interactive Bar Chart**")
                            fig_bar = explainer.plot_shap_bar_interactive(X_test, top_n=15)
                            st.plotly_chart(fig_bar, use_container_width=True, key="shap_bar_chart")
                        
                        with col2:
                            st.write("**Summary Plot**")
                            fig_summary = explainer.plot_shap_summary_interactive(X_test, max_display=15)
                            st.plotly_chart(fig_summary, use_container_width=True, key="shap_summary_chart")
                        
                        # Instance-level explanation
                        st.subheader("Individual Prediction Explanation")
                        
                        instance_idx = st.number_input(
                            "Select Instance Index", 
                            0, len(X_test)-1, 0
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Waterfall Plot**")
                            fig_waterfall = explainer.plot_shap_waterfall_interactive(X_test, instance_idx)
                            st.plotly_chart(fig_waterfall, use_container_width=True, key="shap_waterfall_chart")
                        
                        with col2:
                            st.write("**Force Plot**")
                            fig_force = explainer.plot_shap_force_interactive(X_test, instance_idx)
                            st.plotly_chart(fig_force, use_container_width=True, key="shap_force_chart")
                        
                        # Feature dependence
                        st.subheader("Feature Dependence Analysis")
                        
                        importance_df = explainer.get_global_importance(X_test, top_n=10)
                        selected_feature = st.selectbox(
                            "Select Feature for Dependence Analysis",
                            importance_df['feature'].tolist()
                        )
                        
                        fig_dependence = explainer.plot_shap_dependence_interactive(
                            X_test, 
                            selected_feature
                        )
                        st.plotly_chart(fig_dependence, use_container_width=True, key="shap_dependence_chart")
                        
                    except Exception as e:
                        st.error(f"Error in SHAP analysis: {str(e)}")
                        logger.error(f"SHAP analysis error: {str(e)}", exc_info=True)
    
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
    
    # ==================== Tab 4: Trends & Analysis ====================
    with tab4:
        st.header("Historical Trend Analysis")
        
        if 'order_date' in filtered_raw_df.columns:
            try:
                # Initialize trend analyzer
                trend_analyzer = TrendAnalyzer(filtered_raw_df)
                
                # Time trends
                st.subheader("Delivery Performance Trends Over Time")
                
                time_period = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"])
                
                if time_period == "Daily":
                    trends = trend_analyzer.analyze_time_trends(period='daily')
                    fig = trend_analyzer.plot_delay_trend(trends)
                elif time_period == "Weekly":
                    trends = trend_analyzer.analyze_time_trends(period='weekly')
                    fig = trend_analyzer.plot_delay_trend(trends)
                else:
                    trends = trend_analyzer.analyze_time_trends(period='monthly')
                    fig = trend_analyzer.plot_delay_trend(trends)
                
                st.plotly_chart(fig, use_container_width=True, key="trends_delay_chart")
                
                # Carrier trends
                st.subheader("Carrier Performance Trends")
                carrier_trends = trend_analyzer.analyze_carrier_trends()
                fig_carrier = trend_analyzer.plot_carrier_performance_trends(carrier_trends)
                st.plotly_chart(fig_carrier, use_container_width=True, key="trends_carrier_chart")
                
                # Trend summary
                st.subheader("Trend Summary")
                summary = trend_analyzer.get_trend_summary(trends)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Overall Trend",
                        summary['trend_direction'],
                        f"{summary['improvement_rate']:.2f}% improvement" if summary['improvement_rate'] > 0 else "No improvement"
                    )
                
                with col2:
                    st.metric(
                        "Current Delay Rate",
                        f"{summary['current_delay_rate']*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Average Delay Rate",
                        f"{summary['avg_delay_rate']*100:.1f}%"
                    )
                
                # Additional trend insights
                st.subheader("Trend Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    if summary['best_day'] is not None:
                        st.success(f"✅ **Best Day**: {summary['best_day']}")
                    else:
                        st.info("No best day data available")
                
                with col2:
                    if summary['worst_day'] is not None:
                        st.warning(f"⚠️ **Worst Day**: {summary['worst_day']}")
                    else:
                        st.info("No worst day data available")
                
            except Exception as e:
                st.error(f"Error in trend analysis: {str(e)}")
                logger.error(f"Trend analysis error: {str(e)}", exc_info=True)
        else:
            st.warning("Order date information not available for trend analysis")
    
    # ==================== Tab 5: Anomaly Detection ====================
    with tab5:
        st.header("Anomaly Detection")
        
        try:
            # Initialize anomaly detector
            contamination = st.slider("Expected Anomaly Rate", 0.01, 0.5, 0.1, 0.01)
            detector = AnomalyDetector(filtered_df, contamination=contamination)
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    # Comprehensive anomaly detection
                    anomalies = detector.get_comprehensive_anomalies()
                    
                    # Store in session
                    st.session_state.anomalies = anomalies
                    st.session_state.detector = detector
                    
                    st.success(f"✅ Detection complete! Found {anomalies['is_anomalous'].sum()} anomalies")
            
            if 'anomalies' in st.session_state:
                anomalies = st.session_state.anomalies
                detector = st.session_state.detector
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Anomalies", f"{anomalies['is_anomalous'].sum():,}")
                col2.metric("ML-Based", f"{anomalies.get('is_anomaly', pd.Series([False])).sum():,}")
                col3.metric("Delivery Time", f"{anomalies.get('has_delivery_anomaly', pd.Series([False])).sum():,}")
                col4.metric("Pattern", f"{anomalies.get('has_pattern_anomaly', pd.Series([False])).sum():,}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Anomaly Overview")
                    fig_overview = detector.plot_anomaly_overview(anomalies)
                    st.plotly_chart(fig_overview, use_container_width=True, key="anomaly_overview_chart")
                
                with col2:
                    st.subheader("Anomaly Score Distribution")
                    fig_scores = detector.plot_anomaly_scores()
                    st.plotly_chart(fig_scores, use_container_width=True, key="anomaly_scores_chart")
                
                # Carrier anomaly analysis
                st.subheader("Anomalies by Carrier")
                fig_carrier = detector.plot_anomalies_by_carrier(anomalies)
                st.plotly_chart(fig_carrier, use_container_width=True, key="anomaly_carrier_chart")
                
                # Show critical anomalies
                st.subheader("Critical Anomalies Requiring Attention")
                
                critical_mask = (
                    (anomalies['anomaly_severity'] == 'CRITICAL') |
                    (anomalies['anomaly_count'] >= 2)
                )
                
                critical_anomalies = filtered_raw_df[critical_mask].copy()
                critical_anomalies['anomaly_count'] = anomalies.loc[critical_mask, 'anomaly_count'].values
                critical_anomalies['severity'] = anomalies.loc[critical_mask, 'anomaly_severity'].values
                
                if len(critical_anomalies) > 0:
                    st.error(f"⚠️ {len(critical_anomalies)} critical anomalies detected")
                    
                    display_cols = ['order_id', 'carrier', 'priority', 'distance_km', 
                                   'actual_delivery_days', 'anomaly_count', 'severity']
                    display_cols = [c for c in display_cols if c in critical_anomalies.columns]
                    
                    st.dataframe(critical_anomalies[display_cols], use_container_width=True)
                else:
                    st.success("✅ No critical anomalies found")
                
        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")
            logger.error(f"Anomaly detection error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
