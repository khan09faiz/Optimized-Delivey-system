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
from customer_insights import CustomerInsights, OrderTracker, get_customer_dashboard_data
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
    if 'show_predictions' not in st.session_state:
        st.session_state.show_predictions = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
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
    
    # Cross-validation option
    use_cv = st.sidebar.checkbox("Use Cross-Validation", value=True, 
                                  help="Recommended to prevent overfitting")
    use_smote = st.sidebar.checkbox("Handle Class Imbalance (SMOTE)", value=True,
                                     help="Balance classes if data is skewed")
    
    if st.sidebar.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            try:
                # Get feature columns (only numeric)
                feature_cols = engineer.get_feature_columns(filtered_df, target_col='delay_flag')
                
                # Prepare data
                X = filtered_df[feature_cols].fillna(0)
                y = filtered_df['delay_flag']
                
                # Create trainer
                trainer = ModelTrainer()
                
                # Check class imbalance
                imbalance_info = trainer.check_class_imbalance(y)
                st.sidebar.info(f"📊 Class Distribution:\n"
                               f"Class 0: {imbalance_info['class_percentages'][0]:.1f}%\n"
                               f"Class 1: {imbalance_info['class_percentages'][1]:.1f}%\n"
                               f"Ratio: {imbalance_info['imbalance_ratio']:.2f}")
                
                if use_cv:
                    # Cross-validation approach
                    st.sidebar.write("Running 5-fold cross-validation...")
                    
                    cv_results = trainer.train_with_cross_validation_detailed(
                        X, y, 
                        model_type='RandomForest',
                        n_splits=5,
                        use_smote=use_smote
                    )
                    
                    st.sidebar.success(f"✅ CV Mean ROC-AUC: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
                    
                    # Store CV results
                    st.session_state.cv_results = cv_results
                    
                    # Train final model on full data
                    if use_smote and imbalance_info['is_imbalanced']:
                        X_balanced, y_balanced = trainer.apply_smote(X, y)
                    else:
                        X_balanced, y_balanced = X, y
                    
                    # Split for evaluation
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
                    )
                else:
                    # Traditional train/test split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    if use_smote:
                        X_train, y_train = trainer.apply_smote(X_train, y_train)
                
                # Train final models
                results = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
                
                # Store in session
                st.session_state.trainer = trainer
                st.session_state.feature_cols = feature_cols
                st.session_state.imbalance_info = imbalance_info
                
                # Set flag to switch to predictions tab
                st.session_state.show_predictions = True
                
                st.sidebar.success("✅ Models trained successfully! Switching to Predictions...")
                
                # Force a rerun to trigger tab switch
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Training error: {str(e)}")
                logger.error(f"Training error: {str(e)}", exc_info=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🎯 Predictions", 
        "💡 Recommendations",
        "📈 Trends & Analysis",
        "🔍 Anomaly Detection",
        "👤 Customer Insights & Tracking"
    ])
    
    # Auto-switch to Predictions tab after training
    if st.session_state.get('show_predictions', False):
        st.session_state.show_predictions = False  # Reset flag
        st.markdown("""
        <script>
            // Find and click the Predictions tab
            const tabs = parent.document.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs.length >= 2) {
                tabs[1].click();  // Index 1 is the Predictions tab
            }
        </script>
        """, unsafe_allow_html=True)
    
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
        
        # Row 1: Core Metrics
        st.subheader("📊 Delivery Performance Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Delay Distribution**")
            fig = viz.plot_delay_distribution(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_delay_dist")
        
        with col2:
            st.write("**Carrier Performance**")
            fig = viz.plot_carrier_performance(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_carrier_perf")
        
        # Row 2: Priority and Distance
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Priority Analysis**")
            fig = viz.plot_priority_analysis(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_priority")
        
        with col4:
            st.write("**Distance vs Delay**")
            fig = viz.plot_distance_vs_delay(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_distance")
        
        st.markdown("---")
        
        # Row 3: New Visualizations
        st.subheader("💰 Order Value & Customer Insights")
        col5, col6 = st.columns(2)
        with col5:
            st.write("**Order Value Distribution**")
            fig = viz.plot_order_value_distribution(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_order_value")
        
        with col6:
            st.write("**Customer Segment Performance**")
            fig = viz.plot_customer_segment_analysis(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_customer_segment")
        
        st.markdown("---")
        
        # Row 4: Operational Factors
        st.subheader("🚦 Operational Impact Analysis")
        col7, col8 = st.columns(2)
        with col7:
            st.write("**Traffic Impact on Deliveries**")
            fig = viz.plot_traffic_impact(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_traffic")
        
        with col8:
            st.write("**Weather Impact**")
            fig = viz.plot_weather_impact(filtered_df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_weather")
        
        # Row 5: Delivery Time Analysis
        st.markdown("---")
        st.subheader("⏱️ Delivery Time Performance")
        fig = viz.plot_delivery_time_analysis(filtered_df)
        st.plotly_chart(fig, use_container_width=True, key="dashboard_delivery_time")
        
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
            
            # Show Model Validation Info
            with st.expander("📊 Model Validation & Class Balance Info", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Class Distribution**")
                    if 'imbalance_info' in st.session_state:
                        imb_info = st.session_state.imbalance_info
                        st.metric("Imbalance Ratio", f"{imb_info['imbalance_ratio']:.2f}")
                        st.write(f"Class 0: {imb_info['class_percentages'][0]:.1f}%")
                        st.write(f"Class 1: {imb_info['class_percentages'][1]:.1f}%")
                        
                        if imb_info['is_imbalanced']:
                            st.warning("⚠️ Dataset is imbalanced")
                        else:
                            st.success("✅ Dataset is balanced")
                
                with col2:
                    st.write("**Cross-Validation Results**")
                    if 'cv_results' in st.session_state:
                        cv_res = st.session_state.cv_results
                        st.metric("Mean ROC-AUC", 
                                f"{cv_res['mean_score']:.4f} ± {cv_res['std_score']:.4f}")
                        
                        # Show fold scores
                        fold_scores = cv_res['fold_scores']
                        st.write("Fold Scores:")
                        for i, score in enumerate(fold_scores, 1):
                            st.write(f"  Fold {i}: {score:.4f}")
                        
                        if cv_res['std_score'] > 0.1:
                            st.warning("⚠️ High variance - possible overfitting")
                        else:
                            st.success("✅ Good model stability")
                    else:
                        st.info("Cross-validation not performed")
            
            st.markdown("---")
            
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
                        
                        # Feature Impact Overview
                        st.subheader("📊 Feature Impact Analysis")
                        
                        # Simple clean visualization
                        fig_impact = explainer.plot_feature_impact_simple(X_test, top_n=15)
                        st.plotly_chart(fig_impact, use_container_width=True, key="shap_impact_simple")
                        
                        # Detailed views in expander
                        with st.expander("🔍 Show Detailed SHAP Visualizations"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Bar Chart View**")
                                fig_bar = explainer.plot_shap_bar_interactive(X_test, top_n=15)
                                st.plotly_chart(fig_bar, use_container_width=True, key="shap_bar_chart")
                            
                            with col2:
                                st.write("**Summary Plot**")
                                fig_summary = explainer.plot_shap_summary_interactive(X_test, max_display=15)
                                st.plotly_chart(fig_summary, use_container_width=True, key="shap_summary_chart")
                        
                        # Feature dependence
                        st.subheader("🔗 Feature Dependence Analysis")
                        st.caption("See how each feature value affects predictions")
                        
                        importance_df = explainer.get_global_importance(X_test, top_n=10)
                        selected_feature = st.selectbox(
                            "Select Feature to Analyze",
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
    
    # ==================== Tab 6: Customer Insights & Tracking ====================
    with tab6:
        st.header("👤 Customer Insights & Order Tracking")
        st.markdown("**Personalized recommendations and real-time order tracking for customers**")
        
        # Order Selection
        st.subheader("Select Your Order")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            order_options = filtered_raw_df['order_id'].unique() if 'order_id' in filtered_raw_df.columns else []
            selected_order_id = st.selectbox(
                "Choose Order ID",
                options=order_options,
                help="Select an order to view detailed insights and tracking"
            )
        
        with col2:
            st.info(f"**{len(order_options)}** orders available")
        
        if selected_order_id:
            # Get order data
            order_data = filtered_raw_df[filtered_raw_df['order_id'] == selected_order_id].iloc[0]
            
            # Add delay probability if predictions exist
            if 'trainer' in st.session_state:
                order_features = filtered_df[filtered_df.index == order_data.name][st.session_state.feature_cols].fillna(0)
                _, probabilities = st.session_state.trainer.predict(order_features)
                order_data['delay_probability'] = probabilities[0]
            else:
                order_data['delay_probability'] = 0.0
            
            # Get comprehensive insights
            insights_gen = CustomerInsights()
            tracker = OrderTracker()
            
            st.markdown("---")
            
            # ============ Section 1: Order Status Overview ============
            st.subheader("📦 Order Status Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Order ID", selected_order_id)
            col2.metric("Carrier", order_data.get('carrier', 'N/A'))
            col3.metric("Priority", order_data.get('priority', 'Standard'))
            
            delay_prob = order_data.get('delay_probability', 0)
            risk_cat = get_risk_category(delay_prob)
            risk_color = '🔴' if risk_cat == 'High' else '🟡' if risk_cat == 'Moderate' else '🟢'
            col4.metric("Delay Risk", f"{risk_color} {risk_cat}", f"{delay_prob:.1%}")
            
            st.markdown("---")
            
            # ============ Section 2: Real-Time Tracking ============
            st.subheader("🗺️ Live Order Tracking")
            
            tab_tracking, tab_timeline = st.tabs(["📍 Map View", "⏱️ Timeline"])
            
            with tab_tracking:
                # Show simulated map
                fig_map = tracker.plot_real_time_location(order_data)
                st.plotly_chart(fig_map, use_container_width=True, key="tracking_map")
                
                # Tracking insights
                timeline = tracker.get_order_timeline(order_data)
                tracking_insights = tracker.get_tracking_insights(order_data, timeline)
                
                st.markdown("### 📊 Tracking Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Current Stage:** {tracking_insights['current_stage']}")
                    st.success(f"**Next Action:** {tracking_insights['next_action']}")
                
                with col2:
                    est_delivery = tracking_insights.get('estimated_delivery')
                    if est_delivery:
                        st.warning(f"**Estimated Delivery:** {est_delivery.strftime('%b %d, %I:%M %p')}")
                    
                    risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                    st.metric("Delay Risk Level", f"{risk_emoji.get(tracking_insights['delay_risk'], '🟢')} {tracking_insights['delay_risk'].upper()}")
                
                # Alerts
                if tracking_insights['alerts']:
                    st.markdown("### 🔔 Active Alerts")
                    for alert in tracking_insights['alerts']:
                        if '⚠️' in alert or '🚨' in alert:
                            st.warning(alert)
                        elif '✅' in alert:
                            st.success(alert)
                        else:
                            st.info(alert)
            
            with tab_timeline:
                # Show timeline
                fig_timeline = tracker.plot_order_timeline(timeline)
                st.plotly_chart(fig_timeline, use_container_width=True, key="order_timeline")
                
                # Timeline legend
                st.markdown("""
                **Timeline Status:**
                - ✓ **Completed** - Stage finished
                - ⏳ **In Progress** - Currently at this stage
                - ○ **Pending** - Upcoming stage
                """)
            
            st.markdown("---")
            
            # ============ Section 3: Improvement Opportunities ============
            st.subheader("🎯 How to Improve Your Delivery Success")
            
            improvement_data = insights_gen.get_order_improvement_score(order_data)
            
            # Score card
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Your Score", f"{improvement_data['current_score']}/100", 
                       help="Based on order characteristics and past behavior")
            col2.metric("Grade", improvement_data['grade'],
                       help="A+ = Excellent, A = Good, B = Average, C = Needs Improvement")
            col3.metric("Improvement Potential", f"+{improvement_data['improvement_potential']}%",
                       help="How much you can improve delivery success")
            
            # Improvement chart
            if improvement_data['factors']:
                st.markdown("### 📈 What You Can Improve")
                fig_improvement = insights_gen.plot_improvement_opportunities(improvement_data)
                st.plotly_chart(fig_improvement, use_container_width=True, key="improvement_chart")
                
                # Detailed recommendations
                st.markdown("### 💡 Actionable Steps")
                for i, factor in enumerate(improvement_data['factors'], 1):
                    with st.expander(f"{i}. {factor['factor']} (Impact: {abs(factor['impact'])}%)"):
                        st.write(f"**What to do:** {factor['fix']}")
                        st.progress(abs(factor['impact']) / 100)
            else:
                st.success("🌟 Perfect! You're already following all best practices!")
            
            st.markdown("---")
            
            # ============ Section 4: Best Practices Guide ============
            st.subheader("📚 Personalized Best Practices")
            st.caption("Tips customized for your order")
            
            best_practices = insights_gen.get_customer_best_practices(order_data)
            
            tab_before, tab_during, tab_communication, tab_general = st.tabs([
                "📅 Before Ordering",
                "📦 During Delivery", 
                "💬 Communication",
                "🌟 General Tips"
            ])
            
            with tab_before:
                st.markdown("### Things to Consider Before Placing Orders")
                for tip in best_practices['before_ordering']:
                    st.markdown(f"- {tip}")
            
            with tab_during:
                st.markdown("### What to Do While Your Order is Being Delivered")
                for tip in best_practices['during_delivery']:
                    st.markdown(f"- {tip}")
            
            with tab_communication:
                st.markdown("### How to Communicate Effectively")
                for tip in best_practices['communication']:
                    st.markdown(f"- {tip}")
            
            with tab_general:
                st.markdown("### General Best Practices")
                for tip in best_practices['general_tips']:
                    st.markdown(f"- {tip}")
            
            st.markdown("---")
            
            # ============ Section 5: Quick Reference Card ============
            st.subheader("📋 Quick Reference Card")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📞 Customer Support**")
                st.info("""
                - **Carrier:** {carrier}
                - **Tracking ID:** {order_id}
                - **Support:** 1800-XXX-XXXX
                - **Email:** support@carrier.com
                """.format(
                    carrier=order_data.get('carrier', 'N/A'),
                    order_id=selected_order_id
                ))
            
            with col2:
                st.markdown("**📦 Delivery Details**")
                st.info(f"""
                - **Distance:** {order_data.get('distance_km', 0):.1f} km
                - **Priority:** {order_data.get('priority', 'Standard')}
                - **Value:** {format_currency(order_data.get('order_value', 0))}
                - **Segment:** {order_data.get('customer_segment', 'N/A')}
                """)
            
            with col3:
                st.markdown("**⚡ Quick Actions**")
                if st.button("🔔 Enable Notifications", key="enable_notif"):
                    st.success("✅ Notifications enabled!")
                
                if st.button("📞 Call Carrier", key="call_carrier"):
                    st.info(f"📞 Dialing {order_data.get('carrier', 'carrier')} support...")
                
                if st.button("📧 Email Updates", key="email_updates"):
                    st.success("✅ Email alerts activated!")
            
            st.markdown("---")
            
            # ============ Section 6: Order Summary ============
            with st.expander("📊 Complete Order Summary"):
                order_summary_df = pd.DataFrame([{
                    'Field': key,
                    'Value': str(value)
                } for key, value in order_data.items() if not key.startswith('_')])
                
                st.dataframe(order_summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
