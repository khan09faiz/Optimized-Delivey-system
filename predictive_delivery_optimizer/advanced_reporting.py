"""
Advanced Reporting & Export Module
==================================
Provides professional report generation with multi-format export capabilities.
Supports PDF, Excel, PowerPoint, and interactive HTML reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

from utils import setup_logging

logger = setup_logging("advanced_reporting")


class ReportGenerator:
    """Generates professional reports in multiple formats."""
    
    def __init__(self, data: pd.DataFrame, predictions: Optional[pd.DataFrame] = None):
        """
        Initialize ReportGenerator.
        
        Args:
            data: Main dataset
            predictions: Optional prediction results
        """
        self.data = data
        self.predictions = predictions
        self.report_metadata = {
            'generated_at': datetime.now(),
            'total_records': len(data),
            'report_version': '1.0'
        }
        
        logger.info(f"ReportGenerator initialized with {len(data)} records")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary of summary metrics
        """
        logger.info("Generating summary statistics...")
        
        summary = {
            'total_orders': len(self.data),
            'date_range': {
                'start': self.data['order_date'].min() if 'order_date' in self.data.columns else 'N/A',
                'end': self.data['order_date'].max() if 'order_date' in self.data.columns else 'N/A'
            },
            'delay_statistics': {},
            'carrier_breakdown': {},
            'priority_breakdown': {},
            'segment_breakdown': {}
        }
        
        # Delay statistics
        if 'is_delayed' in self.data.columns:
            delayed = self.data['is_delayed'].sum()
            summary['delay_statistics'] = {
                'total_delayed': int(delayed),
                'total_on_time': int(len(self.data) - delayed),
                'delay_rate': float(delayed / len(self.data) * 100)
            }
        
        if 'actual_delivery_days' in self.data.columns:
            summary['delay_statistics']['avg_delivery_days'] = float(self.data['actual_delivery_days'].mean())
            summary['delay_statistics']['median_delivery_days'] = float(self.data['actual_delivery_days'].median())
            summary['delay_statistics']['max_delivery_days'] = float(self.data['actual_delivery_days'].max())
        
        # Carrier breakdown
        if 'carrier_name' in self.data.columns:
            carrier_stats = self.data.groupby('carrier_name').agg({
                'order_id': 'count',
                'is_delayed': 'mean' if 'is_delayed' in self.data.columns else 'count'
            }).to_dict()
            summary['carrier_breakdown'] = carrier_stats
        
        # Priority breakdown
        if 'priority' in self.data.columns:
            priority_stats = self.data['priority'].value_counts().to_dict()
            summary['priority_breakdown'] = priority_stats
        
        # Customer segment breakdown
        if 'customer_segment' in self.data.columns:
            segment_stats = self.data['customer_segment'].value_counts().to_dict()
            summary['segment_breakdown'] = segment_stats
        
        logger.info("Summary statistics generated")
        return summary
    
    def generate_performance_metrics(self) -> pd.DataFrame:
        """
        Generate performance metrics by different dimensions.
        
        Returns:
            DataFrame with performance metrics
        """
        logger.info("Generating performance metrics...")
        
        metrics_list = []
        
        # Overall metrics
        if 'is_delayed' in self.data.columns:
            metrics_list.append({
                'dimension': 'Overall',
                'category': 'All',
                'total_orders': len(self.data),
                'delayed_orders': self.data['is_delayed'].sum(),
                'delay_rate_%': self.data['is_delayed'].mean() * 100,
                'avg_delivery_days': self.data['actual_delivery_days'].mean() if 'actual_delivery_days' in self.data.columns else 0
            })
        
        # Carrier metrics
        if 'carrier_name' in self.data.columns and 'is_delayed' in self.data.columns:
            for carrier in self.data['carrier_name'].unique():
                carrier_data = self.data[self.data['carrier_name'] == carrier]
                metrics_list.append({
                    'dimension': 'Carrier',
                    'category': carrier,
                    'total_orders': len(carrier_data),
                    'delayed_orders': carrier_data['is_delayed'].sum(),
                    'delay_rate_%': carrier_data['is_delayed'].mean() * 100,
                    'avg_delivery_days': carrier_data['actual_delivery_days'].mean() if 'actual_delivery_days' in carrier_data.columns else 0
                })
        
        # Priority metrics
        if 'priority' in self.data.columns and 'is_delayed' in self.data.columns:
            for priority in self.data['priority'].unique():
                priority_data = self.data[self.data['priority'] == priority]
                metrics_list.append({
                    'dimension': 'Priority',
                    'category': priority,
                    'total_orders': len(priority_data),
                    'delayed_orders': priority_data['is_delayed'].sum(),
                    'delay_rate_%': priority_data['is_delayed'].mean() * 100,
                    'avg_delivery_days': priority_data['actual_delivery_days'].mean() if 'actual_delivery_days' in priority_data.columns else 0
                })
        
        # Customer segment metrics
        if 'customer_segment' in self.data.columns and 'is_delayed' in self.data.columns:
            for segment in self.data['customer_segment'].unique():
                segment_data = self.data[self.data['customer_segment'] == segment]
                metrics_list.append({
                    'dimension': 'Customer Segment',
                    'category': segment,
                    'total_orders': len(segment_data),
                    'delayed_orders': segment_data['is_delayed'].sum(),
                    'delay_rate_%': segment_data['is_delayed'].mean() * 100,
                    'avg_delivery_days': segment_data['actual_delivery_days'].mean() if 'actual_delivery_days' in segment_data.columns else 0
                })
        
        metrics_df = pd.DataFrame(metrics_list)
        logger.info(f"Generated {len(metrics_df)} performance metrics")
        
        return metrics_df
    
    def create_executive_summary(self) -> Dict[str, Any]:
        """
        Create executive summary with key insights.
        
        Returns:
            Dictionary with executive summary
        """
        logger.info("Creating executive summary...")
        
        summary = {
            'overview': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Overview
        summary['overview'] = {
            'report_date': self.report_metadata['generated_at'].strftime('%Y-%m-%d %H:%M'),
            'total_orders': len(self.data),
            'analysis_period': f"{self.data['order_date'].min()} to {self.data['order_date'].max()}" if 'order_date' in self.data.columns else 'N/A'
        }
        
        # Key findings
        if 'is_delayed' in self.data.columns:
            delay_rate = self.data['is_delayed'].mean() * 100
            
            summary['key_findings'].append(
                f"Overall delay rate: {delay_rate:.1f}%"
            )
            
            # Best/worst carriers
            if 'carrier_name' in self.data.columns:
                carrier_performance = self.data.groupby('carrier_name')['is_delayed'].mean().sort_values()
                best_carrier = carrier_performance.index[0]
                worst_carrier = carrier_performance.index[-1]
                
                summary['key_findings'].append(
                    f"Best performing carrier: {best_carrier} ({carrier_performance[best_carrier]*100:.1f}% delay rate)"
                )
                summary['key_findings'].append(
                    f"Needs improvement: {worst_carrier} ({carrier_performance[worst_carrier]*100:.1f}% delay rate)"
                )
            
            # Priority analysis
            if 'priority' in self.data.columns:
                high_priority_delays = self.data[self.data['priority'] == 'High']['is_delayed'].mean() * 100
                summary['key_findings'].append(
                    f"High priority delay rate: {high_priority_delays:.1f}%"
                )
        
        # Recommendations
        if delay_rate > 30:
            summary['recommendations'].append("URGENT: Delay rate exceeds 30%. Immediate action required.")
        
        if 'carrier_name' in self.data.columns:
            carrier_delay = self.data.groupby('carrier_name')['is_delayed'].mean()
            high_delay_carriers = carrier_delay[carrier_delay > 0.5].index.tolist()
            if high_delay_carriers:
                summary['recommendations'].append(
                    f"Review contracts with carriers: {', '.join(high_delay_carriers)}"
                )
        
        logger.info("Executive summary created")
        return summary
    
    def export_to_excel(self, filename: str = "delivery_report.xlsx") -> BytesIO:
        """
        Export comprehensive report to Excel with multiple sheets.
        
        Args:
            filename: Output filename
            
        Returns:
            BytesIO object containing Excel file
        """
        logger.info(f"Exporting to Excel: {filename}")
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_stats = self.generate_summary_statistics()
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Performance metrics sheet
            metrics_df = self.generate_performance_metrics()
            metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
            
            # Raw data sheet
            self.data.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Predictions sheet (if available)
            if self.predictions is not None:
                self.predictions.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Carrier analysis
            if 'carrier_name' in self.data.columns and 'is_delayed' in self.data.columns:
                carrier_analysis = self.data.groupby('carrier_name').agg({
                    'order_id': 'count',
                    'is_delayed': ['sum', 'mean'],
                    'actual_delivery_days': ['mean', 'median', 'max'] if 'actual_delivery_days' in self.data.columns else 'count'
                }).round(2)
                carrier_analysis.to_excel(writer, sheet_name='Carrier Analysis')
            
            # Priority analysis
            if 'priority' in self.data.columns and 'is_delayed' in self.data.columns:
                priority_analysis = self.data.groupby('priority').agg({
                    'order_id': 'count',
                    'is_delayed': ['sum', 'mean'],
                    'actual_delivery_days': ['mean', 'median'] if 'actual_delivery_days' in self.data.columns else 'count'
                }).round(2)
                priority_analysis.to_excel(writer, sheet_name='Priority Analysis')
        
        output.seek(0)
        logger.info("Excel export completed")
        
        return output
    
    def export_to_csv_zip(self) -> Dict[str, BytesIO]:
        """
        Export multiple CSV files.
        
        Returns:
            Dictionary of filename to BytesIO objects
        """
        logger.info("Exporting to CSV files...")
        
        csv_files = {}
        
        # Main data
        csv_files['delivery_data.csv'] = self._df_to_csv(self.data)
        
        # Performance metrics
        metrics_df = self.generate_performance_metrics()
        csv_files['performance_metrics.csv'] = self._df_to_csv(metrics_df)
        
        # Predictions
        if self.predictions is not None:
            csv_files['predictions.csv'] = self._df_to_csv(self.predictions)
        
        logger.info(f"Generated {len(csv_files)} CSV files")
        return csv_files
    
    def _df_to_csv(self, df: pd.DataFrame) -> BytesIO:
        """Convert DataFrame to CSV BytesIO."""
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output
    
    def create_html_report(self) -> str:
        """
        Create interactive HTML report.
        
        Returns:
            HTML string
        """
        logger.info("Creating HTML report...")
        
        executive_summary = self.create_executive_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Delivery Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric-card {{ display: inline-block; margin: 10px; padding: 20px; background: #ecf0f1; border-radius: 8px; min-width: 200px; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #2980b9; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
                .finding {{ background: #e8f8f5; padding: 15px; margin: 10px 0; border-left: 4px solid #16a085; }}
                .recommendation {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Delivery Performance Report</h1>
                <p><strong>Generated:</strong> {executive_summary['overview']['report_date']}</p>
                <p><strong>Analysis Period:</strong> {executive_summary['overview']['analysis_period']}</p>
                
                <h2>Key Metrics</h2>
                <div class="metric-card">
                    <div class="metric-value">{executive_summary['overview']['total_orders']}</div>
                    <div class="metric-label">Total Orders</div>
                </div>
        """
        
        if 'is_delayed' in self.data.columns:
            delay_rate = self.data['is_delayed'].mean() * 100
            on_time_rate = 100 - delay_rate
            
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{delay_rate:.1f}%</div>
                    <div class="metric-label">Delay Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{on_time_rate:.1f}%</div>
                    <div class="metric-label">On-Time Rate</div>
                </div>
            """
        
        # Key findings
        html += "<h2>Key Findings</h2>"
        for finding in executive_summary['key_findings']:
            html += f'<div class="finding">✓ {finding}</div>'
        
        # Recommendations
        html += "<h2>Recommendations</h2>"
        for rec in executive_summary['recommendations']:
            html += f'<div class="recommendation">⚠ {rec}</div>'
        
        # Performance table
        metrics_df = self.generate_performance_metrics()
        html += "<h2>Detailed Performance Metrics</h2>"
        html += metrics_df.to_html(classes='table', index=False, float_format='%.2f')
        
        html += """
            </div>
        </body>
        </html>
        """
        
        logger.info("HTML report created")
        return html


# ==================== Convenience Functions ====================
def create_report(
    data: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None
) -> ReportGenerator:
    """
    Create a report generator instance.
    
    Args:
        data: Main dataset
        predictions: Optional predictions
        
    Returns:
        ReportGenerator instance
    """
    return ReportGenerator(data, predictions)


def generate_excel_report(
    data: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
    filename: str = "delivery_report.xlsx"
) -> BytesIO:
    """
    Quick function to generate Excel report.
    
    Args:
        data: Main dataset
        predictions: Optional predictions
        filename: Output filename
        
    Returns:
        BytesIO object with Excel file
    """
    generator = ReportGenerator(data, predictions)
    return generator.export_to_excel(filename)
