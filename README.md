# Optimized Delivery System

A comprehensive **Predictive Delivery Optimizer** built with Python 3.10+ and Streamlit. This application analyzes 7 interconnected datasets to provide insights, predictions, and optimization recommendations for delivery operations.

## 🚀 Features

- **Multi-Dataset Analysis**: Analyzes 7 interconnected datasets:
  - `orders.csv` - Order information
  - `delivery_performance.csv` - Delivery metrics and performance
  - `routes_distance.csv` - Route and distance data
  - `vehicle_fleet.csv` - Fleet management data
  - `warehouse_inventory.csv` - Warehouse and inventory information
  - `customer_feedback.csv` - Customer ratings and feedback
  - `cost_breakdown.csv` - Cost analysis data

- **Predictive Modeling**: Machine learning models for delivery delay and cost prediction
- **Feature Engineering**: Automated feature extraction and engineering from raw data
- **Model Explainability**: Interpretable ML with feature importance and model explanations
- **Optimization Recommendations**: AI-powered recommendations for:
  - Route optimization
  - Fleet management
  - Cost reduction
  - Customer satisfaction improvement
- **Interactive Dashboard**: Streamlit-based UI with visualizations and insights

## 📁 Architecture

```
/predictive_delivery_optimizer
├── data_loader.py              # Data loading and validation
├── feature_engineering.py       # Feature extraction and preprocessing
├── model_training.py           # ML model training and evaluation
├── explainability.py           # Model interpretation
├── recommendation_engine.py    # Optimization recommendations
├── visualization.py            # Data visualizations
├── app.py                      # Streamlit application
└── utils.py                    # Utility functions
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/khan09faiz/Optimized-Delivey-system.git
cd Optimized-Delivey-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create sample data:
The application will automatically generate sample datasets if CSV files are not found in the `data/` directory.

## 🚀 Usage

### Running the Streamlit Application

```bash
streamlit run predictive_delivery_optimizer/app.py
```

The application will be available at `http://localhost:8501`

### Using the Python API

```python
from predictive_delivery_optimizer import DataLoader, FeatureEngineer, ModelTrainer

# Load data
loader = DataLoader(data_dir='data')
datasets = loader.load_all_datasets()

# Feature engineering
engineer = FeatureEngineer(datasets)
X, y = engineer.prepare_features_for_modeling()

# Train models
trainer = ModelTrainer()
trainer.train_models(X, y)

# Get best model
best_model_name, best_model = trainer.get_best_model()
```

## 📊 Dashboard Features

### 1. Dashboard Overview
- Key performance metrics
- Delivery performance charts
- Cost breakdown visualization
- Customer feedback analysis
- Fleet status overview

### 2. Data Analysis
- Explore individual datasets
- Statistical summaries
- Data quality metrics
- Interactive visualizations

### 3. Model Performance
- Model comparison and metrics
- Feature importance analysis
- Predictions vs actual values
- Model explainability insights

### 4. Recommendations
- Prioritized optimization recommendations
- Route optimization suggestions
- Fleet management improvements
- Cost reduction strategies
- Customer satisfaction enhancements

### 5. Dataset Info
- Dataset schemas
- Data quality metrics
- Column descriptions

## 🔧 Configuration

### Data Directory
By default, the application looks for CSV files in the `data/` directory. You can change this by:

```python
loader = DataLoader(data_dir='your/custom/path')
```

### Model Parameters
Modify model parameters in `model_training.py`:

```python
self.models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}
```

## 📈 Sample Data

If you don't have your own datasets, the application will automatically generate sample data with realistic patterns:

- 100 orders
- 100 delivery records
- 30 routes
- 50 vehicles
- 20 warehouses
- 100 customer feedback entries
- 100 cost records

## 🧪 Testing

The application includes built-in validation and error handling. To test the system:

1. Run the application with sample data
2. Navigate through all dashboard pages
3. Verify that visualizations load correctly
4. Check that models train successfully
5. Review generated recommendations

## 📝 Requirements

- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **scikit-learn** >= 1.3.0 - Machine learning
- **streamlit** >= 1.28.0 - Web interface
- **plotly** >= 5.17.0 - Interactive visualizations
- **joblib** >= 1.3.0 - Model persistence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

Delivery Optimizer Team

## 🔗 Links

- [GitHub Repository](https://github.com/khan09faiz/Optimized-Delivey-system)
- [Issue Tracker](https://github.com/khan09faiz/Optimized-Delivey-system/issues)

## 🙏 Acknowledgments

- Built with Streamlit
- Machine learning powered by scikit-learn
- Visualizations created with Plotly