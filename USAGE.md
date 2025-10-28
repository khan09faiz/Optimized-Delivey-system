# Usage Guide - Predictive Delivery Optimizer

This guide will help you get started with the Predictive Delivery Optimizer application.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/khan09faiz/Optimized-Delivey-system.git
cd Optimized-Delivey-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

**Option A: Using the run script (Recommended)**
```bash
./run.sh
```

**Option B: Direct command**
```bash
streamlit run predictive_delivery_optimizer/app.py
```

The application will automatically open in your browser at `http://localhost:8501`

### 3. Running Tests

To verify the system is working correctly:
```bash
python test_system.py
```

## Using Your Own Data

### Data Format

The application expects 7 CSV files in the `data/` directory:

#### 1. orders.csv
Required columns:
- `order_id`: Unique order identifier
- `customer_id`: Customer identifier
- `order_date`: Date order was placed
- `delivery_date`: Date order was/will be delivered
- `origin`: Origin location
- `destination`: Destination location
- `weight`: Package weight in kg
- `priority`: Order priority (High/Medium/Low)

#### 2. delivery_performance.csv
Required columns:
- `delivery_id`: Unique delivery identifier
- `order_id`: Associated order ID
- `actual_delivery_time`: Actual delivery time in hours
- `scheduled_delivery_time`: Scheduled delivery time in hours
- `status`: Delivery status (Delivered/In Transit/Delayed)
- `delay_minutes`: Delay in minutes (negative means early)

#### 3. routes_distance.csv
Required columns:
- `route_id`: Unique route identifier
- `origin`: Origin location
- `destination`: Destination location
- `distance_km`: Distance in kilometers
- `estimated_time_hours`: Estimated travel time in hours
- `traffic_level`: Traffic level (Low/Medium/High)

#### 4. vehicle_fleet.csv
Required columns:
- `vehicle_id`: Unique vehicle identifier
- `vehicle_type`: Type of vehicle (Truck/Van/Motorcycle)
- `capacity_kg`: Vehicle capacity in kg
- `fuel_efficiency`: Fuel efficiency (km per liter)
- `maintenance_status`: Maintenance status (Good/Fair/Needs Service)
- `availability`: Availability status (Available/In Use/Maintenance)

#### 5. warehouse_inventory.csv
Required columns:
- `warehouse_id`: Unique warehouse identifier
- `location`: Warehouse location
- `capacity`: Total capacity
- `current_stock`: Current stock level
- `product_type`: Type of products stored

#### 6. customer_feedback.csv
Required columns:
- `feedback_id`: Unique feedback identifier
- `order_id`: Associated order ID
- `customer_id`: Customer identifier
- `rating`: Overall rating (1-5)
- `delivery_rating`: Delivery-specific rating (1-5)
- `comments`: Customer comments

#### 7. cost_breakdown.csv
Required columns:
- `cost_id`: Unique cost identifier
- `order_id`: Associated order ID
- `fuel_cost`: Fuel cost
- `labor_cost`: Labor cost
- `maintenance_cost`: Maintenance cost
- `total_cost`: Total cost

### Loading Your Data

1. Create a `data/` directory in the project root
2. Place your CSV files in the `data/` directory
3. Ensure files follow the naming convention and have required columns
4. Run the application

If files are missing, the system will automatically generate sample data.

## Application Features

### Dashboard Page

The main dashboard displays:
- Key performance metrics (deliveries, on-time rate, delays, ratings)
- Delivery performance visualizations
- Cost breakdown charts
- Customer feedback analysis
- Fleet status overview

### Data Analysis Page

Explore individual datasets:
- Select any dataset from the dropdown
- View data preview and statistics
- Interactive visualizations specific to each dataset
- Missing value analysis

### Model Performance Page

Machine learning insights:
- Comparison of different ML models (Random Forest, Gradient Boosting, Linear Regression)
- Feature importance analysis
- Model accuracy metrics (MAE, MSE, RMSE, R²)
- Predictions vs actual values visualization
- Model explainability information

### Recommendations Page

AI-powered optimization recommendations:
- **Route Optimizations**: Suggestions for improving delivery routes
- **Fleet Optimizations**: Fleet management and maintenance recommendations
- **Cost Optimizations**: Strategies to reduce operational costs
- **Customer Satisfaction**: Actions to improve customer experience

Recommendations are prioritized as High, Medium, or Low priority.

### Dataset Info Page

View metadata about loaded datasets:
- Number of rows and columns
- Missing values count
- Memory usage
- Required schema for each dataset

## Programming Interface

You can also use the application programmatically:

```python
from predictive_delivery_optimizer import (
    DataLoader,
    FeatureEngineer,
    ModelTrainer,
    RecommendationEngine,
    Visualizer
)

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
print(f"Best model: {best_model_name}")

# Generate recommendations
rec_engine = RecommendationEngine(datasets)
recommendations = rec_engine.generate_all_recommendations()

# Create visualizations
visualizer = Visualizer(datasets)
fig = visualizer.plot_delivery_performance()
```

## Customization

### Changing Model Parameters

Edit `predictive_delivery_optimizer/model_training.py`:

```python
self.models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200,  # Increase trees
        max_depth=15,      # Limit depth
        random_state=42
    ),
    # ... other models
}
```

### Adding New Models

In `model_training.py`, add to the `models` dictionary:

```python
from sklearn.svm import SVR

self.models['SVM'] = SVR(kernel='rbf')
```

### Customizing Visualizations

Edit `predictive_delivery_optimizer/visualization.py` to modify charts:

```python
def plot_custom_chart(self):
    fig = go.Figure()
    # Your custom plotly code
    return fig
```

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
**Solution**: Install requirements: `pip install -r requirements.txt`

**Issue**: Data validation errors
**Solution**: Ensure your CSV files have all required columns

**Issue**: Streamlit won't start
**Solution**: Check if port 8501 is available: `lsof -i :8501`

**Issue**: Poor model performance
**Solution**: Check data quality and increase sample size (>100 records recommended)

## Performance Tips

1. **Data Size**: The system works best with 100+ samples per dataset
2. **Feature Engineering**: Ensure dates are in valid datetime format
3. **Model Training**: With large datasets, training may take a few minutes
4. **Caching**: Streamlit caches data and models - use the "Clear Cache" option if you update data files

## Support

For issues or questions:
- Create an issue on [GitHub](https://github.com/khan09faiz/Optimized-Delivey-system/issues)
- Check the README.md for detailed documentation
- Review the test_system.py for usage examples

## Next Steps

1. Prepare your delivery data in the required format
2. Place CSV files in the `data/` directory
3. Run the application
4. Explore insights and recommendations
5. Implement suggested optimizations
6. Monitor improvements over time

Happy optimizing! 🚚📊
