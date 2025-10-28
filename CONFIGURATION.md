# Configuration Example for Predictive Delivery Optimizer

This file shows example configurations you can use to customize the application.

## Model Configuration

### Example: Custom Random Forest Parameters
```python
# In predictive_delivery_optimizer/model_training.py

from sklearn.ensemble import RandomForestRegressor

self.models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200,        # Number of trees
        max_depth=20,            # Maximum tree depth
        min_samples_split=5,     # Minimum samples to split
        min_samples_leaf=2,      # Minimum samples per leaf
        random_state=42,
        n_jobs=-1                # Use all CPU cores
    )
}
```

### Example: Add XGBoost Model
```python
# Install: pip install xgboost
from xgboost import XGBRegressor

self.models['XGBoost'] = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

## Data Loader Configuration

### Example: Custom Data Directory
```python
# When initializing DataLoader
loader = DataLoader(data_dir='/path/to/your/data')
```

### Example: Add Custom Dataset
```python
# In predictive_delivery_optimizer/data_loader.py

# Add to schema dictionary
self.schema['custom_dataset'] = [
    'column1', 'column2', 'column3'
]
```

## Feature Engineering Configuration

### Example: Custom Feature Creation
```python
# In predictive_delivery_optimizer/feature_engineering.py

def create_custom_features(self, df):
    # Add your custom features
    df['custom_ratio'] = df['col1'] / (df['col2'] + 1)
    df['custom_product'] = df['col3'] * df['col4']
    return df
```

## Streamlit Configuration

### Example: Custom Theme
```toml
# In .streamlit/config.toml

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "monospace"

[server]
port = 8080
maxUploadSize = 200
```

## Visualization Configuration

### Example: Custom Color Scheme
```python
# In predictive_delivery_optimizer/visualization.py

import plotly.express as px

self.color_scheme = px.colors.qualitative.Vivid
# Or use custom colors
self.color_scheme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
```

### Example: Custom Plot Settings
```python
def plot_custom_chart(self):
    fig = go.Figure()
    
    # Your data
    fig.add_trace(go.Bar(
        x=data.index,
        y=data.values,
        marker_color=self.color_scheme[0]
    ))
    
    # Custom layout
    fig.update_layout(
        title_text="My Custom Chart",
        title_font_size=24,
        height=600,
        template="plotly_white"
    )
    
    return fig
```

## Recommendation Engine Configuration

### Example: Custom Threshold Values
```python
# In predictive_delivery_optimizer/recommendation_engine.py

def recommend_route_optimizations(self):
    # Custom thresholds
    HIGH_TRAFFIC_THRESHOLD = 0.8  # 80th percentile
    LONG_DISTANCE_THRESHOLD = 150  # km
    
    # Your recommendation logic
    # ...
```

## Logging Configuration

### Example: Custom Logging Level
```python
# In predictive_delivery_optimizer/utils.py

import logging

logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## Sample Data Configuration

### Example: Adjust Sample Data Size
```python
# In predictive_delivery_optimizer/data_loader.py

def _create_sample_data(self, dataset_name: str):
    n_samples = 500  # Increase from default 100
    # ... rest of the code
```

## Database Integration Example

### Example: Load from PostgreSQL
```python
# Install: pip install psycopg2-binary

import pandas as pd
import psycopg2

def load_from_database(self):
    conn = psycopg2.connect(
        host="localhost",
        database="delivery_db",
        user="user",
        password="password"
    )
    
    query = "SELECT * FROM orders"
    df = pd.read_sql(query, conn)
    
    conn.close()
    return df
```

## API Integration Example

### Example: Export Recommendations to API
```python
# Install: pip install requests

import requests

def send_recommendations_to_api(recommendations):
    url = "https://api.example.com/recommendations"
    headers = {"Authorization": "Bearer YOUR_TOKEN"}
    
    response = requests.post(
        url,
        json=recommendations,
        headers=headers
    )
    
    return response.json()
```

## Performance Optimization

### Example: Enable Parallel Processing
```python
# For model training
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(train_single_model)(model, X, y)
    for model in self.models.values()
)
```

### Example: Reduce Memory Usage
```python
# Optimize data types
df['order_id'] = df['order_id'].astype('category')
df['weight'] = df['weight'].astype('float32')
```

## Testing Configuration

### Example: Custom Test Data
```python
# Create test data
import numpy as np

test_data = {
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randn(100)
}
```

## Notes

- Always backup your data before modifying configurations
- Test changes in a development environment first
- Some configurations may require additional package installations
- Refer to the official documentation of libraries for detailed parameter descriptions
