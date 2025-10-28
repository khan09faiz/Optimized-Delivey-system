#!/bin/bash
# Run script for the Predictive Delivery Optimizer

echo "======================================================"
echo "  Predictive Delivery Optimizer"
echo "======================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting Streamlit application..."
echo "The app will be available at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run predictive_delivery_optimizer/app.py
