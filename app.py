from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from stock_handler import StockHandler
from efficient_frontier import EfficientFrontier

app = Flask(__name__)
stock_handler = StockHandler()
ef_calculator = EfficientFrontier(stock_handler)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/efficient-frontier', methods=['GET'])
def get_efficient_frontier():
    """Return the efficient frontier data."""
    ef_data = ef_calculator.get_efficient_frontier()
    return jsonify(ef_data)

@app.route('/api/portfolio', methods=['POST'])
def calculate_portfolio():
    """Calculate portfolio metrics based on provided weights."""
    data = request.get_json()
    weights = data.get('weights', {})
    
    # Convert dict to proper format if needed
    if isinstance(weights, dict):
        # Check if this is the new grid-based allocation format
        is_grid_allocation = any(key.startswith(('equity_', 'fixed_')) for key in weights.keys())
        
        if is_grid_allocation:
            # For grid allocation, we use the weights directly
            # But we need to ensure all known assets have a weight
            weight_dict = {asset: 0.0 for asset in stock_handler.get_assets()}
            
            # Update with provided weights (may include assets not in stock_handler)
            # We'll keep only the known assets
            for asset, weight in weights.items():
                if asset in stock_handler.get_assets():
                    weight_dict[asset] = weight
            
            # Convert to ordered list matching assets in stock_handler
            weight_list = [weight_dict.get(asset, 0) for asset in stock_handler.get_assets()]
            
        else:
            # Standard format - convert {asset: weight} to ordered list matching the assets in stock_handler
            weight_list = [weights.get(asset, 0) for asset in stock_handler.get_assets()]
        
        # Normalize weights
        weight_sum = sum(weight_list)
        if weight_sum > 0:
            weight_list = np.array(weight_list) / weight_sum
        weights = weight_list.tolist()
    
    portfolio_data = ef_calculator.calculate_portfolio_metrics(weights)
    return jsonify(portfolio_data)

@app.route('/api/optimize', methods=['GET'])
def optimize_portfolio():
    """Return an optimized portfolio."""
    optimization_type = request.args.get('type', 'max_sharpe')
    optimized_data = ef_calculator.optimize_portfolio(optimization_type)
    return jsonify(optimized_data)

@app.route('/api/assets', methods=['GET'])
def get_assets():
    """Return the list of available assets."""
    return jsonify(stock_handler.get_assets())

if __name__ == '__main__':
    app.run(debug=True) 