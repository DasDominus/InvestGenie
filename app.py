from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from stock_handler import StockHandler
from fixed_income_handler import FixedIncomeHandler
from efficient_frontier import EfficientFrontier

app = Flask(__name__)
stock_handler = StockHandler()
fixed_income_handler = FixedIncomeHandler()

# Create a combined handler by merging data from both handlers
class CombinedHandler:
    def __init__(self, stock_handler, fixed_income_handler):
        self.stock_handler = stock_handler
        self.fixed_income_handler = fixed_income_handler
        
        # Combine assets
        self.equity_assets = stock_handler.get_assets()
        self.fixed_income_assets = fixed_income_handler.get_assets()
        
        # Create asset type mapping
        self.asset_types = {}
        for asset in self.equity_assets:
            self.asset_types[asset] = 'equity'
        for asset in self.fixed_income_assets:
            self.asset_types[asset] = 'fixed_income'
        
        # Combine all assets
        self.assets = self.equity_assets + self.fixed_income_assets
        
        # Combine price data
        self._combine_price_data()
    
    def _combine_price_data(self):
        """Combine price data from both handlers."""
        stock_prices = self.stock_handler.get_prices()
        fixed_income_prices = self.fixed_income_handler.get_prices()
        
        # Reset the index to date only (without time) to make matching more reliable
        stock_prices.index = pd.to_datetime(stock_prices.index.date)
        fixed_income_prices.index = pd.to_datetime(fixed_income_prices.index.date)
        
        # Find the intersection of dates (common trading days)
        common_dates = stock_prices.index.intersection(fixed_income_prices.index)
        
        # Filter both DataFrames to only include common dates
        stock_prices = stock_prices.loc[common_dates]
        fixed_income_prices = fixed_income_prices.loc[common_dates]
        
        # Combine the DataFrames
        self.prices = pd.concat([stock_prices, fixed_income_prices], axis=1)
    
    def get_assets(self):
        """Return the list of available assets."""
        return self.assets
    
    def get_equity_assets(self):
        """Return just the equity assets."""
        return self.equity_assets
    
    def get_fixed_income_assets(self):
        """Return just the fixed income assets."""
        return self.fixed_income_assets
    
    def get_asset_type(self, asset):
        """Return the type of an asset."""
        return self.asset_types.get(asset)
    
    def get_prices(self):
        """Return the combined historical price data."""
        return self.prices
    
    def get_returns(self):
        """Calculate and return daily returns."""
        return self.prices.pct_change().dropna()
    
    def get_mean_returns(self):
        """Calculate and return mean returns."""
        returns = self.get_returns()
        return returns.mean()
    
    def get_cov_matrix(self):
        """Calculate and return the covariance matrix."""
        returns = self.get_returns()
        return returns.cov()
    
    def get_asset_names_display(self):
        """Return more readable names for display in UI."""
        equity_names = self.stock_handler.get_asset_names_display()
        fixed_income_names = self.fixed_income_handler.get_asset_names_display()
        return {**equity_names, **fixed_income_names}

# Create the combined handler
combined_handler = CombinedHandler(stock_handler, fixed_income_handler)
ef_calculator = EfficientFrontier(combined_handler)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/efficient-frontier', methods=['GET'])
def get_efficient_frontier():
    """Return the efficient frontier data."""
    ef_data = ef_calculator.get_efficient_frontier()
    
    # Add asset display names
    asset_names = combined_handler.get_asset_names_display()
    
    # Add asset type information
    asset_types = {}
    for asset in combined_handler.get_assets():
        asset_types[asset] = combined_handler.get_asset_type(asset)
    
    ef_data['asset_names'] = asset_names
    ef_data['asset_types'] = asset_types
    ef_data['equity_assets'] = combined_handler.get_equity_assets()
    ef_data['fixed_income_assets'] = combined_handler.get_fixed_income_assets()
    
    return jsonify(ef_data)

@app.route('/api/portfolio', methods=['POST'])
def calculate_portfolio():
    """Calculate portfolio metrics based on provided weights."""
    data = request.get_json()
    weights = data.get('weights', {})
    
    # Convert dict to proper format if needed
    if isinstance(weights, dict):
        # Convert {asset: weight} to ordered list matching the assets in combined_handler
        weight_list = [weights.get(asset, 0) for asset in combined_handler.get_assets()]
        # Normalize weights
        weight_list = np.array(weight_list) / sum(weight_list) if sum(weight_list) > 0 else weight_list
        weights = weight_list.tolist()
    
    portfolio_data = ef_calculator.calculate_portfolio_metrics(weights)
    
    # Add asset type information
    for asset in portfolio_data['weights']:
        asset_type = combined_handler.get_asset_type(asset)
        if asset_type:
            portfolio_data['weights'][asset] = {
                'weight': portfolio_data['weights'][asset],
                'type': asset_type
            }
    
    # Calculate total weights by asset type
    equity_weight = sum(portfolio_data['weights'][asset]['weight'] 
                        for asset in combined_handler.get_equity_assets() 
                        if asset in portfolio_data['weights'])
    
    fixed_income_weight = sum(portfolio_data['weights'][asset]['weight'] 
                              for asset in combined_handler.get_fixed_income_assets() 
                              if asset in portfolio_data['weights'])
    
    portfolio_data['asset_type_weights'] = {
        'equity': equity_weight,
        'fixed_income': fixed_income_weight
    }
    
    return jsonify(portfolio_data)

@app.route('/api/optimize', methods=['GET'])
def optimize_portfolio():
    """Return an optimized portfolio."""
    optimization_type = request.args.get('type', 'max_sharpe')
    
    # Get constraints if provided
    min_equity = request.args.get('min_equity', None)
    max_equity = request.args.get('max_equity', None)
    min_fixed_income = request.args.get('min_fixed_income', None)
    max_fixed_income = request.args.get('max_fixed_income', None)
    
    constraints = {}
    if min_equity is not None:
        constraints['min_equity'] = float(min_equity)
    if max_equity is not None:
        constraints['max_equity'] = float(max_equity)
    if min_fixed_income is not None:
        constraints['min_fixed_income'] = float(min_fixed_income)
    if max_fixed_income is not None:
        constraints['max_fixed_income'] = float(max_fixed_income)
    
    optimized_data = ef_calculator.optimize_portfolio(optimization_type, constraints)
    
    # Add asset display names
    asset_names = combined_handler.get_asset_names_display()
    optimized_data['asset_names'] = asset_names
    
    return jsonify(optimized_data)

@app.route('/api/assets', methods=['GET'])
def get_assets():
    """Return the list of available assets with their types and display names."""
    assets = combined_handler.get_assets()
    asset_names = combined_handler.get_asset_names_display()
    
    asset_data = {
        'assets': assets,
        'display_names': asset_names,
        'equity_assets': combined_handler.get_equity_assets(),
        'fixed_income_assets': combined_handler.get_fixed_income_assets()
    }
    
    return jsonify(asset_data)

if __name__ == '__main__':
    app.run(debug=True) 