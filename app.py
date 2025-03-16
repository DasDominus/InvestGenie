from flask import Flask, render_template, request, jsonify, send_file
import json
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import csv
import io
import os
from datetime import datetime

from stock_handler import StockHandler
from efficient_frontier import EfficientFrontier

app = Flask(__name__)
stock_handler = StockHandler()
ef_calculator = EfficientFrontier(stock_handler)

# Create directory for saved allocations if it doesn't exist
ALLOCATIONS_DIR = 'allocations'
if not os.path.exists(ALLOCATIONS_DIR):
    os.makedirs(ALLOCATIONS_DIR)

# Create directory for provider allocations if it doesn't exist
PROVIDERS_DIR = 'providers'
if not os.path.exists(PROVIDERS_DIR):
    os.makedirs(PROVIDERS_DIR)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/providers')
def providers():
    """Render the providers page."""
    return render_template('providers.html')

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

@app.route('/api/allocations/save', methods=['POST'])
def save_allocation():
    """Save the current allocation to a CSV file on the server."""
    try:
        data = request.get_json()
        allocation = data.get('allocation', [])
        
        if not allocation:
            return jsonify({'success': False, 'message': 'No allocation data provided'}), 400
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"allocation_{timestamp}.csv"
        filepath = os.path.join(ALLOCATIONS_DIR, filename)
        
        # Save the allocation to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['type', 'size_term', 'style_risk', 'amount'])
            for item in allocation:
                writer.writerow([item['type'], item['size_term'], item['style_risk'], item['amount']])
        
        return jsonify({'success': True, 'filename': filename, 'message': 'Allocation saved successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error saving allocation: {str(e)}'}), 500

@app.route('/api/allocations/load/<filename>', methods=['GET'])
def load_allocation(filename):
    """Load an allocation from a CSV file on the server."""
    try:
        filepath = os.path.join(ALLOCATIONS_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'}), 404
        
        # Read the allocation from CSV
        allocation = []
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                allocation.append(row)
        
        return jsonify({'success': True, 'allocation': allocation})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading allocation: {str(e)}'}), 500

@app.route('/api/allocations/list', methods=['GET'])
def list_allocations():
    """List all saved allocations."""
    try:
        files = []
        for filename in os.listdir(ALLOCATIONS_DIR):
            if filename.endswith('.csv'):
                filepath = os.path.join(ALLOCATIONS_DIR, filename)
                timestamp = os.path.getmtime(filepath)
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                files.append({
                    'filename': filename,
                    'date': date
                })
        
        # Sort by date (newest first)
        files.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({'success': True, 'files': files})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error listing allocations: {str(e)}'}), 500

# Provider-specific routes
@app.route('/api/providers/save', methods=['POST'])
def save_providers():
    """Save provider allocations to a JSON file on the server."""
    try:
        data = request.get_json()
        providers = data.get('providers', [])
        year = data.get('year', '')
        quarter = data.get('quarter', '')
        
        if not providers:
            return jsonify({'success': False, 'message': 'No provider data provided'}), 400
        
        if not year or not quarter:
            return jsonify({'success': False, 'message': 'Year and quarter are required'}), 400
        
        # Create a directory for the year if it doesn't exist
        year_dir = os.path.join(PROVIDERS_DIR, str(year))
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
        
        # Save to a JSON file named by quarter
        filepath = os.path.join(year_dir, f'providers_q{quarter}.json')
        
        with open(filepath, 'w') as jsonfile:
            json.dump(providers, jsonfile)
        
        return jsonify({'success': True, 'message': f'Providers saved successfully for {year} Q{quarter}'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error saving providers: {str(e)}'}), 500

@app.route('/api/providers/list', methods=['GET'])
def list_providers():
    """List all saved providers for a specific year and quarter."""
    try:
        year = request.args.get('year')
        quarter = request.args.get('quarter')
        
        if not year or not quarter:
            # Return a list of available years and quarters
            years_quarters = []
            if os.path.exists(PROVIDERS_DIR):
                for year_dir in os.listdir(PROVIDERS_DIR):
                    year_path = os.path.join(PROVIDERS_DIR, year_dir)
                    if os.path.isdir(year_path):
                        for file in os.listdir(year_path):
                            if file.startswith('providers_q') and file.endswith('.json'):
                                quarter = file.replace('providers_q', '').replace('.json', '')
                                years_quarters.append({'year': year_dir, 'quarter': quarter})
            
            # Sort by year and quarter (newest first)
            years_quarters.sort(key=lambda x: (x['year'], x['quarter']), reverse=True)
            
            return jsonify({'success': True, 'periods': years_quarters})
        
        # Load providers for the specified year and quarter
        filepath = os.path.join(PROVIDERS_DIR, str(year), f'providers_q{quarter}.json')
        
        if not os.path.exists(filepath):
            return jsonify({'success': True, 'providers': []})
        
        with open(filepath, 'r') as jsonfile:
            providers = json.load(jsonfile)
        
        return jsonify({'success': True, 'providers': providers})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error listing providers: {str(e)}'}), 500

@app.route('/api/providers/periods', methods=['GET'])
def list_periods():
    """List all available years and quarters."""
    try:
        periods = []
        if os.path.exists(PROVIDERS_DIR):
            for year_dir in os.listdir(PROVIDERS_DIR):
                year_path = os.path.join(PROVIDERS_DIR, year_dir)
                if os.path.isdir(year_path):
                    quarters = []
                    for file in os.listdir(year_path):
                        if file.startswith('providers_q') and file.endswith('.json'):
                            quarter = file.replace('providers_q', '').replace('.json', '')
                            quarters.append(quarter)
                    
                    if quarters:
                        quarters.sort()
                        periods.append({'year': year_dir, 'quarters': quarters})
        
        # Sort by year (newest first)
        periods.sort(key=lambda x: x['year'], reverse=True)
        
        return jsonify({'success': True, 'periods': periods})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error listing periods: {str(e)}'}), 500

@app.route('/api/providers/aggregate', methods=['GET'])
def aggregate_providers():
    """Aggregate allocations from all providers for a specific year and quarter."""
    try:
        year = request.args.get('year')
        quarter = request.args.get('quarter')
        
        if not year or not quarter:
            # If no year/quarter specified, try to use the most recent period
            if os.path.exists(PROVIDERS_DIR):
                years = sorted([y for y in os.listdir(PROVIDERS_DIR) if os.path.isdir(os.path.join(PROVIDERS_DIR, y))], reverse=True)
                
                if years:
                    year_dir = os.path.join(PROVIDERS_DIR, years[0])
                    quarter_files = [f for f in os.listdir(year_dir) if f.startswith('providers_q') and f.endswith('.json')]
                    
                    if quarter_files:
                        quarter_files.sort(reverse=True)
                        quarter = quarter_files[0].replace('providers_q', '').replace('.json', '')
                        year = years[0]
                    else:
                        return jsonify({'success': True, 'allocation': {'equity': {}, 'fixed_income': {}, 'cash': {}}, 'period': None})
                else:
                    return jsonify({'success': True, 'allocation': {'equity': {}, 'fixed_income': {}, 'cash': {}}, 'period': None})
        
        filepath = os.path.join(PROVIDERS_DIR, str(year), f'providers_q{quarter}.json')
        
        if not os.path.exists(filepath):
            return jsonify({'success': True, 'allocation': {'equity': {}, 'fixed_income': {}, 'cash': {}}, 'period': {'year': year, 'quarter': quarter}})
        
        with open(filepath, 'r') as jsonfile:
            providers = json.load(jsonfile)
        
        # Aggregate allocations
        aggregated = {
            'equity': {},
            'fixed_income': {},
            'cash': {}
        }
        
        for provider in providers:
            # Aggregate equity allocations
            for key, value in provider.get('allocation', {}).get('equity', {}).items():
                if key in aggregated['equity']:
                    aggregated['equity'][key] += value
                else:
                    aggregated['equity'][key] = value
            
            # Aggregate fixed income allocations
            for key, value in provider.get('allocation', {}).get('fixed_income', {}).items():
                if key in aggregated['fixed_income']:
                    aggregated['fixed_income'][key] += value
                else:
                    aggregated['fixed_income'][key] = value
            
            # Aggregate cash allocations
            for key, value_obj in provider.get('allocation', {}).get('cash', {}).items():
                if isinstance(value_obj, dict) and 'amount' in value_obj:
                    amount = value_obj['amount']
                    if key in aggregated['cash']:
                        aggregated['cash'][key] += amount
                    else:
                        aggregated['cash'][key] = amount
        
        return jsonify({'success': True, 'allocation': aggregated, 'period': {'year': year, 'quarter': quarter}})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error aggregating providers: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 