from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import json
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import csv
import io
import os
import markdown
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

# Create directory for IPS if it doesn't exist
IPS_DIR = 'ips'
if not os.path.exists(IPS_DIR):
    os.makedirs(IPS_DIR)

@app.route('/')
def index():
    """Render the IPS page as the main index page."""
    ips_file = os.path.join(IPS_DIR, 'investment_policy_statement.md')
    
    if os.path.exists(ips_file):
        # IPS exists, display it
        with open(ips_file, 'r') as f:
            ips_content = f.read()
        
        # Convert markdown to HTML
        ips_html = markdown.markdown(ips_content)
        
        # Check if IPS is signed
        is_signed = "## Signature" in ips_content
        
        return render_template('ips.html', 
                              ips_html=ips_html, 
                              is_signed=is_signed,
                              ips_content=ips_content)
    else:
        # IPS doesn't exist, show the creation form
        return render_template('create_ips.html')

@app.route('/dashboard')
def dashboard():
    """Render the portfolio dashboard page."""
    return render_template('dashboard.html')

@app.route('/providers')
def providers():
    """Render the providers page."""
    return render_template('providers.html')

@app.route('/create_ips', methods=['POST'])
def create_ips():
    """Create a new Investment Policy Statement."""
    ips_content = request.form.get('ips_content', '')
    
    ips_file = os.path.join(IPS_DIR, 'investment_policy_statement.md')
    with open(ips_file, 'w') as f:
        f.write(ips_content)
    
    return redirect(url_for('index'))

@app.route('/sign_ips', methods=['POST'])
def sign_ips():
    """Sign the Investment Policy Statement."""
    name = request.form.get('name', '')
    date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    ips_file = os.path.join(IPS_DIR, 'investment_policy_statement.md')
    
    if os.path.exists(ips_file):
        with open(ips_file, 'r') as f:
            ips_content = f.read()
        
        # Add signature if not already signed
        if "## Signature" not in ips_content:
            signature = f"\n\n## Signature\n\nSigned by: {name}\nDate: {date}"
            ips_content += signature
            
            with open(ips_file, 'w') as f:
                f.write(ips_content)
    
    return redirect(url_for('index'))

@app.route('/adjust_ips', methods=['POST'])
def adjust_ips():
    """Allow adjustment of the IPS after it's been signed."""
    ips_file = os.path.join(IPS_DIR, 'investment_policy_statement.md')
    
    if os.path.exists(ips_file):
        with open(ips_file, 'r') as f:
            ips_content = f.read()
            
        # Remove signature section if it exists
        if "## Signature" in ips_content:
            ips_content = ips_content.split("## Signature")[0].strip()
        
        return render_template('adjust_ips.html', ips_content=ips_content)
    
    return redirect(url_for('index'))

@app.route('/fund_stats')
def fund_stats():
    """Render the fund statistics page."""
    # Get all tickers from stock_handler
    all_tickers = stock_handler.get_all_tickers()
    
    # Get fund descriptions for better display
    fund_descriptions = stock_handler.get_fund_descriptions()
    
    selected_ticker = request.args.get('ticker', all_tickers[0] if all_tickers else None)
    
    return render_template('fund_stats.html', 
                          tickers=all_tickers,
                          fund_descriptions=fund_descriptions,
                          selected_ticker=selected_ticker)

@app.route('/api/efficient-frontier', methods=['GET'])
def get_efficient_frontier():
    """Return the efficient frontier data."""
    try:
        print("=== /api/efficient-frontier endpoint called ===")
        print("Calculating efficient frontier...")
        
        start_time = datetime.now()
        ef_data = ef_calculator.get_efficient_frontier()
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Efficient frontier calculation completed in {calculation_time:.2f} seconds")
        print(f"Efficient frontier data points: {len(ef_data['efficient_frontier']['returns'])}")
        print(f"Random portfolios data points: {len(ef_data['random_portfolios']['returns'])}")
        print(f"Min volatility portfolio return: {ef_data['min_volatility']['return']:.4f}")
        print(f"Min volatility portfolio volatility: {ef_data['min_volatility']['volatility']:.4f}")
        print(f"Max Sharpe portfolio return: {ef_data['max_sharpe']['return']:.4f}")
        print(f"Max Sharpe portfolio volatility: {ef_data['max_sharpe']['volatility']:.4f}")
        
        return jsonify(ef_data)
    except Exception as e:
        print(f"Error calculating efficient frontier: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to calculate efficient frontier'
        }), 500

@app.route('/api/stock-bond-frontier', methods=['GET'])
def get_stock_bond_frontier():
    """Return the stock-bond composition frontier data for SPY and BMOAX."""
    try:
        print("=== /api/stock-bond-frontier endpoint called ===")
        
        # Get number of points from query params, default to 11
        num_points = request.args.get('points', default=50, type=int)
        print(f"Requested {num_points} data points for stock-bond frontier")
        
        # Calculate the stock-bond frontier
        start_time = datetime.now()
        frontier_data = ef_calculator.get_stock_bond_frontier(num_points=num_points)
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Stock-bond frontier calculation completed in {calculation_time:.2f} seconds")
        
        # Log some basic information about the returned data
        if 'raw_data' in frontier_data and frontier_data['raw_data']:
            data_points = len(frontier_data['raw_data'])
            print(f"Stock-bond frontier data points: {data_points}")
            
            # Log sample data points (first, middle, last)
            if data_points > 0:
                first_point = frontier_data['raw_data'][0]
                print(f"First point (100% Bond): Return={first_point['return']:.4f}, Volatility={first_point['volatility']:.4f}")
                
                if data_points > 2:
                    mid_point = frontier_data['raw_data'][data_points // 2]
                    print(f"Middle point (~50/50): Return={mid_point['return']:.4f}, Volatility={mid_point['volatility']:.4f}")
                
                last_point = frontier_data['raw_data'][data_points - 1]
                print(f"Last point (100% Stock): Return={last_point['return']:.4f}, Volatility={last_point['volatility']:.4f}")
        
        return jsonify(frontier_data)
    except Exception as e:
        print(f"Error calculating stock-bond frontier: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to calculate stock-bond frontier'
        }), 500

@app.route('/api/portfolio', methods=['POST'])
def calculate_portfolio():
    """Calculate portfolio metrics based on provided weights."""
    try:
        print("=== /api/portfolio endpoint called ===")
        data = request.get_json()
        print(f"Received data: {json.dumps(data, indent=2)}")
        
        weights = data.get('weights', {})
        print(f"Extracted weights: {json.dumps(weights, indent=2)}")
        
        # Check format of weights
        if isinstance(weights, dict):
            print("Weights provided as a dictionary")
            # Get the correct assets list
            assets = stock_handler.get_assets()
            print(f"Available assets: {assets}")
            
            # Map asset categories to actual asset tickers
            # This mapping should be configured based on your actual asset allocation
            asset_mapping = {
                'equity_small_growth': 'FCPGX',
                'equity_small_blend': 'SWSSX',
                'equity_small_value': 'VBR',
                'equity_mid_growth': 'IJK',
                'equity_mid_blend': 'VEXAX',
                'equity_mid_value': 'SWMCX',
                'equity_large_growth': 'FSPGX',
                'equity_large_blend': 'SWTSX',
                'equity_large_value': 'SWLVX',
                'fixed_short_low': 'FSHBX',
                'fixed_short_mid': 'SJNK',
                'fixed_short_high': 'ISTB',
                'fixed_mid_low': 'FBNDX',
                'fixed_mid_mid': 'LSYAX',
                'fixed_mid_high': 'VGIT',
                'fixed_long_low': 'VCLT',
                'fixed_long_mid': 'JNK',
                'fixed_long_high': 'VGLT',
                'cash_cash': 'CASH',
                'cash_money_market': 'SPAXX',
                'cash_other_liquid': 'SPAXX'
            }
            
            # Create a ticker-based weight dictionary
            ticker_weights = {}
            for category, weight in weights.items():
                if category in asset_mapping:
                    ticker = asset_mapping[category]
                    if ticker not in ticker_weights:
                        ticker_weights[ticker] = 0
                    ticker_weights[ticker] += weight
                    print(f"Mapped {category} with weight {weight} to ticker {ticker}")
                else:
                    print(f"Warning: No mapping found for {category}")
            
            print(f"Ticker weights: {json.dumps(ticker_weights, indent=2)}")
            
            # Convert to weight list in the order of assets
            weight_list = [ticker_weights.get(asset, 0) for asset in assets]
            print(f"Weight list before normalization: {weight_list}")
            
            # Normalize weights
            weight_sum = sum(weight_list)
            print(f"Weight sum: {weight_sum}")
            
            if weight_sum > 0:
                weight_list = np.array(weight_list) / weight_sum
                weights = weight_list.tolist()
                print(f"Normalized weights: {weights}")
            else:
                weights = weight_list
                print("Weight sum is zero, not normalizing")
        else:
            print(f"Weights not provided as a dictionary: {type(weights)}")
        
        print("Calculating portfolio metrics...")
        portfolio_data = ef_calculator.calculate_portfolio_metrics(weights)
        print(f"Portfolio calculation result: {json.dumps(portfolio_data, indent=2)}")
        
        return jsonify(portfolio_data)
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to calculate portfolio metrics'
        }), 500

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