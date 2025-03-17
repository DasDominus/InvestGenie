import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

class EfficientFrontier:
    """
    Class to calculate and optimize portfolios based on the efficient frontier.
    """
    
    def __init__(self, stock_handler):
        """Initialize with a stock handler object."""
        self.stock_handler = stock_handler
        self.assets = stock_handler.get_assets()
        
        # Calculate returns and covariance matrix
        self.returns = stock_handler.get_returns()
        self.mean_returns = stock_handler.get_mean_returns()
        self.cov_matrix = stock_handler.get_cov_matrix()
        
        # Handle potential duplicate assets
        self._handle_duplicates()
        
        # Calculate annual returns and covariance (assuming 252 trading days per year)
        self.annual_returns = self.mean_returns * 252
        self.annual_cov_matrix = self.cov_matrix * 252
        
        # Set risk-free rate
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def _handle_duplicates(self):
        """Handle duplicate assets to ensure consistent dimensions."""
        # Check if there are duplicate assets
        unique_assets = list(dict.fromkeys(self.assets))
        
        if len(unique_assets) != len(self.assets):
            print(f"Found duplicate assets. Original count: {len(self.assets)}, Unique count: {len(unique_assets)}")
            
            # Create a mapping of original indices to unique assets
            asset_indices = {}
            for i, asset in enumerate(self.assets):
                if asset not in asset_indices:
                    asset_indices[asset] = []
                asset_indices[asset].append(i)
            
            # Use unique assets list instead
            self.original_assets = self.assets.copy()
            self.assets = unique_assets
            
            # For assets that appear multiple times, combine their weights proportionally in later calculations
            self.asset_mapping = {asset: asset_indices[asset] for asset in unique_assets}
        else:
            self.original_assets = self.assets
            self.asset_mapping = {asset: [i] for i, asset in enumerate(self.assets)}
    
    def calculate_portfolio_metrics(self, weights):
        """
        Calculate expected return, volatility and Sharpe ratio for a given portfolio.
        
        Args:
            weights: List of weights for each asset
            
        Returns:
            dict: Dictionary containing portfolio metrics
        """
        weights = np.array(weights)
        
        # Expected portfolio return
        portfolio_return = np.sum(self.annual_returns * weights)
        
        # Expected portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov_matrix, weights)))
        
        # Sharpe ratio (assuming risk-free rate of 0.02 or 2%)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Map weights back to original assets if there were duplicates
        if hasattr(self, 'original_assets') and len(self.original_assets) != len(self.assets):
            original_weights = []
            for asset in self.original_assets:
                # Find the corresponding unique asset
                idx = self.assets.index(asset)
                original_weights.append(weights[idx])
            
            # Clean up any very small values that might cause JSON formatting issues
            cleaned_weights = [max(0, w) if abs(w) < 1e-10 else w for w in original_weights]
            weights_dict = {asset: float(weight) for asset, weight in zip(self.original_assets, cleaned_weights)}
        else:
            # Clean up any very small values that might cause JSON formatting issues
            cleaned_weights = [max(0, w) if abs(w) < 1e-10 else w for w in weights]
            weights_dict = {asset: float(weight) for asset, weight in zip(self.assets, cleaned_weights)}
        
        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'weights': weights_dict
        }
    
    def get_efficient_frontier(self, points=50):
        """
        Calculate the efficient frontier points.
        
        Args:
            points: Number of points to calculate
            
        Returns:
            dict: Dictionary containing efficient frontier data
        """
        # Get min volatility and max sharpe portfolios
        min_vol_weights = self._optimize('min_volatility')
        max_sharpe_weights = self._optimize('max_sharpe')
        
        min_vol_metrics = self.calculate_portfolio_metrics(min_vol_weights)
        max_sharpe_metrics = self.calculate_portfolio_metrics(max_sharpe_weights)
        
        # Generate efficient frontier by targeting returns between min vol and max return
        target_returns = np.linspace(
            min_vol_metrics['expected_return'] * 0.7,
            max_sharpe_metrics['expected_return'] * 1.3,
            points
        )
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                weights = self._optimize('efficient_return', target_return=target_return)
                metrics = self.calculate_portfolio_metrics(weights)
                efficient_portfolios.append(metrics)
            except:
                # Skip points that can't be optimized
                continue
        
        # Extract data for chart
        ef_returns = [p['expected_return'] for p in efficient_portfolios]
        ef_volatilities = [p['volatility'] for p in efficient_portfolios]
        ef_sharpe_ratios = [p['sharpe_ratio'] for p in efficient_portfolios]
        
        # Generate 100 random portfolios
        random_portfolios = self._generate_random_portfolios(100)
        
        # Extract data for random portfolios
        random_returns = [p['expected_return'] for p in random_portfolios]
        random_volatilities = [p['volatility'] for p in random_portfolios]
        random_sharpe_ratios = [p['sharpe_ratio'] for p in random_portfolios]
        
        # Helper function to clean very small values for safe JSON serialization
        def clean_small_value(value):
            if isinstance(value, (int, float)) and abs(value) < 1e-10:
                return 0.0
            return value
        
        # Clean up any very small values to avoid JSON issues
        min_vol_return = clean_small_value(min_vol_metrics['expected_return'])
        min_vol_volatility = clean_small_value(min_vol_metrics['volatility'])
        min_vol_sharpe = clean_small_value(min_vol_metrics['sharpe_ratio'])
        
        max_sharpe_return = clean_small_value(max_sharpe_metrics['expected_return'])
        max_sharpe_volatility = clean_small_value(max_sharpe_metrics['volatility'])
        max_sharpe_sharpe = clean_small_value(max_sharpe_metrics['sharpe_ratio'])
        
        # Prepare response data
        response = {
            'efficient_frontier': {
                'returns': [clean_small_value(value) for value in ef_returns],
                'volatilities': [clean_small_value(value) for value in ef_volatilities],
                'sharpe_ratios': [clean_small_value(value) for value in ef_sharpe_ratios]
            },
            'random_portfolios': {
                'returns': [clean_small_value(value) for value in random_returns],
                'volatilities': [clean_small_value(value) for value in random_volatilities],
                'sharpe_ratios': [clean_small_value(value) for value in random_sharpe_ratios]
            },
            'min_volatility': {
                'return': min_vol_return,
                'volatility': min_vol_volatility,
                'sharpe_ratio': min_vol_sharpe,
                'weights': min_vol_metrics['weights']
            },
            'max_sharpe': {
                'return': max_sharpe_return,
                'volatility': max_sharpe_volatility,
                'sharpe_ratio': max_sharpe_sharpe,
                'weights': max_sharpe_metrics['weights']
            }
        }
        
        return response
    
    def get_stock_bond_frontier(self, num_points=11):
        """
        Calculate return and standard deviation for various compositions of 
        SPY (stock) and BMOAX (bond).
        
        Args:
            num_points: Number of portfolio allocations to generate
                       (default 11 for 0% to 100% in 10% increments)
                       
        Returns:
            Dictionary with the frontier data for Plotly
        """
        print(f"Calculating stock-bond frontier with {num_points} points")
        
        # Download the historical data for SPY and BMOAX
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            print("Fetching historical data for SPY and BMOAX")
            # Create date range for the last 5 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            print(f"Date range: {start_date} to {end_date}")
            
            # Fetch data for SPY and BMOAX
            data = yf.download(
                tickers=["SPY", "BMOAX"],
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False
            )
            
            print(f"Downloaded data shape: {data.shape}")
            
            # Extract closing prices
            prices = pd.DataFrame()
            
            # Handle multi-level columns
            if ("SPY", "Close") in data.columns and ("BMOAX", "Close") in data.columns:
                print("Successfully extracted SPY and BMOAX closing prices")
                prices["SPY"] = data[("SPY", "Close")]
                prices["BMOAX"] = data[("BMOAX", "Close")]
            else:
                print("Could not find SPY and BMOAX closing prices in downloaded data")
                print(f"Available columns: {data.columns}")
                # Fallback to mock data if we can't get the actual data
                print("Falling back to mock data")
                return self._generate_mock_stock_bond_frontier(num_points)
            
            # Drop any rows with missing values
            original_row_count = len(prices)
            prices = prices.dropna()
            print(f"Dropped {original_row_count - len(prices)} rows with NaN values, {len(prices)} rows remaining")
            
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            print(f"Calculated returns with shape: {returns.shape}")
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            print(f"Mean daily returns: SPY={mean_returns['SPY']:.6f}, BMOAX={mean_returns['BMOAX']:.6f}")
            print(f"Daily return correlation: {returns.corr().iloc[0,1]:.6f}")
            
            # Calculate annual returns and covariance (assuming 252 trading days)
            annual_returns = mean_returns * 252
            annual_cov_matrix = cov_matrix * 252
            print(f"Annualized returns: SPY={annual_returns['SPY']:.4f} ({annual_returns['SPY']*100:.2f}%), " +
                  f"BMOAX={annual_returns['BMOAX']:.4f} ({annual_returns['BMOAX']*100:.2f}%)")
            print(f"Annualized volatility: SPY={np.sqrt(annual_cov_matrix.iloc[0,0]):.4f} ({np.sqrt(annual_cov_matrix.iloc[0,0])*100:.2f}%), " +
                  f"BMOAX={np.sqrt(annual_cov_matrix.iloc[1,1]):.4f} ({np.sqrt(annual_cov_matrix.iloc[1,1])*100:.2f}%)")
            
            # Generate weights for the stock-bond allocations
            weights_list = []
            for i in range(num_points):
                stock_weight = i / (num_points - 1) if num_points > 1 else 0
                bond_weight = 1 - stock_weight
                weights_list.append([stock_weight, bond_weight])
            
            print(f"Generated {len(weights_list)} weight combinations")
            
            # Calculate portfolio metrics for each weight allocation
            portfolio_data = []
            for weights in weights_list:
                # Calculate portfolio return
                portfolio_return = np.sum(annual_returns * weights)
                
                # Calculate portfolio volatility (standard deviation)
                portfolio_std = np.sqrt(weights @ annual_cov_matrix @ weights)
                
                # Calculate Sharpe ratio
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
                
                # Calculate stock percentage for labeling
                stock_pct = weights[0] * 100
                
                portfolio_data.append({
                    'return': portfolio_return,
                    'volatility': portfolio_std,
                    'sharpe': sharpe_ratio,
                    'weights': {
                        'SPY': weights[0],
                        'BMOAX': weights[1]
                    },
                    'stock_pct': stock_pct
                })
            
            print("Portfolio metrics calculated for all weight combinations")
            # Log a sample of the data
            print(f"Sample metrics for 50/50 portfolio (if available):")
            mid_point = len(portfolio_data) // 2
            if 0 <= mid_point < len(portfolio_data):
                print(f"  Return: {portfolio_data[mid_point]['return']:.4f} ({portfolio_data[mid_point]['return']*100:.2f}%)")
                print(f"  Volatility: {portfolio_data[mid_point]['volatility']:.4f} ({portfolio_data[mid_point]['volatility']*100:.2f}%)")
                print(f"  Sharpe Ratio: {portfolio_data[mid_point]['sharpe']:.4f}")
            
            # Format the data for Plotly
            stock_bond_data = {
                'x': [p['volatility'] for p in portfolio_data],
                'y': [p['return'] for p in portfolio_data],
                'mode': 'lines+markers',
                'name': 'Stock-Bond Frontier',
                'type': 'scatter',
                'line': {'color': 'blue', 'width': 2},
                'marker': {'size': 8},
                'text': [f"Stock: {p['stock_pct']:.0f}%, Bond: {(100-p['stock_pct']):.0f}%" 
                         for p in portfolio_data],
                'hoverinfo': 'text+x+y',
                'hovertemplate': 'Volatility: %{x:.4f}<br>Return: %{y:.4f}<br>%{text}'
            }
            
            print("Stock-bond frontier data formatted successfully for Plotly")
            return {
                'stock_bond_frontier': stock_bond_data,
                'raw_data': portfolio_data
            }
            
        except Exception as e:
            print(f"Error generating stock-bond frontier: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to mock data
            print("Falling back to mock data due to error")
            return self._generate_mock_stock_bond_frontier(num_points)
    
    def _generate_mock_stock_bond_frontier(self, num_points=11):
        """
        Generate mock data for stock-bond frontier when real data is unavailable.
        """
        print(f"Generating mock stock-bond frontier data with {num_points} points")
        
        # Typical annual return and volatility values
        spy_return = 0.10  # 10% annual return for stocks
        spy_vol = 0.18     # 18% annual volatility for stocks
        bmoax_return = 0.04  # 4% annual return for bonds
        bmoax_vol = 0.05     # 5% annual volatility for bonds
        
        # Correlation between stocks and bonds (typically negative or low)
        correlation = -0.1
        
        print(f"Mock data parameters:")
        print(f"  SPY return: {spy_return:.4f} ({spy_return*100:.1f}%)")
        print(f"  SPY volatility: {spy_vol:.4f} ({spy_vol*100:.1f}%)")
        print(f"  BMOAX return: {bmoax_return:.4f} ({bmoax_return*100:.1f}%)")
        print(f"  BMOAX volatility: {bmoax_vol:.4f} ({bmoax_vol*100:.1f}%)")
        print(f"  Correlation: {correlation:.4f}")
        
        # Generate weights for the stock-bond allocations
        weights_list = []
        for i in range(num_points):
            stock_weight = i / (num_points - 1) if num_points > 1 else 0
            bond_weight = 1 - stock_weight
            weights_list.append([stock_weight, bond_weight])
        
        print(f"Generated {len(weights_list)} weight combinations")
        
        # Calculate portfolio metrics for each weight allocation
        portfolio_data = []
        for weights in weights_list:
            # Calculate weighted return
            portfolio_return = weights[0] * spy_return + weights[1] * bmoax_return
            
            # Calculate portfolio volatility using correlation
            # Formula: σp² = w1²σ1² + w2²σ2² + 2w1w2σ1σ2ρ12
            portfolio_var = (weights[0]**2 * spy_vol**2 + 
                            weights[1]**2 * bmoax_vol**2 + 
                            2 * weights[0] * weights[1] * spy_vol * bmoax_vol * correlation)
            portfolio_std = np.sqrt(portfolio_var)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            # Calculate stock percentage for labeling
            stock_pct = weights[0] * 100
            
            portfolio_data.append({
                'return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe': sharpe_ratio,
                'weights': {
                    'SPY': weights[0],
                    'BMOAX': weights[1]
                },
                'stock_pct': stock_pct
            })
        
        print("Portfolio metrics calculated for all mock weight combinations")
        
        # Log a sample of the data
        print(f"Sample mock metrics for 50/50 portfolio (if available):")
        mid_point = len(portfolio_data) // 2
        if 0 <= mid_point < len(portfolio_data):
            print(f"  Return: {portfolio_data[mid_point]['return']:.4f} ({portfolio_data[mid_point]['return']*100:.2f}%)")
            print(f"  Volatility: {portfolio_data[mid_point]['volatility']:.4f} ({portfolio_data[mid_point]['volatility']*100:.2f}%)")
            print(f"  Sharpe Ratio: {portfolio_data[mid_point]['sharpe']:.4f}")
        
        # Format the data for Plotly
        stock_bond_data = {
            'x': [p['volatility'] for p in portfolio_data],
            'y': [p['return'] for p in portfolio_data],
            'mode': 'lines+markers',
            'name': 'Stock-Bond Frontier',
            'type': 'scatter',
            'line': {'color': 'blue', 'width': 2},
            'marker': {'size': 8},
            'text': [f"Stock: {p['stock_pct']:.0f}%, Bond: {(100-p['stock_pct']):.0f}%" 
                     for p in portfolio_data],
            'hoverinfo': 'text+x+y',
            'hovertemplate': 'Volatility: %{x:.4f}<br>Return: %{y:.4f}<br>%{text}'
        }
        
        print("Mock stock-bond frontier data formatted successfully for Plotly")
        
        return {
            'stock_bond_frontier': stock_bond_data,
            'raw_data': portfolio_data
        }
    
    def _generate_random_portfolios(self, num_portfolios=100):
        """Generate random portfolios for comparison."""
        results = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(self.assets))
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate portfolio metrics
            metrics = self.calculate_portfolio_metrics(weights)
            results.append(metrics)
        
        return results
    
    def _portfolio_volatility(self, weights):
        """Calculate portfolio volatility (objective function for min_volatility optimization)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.annual_cov_matrix, weights)))
    
    def _portfolio_return(self, weights):
        """Calculate portfolio return."""
        return np.sum(weights * self.annual_returns)
    
    def _negative_sharpe_ratio(self, weights):
        """Calculate negative Sharpe ratio (objective function for max_sharpe optimization)."""
        p_return = self._portfolio_return(weights)
        p_volatility = self._portfolio_volatility(weights)
        return -(p_return - self.risk_free_rate) / p_volatility
    
    def _portfolio_return_constraint(self, weights, target_return):
        """Constraint function for efficient_return: return = target_return."""
        return self._portfolio_return(weights) - target_return
    
    def _optimize(self, objective='max_sharpe', target_return=None):
        """
        Optimize portfolio based on objective.
        
        Args:
            objective: One of 'min_volatility', 'max_sharpe', 'efficient_return'
            target_return: Target return for 'efficient_return' objective
            
        Returns:
            list: Optimized weights
        """
        num_assets = len(self.assets)
        args = ()
        constraints = []
        
        # Constraint: sum of weights = 1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Set objective function
        if objective == 'min_volatility':
            objective_function = self._portfolio_volatility
        elif objective == 'max_sharpe':
            objective_function = self._negative_sharpe_ratio
        elif objective == 'efficient_return':
            if target_return is None:
                raise ValueError("Target return must be provided for 'efficient_return' objective")
            objective_function = self._portfolio_volatility
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self._portfolio_return_constraint(x, target_return),
                'args': (target_return,)
            })
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Initial guess: equal weights
        initial_weights = np.ones(num_assets) / num_assets
        
        # Bounds: each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Optimize
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            args=args
        )
        
        # Check if optimization succeeded
        if not result['success']:
            raise RuntimeError(f"Optimization failed: {result['message']}")
        
        # Clean up the weights to avoid very small values that might cause issues
        weights = result['x']
        clean_weights = np.array([max(0, w) if abs(w) < 1e-10 else w for w in weights])
        
        # Re-normalize to ensure they still sum to 1
        if np.sum(clean_weights) > 0:
            clean_weights = clean_weights / np.sum(clean_weights)
        
        # Return the optimized weights
        return clean_weights
    
    def optimize_portfolio(self, objective='max_sharpe'):
        """
        Optimize portfolio and return metrics.
        
        Args:
            objective: One of 'min_volatility', 'max_sharpe'
            
        Returns:
            dict: Optimized portfolio metrics
        """
        weights = self._optimize(objective)
        metrics = self.calculate_portfolio_metrics(weights)
        
        return {
            'objective': objective,
            'metrics': metrics
        } 