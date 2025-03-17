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