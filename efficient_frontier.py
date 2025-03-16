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
        
        # Calculate annual returns and covariance (assuming 252 trading days per year)
        self.annual_returns = self.mean_returns * 252
        self.annual_cov_matrix = self.cov_matrix * 252
        
        # Set risk-free rate
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
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
        
        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'weights': {asset: float(weight) for asset, weight in zip(self.assets, weights)}
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
        
        # Prepare return data
        return {
            'efficient_frontier': {
                'returns': ef_returns,
                'volatilities': ef_volatilities,
                'sharpe_ratios': ef_sharpe_ratios
            },
            'min_volatility': {
                'return': min_vol_metrics['expected_return'],
                'volatility': min_vol_metrics['volatility'],
                'sharpe_ratio': min_vol_metrics['sharpe_ratio'],
                'weights': min_vol_metrics['weights']
            },
            'max_sharpe': {
                'return': max_sharpe_metrics['expected_return'],
                'volatility': max_sharpe_metrics['volatility'],
                'sharpe_ratio': max_sharpe_metrics['sharpe_ratio'],
                'weights': max_sharpe_metrics['weights']
            },
            'random_portfolios': {
                'returns': [p['expected_return'] for p in random_portfolios],
                'volatilities': [p['volatility'] for p in random_portfolios],
                'sharpe_ratios': [p['sharpe_ratio'] for p in random_portfolios]
            },
            'assets': self.assets
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
        
        # Return the optimized weights
        return result['x']
    
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