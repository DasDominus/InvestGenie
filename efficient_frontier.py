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
    
    def __init__(self, handler):
        """Initialize with a data handler object."""
        self.handler = handler
        self.assets = handler.get_assets()
        
        # Get asset types if available
        self.equity_assets = []
        self.fixed_income_assets = []
        if hasattr(handler, 'get_equity_assets') and hasattr(handler, 'get_fixed_income_assets'):
            self.equity_assets = handler.get_equity_assets()
            self.fixed_income_assets = handler.get_fixed_income_assets()
        
        # Calculate returns and covariance matrix
        self.returns = handler.get_returns()
        self.mean_returns = handler.get_mean_returns()
        self.cov_matrix = handler.get_cov_matrix()
        
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
    
    def _get_asset_type_constraint(self, weights, asset_type, min_val=None, max_val=None):
        """
        Create a constraint function for an asset type's allocation.
        
        Args:
            weights: The portfolio weights
            asset_type: 'equity' or 'fixed_income'
            min_val: Minimum allocation to this asset type (0-1)
            max_val: Maximum allocation to this asset type (0-1)
            
        Returns:
            Function that will be > 0 if the constraint is satisfied
        """
        assets = self.equity_assets if asset_type == 'equity' else self.fixed_income_assets
        asset_indices = [self.assets.index(asset) for asset in assets if asset in self.assets]
        
        # Sum the weights for this asset type
        type_weight = sum(weights[i] for i in asset_indices)
        
        if min_val is not None and max_val is not None:
            # Both min and max constraints
            return min_val <= type_weight <= max_val
        elif min_val is not None:
            # Just min constraint
            return type_weight >= min_val
        elif max_val is not None:
            # Just max constraint
            return type_weight <= max_val
        else:
            # No constraint
            return True
    
    def _equity_min_constraint(self, weights, min_val):
        """Constraint function for minimum equity allocation."""
        indices = [self.assets.index(asset) for asset in self.equity_assets if asset in self.assets]
        return sum(weights[i] for i in indices) - min_val
    
    def _equity_max_constraint(self, weights, max_val):
        """Constraint function for maximum equity allocation."""
        indices = [self.assets.index(asset) for asset in self.equity_assets if asset in self.assets]
        return max_val - sum(weights[i] for i in indices)
    
    def _fixed_income_min_constraint(self, weights, min_val):
        """Constraint function for minimum fixed income allocation."""
        indices = [self.assets.index(asset) for asset in self.fixed_income_assets if asset in self.assets]
        return sum(weights[i] for i in indices) - min_val
    
    def _fixed_income_max_constraint(self, weights, max_val):
        """Constraint function for maximum fixed income allocation."""
        indices = [self.assets.index(asset) for asset in self.fixed_income_assets if asset in self.assets]
        return max_val - sum(weights[i] for i in indices)
    
    def _optimize(self, objective='max_sharpe', target_return=None, constraints=None):
        """
        Optimize portfolio based on objective.
        
        Args:
            objective: One of 'min_volatility', 'max_sharpe', 'efficient_return'
            target_return: Target return for 'efficient_return' objective
            constraints: Dict with asset type constraints (min_equity, max_equity, etc.)
            
        Returns:
            list: Optimized weights
        """
        num_assets = len(self.assets)
        args = ()
        opt_constraints = []
        
        # Default constraints
        if constraints is None:
            constraints = {}
        
        # Constraint: sum of weights = 1
        opt_constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Add asset type constraints if provided
        if self.equity_assets and 'min_equity' in constraints:
            opt_constraints.append({
                'type': 'ineq',
                'fun': self._equity_min_constraint,
                'args': (constraints['min_equity'],)
            })
        
        if self.equity_assets and 'max_equity' in constraints:
            opt_constraints.append({
                'type': 'ineq',
                'fun': self._equity_max_constraint,
                'args': (constraints['max_equity'],)
            })
        
        if self.fixed_income_assets and 'min_fixed_income' in constraints:
            opt_constraints.append({
                'type': 'ineq',
                'fun': self._fixed_income_min_constraint,
                'args': (constraints['min_fixed_income'],)
            })
        
        if self.fixed_income_assets and 'max_fixed_income' in constraints:
            opt_constraints.append({
                'type': 'ineq',
                'fun': self._fixed_income_max_constraint,
                'args': (constraints['max_fixed_income'],)
            })
        
        # Set objective function
        if objective == 'min_volatility':
            objective_function = self._portfolio_volatility
        elif objective == 'max_sharpe':
            objective_function = self._negative_sharpe_ratio
        elif objective == 'efficient_return':
            if target_return is None:
                raise ValueError("Target return must be provided for 'efficient_return' objective")
            objective_function = self._portfolio_volatility
            opt_constraints.append({
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
            constraints=opt_constraints,
            args=args
        )
        
        # Check if optimization succeeded
        if not result['success']:
            raise RuntimeError(f"Optimization failed: {result['message']}")
        
        # Return the optimized weights
        return result['x']
    
    def optimize_portfolio(self, objective='max_sharpe', constraints=None):
        """
        Optimize portfolio and return metrics.
        
        Args:
            objective: One of 'min_volatility', 'max_sharpe'
            constraints: Dict with asset type constraints
            
        Returns:
            dict: Optimized portfolio metrics
        """
        weights = self._optimize(objective, constraints=constraints)
        metrics = self.calculate_portfolio_metrics(weights)
        
        # Calculate asset type allocations
        if self.equity_assets and self.fixed_income_assets:
            equity_weight = sum(metrics['weights'][asset] for asset in self.equity_assets if asset in metrics['weights'])
            fixed_income_weight = sum(metrics['weights'][asset] for asset in self.fixed_income_assets if asset in metrics['weights'])
            
            metrics['asset_type_weights'] = {
                'equity': equity_weight,
                'fixed_income': fixed_income_weight
            }
        
        return {
            'objective': objective,
            'metrics': metrics,
            'constraints': constraints
        } 