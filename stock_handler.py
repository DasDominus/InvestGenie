import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class StockHandler:
    """
    Mock handler for generating ETF and fund data.
    In a real application, this would fetch data from an API like Yahoo Finance or Alpha Vantage.
    """
    
    def __init__(self):
        # Define mock funds/ETFs
        self.assets = [
            'US_LARGE_CAP',    # S&P 500 equivalent
            'US_MID_CAP',      # Mid-cap index
            'US_SMALL_CAP',    # Russell 2000 equivalent
            'INTL_DEVELOPED',  # MSCI EAFE equivalent
            'INTL_EMERGING',   # MSCI Emerging Markets
            'REAL_ESTATE',     # REIT index
            'COMMODITIES'      # Broad commodity index
        ]
        
        # Generate mock historical data
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock historical price data for assets."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create date range for the last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
        
        # Generate price data
        self.prices = pd.DataFrame(index=dates)
        
        # Different mean returns and volatilities for diversification
        mean_returns = {
            'US_LARGE_CAP': 0.0010,      # 25.2% annual
            'US_MID_CAP': 0.0011,        # 27.7% annual
            'US_SMALL_CAP': 0.0012,      # 30.2% annual
            'INTL_DEVELOPED': 0.0008,    # 20.2% annual
            'INTL_EMERGING': 0.0009,     # 22.7% annual
            'REAL_ESTATE': 0.0009,       # 22.7% annual
            'COMMODITIES': 0.0006        # 15.1% annual
        }
        
        volatilities = {
            'US_LARGE_CAP': 0.010,       # 15.9% annual
            'US_MID_CAP': 0.012,         # 19.0% annual
            'US_SMALL_CAP': 0.015,       # 23.8% annual
            'INTL_DEVELOPED': 0.011,     # 17.5% annual
            'INTL_EMERGING': 0.016,      # 25.4% annual
            'REAL_ESTATE': 0.013,        # 20.6% annual
            'COMMODITIES': 0.014         # 22.2% annual
        }
        
        # Initialize starting prices
        start_prices = {
            'US_LARGE_CAP': 100.0,
            'US_MID_CAP': 100.0,
            'US_SMALL_CAP': 100.0,
            'INTL_DEVELOPED': 100.0,
            'INTL_EMERGING': 100.0,
            'REAL_ESTATE': 100.0,
            'COMMODITIES': 100.0
        }
        
        # Generate price paths with correlations
        # Create a correlation matrix
        corr_matrix = np.array([
            # US_LC  US_MC  US_SC  INTL_D INTL_E RE     COMMO
            [1.00,  0.90,  0.85,  0.70,  0.65,  0.65,  0.40],  # US_LARGE_CAP
            [0.90,  1.00,  0.90,  0.65,  0.60,  0.60,  0.40],  # US_MID_CAP
            [0.85,  0.90,  1.00,  0.60,  0.65,  0.55,  0.35],  # US_SMALL_CAP
            [0.70,  0.65,  0.60,  1.00,  0.80,  0.60,  0.45],  # INTL_DEVELOPED
            [0.65,  0.60,  0.65,  0.80,  1.00,  0.55,  0.50],  # INTL_EMERGING
            [0.65,  0.60,  0.55,  0.60,  0.55,  1.00,  0.35],  # REAL_ESTATE
            [0.40,  0.40,  0.35,  0.45,  0.50,  0.35,  1.00],  # COMMODITIES
        ])
        
        # Generate correlated random returns
        n_days = len(dates)
        n_assets = len(self.assets)
        
        # Generate uncorrelated random returns
        uncorrelated_returns = np.random.normal(
            size=(n_days, n_assets)
        )
        
        # Cholesky decomposition of correlation matrix
        cholesky = np.linalg.cholesky(corr_matrix)
        
        # Correlate the returns
        correlated_returns = np.dot(uncorrelated_returns, cholesky.T)
        
        # Scale returns and add mean
        for i, asset in enumerate(self.assets):
            vol = volatilities[asset]
            mean = mean_returns[asset]
            correlated_returns[:, i] = correlated_returns[:, i] * vol + mean
        
        # Convert returns to prices
        for i, asset in enumerate(self.assets):
            price = start_prices[asset]
            prices = [price]
            
            for ret in correlated_returns[:, i]:
                price = price * (1 + ret)
                prices.append(price)
            
            self.prices[asset] = prices[:n_days]
    
    def get_assets(self):
        """Return the list of available assets."""
        return self.assets
    
    def get_prices(self):
        """Return the historical price data."""
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
        return {
            'US_LARGE_CAP': 'US Large Cap ETF',
            'US_MID_CAP': 'US Mid Cap ETF',
            'US_SMALL_CAP': 'US Small Cap ETF',
            'INTL_DEVELOPED': 'International Developed ETF',
            'INTL_EMERGING': 'Emerging Markets ETF',
            'REAL_ESTATE': 'Real Estate ETF',
            'COMMODITIES': 'Commodities ETF'
        } 