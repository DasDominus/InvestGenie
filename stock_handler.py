import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class StockHandler:
    """
    Mock handler for generating stock data.
    In a real application, this would fetch data from an API like Yahoo Finance or Alpha Vantage.
    """
    
    def __init__(self):
        # Define mock assets
        self.assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK.B', 'JNJ', 'JPM', 'V', 'PG']
        
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
            'AAPL': 0.0015, 'MSFT': 0.0014, 'AMZN': 0.0016, 'GOOGL': 0.0012, 'META': 0.0013,
            'BRK.B': 0.0008, 'JNJ': 0.0006, 'JPM': 0.0009, 'V': 0.0011, 'PG': 0.0007
        }
        
        volatilities = {
            'AAPL': 0.018, 'MSFT': 0.017, 'AMZN': 0.022, 'GOOGL': 0.019, 'META': 0.025,
            'BRK.B': 0.014, 'JNJ': 0.012, 'JPM': 0.020, 'V': 0.016, 'PG': 0.011
        }
        
        # Initialize starting prices
        start_prices = {
            'AAPL': 100.0, 'MSFT': 150.0, 'AMZN': 1800.0, 'GOOGL': 1200.0, 'META': 200.0,
            'BRK.B': 250.0, 'JNJ': 130.0, 'JPM': 110.0, 'V': 180.0, 'PG': 120.0
        }
        
        # Generate price paths with correlations
        # Create a correlation matrix
        corr_matrix = np.array([
            [1.0,  0.7,  0.6,  0.7,  0.6,  0.4,  0.3,  0.5,  0.5,  0.3],  # AAPL
            [0.7,  1.0,  0.6,  0.8,  0.6,  0.4,  0.3,  0.4,  0.5,  0.3],  # MSFT
            [0.6,  0.6,  1.0,  0.7,  0.7,  0.3,  0.2,  0.4,  0.4,  0.2],  # AMZN
            [0.7,  0.8,  0.7,  1.0,  0.7,  0.3,  0.2,  0.3,  0.4,  0.2],  # GOOGL
            [0.6,  0.6,  0.7,  0.7,  1.0,  0.3,  0.2,  0.4,  0.4,  0.2],  # META
            [0.4,  0.4,  0.3,  0.3,  0.3,  1.0,  0.4,  0.5,  0.4,  0.3],  # BRK.B
            [0.3,  0.3,  0.2,  0.2,  0.2,  0.4,  1.0,  0.3,  0.3,  0.5],  # JNJ
            [0.5,  0.4,  0.4,  0.3,  0.4,  0.5,  0.3,  1.0,  0.6,  0.3],  # JPM
            [0.5,  0.5,  0.4,  0.4,  0.4,  0.4,  0.3,  0.6,  1.0,  0.3],  # V
            [0.3,  0.3,  0.2,  0.2,  0.2,  0.3,  0.5,  0.3,  0.3,  1.0],  # PG
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