import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FixedIncomeHandler:
    """
    Mock handler for generating fixed income data.
    In a real application, this would fetch data from an API like Yahoo Finance or a bond data provider.
    """
    
    def __init__(self):
        # Define mock fixed income assets
        self.assets = [
            'TREASURY_1_3YR',    # Short-term Treasury bonds
            'TREASURY_7_10YR',   # Medium-term Treasury bonds
            'TREASURY_20YR_PLUS', # Long-term Treasury bonds
            'CORPORATE_SHORT',   # Short-term corporate bonds
            'CORPORATE_INTERMED', # Intermediate corporate bonds
            'HIGH_YIELD',        # High-yield corporate bonds
            'MUNICIPAL',         # Municipal bonds
            'INTERNATIONAL'      # International bonds
        ]
        
        # Generate mock historical data
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock historical price data for fixed income assets."""
        # Set random seed for reproducibility
        np.random.seed(43)  # Different seed from stock handler
        
        # Create date range for the last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
        
        # Generate price data
        self.prices = pd.DataFrame(index=dates)
        
        # Different mean returns and volatilities
        # Fixed income typically has lower returns and volatility than stocks
        mean_returns = {
            'TREASURY_1_3YR': 0.00008,     # 2.0% annual
            'TREASURY_7_10YR': 0.00010,    # 2.5% annual
            'TREASURY_20YR_PLUS': 0.00012, # 3.0% annual
            'CORPORATE_SHORT': 0.00010,    # 2.5% annual
            'CORPORATE_INTERMED': 0.00014, # 3.5% annual
            'HIGH_YIELD': 0.00020,         # 5.0% annual
            'MUNICIPAL': 0.00009,          # 2.3% annual
            'INTERNATIONAL': 0.00012       # 3.0% annual
        }
        
        volatilities = {
            'TREASURY_1_3YR': 0.0020,      # 3.2% annual
            'TREASURY_7_10YR': 0.0035,     # 5.6% annual
            'TREASURY_20YR_PLUS': 0.0060,  # 9.5% annual
            'CORPORATE_SHORT': 0.0025,     # 4.0% annual
            'CORPORATE_INTERMED': 0.0040,  # 6.3% annual
            'HIGH_YIELD': 0.0080,          # 12.7% annual
            'MUNICIPAL': 0.0030,           # 4.8% annual
            'INTERNATIONAL': 0.0045        # 7.1% annual
        }
        
        # Initialize starting prices
        start_prices = {
            'TREASURY_1_3YR': 100.0,
            'TREASURY_7_10YR': 100.0,
            'TREASURY_20YR_PLUS': 100.0,
            'CORPORATE_SHORT': 100.0,
            'CORPORATE_INTERMED': 100.0,
            'HIGH_YIELD': 100.0,
            'MUNICIPAL': 100.0,
            'INTERNATIONAL': 100.0
        }
        
        # Generate price paths with correlations
        # Create a correlation matrix
        corr_matrix = np.array([
            # T1-3  T7-10 T20+  CS    CI    HY    MUNI  INTL
            [1.00, 0.85, 0.65, 0.80, 0.60, 0.30, 0.70, 0.50],  # TREASURY_1_3YR
            [0.85, 1.00, 0.90, 0.75, 0.80, 0.40, 0.75, 0.60],  # TREASURY_7_10YR
            [0.65, 0.90, 1.00, 0.60, 0.75, 0.35, 0.65, 0.55],  # TREASURY_20YR_PLUS
            [0.80, 0.75, 0.60, 1.00, 0.85, 0.60, 0.75, 0.55],  # CORPORATE_SHORT
            [0.60, 0.80, 0.75, 0.85, 1.00, 0.70, 0.70, 0.65],  # CORPORATE_INTERMED
            [0.30, 0.40, 0.35, 0.60, 0.70, 1.00, 0.50, 0.60],  # HIGH_YIELD
            [0.70, 0.75, 0.65, 0.75, 0.70, 0.50, 1.00, 0.60],  # MUNICIPAL
            [0.50, 0.60, 0.55, 0.55, 0.65, 0.60, 0.60, 1.00],  # INTERNATIONAL
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
            'TREASURY_1_3YR': 'Short-Term Treasury Bonds',
            'TREASURY_7_10YR': 'Medium-Term Treasury Bonds',
            'TREASURY_20YR_PLUS': 'Long-Term Treasury Bonds',
            'CORPORATE_SHORT': 'Short-Term Corporate Bonds',
            'CORPORATE_INTERMED': 'Intermediate Corporate Bonds',
            'HIGH_YIELD': 'High-Yield Corporate Bonds',
            'MUNICIPAL': 'Municipal Bonds',
            'INTERNATIONAL': 'International Bonds'
        } 