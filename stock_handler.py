import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.linalg import eigh

class StockHandler:
    """
    Mock handler for generating ETF and fund data.
    In a real application, this would fetch data from an API like Yahoo Finance or Alpha Vantage.
    """
    
    def __init__(self):
        # Define mock funds/ETFs with full Morningstar style box (3x3 grid)
        self.assets = [
            # Large Cap (3 styles)
            'US_LARGE_CAP_GROWTH',
            'US_LARGE_CAP_BLEND',
            'US_LARGE_CAP_VALUE',
            
            # Mid Cap (3 styles)
            'US_MID_CAP_GROWTH',
            'US_MID_CAP_BLEND',
            'US_MID_CAP_VALUE',
            
            # Small Cap (3 styles)
            'US_SMALL_CAP_GROWTH',
            'US_SMALL_CAP_BLEND',
            'US_SMALL_CAP_VALUE',
            
            # Other asset classes (kept from original)
            'INTL_DEVELOPED',  # MSCI EAFE equivalent
            'INTL_EMERGING',   # MSCI Emerging Markets
            'REAL_ESTATE',     # REIT index
            'COMMODITIES'      # Broad commodity index
        ]
        
        # Generate mock historical data
        self._generate_mock_data()
    
    def _make_positive_definite(self, matrix, epsilon=1e-6):
        """
        Convert a matrix to the nearest positive definite matrix
        using eigenvalue adjustment.
        """
        # Ensure matrix is symmetric
        n = matrix.shape[0]
        symmetric_matrix = (matrix + matrix.T) / 2
        
        # Fix diagonal to ensure all 1's
        np.fill_diagonal(symmetric_matrix, 1.0)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(symmetric_matrix)
        
        # Replace any eigenvalue less than epsilon with epsilon
        eigenvalues = np.maximum(eigenvalues, epsilon)
        
        # Reconstruct the matrix
        pd_matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
        
        # Rescale to ensure it's a correlation matrix with 1's on diagonal
        d = np.sqrt(np.diag(pd_matrix))
        pd_matrix = pd_matrix / np.outer(d, d)
        np.fill_diagonal(pd_matrix, 1.0)
        
        return pd_matrix
    
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
            # Large Cap
            'US_LARGE_CAP_GROWTH': 0.0011,  # 27.7% annual
            'US_LARGE_CAP_BLEND': 0.0010,   # 25.2% annual
            'US_LARGE_CAP_VALUE': 0.0009,   # 22.7% annual
            
            # Mid Cap
            'US_MID_CAP_GROWTH': 0.0012,    # 30.2% annual
            'US_MID_CAP_BLEND': 0.0011,     # 27.7% annual
            'US_MID_CAP_VALUE': 0.0010,     # 25.2% annual
            
            # Small Cap
            'US_SMALL_CAP_GROWTH': 0.0013,  # 32.8% annual
            'US_SMALL_CAP_BLEND': 0.0012,   # 30.2% annual
            'US_SMALL_CAP_VALUE': 0.0011,   # 27.7% annual
            
            # Other assets
            'INTL_DEVELOPED': 0.0008,       # 20.2% annual
            'INTL_EMERGING': 0.0009,        # 22.7% annual
            'REAL_ESTATE': 0.0009,          # 22.7% annual
            'COMMODITIES': 0.0006           # 15.1% annual
        }
        
        volatilities = {
            # Large Cap
            'US_LARGE_CAP_GROWTH': 0.011,   # 17.5% annual
            'US_LARGE_CAP_BLEND': 0.010,    # 15.9% annual
            'US_LARGE_CAP_VALUE': 0.009,    # 14.3% annual
            
            # Mid Cap
            'US_MID_CAP_GROWTH': 0.013,     # 20.6% annual
            'US_MID_CAP_BLEND': 0.012,      # 19.0% annual
            'US_MID_CAP_VALUE': 0.011,      # 17.5% annual
            
            # Small Cap
            'US_SMALL_CAP_GROWTH': 0.017,   # 27.0% annual
            'US_SMALL_CAP_BLEND': 0.015,    # 23.8% annual
            'US_SMALL_CAP_VALUE': 0.014,    # 22.2% annual
            
            # Other assets
            'INTL_DEVELOPED': 0.011,        # 17.5% annual
            'INTL_EMERGING': 0.016,         # 25.4% annual
            'REAL_ESTATE': 0.013,           # 20.6% annual
            'COMMODITIES': 0.014            # 22.2% annual
        }
        
        # Initialize starting prices
        start_prices = {asset: 100.0 for asset in self.assets}
        
        # Generate price paths with correlations
        # Create a correlation matrix (13x13 now)
        # Format: US_LG, US_LB, US_LV, US_MG, US_MB, US_MV, US_SG, US_SB, US_SV, INT_D, INT_E, RE, COMMO
        initial_corr_matrix = np.array([
            # US_LG  US_LB  US_LV  US_MG  US_MB  US_MV  US_SG  US_SB  US_SV  INT_D  INT_E  RE     COMMO
            [1.00,  0.90,  0.80,  0.85,  0.75,  0.70,  0.70,  0.65,  0.60,  0.65,  0.60,  0.60,  0.40],  # US_LARGE_GROWTH
            [0.90,  1.00,  0.90,  0.80,  0.85,  0.75,  0.65,  0.70,  0.65,  0.70,  0.65,  0.65,  0.40],  # US_LARGE_BLEND
            [0.80,  0.90,  1.00,  0.70,  0.80,  0.85,  0.60,  0.65,  0.70,  0.70,  0.65,  0.65,  0.40],  # US_LARGE_VALUE
            [0.85,  0.80,  0.70,  1.00,  0.90,  0.80,  0.85,  0.75,  0.70,  0.60,  0.55,  0.55,  0.35],  # US_MID_GROWTH
            [0.75,  0.85,  0.80,  0.90,  1.00,  0.90,  0.75,  0.85,  0.75,  0.65,  0.60,  0.60,  0.35],  # US_MID_BLEND
            [0.70,  0.75,  0.85,  0.80,  0.90,  1.00,  0.70,  0.75,  0.85,  0.65,  0.60,  0.60,  0.35],  # US_MID_VALUE
            [0.70,  0.65,  0.60,  0.85,  0.75,  0.70,  1.00,  0.90,  0.80,  0.55,  0.60,  0.50,  0.30],  # US_SMALL_GROWTH
            [0.65,  0.70,  0.65,  0.75,  0.85,  0.75,  0.90,  1.00,  0.90,  0.60,  0.65,  0.55,  0.30],  # US_SMALL_BLEND
            [0.60,  0.65,  0.70,  0.70,  0.75,  0.85,  0.80,  0.90,  1.00,  0.60,  0.65,  0.55,  0.30],  # US_SMALL_VALUE
            [0.65,  0.70,  0.70,  0.60,  0.65,  0.65,  0.55,  0.60,  0.60,  1.00,  0.80,  0.60,  0.45],  # INTL_DEVELOPED
            [0.60,  0.65,  0.65,  0.55,  0.60,  0.60,  0.60,  0.65,  0.65,  0.80,  1.00,  0.55,  0.50],  # INTL_EMERGING
            [0.60,  0.65,  0.65,  0.55,  0.60,  0.60,  0.50,  0.55,  0.55,  0.60,  0.55,  1.00,  0.35],  # REAL_ESTATE
            [0.40,  0.40,  0.40,  0.35,  0.35,  0.35,  0.30,  0.30,  0.30,  0.45,  0.50,  0.35,  1.00],  # COMMODITIES
        ])
        
        # Make sure the correlation matrix is positive definite
        corr_matrix = self._make_positive_definite(initial_corr_matrix)
        
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
    
    def get_morningstar_style_box(self):
        """Return the Morningstar style box categories and their corresponding assets."""
        return {
            'large_growth': 'US_LARGE_CAP_GROWTH',
            'large_blend': 'US_LARGE_CAP_BLEND',
            'large_value': 'US_LARGE_CAP_VALUE',
            'mid_growth': 'US_MID_CAP_GROWTH',
            'mid_blend': 'US_MID_CAP_BLEND',
            'mid_value': 'US_MID_CAP_VALUE',
            'small_growth': 'US_SMALL_CAP_GROWTH',
            'small_blend': 'US_SMALL_CAP_BLEND',
            'small_value': 'US_SMALL_CAP_VALUE'
        }
        
    def get_asset_names_display(self):
        """Return more readable names for display in UI."""
        return {
            'US_LARGE_CAP_GROWTH': 'US Large Cap Growth',
            'US_LARGE_CAP_BLEND': 'US Large Cap Blend',
            'US_LARGE_CAP_VALUE': 'US Large Cap Value',
            'US_MID_CAP_GROWTH': 'US Mid Cap Growth',
            'US_MID_CAP_BLEND': 'US Mid Cap Blend',
            'US_MID_CAP_VALUE': 'US Mid Cap Value',
            'US_SMALL_CAP_GROWTH': 'US Small Cap Growth',
            'US_SMALL_CAP_BLEND': 'US Small Cap Blend',
            'US_SMALL_CAP_VALUE': 'US Small Cap Value',
            'INTL_DEVELOPED': 'International Developed',
            'INTL_EMERGING': 'Emerging Markets',
            'REAL_ESTATE': 'Real Estate',
            'COMMODITIES': 'Commodities'
        } 