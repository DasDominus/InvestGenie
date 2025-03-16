import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class StockHandler:
    """
    Mock handler for generating stock data.
    In a real application, this would fetch data from an API like Yahoo Finance or Alpha Vantage.
    """
    
    def __init__(self):
        # Define mock assets - original assets
        self.original_assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK.B', 'JNJ', 'JPM', 'V', 'PG']
        
        # Define grid-based allocation assets
        self.equity_assets = [
            'equity_small_growth', 'equity_small_blend', 'equity_small_value',
            'equity_mid_growth', 'equity_mid_blend', 'equity_mid_value',
            'equity_large_growth', 'equity_large_blend', 'equity_large_value'
        ]
        
        self.fixed_income_assets = [
            'fixed_short_low', 'fixed_short_mid', 'fixed_short_high',
            'fixed_mid_low', 'fixed_mid_mid', 'fixed_mid_high',
            'fixed_long_low', 'fixed_long_mid', 'fixed_long_high'
        ]
        
        # Combine all assets
        self.assets = self.original_assets + self.equity_assets + self.fixed_income_assets
        
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
        
        # Define mean returns and volatilities for original assets
        mean_returns = {
            'AAPL': 0.0015, 'MSFT': 0.0014, 'AMZN': 0.0016, 'GOOGL': 0.0012, 'META': 0.0013,
            'BRK.B': 0.0008, 'JNJ': 0.0006, 'JPM': 0.0009, 'V': 0.0011, 'PG': 0.0007
        }
        
        volatilities = {
            'AAPL': 0.018, 'MSFT': 0.017, 'AMZN': 0.022, 'GOOGL': 0.019, 'META': 0.025,
            'BRK.B': 0.014, 'JNJ': 0.012, 'JPM': 0.020, 'V': 0.016, 'PG': 0.011
        }
        
        # Define mean returns and volatilities for grid-based assets
        # Equity assets - different risk/return profiles based on categories
        for asset in self.equity_assets:
            _, size, style = asset.split('_')
            
            # Base return and volatility by size
            if size == 'small':
                base_return = 0.0018
                base_vol = 0.022
            elif size == 'mid':
                base_return = 0.0014
                base_vol = 0.018
            else:  # large
                base_return = 0.0010
                base_vol = 0.015
            
            # Adjust by style
            if style == 'growth':
                mean_returns[asset] = base_return * 1.3
                volatilities[asset] = base_vol * 1.2
            elif style == 'blend':
                mean_returns[asset] = base_return
                volatilities[asset] = base_vol
            else:  # value
                mean_returns[asset] = base_return * 0.8
                volatilities[asset] = base_vol * 0.9
        
        # Fixed income assets - different risk/return profiles based on categories
        for asset in self.fixed_income_assets:
            _, term, risk = asset.split('_')
            
            # Base return and volatility by term
            if term == 'short':
                base_return = 0.0004
                base_vol = 0.005
            elif term == 'mid':
                base_return = 0.0006
                base_vol = 0.008
            else:  # long
                base_return = 0.0008
                base_vol = 0.012
            
            # Adjust by risk level
            if risk == 'low':
                mean_returns[asset] = base_return * 0.8
                volatilities[asset] = base_vol * 0.7
            elif risk == 'mid':
                mean_returns[asset] = base_return
                volatilities[asset] = base_vol
            else:  # high
                mean_returns[asset] = base_return * 1.5
                volatilities[asset] = base_vol * 1.4
        
        # Initialize starting prices
        start_prices = {asset: 100.0 for asset in self.assets}
        
        # Update with custom starting prices for original assets
        start_prices.update({
            'AAPL': 100.0, 'MSFT': 150.0, 'AMZN': 1800.0, 'GOOGL': 1200.0, 'META': 200.0,
            'BRK.B': 250.0, 'JNJ': 130.0, 'JPM': 110.0, 'V': 180.0, 'PG': 120.0
        })
        
        # Generate price paths with correlations
        # Create a correlation matrix
        corr_matrix = np.array([
            [1.00, 0.75, 0.65, 0.70, 0.60, 0.45, 0.30, 0.40, 0.35, 0.25],  # AAPL
            [0.75, 1.00, 0.60, 0.75, 0.65, 0.40, 0.25, 0.35, 0.40, 0.20],  # MSFT
            [0.65, 0.60, 1.00, 0.65, 0.60, 0.35, 0.20, 0.30, 0.45, 0.15],  # AMZN
            [0.70, 0.75, 0.65, 1.00, 0.70, 0.30, 0.25, 0.35, 0.40, 0.20],  # GOOGL
            [0.60, 0.65, 0.60, 0.70, 1.00, 0.30, 0.20, 0.40, 0.35, 0.15],  # META
            [0.45, 0.40, 0.35, 0.30, 0.30, 1.00, 0.50, 0.55, 0.45, 0.40],  # BRK.B
            [0.30, 0.25, 0.20, 0.25, 0.20, 0.50, 1.00, 0.45, 0.40, 0.60],  # JNJ
            [0.40, 0.35, 0.30, 0.35, 0.40, 0.55, 0.45, 1.00, 0.60, 0.35],  # JPM
            [0.35, 0.40, 0.45, 0.40, 0.35, 0.45, 0.40, 0.60, 1.00, 0.30],  # V
            [0.25, 0.20, 0.15, 0.20, 0.15, 0.40, 0.60, 0.35, 0.30, 1.00]   # PG
        ])
        
        # Extend correlation matrix for all assets
        n_total_assets = len(self.assets)
        n_original_assets = len(self.original_assets)
        extended_corr = np.eye(n_total_assets)
        
        # Copy original correlation matrix
        extended_corr[:n_original_assets, :n_original_assets] = corr_matrix
        
        # Set correlations for equity assets
        # Define indices ranges
        equity_indices = range(n_original_assets, n_original_assets + len(self.equity_assets))
        fixed_income_indices = range(n_original_assets + len(self.equity_assets), n_total_assets)
        
        # Equity assets correlate higher with each other than with other asset types
        for i in equity_indices:
            equity_asset = self.assets[i]
            
            # Correlate with original assets - higher with tech stocks for growth assets
            for j in range(n_original_assets):
                original_asset = self.original_assets[j]
                
                # Parse category
                _, size, style = equity_asset.split('_')
                
                # Growth assets correlate more with tech stocks
                if style == 'growth' and original_asset in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
                    corr_value = 0.6
                # Value assets correlate more with value stocks
                elif style == 'value' and original_asset in ['BRK.B', 'JNJ', 'JPM', 'PG']:
                    corr_value = 0.55
                else:
                    corr_value = 0.3
                
                extended_corr[i, j] = corr_value
                extended_corr[j, i] = corr_value
            
            # Correlate with other equity assets
            for j in equity_indices:
                if i != j:
                    equity_asset_j = self.assets[j]
                    _, size_j, style_j = equity_asset_j.split('_')
                    
                    # Same size/style categories correlate higher
                    if size == size_j and style == style_j:
                        corr_value = 0.85
                    elif size == size_j or style == style_j:
                        corr_value = 0.7
                    else:
                        corr_value = 0.5
                    
                    extended_corr[i, j] = corr_value
                    extended_corr[j, i] = corr_value
        
        # Fixed income assets correlate higher with each other
        for i in fixed_income_indices:
            fixed_asset = self.assets[i]
            
            # Correlate with original assets - generally lower correlation with stocks
            for j in range(n_original_assets):
                corr_value = 0.2
                extended_corr[i, j] = corr_value
                extended_corr[j, i] = corr_value
            
            # Correlate with equity assets - generally moderate inverse correlation
            for j in equity_indices:
                corr_value = -0.15
                extended_corr[i, j] = corr_value
                extended_corr[j, i] = corr_value
            
            # Correlate with other fixed income assets
            for j in fixed_income_indices:
                if i != j:
                    fixed_asset_j = self.assets[j]
                    _, term, risk = fixed_asset.split('_')
                    _, term_j, risk_j = fixed_asset_j.split('_')
                    
                    # Same term/risk categories correlate higher
                    if term == term_j and risk == risk_j:
                        corr_value = 0.85
                    elif term == term_j or risk == risk_j:
                        corr_value = 0.7
                    else:
                        corr_value = 0.4
                    
                    extended_corr[i, j] = corr_value
                    extended_corr[j, i] = corr_value
        
        # Generate correlated returns
        np.random.seed(42)
        n_days = len(dates)
        
        # Using Cholesky decomposition to generate correlated random variables
        L = np.linalg.cholesky(extended_corr)
        
        # Generate independent random variables
        uncorrelated_returns = np.random.normal(size=(n_days, n_total_assets))
        
        # Transform to correlated random variables
        correlated_returns = uncorrelated_returns @ L.T
        
        # Adjust for desired mean and volatility
        for i, asset in enumerate(self.assets):
            daily_mean = mean_returns[asset]
            daily_vol = volatilities[asset]
            
            # Convert to daily returns with correct mean and volatility
            asset_returns = correlated_returns[:, i]
            asset_returns = (asset_returns * daily_vol) + daily_mean
            
            # Generate price path
            price = start_prices[asset]
            prices = [price]
            
            for ret in asset_returns:
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