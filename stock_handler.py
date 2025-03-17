import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class StockHandler:
    """
    Handler for retrieving and processing stock data using Yahoo Finance API.
    """
    
    def __init__(self):
        # Define the fund categories and their representative tickers as specified
        
        # Equity funds
        self.equity_large_value = "SWLVX"  # Schwab U.S. Large-Cap Value Index Fund
        self.equity_large_blend = "SWTSX"  # Schwab Total Stock Market Index Fund
        self.equity_large_growth = "FSPGX"  # Fidelity® Large Cap Growth Index Fund
        
        self.equity_mid_value = "SWMCX"  # Schwab U.S. Mid-Cap Index Fund
        self.equity_mid_blend = "VEXAX"  # Vanguard Extended Market Index Fund Admiral Shares
        self.equity_mid_growth = "IJK"  # iShares S&P Mid-Cap 400 Growth ETF
        
        self.equity_small_value = "VBR"  # Vanguard Small-Cap Value ETF
        self.equity_small_blend = "SWSSX"  # Schwab Small-Cap Index Fund
        self.equity_small_growth = "FCPGX"  # Fidelity Small Cap Growth
        
        # Fixed income funds
        self.fixed_short_low = "FSHBX"  # Fidelity® Short-Term Bond Fund (Investment Grade)
        self.fixed_short_mid = "SJNK"  # SPDR® Bloomberg Short Term High Yield Bond ETF (High Yield/Junk)
        self.fixed_short_high = "ISTB"  # iShares Core 1-5 Year USD Bond ETF (U.S. Government)
        
        self.fixed_mid_low = "FBNDX"  # Fidelity® Investment Grade Bond Fund
        self.fixed_mid_mid = "LSYAX"  # Lord Abbett Short Duration High Yield Fund
        self.fixed_mid_high = "VGIT"  # Vanguard Intermediate-Term Government Bond ETF
        
        self.fixed_long_low = "VCLT"  # Vanguard Long-Term Corporate Bond ETF
        self.fixed_long_mid = "JNK"  # SPDR® Bloomberg High Yield Bond ETF
        self.fixed_long_high = "VGLT"  # Vanguard Long-Term Government Bond ETF
        
        # Organize funds into categories
        self.equity_assets = [
            self.equity_small_growth, self.equity_small_blend, self.equity_small_value,
            self.equity_mid_growth, self.equity_mid_blend, self.equity_mid_value,
            self.equity_large_growth, self.equity_large_blend, self.equity_large_value
        ]
        
        self.fixed_income_assets = [
            self.fixed_short_low, self.fixed_short_mid, self.fixed_short_high,
            self.fixed_mid_low, self.fixed_mid_mid, self.fixed_mid_high,
            self.fixed_long_low, self.fixed_long_mid, self.fixed_long_high
        ]
        
        # Define fund descriptions for display
        self.fund_descriptions = {
            # Equity - Large Cap
            "SWLVX": "Schwab U.S. Large-Cap Value Index Fund (Large-Cap Value)",
            "SWTSX": "Schwab Total Stock Market Index Fund (Large-Cap Blend)",
            "FSPGX": "Fidelity® Large Cap Growth Index Fund (Large-Cap Growth)",
            
            # Equity - Mid Cap
            "SWMCX": "Schwab U.S. Mid-Cap Index Fund (Mid-Cap Value)",
            "VEXAX": "Vanguard Extended Market Index Fund Admiral Shares (Mid-Cap Blend)",
            "JSMD": "Janus Henderson Small/Mid Cap Growth Alpha ETF (Mid/Small-Cap Growth)",
            
            # Equity - Small Cap
            "VBR": "Vanguard Small-Cap Value ETF (Small-Cap Value)",
            "SWSSX": "Schwab Small-Cap Index Fund (Small-Cap Blend)",
            
            # Fixed Income - Short Term
            "FSHBX": "Fidelity® Short-Term Bond Fund (Short-Term, Investment Grade)",
            "SJNK": "SPDR® Bloomberg Short Term High Yield Bond ETF (Short-Term, High Yield)",
            "ISTB": "iShares Core 1-5 Year USD Bond ETF (Short-Term, U.S. Government)",
            
            # Fixed Income - Intermediate Term
            "FBNDX": "Fidelity® Investment Grade Bond Fund (Intermediate-Term, Investment Grade)",
            "LSYAX": "Lord Abbett Short Duration High Yield Fund (Intermediate-Term, High Yield)",
            "VGIT": "Vanguard Intermediate-Term Government Bond ETF (Intermediate-Term, U.S. Government)",
            
            # Fixed Income - Long Term
            "VCLT": "Vanguard Long-Term Corporate Bond ETF (Long-Term, Investment Grade)",
            "JNK": "SPDR® Bloomberg High Yield Bond ETF (Long-Term, High Yield)",
            "VGLT": "Vanguard Long-Term Government Bond ETF (Long-Term, U.S. Government)"
        }
        
        # Store fund data in memory
        self.fund_data = {}
        
        # Call the data fetching method
        self._fetch_data()
    
    def _fetch_data(self):
        """Fetch historical price data for assets from Yahoo Finance."""
        try:
            # Create date range for the last 5 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            
            # Fetch data for all assets
            data = yf.download(
                tickers=self.equity_assets + self.fixed_income_assets,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False
            )
            
            # Extract closing prices
            # If only one ticker is returned, the data structure is different
            if len(self.equity_assets + self.fixed_income_assets) == 1:
                self.prices = pd.DataFrame(data["Close"])
                self.prices.columns = [self.equity_assets[0] if self.equity_assets else self.fixed_income_assets[0]]
            else:
                # Handle multi-level columns when multiple tickers are fetched
                self.prices = pd.DataFrame()
                for ticker in self.equity_assets + self.fixed_income_assets:
                    try:
                        # Try to get Close price from multi-index
                        if (ticker, 'Close') in data.columns:
                            self.prices[ticker] = data[(ticker, 'Close')]
                        # If ticker doesn't exist in data, use NaN values
                        else:
                            print(f"Warning: No data found for {ticker}")
                            self.prices[ticker] = np.nan
                    except Exception as e:
                        print(f"Error extracting data for {ticker}: {e}")
                        self.prices[ticker] = np.nan
            
            # Drop any rows with all missing values
            self.prices = self.prices.dropna(how='all')
            
            # Fill any remaining NaN values with forward fill followed by backward fill
            # Fix for deprecated fillna method warning
            self.prices = self.prices.ffill().bfill()
            
            # Calculate daily returns
            self.returns = self.prices.pct_change().dropna()
            
            # Calculate mean returns and covariance matrix
            self.mean_returns = self.returns.mean()
            self.cov_matrix = self.returns.cov()
            
            print(f"Successfully fetched data for {len(self.equity_assets + self.fixed_income_assets)} assets")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Fallback to mock data if API fails
            self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock historical price data for assets as a fallback."""
        print("Generating mock data as fallback...")
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create date range for the last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
        
        # Generate price data
        self.prices = pd.DataFrame(index=dates)
        
        # Generate realistic returns for each asset category
        for asset in self.equity_assets + self.fixed_income_assets:
            # Assign realistic returns and volatilities based on asset class
            if asset in self.equity_assets:
                if 'large' in asset:
                    mean_return = 0.0008  # ~20% annual
                    volatility = 0.012    # ~19% annual
                elif 'mid' in asset:
                    mean_return = 0.0009  # ~22% annual
                    volatility = 0.014    # ~22% annual
                else:  # small
                    mean_return = 0.0010  # ~25% annual
                    volatility = 0.016    # ~25% annual
            else:  # fixed income
                if 'short' in asset:
                    mean_return = 0.0002  # ~5% annual
                    volatility = 0.002    # ~3% annual
                elif 'mid' in asset:
                    mean_return = 0.0003  # ~7% annual
                    volatility = 0.004    # ~6% annual
                else:  # long
                    mean_return = 0.0004  # ~10% annual
                    volatility = 0.007    # ~11% annual
            
            # Generate log returns
            log_returns = np.random.normal(mean_return, volatility, len(dates))
            
            # Convert to price series starting at 100
            prices = 100 * np.exp(np.cumsum(log_returns))
            self.prices[asset] = prices
        
        # Calculate returns and statistics
        self.returns = self.prices.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print("Mock data generation complete.")
        
    def get_assets(self):
        """Return the list of assets."""
        return self.equity_assets + self.fixed_income_assets
    
    def get_prices(self):
        """Return the price data."""
        return self.prices
    
    def get_returns(self):
        """Return the returns data."""
        return self.returns
    
    def get_mean_returns(self):
        """Return the mean returns."""
        return self.mean_returns
    
    def get_cov_matrix(self):
        """Return the covariance matrix."""
        return self.cov_matrix
    
    def get_all_tickers(self):
        """Return all tickers used in the application."""
        return self.equity_assets + self.fixed_income_assets
    
    def get_fund_descriptions(self):
        """Return the descriptions of all funds."""
        return self.fund_descriptions
    
    def get_fund_data(self, ticker):
        """Return the stored data for a specific fund."""
        if ticker in self.fund_data:
            return self.fund_data[ticker]
        return None 