# Portfolio Efficient Frontier Analyzer

A local Python-based tool for portfolio optimization and efficient frontier analysis. This application helps you visualize where your asset allocation portfolio lies on the efficient frontier and lets you adjust different allocations to see how your portfolio moves.

## Features

- Visualize the efficient frontier for a set of assets
- Calculate the risk, return, and Sharpe ratio of your portfolio
- Adjust asset allocations and see real-time changes on the efficient frontier
- Optimize your portfolio for maximum Sharpe ratio or minimum volatility
- Compare your portfolio to random portfolios and the efficient frontier

## Requirements

- Python 3.6 or higher
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Start the server:

```
python app.py
```

2. Open a web browser and navigate to `http://127.0.0.1:5000/`

3. The application will load with an initial set of assets and their data
   - The chart shows the efficient frontier, random portfolios, and key points
   - Adjust the sliders to change asset allocations
   - Click 'Calculate Portfolio' to see where your portfolio falls on the frontier
   - Use the optimization buttons to find optimal portfolios

## Customization

The current implementation uses mock data for stock prices. In a real-world scenario, you would modify the `StockHandler` class to fetch real market data from an API like:

- Yahoo Finance API
- Alpha Vantage
- IEX Cloud
- Quandl

## Technical Details

The application uses:
- Flask for the web server
- PyPortfolioOpt for portfolio optimization calculations
- Plotly for interactive data visualization
- NumPy and Pandas for data manipulation

## Mathematical Background

The application uses Modern Portfolio Theory to calculate:
- Expected returns based on historical data
- Portfolio volatility using covariance between assets
- Sharpe ratio for risk-adjusted performance
- The efficient frontier representing optimal portfolios

## License

MIT 