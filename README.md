# Portfolio Efficient Frontier Analysis Tool

A web-based tool for portfolio optimization using Modern Portfolio Theory and the Efficient Frontier. This application allows users to analyze different portfolio allocations and see how they perform in terms of expected return, volatility, and Sharpe ratio.

## Features

- **Efficient Frontier Visualization**: Displays a graph of the efficient frontier along with random portfolios colored by Sharpe ratio
- **Asset Allocation Grid**: Allows setting allocations using dollar values for different asset classes
- **Portfolio Optimization**: One-click optimization for maximum Sharpe ratio or minimum volatility
- **Allocation Saving/Loading**: Import and export portfolio allocations as CSV files

## Asset Allocation Grid Structure

### Equity Allocation

The equity allocation grid follows a 3×3 matrix structure:

| | Growth | Blend | Value |
|----------|----------|----------|----------|
| Small Cap | $ | $ | $ |
| Mid Cap | $ | $ | $ |
| Large Cap | $ | $ | $ |

### Fixed Income Allocation

The fixed income allocation grid follows a 3×3 matrix structure:

| | Low Risk | Mid Risk | High Risk |
|----------|----------|----------|----------|
| Short Term | $ | $ | $ |
| Mid Term | $ | $ | $ |
| Long Term | $ | $ | $ |

## Usage Instructions

1. **View the Efficient Frontier**:
   - The graph displays the efficient frontier, random portfolios, and optimal points
   - Hover over points to see additional details

2. **Set Asset Allocation**:
   - Enter dollar values in the grid cells for the asset classes you want to allocate to
   - Totals will automatically calculate for each row, column, and overall
   - Click "Update Portfolio" to see how your allocation performs

3. **Optimize Your Portfolio**:
   - Click "Max Sharpe Ratio" for the portfolio with the best risk-adjusted return
   - Click "Min Volatility" for the portfolio with the lowest risk
   - The allocation grid will automatically update with the optimized values

4. **Save and Load Allocations**:
   - Click "Save CSV" to save your current allocation to a CSV file
   - Use the file input and "Load" button to import an allocation from a CSV file
   - The system also automatically saves allocations on the server that can be loaded later

## CSV File Format

The CSV file follows this format:

```
type,size_term,style_risk,amount
equity,small,growth,5000
equity,large,value,10000
fixed,short,low,3000
...
```

Where:
- `type`: Either "equity" or "fixed"
- `size_term`: Size for equity (small, mid, large) or term for fixed income (short, mid, long)
- `style_risk`: Style for equity (growth, blend, value) or risk for fixed income (low, mid, high)
- `amount`: Dollar amount allocated

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open a browser and go to `http://localhost:5000`

## Dependencies

- Flask
- NumPy
- pandas
- Plotly

## Note

This application uses mock data for demonstration purposes. In a real investment scenario, you would connect to actual market data sources. 