import pandas as pd
import numpy as np

class MorningstarStyleHandler:
    """
    Handler for Morningstar style box allocations. 
    Allows specifying monetary values for each of the 9 style box categories
    and calculating portfolio allocations based on these values.
    """
    
    def __init__(self, stock_handler):
        """
        Initialize with a stock handler that provides the 9 Morningstar style box categories.
        
        Args:
            stock_handler: An instance of StockHandler with Morningstar style box assets
        """
        self.stock_handler = stock_handler
        self.style_box_assets = stock_handler.get_morningstar_style_box()
        
        # Initialize monetary allocations to zero
        self.monetary_allocations = {
            'large_growth': 0.0,
            'large_blend': 0.0,
            'large_value': 0.0,
            'mid_growth': 0.0,
            'mid_blend': 0.0,
            'mid_value': 0.0,
            'small_growth': 0.0,
            'small_blend': 0.0,
            'small_value': 0.0
        }
        
        # Get display names for UI
        self.display_names = {
            'large_growth': 'Large Cap Growth',
            'large_blend': 'Large Cap Blend',
            'large_value': 'Large Cap Value',
            'mid_growth': 'Mid Cap Growth',
            'mid_blend': 'Mid Cap Blend',
            'mid_value': 'Mid Cap Value',
            'small_growth': 'Small Cap Growth',
            'small_blend': 'Small Cap Blend',
            'small_value': 'Small Cap Value'
        }
    
    def set_monetary_allocation(self, style_category, amount_usd):
        """
        Set the monetary allocation for a specific style box category.
        
        Args:
            style_category: The style box category (e.g., 'large_growth')
            amount_usd: The monetary amount in USD to allocate to this category
            
        Returns:
            True if successful, False if category not found
        """
        if style_category in self.monetary_allocations:
            self.monetary_allocations[style_category] = float(amount_usd)
            return True
        return False
    
    def set_monetary_allocations(self, allocations_dict):
        """
        Set monetary allocations for multiple style box categories at once.
        
        Args:
            allocations_dict: Dictionary mapping style categories to USD amounts
            
        Returns:
            Dictionary with results of each allocation attempt
        """
        results = {}
        for category, amount in allocations_dict.items():
            results[category] = self.set_monetary_allocation(category, amount)
        return results
    
    def get_total_portfolio_value(self):
        """
        Calculate the total portfolio value in USD.
        
        Returns:
            Total value of all style box allocations
        """
        return sum(self.monetary_allocations.values())
    
    def get_percentage_allocations(self):
        """
        Calculate the percentage allocation for each style box category.
        
        Returns:
            Dictionary mapping style categories to percentage allocations
        """
        total = self.get_total_portfolio_value()
        if total == 0:
            return {category: 0.0 for category in self.monetary_allocations}
        
        return {category: amount / total 
                for category, amount in self.monetary_allocations.items()}
    
    def get_asset_weights(self):
        """
        Convert style box allocations to asset weights for portfolio calculations.
        
        Returns:
            Dictionary mapping asset symbols to weight percentages
        """
        percentages = self.get_percentage_allocations()
        
        # Map style box categories to actual assets
        asset_weights = {}
        for style, percentage in percentages.items():
            asset = self.style_box_assets.get(style)
            if asset:
                asset_weights[asset] = percentage
        
        return asset_weights
    
    def get_asset_monetary_values(self):
        """
        Get the monetary value allocated to each asset.
        
        Returns:
            Dictionary mapping asset symbols to USD values
        """
        asset_values = {}
        for style, amount in self.monetary_allocations.items():
            asset = self.style_box_assets.get(style)
            if asset:
                asset_values[asset] = amount
        
        return asset_values
    
    def get_style_box_grid(self):
        """
        Return the style box as a 3x3 grid with monetary values.
        
        Returns:
            DataFrame representing the 3x3 style box grid with monetary values
        """
        # Create the grid
        grid = pd.DataFrame(
            index=['Large Cap', 'Mid Cap', 'Small Cap'],
            columns=['Growth', 'Blend', 'Value']
        )
        
        # Fill in the values
        grid.loc['Large Cap', 'Growth'] = self.monetary_allocations['large_growth']
        grid.loc['Large Cap', 'Blend'] = self.monetary_allocations['large_blend']
        grid.loc['Large Cap', 'Value'] = self.monetary_allocations['large_value']
        
        grid.loc['Mid Cap', 'Growth'] = self.monetary_allocations['mid_growth']
        grid.loc['Mid Cap', 'Blend'] = self.monetary_allocations['mid_blend']
        grid.loc['Mid Cap', 'Value'] = self.monetary_allocations['mid_value']
        
        grid.loc['Small Cap', 'Growth'] = self.monetary_allocations['small_growth']
        grid.loc['Small Cap', 'Blend'] = self.monetary_allocations['small_blend']
        grid.loc['Small Cap', 'Value'] = self.monetary_allocations['small_value']
        
        return grid
    
    def get_style_box_summary(self):
        """
        Get summary statistics for the style box allocations.
        
        Returns:
            Dictionary with style box summary information
        """
        grid = self.get_style_box_grid()
        
        # Calculate row and column totals
        row_totals = grid.sum(axis=1)
        col_totals = grid.sum(axis=0)
        total = grid.values.sum()
        
        # Calculate percentages
        row_percentages = row_totals / total if total > 0 else row_totals * 0
        col_percentages = col_totals / total if total > 0 else col_totals * 0
        
        return {
            'grid': grid.to_dict(),
            'row_totals': {
                'values': row_totals.to_dict(),
                'percentages': row_percentages.to_dict()
            },
            'column_totals': {
                'values': col_totals.to_dict(),
                'percentages': col_percentages.to_dict()
            },
            'total_value': total,
            'display_names': self.display_names
        } 