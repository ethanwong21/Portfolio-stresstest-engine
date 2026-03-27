import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FactorEngine:
    def __init__(self, config):
        """
        Initializes the Factor Engine.
        """
        self.rolling_window = config.rolling_window_days

    def compute_exposures(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the sensitivities (betas) of each asset to the market factors 
        using ordinary least squares (OLS) regression over the configured rolling window.
        
        Args:
            asset_returns (pd.DataFrame): Historical asset returns (dates as index).
            factor_returns (pd.DataFrame): Historical factor returns (dates as index).
            
        Returns:
            pd.DataFrame: Matrix where rows are assets and columns are factor exposures.
        """
        # Align dates
        aligned_data = pd.concat([asset_returns, factor_returns], axis=1, join='inner').dropna()
        if len(aligned_data) < self.rolling_window:
            logger.warning(f"Data history ({len(aligned_data)} days) is less than rolling window ({self.rolling_window} days). Using all available data.")
            window_data = aligned_data
        else:
            # Use only the most recent 'rolling_window' observations for the current exposure
            window_data = aligned_data.tail(self.rolling_window)
        
        asset_cols = asset_returns.columns
        factor_cols = factor_returns.columns
        
        X = window_data[factor_cols].values
        
        exposures = {}
        for asset in asset_cols:
            y = window_data[asset].values
            # Run OLS regression
            model = LinearRegression()
            model.fit(X, y)
            # Store betas (coefficients)
            exposures[asset] = dict(zip(factor_cols, model.coef_))
            
        exposures_df = pd.DataFrame.from_dict(exposures, orient='index')
        return exposures_df
