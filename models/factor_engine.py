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

    def get_asset_exposures(self, ticker: str, asset_class: str = "") -> dict:
        """
        Returns a dictionary of factor sensitivities for a single asset.
        Supports: market, rates, inflation, commodities.
        """
        ticker = str(ticker).upper()
        asset_class = str(asset_class).upper()
        
        # Default Beta (Broad Market Exposure)
        beta = {'market': 1.0, 'rates': -0.2, 'inflation': -0.1, 'commodities': 0.0}
        
        # 1. Ticker-based overrides (Specific assets)
        if any(t in ticker for t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'QQQ', 'META', 'TSLA']):
            beta = {'market': 1.35, 'rates': -0.6, 'inflation': -0.3, 'commodities': 0.0}
        elif any(t in ticker for t in ['XOM', 'CVX', 'BP', 'SHEL', 'OXY', 'XLE']):
            beta = {'market': 0.8, 'rates': 0.3, 'inflation': 0.6, 'commodities': 1.2}
        elif any(t in ticker for t in ['GLD', 'IAU', 'PHYS', 'GDX']):
            beta = {'market': 0.1, 'rates': -0.2, 'inflation': 0.9, 'commodities': 1.5}
        elif any(t in ticker for t in ['TLT', 'AGG', 'BND', 'LQD', 'JNK', 'HYG']):
            beta = {'market': 0.0, 'rates': -5.0, 'inflation': -2.5, 'commodities': 0.0}
        elif any(t in ticker for t in ['TIP', 'VTIP', 'STIP']):
            beta = {'market': 0.0, 'rates': -2.0, 'inflation': 1.0, 'commodities': 0.0}
        elif any(t in ticker for t in ['XLU', 'VPU', 'IDU']):
            beta = {'market': 0.55, 'rates': -1.2, 'inflation': 0.2, 'commodities': 0.0}

        # 2. Sector-based overrides (Fallback if no ticker hit)
        elif 'TECH' in asset_class or 'GROWTH' in asset_class:
            beta = {'market': 1.4, 'rates': -0.7, 'inflation': -0.4, 'commodities': 0.0}
        elif 'ENERGY' in asset_class or 'COMMODITY' in asset_class:
            beta = {'market': 0.75, 'rates': 0.2, 'inflation': 0.7, 'commodities': 1.4}
        elif 'BOND' in asset_class or 'FIXED' in asset_class:
            beta = {'market': 0.0, 'rates': -4.5, 'inflation': -2.0, 'commodities': 0.0}
        elif 'UTILITIES' in asset_class:
            beta = {'market': 0.6, 'rates': -1.5, 'inflation': 0.3, 'commodities': 0.0}
        elif 'HEDGE' in asset_class or 'DEFENSIVE' in asset_class:
            beta = {'market': 0.4, 'rates': -0.1, 'inflation': 0.5, 'commodities': 0.2}

        return beta

    def assign_betas(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the full exposures table for a portfolio.
        """
        exposures = {}
        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']
            asset_class = str(row.get('asset_class', ''))
            
            exposures[ticker] = self.get_asset_exposures(ticker, asset_class)
            
        if not exposures:
            raise ValueError("Exposures table is empty - No valid tickers found in portfolio.")
            
        exposures_df = pd.DataFrame.from_dict(exposures, orient='index')
        return exposures_df

    def compute_exposures(self, asset_returns, factor_returns):
        """
        Computes the factor exposures for a set of assets.
        This method is required by the rolling_backtest module.
        """
        exposures = {}
        for ticker in asset_returns.columns:
            exposures[ticker] = self.get_asset_exposures(ticker)
            
        return pd.DataFrame.from_dict(exposures, orient="index")
