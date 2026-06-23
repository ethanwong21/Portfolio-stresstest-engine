import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

FACTOR_COLUMNS = ['market', 'rates', 'inflation', 'commodities']


class FactorEngine:
    def __init__(self, config):
        self.rolling_window = config.rolling_window_days

    def fit_betas(self, ticker: str, asset_series: pd.Series, factor_returns: pd.DataFrame) -> dict:
        """
        Fits OLS regression betas for a single ticker against factor proxies.
        Falls back to heuristic betas if insufficient data (<60 observations).
        """
        factor_df = factor_returns.copy()
        # Normalize column name: factor data uses 'equity', exposure labels use 'market'
        if 'equity' in factor_df.columns and 'market' not in factor_df.columns:
            factor_df = factor_df.rename(columns={'equity': 'market'})

        available = [f for f in FACTOR_COLUMNS if f in factor_df.columns]
        aligned = factor_df[available].join(asset_series.rename('_y_'), how='inner').dropna()

        if len(aligned) < 60:
            logger.warning(f"{ticker}: only {len(aligned)} obs — using heuristic betas")
            return self.get_asset_exposures(ticker)

        X = aligned[available].values
        y = aligned['_y_'].values
        coefs = LinearRegression(fit_intercept=True).fit(X, y).coef_
        betas = dict(zip(available, coefs))

        # Fill any factor not covered by regression with heuristic value
        heuristic = self.get_asset_exposures(ticker)
        for f in FACTOR_COLUMNS:
            if f not in betas:
                betas[f] = heuristic.get(f, 0.0)

        logger.info(f"OLS betas {ticker}: { {k: round(v, 3) for k, v in betas.items()} }")
        return betas

    def get_asset_exposures(self, ticker: str, asset_class: str = "") -> dict:
        """
        Heuristic factor betas used as fallback when OLS cannot be run.
        """
        ticker = str(ticker).upper()
        asset_class = str(asset_class).upper()

        beta = {'market': 1.0, 'rates': -0.2, 'inflation': -0.1, 'commodities': 0.0}

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

    def assign_betas(self, portfolio_df: pd.DataFrame,
                     asset_returns: pd.DataFrame = None,
                     factor_returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Builds the full exposures table for a portfolio.
        Uses OLS regression when asset_returns and factor_returns are supplied;
        falls back to heuristic betas per ticker otherwise.
        """
        exposures = {}
        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']
            asset_class = str(row.get('asset_class', ''))

            use_ols = (
                asset_returns is not None
                and factor_returns is not None
                and ticker in asset_returns.columns
            )

            if use_ols:
                exposures[ticker] = self.fit_betas(ticker, asset_returns[ticker], factor_returns)
            else:
                exposures[ticker] = self.get_asset_exposures(ticker, asset_class)

        if not exposures:
            raise ValueError("Exposures table is empty — no valid tickers found in portfolio.")

        return pd.DataFrame.from_dict(exposures, orient='index')

    def compute_exposures(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Computes OLS factor exposures for a set of assets.
        Used by the rolling backtest to fit betas on each historical window.
        """
        exposures = {}
        for ticker in asset_returns.columns:
            exposures[ticker] = self.fit_betas(ticker, asset_returns[ticker], factor_returns)
        return pd.DataFrame.from_dict(exposures, orient='index')
