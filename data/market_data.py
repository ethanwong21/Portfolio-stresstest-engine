import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketDataLoader:
    def __init__(self, config):
        """
        Initializes the Market DataLoader.
        
        Args:
            config: MarketDataConfig from the parsed system configuration.
        """
        self.config = config

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches historical data for configured factors.
        
        Returns:
            pd.DataFrame: A dataframe containing daily percentage returns for the configured factors.
        """
        if self.config.source == "yfinance":
            return self._fetch_yfinance_data()
        elif self.config.source == "simulated":
            return self._simulate_data()
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")

    def _fetch_yfinance_data(self) -> pd.DataFrame:
        """
        Downloads data from Yahoo Finance and computes daily returns.
        """
        tickers = list(self.config.factors.values())
        logger.info(f"Fetching market data for tickers: {tickers}")
        
        # Download adjusted closing prices
        data_raw = yf.download(
            tickers, 
            start=self.config.start_date, 
            end=self.config.end_date, 
            progress=False
        )
        if data_raw.empty:
            logger.warning("yfinance returned empty data for factors. Falling back to simulated.")
            return self._simulate_data()
            
        data = data_raw.get('Adj Close', data_raw.get('Close', data_raw))
        
        # If only one ticker is requested, it might not be a multi-index dataframe
        if len(tickers) == 1:
            data = pd.DataFrame(data)
            data.columns = tickers
        
        # Drop missing data
        data = data.dropna()
        
        # Compute daily percentage changes
        returns = data.pct_change().dropna()
        
        # Re-map columns from ticker symbols to logical factor names
        rename_map = {ticker: factor_name for factor_name, ticker in self.config.factors.items()}
        returns = returns.rename(columns=rename_map)
        
        logger.info(f"Successfully loaded {len(returns)} days of historical factor returns.")
        return returns

    def _simulate_data(self) -> pd.DataFrame:
        """
        Generates simulated random walks for the configured factors.
        """
        factors = list(self.config.factors.keys())
        logger.info(f"Simulating daily returns for factors: {factors}")
        
        # Create a date range matching work days
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='B')
        n_days = len(dates)
        
        # Draw from multivar normal distribution (independent for simplicity, could add correlation matrix)
        returns_data = np.random.normal(loc=0.0001, scale=0.01, size=(n_days, len(factors)))
        
        returns = pd.DataFrame(returns_data, index=dates, columns=factors)
        return returns

    def fetch_asset_returns(self, tickers: list) -> pd.DataFrame:
        """
        Fetches historical data for the portfolio assets dynamically.
        """
        if self.config.source == "yfinance":
            logger.info(f"Fetching historical asset data for tickers: {tickers}")
            data_raw = yf.download(
                tickers, 
                start=self.config.start_date, 
                end=self.config.end_date, 
                progress=False
            )
            if data_raw.empty:
                logger.warning("yfinance returned empty data for assets. Falling back to simulated.")
                return self.fetch_asset_returns(tickers, force_simulate=True)
                
            data = data_raw.get('Adj Close', data_raw.get('Close', data_raw))
            
            if len(tickers) == 1:
                data = pd.DataFrame(data)
                data.columns = tickers
                
            returns = data.dropna().pct_change().dropna()
            return returns
        elif self.config.source == "simulated" or locals().get('force_simulate', False):
            logger.info(f"Simulating daily returns for assets: {tickers}")
            dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='B')
            n_days = len(dates)
            returns_data = np.random.normal(loc=0.0005, scale=0.015, size=(n_days, len(tickers)))
            returns = pd.DataFrame(returns_data, index=dates, columns=tickers)
            return returns
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")

if __name__ == "__main__":
    # Smoke test placeholder
    pass
