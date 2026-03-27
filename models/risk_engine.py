import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RiskEngine:
    def __init__(self, portfolio: pd.DataFrame, config):
        """
        Initializes RiskEngine.
        
        Args:
            portfolio (pd.DataFrame): Portfolio data containing weights/market value.
            config: Model parameters config containing var_confidence_level.
        """
        self.portfolio = portfolio.set_index('ticker') if 'ticker' in portfolio.columns else portfolio
        self.confidence_level = config.var_confidence_level

    def compute_scenario_pnl(self, expected_asset_returns: pd.DataFrame) -> dict:
        """
        Computes the weighted portfolio return and dollar PnL under stress scenarios.
        
        Args:
            expected_asset_returns (pd.DataFrame): Predicted asset returns for each scenario (rows=assets, cols=scenario names).
            
        Returns:
            dict: Portfolio-level summary by scenario.
        """
        weights = self.portfolio['weight']
        market_values = self.portfolio.get('market_value', pd.Series(1.0, index=self.portfolio.index))
        # Ensure alignment
        expected_asset_returns = expected_asset_returns.reindex(weights.index).fillna(0)
        
        results = {}
        for scenario in expected_asset_returns.columns:
            asset_returns = expected_asset_returns[scenario]
            
            # Portfolio Return
            port_return = (weights * asset_returns).sum()
            
            # Portfolio Dollar PnL
            dollar_pnl = (market_values * asset_returns).sum()
            
            # Asset Level PnL Contribution
            asset_contribution = (market_values * asset_returns)
            
            results[scenario] = {
                'portfolio_return': port_return,
                'portfolio_dollar_pnl': dollar_pnl,
                'asset_contributions': asset_contribution
            }
            
        return results

    def compute_historical_var(self, asset_returns: pd.DataFrame) -> dict:
        """
        Computes Historical Simulation VaR using historical asset returns and current weights.
        """
        weights = self.portfolio['weight']
        total_mv = self.portfolio.get('market_value', pd.Series(100.0, index=self.portfolio.index)).sum()
        
        # Align
        aligned_returns = asset_returns.reindex(columns=weights.index).fillna(0)
        
        # Compute historical portfolio returns
        port_returns = aligned_returns.dot(weights)
        
        # Calculate VaR
        percentile = (1 - self.confidence_level) * 100
        var_pct = np.percentile(port_returns, percentile)
        
        var_dollar = total_mv * var_pct
        
        # Calculate Max Drawdown
        cum_returns = (1 + port_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        return {
            'var_percent': var_pct,
            'var_dollar': var_dollar,
            'confidence_level': self.confidence_level,
            'max_historical_drawdown': max_drawdown
        }
