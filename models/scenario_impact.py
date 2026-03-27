import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ScenarioImpactModel:
    def __init__(self, factor_exposures: pd.DataFrame):
        """
        Initializes the impact model with current asset factor exposures.
        
        Args:
            factor_exposures (pd.DataFrame): Rows=assets, Cols=factors
        """
        self.exposures = factor_exposures

    def propagate_shocks(self, scenario_shocks: dict) -> pd.DataFrame:
        """
        Applies a dict of factor shocks through the exposure matrix to get asset impacts.
        Formula: Asset_Return = sum(Beta_i * Shock_i)
        
        Args:
            scenario_shocks (dict): { 'factor1': shock_val, ... }
            
        Returns:
            pd.Series: Indexed by asset, values = expected return under this scenario.
        """
        # Ensure shocks align with exposure columns, filling un-shocked factors with 0
        shock_vector = pd.Series(scenario_shocks)
        
        # Realign shock vector to match exposure columns (factors)
        aligned_shocks = shock_vector.reindex(self.exposures.columns).fillna(0)
        
        # Matrix multiplication: Exposures (Assets x Factors) dot Shocks (Factors x 1)
        expected_returns = self.exposures.dot(aligned_shocks)
        
        return expected_returns
