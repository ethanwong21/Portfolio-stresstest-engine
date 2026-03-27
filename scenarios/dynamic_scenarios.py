import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DynamicScenarioGenerator:
    def __init__(self, config, factor_returns: pd.DataFrame):
        """
        Initializes the dynamic scenario generator with statistical macro data.
        
        Args:
            config: DynamicScenariosConfig from AppConfig.
            factor_returns (pd.DataFrame): Historical returns of configured factors.
        """
        self.config = config
        self.factor_returns = factor_returns

    def compute_statistics(self) -> dict:
        """
        Computes the standard deviation (sigma) and mean for each requested factor.
        """
        stats = {}
        for factor in self.config.factors:
            if factor in self.factor_returns.columns:
                series = self.factor_returns[factor]
                stats[factor] = {
                    'mean': series.mean(),
                    'std': series.std()
                }
            else:
                logger.warning(f"Factor '{factor}' not found in market data. Skipping dynamic scenario.")
        return stats

    def generate_dynamic_scenarios(self) -> dict:
        """
        Builds structured dictionary scenarios based on configured sigma magnitudes.
        
        Returns:
            dict: { "Scenario Name": { 'factor': shock_val } }
        """
        if not self.config or not getattr(self.config, 'enable', False):
            return {}
            
        stats = self.compute_statistics()
        dynamic_shocks = {}
        
        for factor, stat in stats.items():
            std = stat['std']
            for level in self.config.sigma_levels:
                # Positive Shock (+σ)
                name_up = f"{factor.capitalize()} Spike (+{level}σ)"
                dynamic_shocks[name_up] = {factor: std * level}
                
                # Negative Shock (-σ)
                name_down = f"{factor.capitalize()} Drawdown (-{level}σ)"
                dynamic_shocks[name_down] = {factor: -std * level}
                
        logger.info(f"Generated {len(dynamic_shocks)} dynamic scenarios based on historical volatility.")
        return dynamic_shocks
