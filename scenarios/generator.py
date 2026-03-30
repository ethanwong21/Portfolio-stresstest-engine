import logging

logger = logging.getLogger(__name__)

class ScenarioGenerator:
    def __init__(self, config):
        """
        Initializes the scenario generator.
        
        Args:
            config: A List[ScenarioConfig] from the app configuration.
        """
        self.scenarios = config

    def get_shocks(self) -> dict:
        """
        Formats all config-defined scenarios into a dictionary of shock vectors.
        Specifically handles the mapping of factor shocks based on macroeconomic logic.
        """
        # Hardcoded factor-based stress engine mapping
        scenario_vectors = {
            "Equity Market Crash": {
                'market': -0.25,
                'rates': -0.01,
                'inflation': -0.02,
                'commodities': -0.10
            },
            "Yield Curve Shock (+100bps)": {
                'market': -0.05,
                'rates': 0.01,
                'inflation': 0.00,
                'commodities': -0.02
            },
            "Stagflation": {
                'market': -0.10,
                'rates': 0.02,
                'inflation': 0.05,
                'commodities': 0.10
            },
            "Inflation Spike (+10)": {
                'market': -0.03,
                'rates': 0.01,
                'inflation': 0.03,
                'commodities': 0.08
            },
            "Rates Spike (+20)": {
                'market': -0.08,
                'rates': 0.02,
                'inflation': 0.01,
                'commodities': -0.05
            }
        }
        
        logger.info(f"Loaded {len(scenario_vectors)} factor-driven stress scenarios.")
        return scenario_vectors
