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
        
        Returns:
            dict: { "Scenario Name": { 'factor1': shock_val, 'factor2': shock_val } }
        """
        scenario_vectors = {}
        for scenario in self.scenarios:
            scenario_vectors[scenario.name] = scenario.shocks
        
        logger.info(f"Loaded {len(scenario_vectors)} stress scenarios from configuration.")
        return scenario_vectors
