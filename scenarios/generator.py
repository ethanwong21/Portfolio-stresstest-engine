import logging

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    def __init__(self, config):
        self.scenarios = config

    def get_shocks(self, include_historical: bool = True) -> dict:
        """
        Returns all scenario shock vectors.

        Synthetic forward-looking scenarios (manually specified) are always
        included. Historically calibrated scenarios (derived from real crisis
        period returns via yfinance) are merged in when include_historical=True.
        """
        scenario_vectors = {
            "Equity Market Crash": {
                "market": -0.25,
                "rates": -0.01,
                "inflation": -0.02,
                "commodities": -0.10,
            },
            "Yield Curve Shock (+100bps)": {
                "market": -0.05,
                "rates": 0.01,
                "inflation": 0.00,
                "commodities": -0.02,
            },
            "Stagflation": {
                "market": -0.10,
                "rates": 0.02,
                "inflation": 0.05,
                "commodities": 0.10,
            },
            "Inflation Spike (+10)": {
                "market": -0.03,
                "rates": 0.01,
                "inflation": 0.03,
                "commodities": 0.08,
            },
            "Rates Spike (+20)": {
                "market": -0.08,
                "rates": 0.02,
                "inflation": 0.01,
                "commodities": -0.05,
            },
        }

        if include_historical:
            try:
                from scenarios.historical import HistoricalScenarioCalibrator
                calibrator = HistoricalScenarioCalibrator()
                historical = calibrator.calibrate()
                scenario_vectors.update(historical)
                logger.info(f"Merged {len(historical)} historically calibrated scenarios.")
            except Exception as exc:
                logger.warning(f"Historical scenario calibration skipped: {exc}")

        logger.info(f"Total scenarios loaded: {len(scenario_vectors)}")
        return scenario_vectors
