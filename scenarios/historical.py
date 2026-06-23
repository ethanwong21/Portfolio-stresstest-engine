import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class HistoricalScenarioCalibrator:
    """
    Calibrates stress scenarios from real historical crisis periods.

    Downloads factor proxy prices and computes the cumulative return of each
    proxy over each named crisis window. The result is used directly as the
    shock vector fed into ScenarioImpactModel, making every 'historical'
    scenario empirically grounded rather than manually guessed.
    """

    FACTOR_PROXIES = {
        "market": "^GSPC",
        "rates": "^TNX",
        "inflation": "TIP",
        "commodities": "GSG",
    }

    CRISIS_PERIODS = {
        "2008 GFC (Sep-Nov)": ("2008-09-01", "2008-11-28"),
        "2020 COVID Crash (Feb-Mar)": ("2020-02-01", "2020-03-31"),
        "2022 Rate Hike Cycle": ("2022-01-01", "2022-12-30"),
        "2001 Dot-Com Bust (Mar-Sep)": ("2001-03-01", "2001-09-28"),
    }

    def calibrate(self) -> dict:
        """
        Downloads price history and returns a dict of historically calibrated
        shock vectors, one per crisis period.

        Returns:
            dict: { "2008 GFC (Sep-Nov)": {"market": -0.38, "rates": 0.12, ...}, ... }
        """
        tickers = list(self.FACTOR_PROXIES.values())

        try:
            raw = yf.download(
                tickers,
                start="2000-01-01",
                end="2024-01-01",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.warning(f"Historical calibration download failed: {exc}")
            return {}

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw

        if prices.empty:
            logger.warning("Historical calibration: no price data returned.")
            return {}

        scenarios = {}
        for name, (start, end) in self.CRISIS_PERIODS.items():
            try:
                period = prices.loc[start:end].dropna(how="all")
                if len(period) < 2:
                    continue

                shocks = {}
                for factor, ticker in self.FACTOR_PROXIES.items():
                    if ticker not in period.columns:
                        continue
                    series = period[ticker].dropna()
                    if len(series) < 2:
                        continue
                    cumulative = round(float((series.iloc[-1] / series.iloc[0]) - 1), 4)
                    shocks[factor] = cumulative

                if shocks:
                    scenarios[name] = shocks
                    logger.info(f"Calibrated '{name}': {shocks}")

            except Exception as exc:
                logger.warning(f"Failed to calibrate '{name}': {exc}")

        logger.info(f"Historical calibration complete - {len(scenarios)} scenarios loaded.")
        return scenarios
