import logging
import argparse
from utils.config import load_config
from data.portfolio import PortfolioLoader
from data.market_data import MarketDataLoader
from models.factor_engine import FactorEngine
from scenarios.generator import ScenarioGenerator
from models.scenario_impact import ScenarioImpactModel
from models.risk_engine import RiskEngine
from outputs.reporting import ReportGenerator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Portfolio Stress Testing and Risk Intelligence Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    args = parser.parse_args()
    
    logger.info("Starting Portfolio Risk Engine")
    
    # 1. Load Configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully.")
    
    # 2. Load Portfolio
    portfolio_loader = PortfolioLoader(config.portfolio)
    portfolio_df = portfolio_loader.load_portfolio()
    
    # 3. Market & Macro Data
    market_loader = MarketDataLoader(config.market_data)
    factor_returns = market_loader.fetch_data()
    
    tickers = portfolio_df['ticker'].tolist() if 'ticker' in portfolio_df.columns else portfolio_df.index.tolist()
    asset_returns = market_loader.fetch_asset_returns(tickers)
    
    # 4. Factor Exposure Engine
    factor_engine = FactorEngine(config.model_parameters)
    exposures_df = factor_engine.compute_exposures(asset_returns, factor_returns)
    logger.info(f"Computed factor exposures for {len(exposures_df)} assets.")
    
    # 5. Scenarios Generation
    scenario_gen = ScenarioGenerator(config.scenarios)
    shocks = scenario_gen.get_shocks()
    
    # 6. Impact Propagation Model
    impact_model = ScenarioImpactModel(exposures_df)
    expected_returns = {}
    for scenario_name, shock_vector in shocks.items():
        expected_returns[scenario_name] = impact_model.propagate_shocks(shock_vector)
    
    import pandas as pd
    expected_returns_df = pd.DataFrame(expected_returns)
    
    # 7. Portfolio Risk Engine
    risk_engine = RiskEngine(portfolio_df, config.model_parameters)
    scenario_pnl = risk_engine.compute_scenario_pnl(expected_returns_df)
    risk_metrics = risk_engine.compute_historical_var(asset_returns)
    
    # 8. Outputs & Reporting
    report_gen = ReportGenerator(config.outputs)
    report_gen.export_scenario_results(scenario_pnl)
    report_gen.export_asset_contributions(scenario_pnl)
    report_gen.plot_scenario_impacts(scenario_pnl)
    report_gen.export_risk_metrics(risk_metrics)
    
    # Export professional Excel report if enabled
    results_dict = {
        'scenario_pnl': scenario_pnl,
        'portfolio': risk_engine.portfolio,
        'exposures': exposures_df,
        'shocks': shocks,
        'risk_metrics': risk_metrics
    }
    report_gen.export_to_excel(results_dict)
    
    logger.info("Portfolio stress testing engine execution completed successfully.")

if __name__ == "__main__":
    main()
