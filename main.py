import logging
import argparse
import os
import pandas as pd
from pathlib import Path
from utils.config import load_config
from data.portfolio import PortfolioLoader
from data.market_data import MarketDataLoader
from models.factor_engine import FactorEngine
from scenarios.generator import ScenarioGenerator
from scenarios.dynamic_scenarios import DynamicScenarioGenerator
from models.scenario_impact import ScenarioImpactModel
from models.risk_engine import RiskEngine
from outputs.reporting import ReportGenerator
from comparison.portfolio_compare import PortfolioComparer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_portfolio_analysis(portfolio_path, config, market_loader, factor_returns, shocks, logger):
    """
    Executes the risk analysis pipeline for a specific portfolio.
    """
    logger.info(f"Analyzing portfolio: {portfolio_path}")
    
    config.portfolio.file_path = portfolio_path
    portfolio_loader = PortfolioLoader(config.portfolio)
    portfolio_df = portfolio_loader.load_portfolio()
    
    tickers = portfolio_df['ticker'].tolist() if 'ticker' in portfolio_df.columns else portfolio_df.index.tolist()
    asset_returns = market_loader.fetch_asset_returns(tickers)
    
    factor_engine = FactorEngine(config.model_parameters)
    exposures_df = factor_engine.compute_exposures(asset_returns, factor_returns)
    
    impact_model = ScenarioImpactModel(exposures_df)
    expected_returns = {}
    for scenario_name, shock_vector in shocks.items():
        expected_returns[scenario_name] = impact_model.propagate_shocks(shock_vector)
    
    expected_returns_df = pd.DataFrame(expected_returns)
    
    risk_engine = RiskEngine(portfolio_df, config.model_parameters)
    scenario_pnl = risk_engine.compute_scenario_pnl(expected_returns_df)
    risk_metrics = risk_engine.compute_historical_var(asset_returns)
    total_val = risk_engine.portfolio.get('market_value', pd.Series([0])).sum()
    
    return scenario_pnl, risk_metrics, total_val, exposures_df, risk_engine.portfolio

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Portfolio Stress Testing and Risk Intelligence Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--portfolio", type=str, help="Path to a single portfolio CSV (overrides config)")
    group.add_argument("--compare", nargs='+', help="Paths to multiple portfolio CSVs for comparison testing")
    
    args = parser.parse_args()
    
    logger.info("Starting Portfolio Risk Engine")
    
    # 1. Load Configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully.")
    
    # 2. Pre-fetch Market & Macro Data
    market_loader = MarketDataLoader(config.market_data)
    factor_returns = market_loader.fetch_data()
    
    # 3. Generate Central Scenarios
    scenario_gen = ScenarioGenerator(config.scenarios)
    shocks = scenario_gen.get_shocks()
    
    if getattr(config, 'dynamic_scenarios', None) and config.dynamic_scenarios.enable:
        dyn_gen = DynamicScenarioGenerator(config.dynamic_scenarios, factor_returns)
        shocks.update(dyn_gen.generate_dynamic_scenarios())
        logger.info(f"Total stress scenarios consolidated for impact modeling: {len(shocks)}.")
        
    # 4. Determine Portfolios to Run
    portfolios_to_run = []
    if args.compare:
        portfolios_to_run = args.compare
    elif args.portfolio:
        portfolios_to_run = [args.portfolio]
    else:
        portfolios_to_run = [config.portfolio.file_path]
        
    # 5. Execute Portfolio Specific Analytics
    report_gen = ReportGenerator(config.outputs)
    comparer = PortfolioComparer(getattr(config, 'comparison', None))
    
    for p_path in portfolios_to_run:
        name = Path(p_path).stem
        scenario_pnl, risk_metrics, total_val, exposures, portfolio = run_portfolio_analysis(
            p_path, config, market_loader, factor_returns, shocks, logger
        )
        comparer.add_portfolio_result(name, scenario_pnl, risk_metrics, total_val)
        
        # Single Portfolio Run Fallthrough
        if not args.compare:
            report_gen.export_scenario_results(scenario_pnl)
            report_gen.export_asset_contributions(scenario_pnl)
            report_gen.plot_scenario_impacts(scenario_pnl)
            report_gen.export_risk_metrics(risk_metrics)
            
            results_dict = {
                'scenario_pnl': scenario_pnl,
                'portfolio': portfolio,
                'exposures': exposures,
                'shocks': shocks,
                'risk_metrics': risk_metrics
            }
            report_gen.export_to_excel(results_dict)
            
    # 6. Execute Multiple Portfolio Comparison Logic
    if args.compare and getattr(config, 'comparison', None) and config.comparison.enable:
        comp_df = comparer.compare_portfolios()
        comparer.print_summary(comp_df)
        
        # We need to dispatch the comparison Dataframe to the exporter alongside the portfolios
        report_gen.export_comparison_excel(comparer.results, comp_df, shocks)

    logger.info("Portfolio stress testing engine execution completed successfully.")

if __name__ == "__main__":
    main()
