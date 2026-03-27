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
    
    # Backtest arguments
    parser.add_argument("--rolling-backtest", action='store_true', help="Execute rolling historical backtest for model validation")
    parser.add_argument("--start-date", type=str, help="Override backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Override backtest end date (YYYY-MM-DD)")
    parser.add_argument("--window-size", type=int, help="Override rolling history window size (integer)")
    
    args = parser.parse_args()
    
    logger.info("Starting Portfolio Risk Engine")
    
    # 1. Load Configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully.")
    
    # 2. Trigger Rolling Backtest Branch Conditionally
    bt_df = None
    bt_metrics = None
    if args.rolling_backtest:
        from backtesting.rolling_backtest import run_rolling_backtest, print_backtest_summary
        
        if args.start_date: config.backtest.start_date = args.start_date
        if args.end_date: config.backtest.end_date = args.end_date
        if args.window_size: config.backtest.window_size = args.window_size
        
        logger.info("Initiating Rolling Time-Based Backtest routines.")
        
        target_port = args.portfolio if args.portfolio else (args.compare[0] if args.compare else config.portfolio.file_path)
        temp_config = config.model_copy(deep=True)
        temp_config.portfolio.file_path = target_port
        
        portfolio_loader = PortfolioLoader(temp_config.portfolio)
        portfolio_df = portfolio_loader.load_portfolio()
        
        market_loader = MarketDataLoader(config.market_data)
        factor_returns = market_loader.fetch_data()
        tickers = portfolio_df['ticker'].tolist() if 'ticker' in portfolio_df.columns else portfolio_df.index.tolist()
        asset_returns = market_loader.fetch_asset_returns(tickers)
        
        bt_df, bt_metrics = run_rolling_backtest(portfolio_df, asset_returns, factor_returns, config.backtest, config.model_parameters)
        print_backtest_summary(bt_metrics)
        
    # 3. Pre-fetch Market & Macro Data (Standard Mode)
    market_loader = MarketDataLoader(config.market_data)
    factor_returns = market_loader.fetch_data()
    
    # 4. Generate Central Scenarios
    scenario_gen = ScenarioGenerator(config.scenarios)
    shocks = scenario_gen.get_shocks()
    
    if getattr(config, 'dynamic_scenarios', None) and config.dynamic_scenarios.enable:
        dyn_gen = DynamicScenarioGenerator(config.dynamic_scenarios, factor_returns)
        shocks.update(dyn_gen.generate_dynamic_scenarios())
        logger.info(f"Total stress scenarios consolidated for impact modeling: {len(shocks)}.")
        
    # 5. Determine Portfolios to Run
    portfolios_to_run = []
    if args.compare:
        portfolios_to_run = args.compare
    elif args.portfolio:
        portfolios_to_run = [args.portfolio]
    else:
        portfolios_to_run = [config.portfolio.file_path]
        
    # 6. Execute Portfolio Specific Analytics
    comparer = PortfolioComparer(getattr(config, 'comparison', None))
    master_results_dict = None
    
    for p_path in portfolios_to_run:
        name = Path(p_path).stem
        scenario_pnl, risk_metrics, total_val, exposures, portfolio = run_portfolio_analysis(
            p_path, config, market_loader, factor_returns, shocks, logger
        )
        comparer.add_portfolio_result(name, scenario_pnl, risk_metrics, total_val)
        
        # Save the first portfolio processed as our MASTER details portfolio
        if master_results_dict is None:
            master_results_dict = {
                'scenario_pnl': scenario_pnl,
                'portfolio': portfolio,
                'exposures': exposures,
                'shocks': shocks,
                'risk_metrics': risk_metrics
            }
            
    # 7. Execute Multiple Portfolio Comparison Logic
    comp_df = None
    if len(portfolios_to_run) > 1 and getattr(config, 'comparison', None) and config.comparison.enable:
        comp_df = comparer.compare_portfolios()
        comparer.print_summary(comp_df)
        
    # 8. EXPORT UNIFIED REPORT
    if master_results_dict:
        report_gen = ReportGenerator(config.outputs)
        report_gen.export_unified_report(master_results_dict, comp_df=comp_df, bt_df=bt_df, bt_metrics=bt_metrics)

    logger.info("Portfolio stress testing engine execution completed successfully.")

if __name__ == "__main__":
    main()
