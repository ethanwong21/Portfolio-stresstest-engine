import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime
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
    """Sets up structured logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("PortfolioRiskEngine")

def resolve_execution_mode(args):
    """
    Deterministically resolves the execution mode based on CLI arguments.
    Returns: str (e.g., 'single', 'comparison', 'single_backtest', 'comparison_backtest')
    """
    is_compare = bool(args.compare)
    is_backtest = bool(args.rolling_backtest)
    
    if is_compare and is_backtest:
        return "comparison_backtest"
    elif is_compare:
        return "comparison"
    elif is_backtest:
        return "single_backtest"
    else:
        return "single"

def validate_inputs(args, portfolios, logger):
    """Performs robust validation on inputs and edge cases."""
    if not portfolios:
        logger.error("No portfolios detected. Provide --portfolio or --compare.")
        sys.exit(1)
        
    for p in portfolios:
        if not os.path.exists(p):
            logger.error(f"Portfolio file not found: {p}")
            sys.exit(1)
            
    if args.compare and len(portfolios) < 2:
        logger.warning("Comparison mode triggered with only one portfolio. Defaulting to single analysis behavior.")

def get_output_filename(config, args):
    """Generates the output filename, optionally appending a timestamp."""
    base_name = args.output if args.output else config.outputs.excel_export.file_name
    
    if config.outputs.excel_export.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = os.path.splitext(base_name)
        return f"{name_parts[0]}_{timestamp}{name_parts[1]}"
    
    return base_name

def run_portfolio_analysis(portfolio_path, config, market_loader, factor_returns, shocks, logger):
    """Executes the core analytics for a specific portfolio."""
    logger.info(f"Running analysis for: {portfolio_path}")
    
    # Load Portfolio
    config.portfolio.file_path = portfolio_path
    loader = PortfolioLoader(config.portfolio)
    p_df = loader.load_portfolio()
    
    # Capture any ingestion warnings
    warnings = getattr(loader, 'warnings', [])
    detected_cols = getattr(loader, 'detected_cols', {})
    original_cols = getattr(loader, 'original_cols', [])
    cleaned_cols = getattr(loader, 'cleaned_cols', [])

    # Fetch Returns

    # Fetch Returns
    tickers = p_df['ticker'].tolist() if 'ticker' in p_df.columns else p_df.index.tolist()
    asset_returns = market_loader.fetch_asset_returns(tickers)
    
    # check data sufficiency
    if len(asset_returns) < 20: # Arbitrary threshold
        logger.warning(f"Portfolio {portfolio_path} has very limited historical data ({len(asset_returns)} rows). Results may be unstable.")

    # Modeling
    f_engine = FactorEngine(config.model_parameters)
    
    # Use deterministic beta assignment for institutional stress testing
    exposures = f_engine.assign_betas(p_df)
    
    # Debug logging for explicit data flow
    logger.info(f"Generated factor exposures for {len(exposures)} assets.")
    if not exposures.empty:
        logger.info(f"Sample exposure ({exposures.index[0]}): {exposures.iloc[0].to_dict()}")

    impact_model = ScenarioImpactModel(exposures)
    expected_rets = {name: impact_model.propagate_shocks(vec) for name, vec in shocks.items()}
    
    risk_engine = RiskEngine(p_df, config.model_parameters)
    scenario_pnl = risk_engine.compute_scenario_pnl(pd.DataFrame(expected_rets))
    risk_metrics = risk_engine.compute_historical_var(asset_returns)
    
    return {
        'scenario_pnl': scenario_pnl,
        'risk_metrics': risk_metrics,
        'exposures': exposures,
        'portfolio': risk_engine.portfolio,
        'ingestion_warnings': warnings,
        'detected_cols': detected_cols,
        'original_cols': original_cols,
        'cleaned_cols': cleaned_cols
    }

def parse_arguments():
    """Defines CLI arguments with argparse."""
    parser = argparse.ArgumentParser(description="Institutional Portfolio Stress Testing Engine")
    
    # Core
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, help="Override output filename")
    parser.add_argument("--dry-run", action='store_true', help="Preview execution steps without computing")
    
    # Portfolios
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--portfolio", type=str, help="Single portfolio path")
    group.add_argument("--compare", nargs='+', help="Multiple portfolio paths")
    
    # Modes
    parser.add_argument("--rolling-backtest", action='store_true', help="Enable historical model validation")
    parser.add_argument("--dynamic", action='store_true', help="Force dynamic data-driven scenarios")
    
    # Overrides
    parser.add_argument("--start-date", type=str, help="Backtest start date")
    parser.add_argument("--end-date", type=str, help="Backtest end date")
    
    return parser.parse_args()

def run_cli():
    """Main CLI entry point."""
    logger = setup_logging()
    args = parse_arguments()
    
    # 1. Initialization
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    portfolios = args.compare if args.compare else ([args.portfolio] if args.portfolio else [config.portfolio.file_path])
    validate_inputs(args, portfolios, logger)
    mode = resolve_execution_mode(args)
    
    # 2. Dry Run Logic
    if args.dry_run:
        print("\n" + "="*40)
        print(f"[DRY RUN] Mode: {mode}")
        print(f"[DRY RUN] Portfolios: {', '.join(portfolios)}")
        print(f"[DRY RUN] Dynamic Scenarios: {args.dynamic or config.dynamic_scenarios.enable}")
        print(f"[DRY RUN] Backtest: {args.rolling_backtest}")
        print("="*40 + "\n")
        return

    logger.info(f"Execution Mode: {mode.upper()}")
    
    # 3. Backtesting sub-routine
    bt_results = None
    if args.rolling_backtest:
        from backtesting.rolling_backtest import run_rolling_backtest, print_backtest_summary
        logger.info("Initializing Rolling Historical Backtest...")
        
        # Settle parameters
        if args.start_date: config.backtest.start_date = args.start_date
        if args.end_date: config.backtest.end_date = args.end_date
        
        target = portfolios[0]
        temp_config = config.model_copy(deep=True)
        temp_config.portfolio.file_path = target
        
        p_loader = PortfolioLoader(temp_config.portfolio)
        p_df = p_loader.load_portfolio()
        m_loader = MarketDataLoader(config.market_data)
        f_rets = m_loader.fetch_data()
        a_rets = m_loader.fetch_asset_returns(p_df['ticker'].tolist() if 'ticker' in p_df.columns else p_df.index.tolist())
        
        bt_df, bt_metrics = run_rolling_backtest(p_df, a_rets, f_rets, config.backtest, config.model_parameters)
        bt_results = (bt_df, bt_metrics)
        print_backtest_summary(bt_metrics)

    # 4. Stress Engine sub-routine
    m_loader = MarketDataLoader(config.market_data)
    f_rets = m_loader.fetch_data()
    
    s_gen = ScenarioGenerator(config.scenarios)
    shocks = s_gen.get_shocks()
    if args.dynamic or config.dynamic_scenarios.enable:
        logger.info("Generating dynamic macro scenarios...")
        d_gen = DynamicScenarioGenerator(config.dynamic_scenarios, f_rets)
        shocks.update(d_gen.generate_dynamic_scenarios())

    comparer = PortfolioComparer(getattr(config, 'comparison', None))
    master_results = None
    
    for p_path in portfolios:
        res = run_portfolio_analysis(p_path, config, m_loader, f_rets, shocks, logger)
        
        # Track for comparison
        name = Path(p_path).stem
        m_val = res['portfolio'].get('market_value', pd.Series([0])).sum()
        comparer.add_portfolio_result(name, res['scenario_pnl'], res['risk_metrics'], m_val)
        
        if master_results is None:
            master_results = res
            master_results['shocks'] = shocks

    # 5. Terminal Summary
    print("\n" + "="*40)
    print("      EXECUTION SUMMARY")
    print("="*40)
    for p_name, p_res in comparer.results.items():
        scenario_data = p_res['scenario_pnl']
        worst_scenario_name = min(scenario_data.keys(), key=lambda k: scenario_data[k]['portfolio_return'])
        worst_scen = scenario_data[worst_scenario_name]
        
        print(f"Portfolio: {p_name}")
        print(f"  Worst Case: {worst_scen['portfolio_return']*100:.2f}% ({worst_scenario_name})")
        print(f"  VaR (95%):  {p_res['risk_metrics'].get('var_percent', 0)*100:.2f}%")
        print("-" * 20)

    # 6. Unified Export
    comp_df = comparer.compare_portfolios() if len(portfolios) > 1 else None
    
    if master_results:
        final_filename = get_output_filename(config, args)
        logger.info(f"Consolidating results into standardized report: {final_filename}")
        
        # Override filename for reporter
        config.outputs.excel_export.file_name = final_filename
        
        report_gen = ReportGenerator(config.outputs)
        report_gen.export_unified_report(
            master_results, 
            comp_df=comp_df, 
            bt_df=bt_results[0] if bt_results else None, 
            bt_metrics=bt_results[1] if bt_results else None
        )

    logger.info("Stress Testing Pipeline COMPLETED.")
