import logging
import pandas as pd
import numpy as np
from models.factor_engine import FactorEngine

logger = logging.getLogger(__name__)

def compute_error_metrics(results_df: pd.DataFrame) -> dict:
    """
    Computes statistical evaluation metrics on the prediction errors generated during backtesting.
    """
    if results_df.empty:
        return {}
        
    mae = np.mean(np.abs(results_df['Error']))
    rmse = np.sqrt(np.mean(results_df['Error']**2))
    
    # Directional Accuracy: Do Predicted Return and Actual Return have the same sign?
    correct_direction = np.sign(results_df['Predicted Return']) == np.sign(results_df['Actual Return'])
    accuracy = correct_direction.mean()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Directional Accuracy': accuracy
    }

def print_backtest_summary(metrics: dict):
    if not metrics:
        return
    print("\n" + "="*40)
    print("      ROLLING BACKTEST RESULTS")
    print("="*40)
    print(f"Mean Absolute Error (MAE):    {metrics['MAE']*100:.2f}%")
    print(f"Root Mean Squared (RMSE):     {metrics['RMSE']*100:.2f}%")
    print(f"Directional Prediction Hit:   {metrics['Directional Accuracy']*100:.2f}%")
    print("="*40 + "\n")

def run_rolling_backtest(portfolio_df: pd.DataFrame, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame, config, model_params_config) -> tuple:
    """
    Executes a historical rolling backtest projecting t -> t+1 portfolio returns.
    Computes historical regressions iteratively preventing any look-ahead bias natively.
    """
    logger.info(f"Starting rolling backtest from {config.start_date} to {config.end_date} at frequency {config.frequency}")
    
    if asset_returns.empty or factor_returns.empty:
        raise ValueError(f"Backtest failed: Empty returns data. Asset rows: {len(asset_returns)}, Factor rows: {len(factor_returns)}")

    # Ensure portfolio is indexed by ticker for alignment
    p_df = portfolio_df.set_index('ticker') if 'ticker' in portfolio_df.columns else portfolio_df
    weights = p_df['weight']
    
    # Resample to requested calculation frequency
    freq_map = {"M": "ME", "D": "D", "W": "W", "Q": "QE"}
    freq = freq_map.get(config.frequency.upper(), "ME")
    
    # Ensure pandas datetime representations
    asset_returns.index = pd.to_datetime(asset_returns.index)
    factor_returns.index = pd.to_datetime(factor_returns.index)
    
    # Secure uniform alignment across series
    common_idx = asset_returns.index.intersection(factor_returns.index)
    if len(common_idx) < config.window_size:
         raise ValueError(f"Insufficient overlapping dates ({len(common_idx)}) between assets and factors for window size {config.window_size}")
         
    asset_returns = asset_returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]
    
    epochs = factor_returns.resample(freq).last().index
    
    # Bound mapping strictly by configuration requests
    mask = (epochs >= pd.to_datetime(config.start_date)) & (epochs <= pd.to_datetime(config.end_date))
    eval_epochs = epochs[mask]
    
    if len(eval_epochs) < 2:
        logger.warning("Backtest range too narrow for the requested frequency. No periods to evaluate.")
        return pd.DataFrame(), {}

    results = []
    
    for i in range(len(eval_epochs) - 1):
        t_current = eval_epochs[i]
        t_next = eval_epochs[i+1]
        
        # 1. Historical Train Masking (STRICTLY NO LOOKAHEAD OVER T_CURRENT)
        hist_asset = asset_returns.loc[:t_current]
        hist_factor = factor_returns.loc[:t_current]
        
        if len(hist_factor) < config.window_size:
            continue
            
        # Frame exact sample window for Betas
        window_asset = hist_asset.iloc[-config.window_size:]
        window_factor = hist_factor.iloc[-config.window_size:]
        
        # 2. Factor Computations isolated entirely upon historical window (t_current context)
        temp_params = model_params_config.model_copy()
        temp_params.rolling_window_days = config.window_size
        factor_engine = FactorEngine(temp_params)
        exposures_df = factor_engine.compute_exposures(window_asset, window_factor)
        
        # 3. Macro Shock determination representing Reality realization mapped (t_current -> t_next)
        next_period_factors = factor_returns.loc[(factor_returns.index > t_current) & (factor_returns.index <= t_next)]
        if next_period_factors.empty:
            continue
            
        realized_factor_shock = (1 + next_period_factors).prod() - 1
        shock_dict = realized_factor_shock.to_dict()
        
        # 4. Extrapolate Model Prediction using exact reality parameters evaluated dynamically
        # Ensure exposures are aligned with weights
        aligned_exposures = exposures_df.reindex(weights.index).fillna(0)
        weighted_exposures = aligned_exposures.mul(weights, axis=0).sum()
        predicted_port_return = sum(weighted_exposures.get(f, 0) * shock for f, shock in shock_dict.items())
        
        # 5. Extract True Measured Return (Control Group reality)
        next_period_assets = asset_returns.loc[(asset_returns.index > t_current) & (asset_returns.index <= t_next)]
        actual_asset_returns = (1 + next_period_assets).prod() - 1
        
        # Align actual returns with weights
        aligned_actuals = actual_asset_returns.reindex(weights.index).fillna(0)
        actual_port_return = (weights * aligned_actuals).sum()
        
        results.append({
            'Date': t_next,
            'Predicted Return': predicted_port_return,
            'Actual Return': actual_port_return,
            'Error': predicted_port_return - actual_port_return
        })
        
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.set_index('Date', inplace=True)
        
        # Prevent degenerate outputs
        if df_results["Actual Return"].abs().sum() == 0:
            logger.error("Backtest Error: Actual returns are all zero. Check asset data pipeline.")
        if df_results["Predicted Return"].abs().sum() == 0:
            logger.error("Backtest Error: Predicted returns are all zero. Check model/factor logic.")
            
        logger.debug(f"Backtest Sample Results:\n{df_results.head()}")
        
    metrics = compute_error_metrics(df_results)
    
    return df_results, metrics
