import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PortfolioComparer:
    def __init__(self, comparison_config):
        """
        Initializes the cross-portfolio comparison engine.
        
        Args:
            comparison_config: Configuration dictating metrics and UI sort behaviors.
        """
        self.config = comparison_config
        self.results = {}

    def add_portfolio_result(self, name: str, scenario_pnl: pd.DataFrame, risk_metrics: dict, total_value: float):
        """
        Stores the raw evaluation outcomes for a given portfolio identifier.
        """
        self.results[name] = {
            'scenario_pnl': scenario_pnl,
            'risk_metrics': risk_metrics,
            'total_value': total_value
        }

    def compare_portfolios(self) -> pd.DataFrame:
        """
        Aggregates stored portfolios, extracting high-level comparative metrics.
        Returns a sorted pandas Dataframe with Resilience Score and Rank.
        """
        comparison_data = []
        for name, data in self.results.items():
            scenario_pnl = data['scenario_pnl']
            risk_metrics = data['risk_metrics']
            
            # Scenario Extremes
            returns = [metrics['portfolio_return'] for metrics in scenario_pnl.values()]
            worst_return = min(returns) if returns else 0.0
            best_return = max(returns) if returns else 0.0
            
            max_drawdown = risk_metrics.get('max_historical_drawdown', 0.0)
            var_value = risk_metrics.get('var_percent', 0.0)
            
            comparison_data.append({
                'Portfolio Name': name,
                'Worst Scenario Return': worst_return,
                'Best Scenario Return': best_return,
                'Max Drawdown': max_drawdown,
                'VaR': var_value,
                'Total Value': data['total_value']
            })
            
        df = pd.DataFrame(comparison_data)
        if df.empty:
            df['Resilience Score'] = []
            df['Rank'] = []
            return df
            
        # Initialize defaults to prevent KeyError
        df['Resilience Score'] = 0.0
        df['Rank'] = 0
            
        # 1. Normalize metrics across portfolios (Min-Max)
        # All inputs (worst return, max ddraw, var) are typically negative.
        # Higher values (closer to zero) are better.
        def normalize(series):
            if series.empty or series.max() == series.min():
                return pd.Series(1.0, index=series.index)
            return (series - series.min()) / (series.max() - series.min())

        w_worst = normalize(df['Worst Scenario Return'])
        w_dd = normalize(df['Max Drawdown'])
        w_var = normalize(df['VaR'])
        
        # 2. Weighted Scoring (Resilience Score: 0-100)
        # Weights: 50% Worst Case, 30% Max Drawdown, 20% VaR
        df['Resilience Score'] = (w_worst * 0.5 + w_dd * 0.3 + w_var * 0.2) * 100
        
        # 3. Create Rank (1 = highest score)
        df['Rank'] = df['Resilience Score'].rank(ascending=False, method='min').astype(int)
        
        # Sort by Rank (ascending)
        df = df.sort_values(by='Rank', ascending=True).reset_index(drop=True)
            
        return df

    def print_summary(self, df: pd.DataFrame):
        """
        Prints a structured summary text table of performance across the dataset.
        """
        print("\n" + "="*40)
        print("      PORTFOLIO COMPARISON RESULTS")
        print("="*40)
        for _, row in df.iterrows():
            name = row['Portfolio Name'].capitalize()
            worst = row['Worst Scenario Return'] * 100
            var = row['VaR'] * 100
            print(f"{name}: Worst = {worst:.2f}%, VaR = {var:.2f}%")
        print("="*40 + "\n")
