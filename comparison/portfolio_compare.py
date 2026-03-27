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
        Returns a sorted pandas Dataframe.
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
        
        # Sort logic
        sort_col = 'Worst Scenario Return'
        if self.config and getattr(self.config, 'sorting_metric', None) == 'worst_scenario_return':
            sort_col = 'Worst Scenario Return'
            
        if not df.empty:
            df = df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)
            
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
