import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PortfolioLoader:
    def __init__(self, config):
        """
        Initializes the PortfolioLoader with configuration mappings.
        
        Args:
            config: A PortfolioConfig object from the parsed pydantic config.
        """
        self.config = config

    def load_portfolio(self) -> pd.DataFrame:
        """
        Loads the portfolio CSV and validates that all configured columns exist.
        
        Returns:
            pd.DataFrame: A normalized portfolio dataframe.
        """
        try:
            df = pd.read_csv(self.config.file_path)
            logger.info(f"Loaded portfolio from {self.config.file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Portfolio file '{self.config.file_path}' not found.")
            
        required_columns = self.config.columns.values()
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Portfolio CSV missing required configured columns: {missing_columns}")
            
        # Optional: standardize column names based on the config keys instead of values
        # This makes it easier to reference them in the code
        rename_map = {v: k for k, v in self.config.columns.items()}
        df = df.rename(columns=rename_map)
        
        # Calculate market value if required cols exist
        if 'price' in df.columns and 'quantity' in df.columns:
            df['market_value'] = df['price'] * df['quantity']
        
        # Calculate weights if missing
        if 'weight' not in df.columns and 'market_value' in df.columns:
            total_mv = df['market_value'].sum()
            df['weight'] = df['market_value'] / total_mv
            
        if abs(df['weight'].sum() - 1.0) > 0.01:
            logger.warning("Portfolio weights do not sum up to 1.0 (100%)")
            
        return df

if __name__ == "__main__":
    # Smoke test placeholder
    pass
