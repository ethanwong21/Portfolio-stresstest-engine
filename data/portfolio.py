import pandas as pd
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
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
        self.warnings = []
        self.detected_cols = {}
        self.original_cols = []
        self.cleaned_cols = []

    def _normalize_col(self, c: str) -> str:
        """Helper to normalize column names: strip, lower, remove special chars."""
        # Strip whitespace, convert to lowercase
        # Remove (%) , () , spaces
        return c.strip().lower().replace('%', '').replace('(', '').replace(')', '').replace(' ', '')

    def load_portfolio(self) -> pd.DataFrame:
        """
        Loads the portfolio CSV and validates that all configured columns exist.
        
        Returns:
            pd.DataFrame: A normalized portfolio dataframe.
        """
        try:
            df = pd.read_csv(self.config.file_path)
            logger.info(f"Loaded raw CSV from {self.config.file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Portfolio file '{self.config.file_path}' not found.")
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}")

        # 1. Flexible Column Detection with Normalization
        self.original_cols = list(df.columns)
        clean_to_orig = {self._normalize_col(c): c for c in df.columns}
        self.cleaned_cols = list(clean_to_orig.keys())

        ticker_keys = ['ticker', 'symbol', 'asset', 'security']
        weight_keys = ['weight', 'allocation', 'percent']
        asset_class_keys = ['assetclass', 'sector', 'industry', 'type']

        # Flexible matching
        ticker_col = next((clean_to_orig[c] for c in self.cleaned_cols if any(k in c for k in ticker_keys)), None)
        weight_col = next((clean_to_orig[c] for c in self.cleaned_cols if any(k in c or c == '%' for k in weight_keys)), None)
        asset_class_col = next((clean_to_orig[c] for c in self.cleaned_cols if any(k in c for k in asset_class_keys)), None)

        if not ticker_col:
            raise ValueError(f"Could not detect ticker column after normalization. Available: {self.cleaned_cols}")
        if not weight_col:
            raise ValueError(f"Could not detect weight column after normalization. Available: {self.cleaned_cols}")

        self.detected_cols = {'ticker': ticker_col, 'weight': weight_col}
        if asset_class_col:
            self.detected_cols['asset_class'] = asset_class_col

        # 2. Standardize & Clean
        cols_to_keep = [ticker_col, weight_col]
        if asset_class_col:
            cols_to_keep.append(asset_class_col)
            
        df = df[cols_to_keep].copy()
        
        rename_map = {ticker_col: 'ticker', weight_col: 'weight'}
        if asset_class_col:
            rename_map[asset_class_col] = 'asset_class'
            
        df = df.rename(columns=rename_map)
        
        # Clean Tickers: strip whitespace, uppercase
        df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
        df = df[df['ticker'] != 'NAN']  # drop empty ticker rows
        
        # Clean Weights: handle percentages and invalid values
        def clean_weight(val):
            if isinstance(val, str):
                val = val.replace('%', '').strip()
            try:
                return float(val)
            except:
                return np.nan

        df['weight'] = df['weight'].apply(clean_weight)
        df = df.dropna(subset=['weight'])
        
        if df.empty:
            raise ValueError("Portfolio is empty after cleaning (no valid tickers and weights found).")

        # 3. Handle Weight Formats (Whole vs Decimal)
        # If any weight is > 1.0, assume they are whole numbers (e.g., 25 instead of 0.25)
        if (df['weight'] > 1.0).any():
            self.warnings.append("Weights detected as whole numbers (e.g., 25 vs 0.25). Rescaling.")
            df['weight'] = df['weight'] / 100.0

        # 4. Aggregation & Validation
        # Sum duplicates
        if df['ticker'].duplicated().any():
            self.warnings.append("Duplicate tickers detected. Aggregating weights.")
            df = df.groupby('ticker')['weight'].sum().reset_index()

        if (df['weight'] < 0).any():
            raise ValueError("Negative weights are not allowed in the portfolio.")

        # 5. Normalization (Enforce sum to 1.0)
        total_w = df['weight'].sum()
        if not np.isclose(total_w, 1.0, atol=1e-4):
            self.warnings.append(f"Weights normalized to sum to 1.0 (Original sum: {total_w:.4f})")
            df['weight'] = df['weight'] / total_w

        return df

if __name__ == "__main__":
    # Smoke test placeholder
    pass
