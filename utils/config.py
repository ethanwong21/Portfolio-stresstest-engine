import yaml
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Dict, List

class PortfolioConfig(BaseModel):
    file_path: str
    columns: Dict[str, str]

class MarketDataConfig(BaseModel):
    source: str
    start_date: str
    end_date: str
    factors: Dict[str, str]

class ModelParametersConfig(BaseModel):
    rolling_window_days: int
    var_confidence_level: float

class ScenarioConfig(BaseModel):
    name: str
    shocks: Dict[str, float]

class ExcelExportConfig(BaseModel):
    enable: bool
    file_name: str
    timestamp: bool = False

class OutputsConfig(BaseModel):
    results_dir: str
    csv_export: bool = False
    png_export: bool = False
    excel_export: ExcelExportConfig

class DynamicScenariosConfig(BaseModel):
    enable: bool
    sigma_levels: List[int]
    factors: List[str]

class ComparisonConfig(BaseModel):
    enable: bool
    sorting_metric: str = "worst_scenario_return"

class BacktestConfig(BaseModel):
    enabled: bool
    frequency: str = "M"
    window_size: int = 60
    start_date: str
    end_date: str

class AppConfig(BaseModel):
    portfolio: PortfolioConfig
    market_data: MarketDataConfig
    model_parameters: ModelParametersConfig
    scenarios: List[ScenarioConfig]
    dynamic_scenarios: DynamicScenariosConfig
    comparison: ComparisonConfig
    backtest: BacktestConfig
    outputs: OutputsConfig

    model_config = ConfigDict(strict=False)

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Loads and validates the configuration from the YAML file.
    
    Args:
        config_path (str): Path to the config file. Defaults to "config.yaml".
        
    Returns:
        AppConfig: Validated Pydantic configuration model.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")
        
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
        
    if not raw_config:
        raise ValueError("Configuration file is empty or invalid.")
        
    return AppConfig.model_validate(raw_config)

if __name__ == "__main__":
    # Test function
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Factors Configured: {list(config.market_data.factors.keys())}")
    print(f"Scenarios Loaded: {len(config.scenarios)}")
