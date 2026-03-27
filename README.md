# Portfolio Stress Testing & Risk Intelligence Engine

A fully configurable, macro-driven portfolio risk analysis system built in Python. This engine evaluates how a portfolio performs under user-defined macroeconomic shock scenarios and provides insights into asset-level risk attribution. It operates without hardcoded values, relying entirely on a declarative YAML configuration and external data sources for maximum flexibility.

## Features

- **Dynamic Portfolio Ingestion**: Load any portfolio structure via CSV, dynamically mapping columns (e.g., ticker, weight, asset class) via configuration.
- **Resilient Market Data**: Fetches historical market and benchmark data using `yfinance`. Automatically falls back to multi-variate continuous Geometric Brownian Motion simulation if API rate limits or database locks occur.
- **Factor Exposure Engine**: Computes dynamic, rolling ordinary least squares (OLS) regression betas to map asset sensitivities against custom macro factors (e.g., Equity, Rates, Credit, Inflation).
- **Scenario Impact Modeling**: Propagates defined macro shocks seamlessly through the exposure matrix to predict hypothetical asset returns and portfolio drawdowns.
- **Risk Aggregation**: Calculates historical Value-at-Risk (VaR) and Maximum Drawdowns based on historical arrays and custom confidence intervals.
- **Institutional Excel Reporting**: Transforms all outputs into a single, professional Microsoft Excel dashboard (`xlsxwriter`). Features comprehensive KPI strips, ranked scenario summaries, conditional formatting, and embedded dynamic charts (Pie/Bar) natively in the spreadsheet.

## Project Structure

```text
Portfolio-stresstest-tool/
├── config.yaml               # Master configuration for all engine parameters
├── main.py                   # Engine entry point and orchestration pipeline
├── sample_portfolio.csv      # Sample portfolio ingestion file
├── requirements.txt          # Python package dependencies
├── data/
│   ├── market_data.py        # yFinance integration and simulation fallback generator
│   └── portfolio.py          # Portfolio CSV parser and validator
├── models/
│   ├── factor_engine.py      # Rolling OLS regression exposure calculator
│   ├── risk_engine.py        # Portfolio-level aggregation and VaR metrics
│   └── scenario_impact.py    # Macro shock propagation math
├── outputs/                  
│   └── reporting.py          # Excel dashboard and KPI summary generator
├── scenarios/
│   └── generator.py          # Scenario dictionary builder from YAML
└── utils/
    └── config.py             # Pydantic configuration validation logic
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Portfolio-stresstest-tool.git
   cd Portfolio-stresstest-tool
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure the Engine:**
   Modify `config.yaml` to set your desired:
   - File input paths (`sample_portfolio.csv`).
   - Market factors (e.g., `^GSPC` for Equity, `^IRX` for Rates).
   - Rolling windows and VaR confidence levels.
   - Stress test shock scenarios (e.g., `Equity Crash`, `Yield Curve Shock`).
   - Output settings (Enable Excel dashboard generation).

2. **Run the Stress Test:**
   ```bash
   python main.py --config config.yaml
   ```

3. **View the Results:**
   Navigate to the `outputs/` directory to open your generated institutional dashboard (`stress_test_dashboard.xlsx`).

## Configuration (`config.yaml`) Example

The system is entirely driven by the YAML configuration. Here is an example of a shock scenario definition:

```yaml
scenarios:
  - name: "Equity Market Crash"
    shocks:
      equity: -0.20
      rates: -0.01  # Flight to safety (rates down 100bps)
      credit: 0.05
      inflation: -0.01

  - name: "Stagflation"
    shocks:
      equity: -0.15
      rates: 0.02
      credit: 0.03
      inflation: 0.04
```

## Dependencies
- `pandas` - Data manipulation
- `numpy` - Vectorized calculations & simulations
- `yfinance` - Historical market data
- `scikit-learn` - OLS Regression
- `PyYAML` - Configuration parsing
- `pydantic` - Settings validation
- `matplotlib` & `seaborn` - Analytics computations
- `xlsxwriter` - Professional Excel formatting
