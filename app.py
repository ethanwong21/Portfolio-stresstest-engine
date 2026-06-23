import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import copy
import numpy as np
import uuid
from pathlib import Path
from datetime import datetime

def safe_format(value, fmt="{:.2f}%"):
    """Return a formatted string for a numeric value or 'N/A' if invalid."""
    if isinstance(value, pd.Series):
        value = value.iloc[0] if len(value) == 1 else "N/A"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "N/A"
    if value is None:
        return "N/A"
    try:
        v = float(value)
        return fmt.format(v) if not pd.isna(v) else "N/A"
    except Exception:
        return "N/A"

from utils.config import load_config
from data.portfolio import PortfolioLoader
from data.market_data import MarketDataLoader
from cli.interface import run_portfolio_analysis
from scenarios.generator import ScenarioGenerator
from scenarios.dynamic_scenarios import DynamicScenarioGenerator
from comparison.portfolio_compare import PortfolioComparer
from outputs.reporting import ReportGenerator
from backtesting.rolling_backtest import run_rolling_backtest

st.set_page_config(
    page_title="Portfolio Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600, show_spinner=False)
def _get_cached_shocks(_s_gen):
    """Cache historical scenario downloads for 1 hour to avoid repeated yfinance calls."""
    return _s_gen.get_shocks(include_historical=True)


def main():
    st.sidebar.header("Dashboard Settings")
    theme = st.sidebar.selectbox("UI Theme", ["Dark", "Light"], index=0)

    if theme == "Dark":
        bg_color, card_bg, text_color, label_color, plotly_template = "#0e1117", "#1e1e1e", "#ffffff", "#cccccc", "plotly_dark"
    else:
        bg_color, card_bg, text_color, label_color, plotly_template = "#f8f9fa", "#ffffff", "#000000", "#333333", "plotly_white"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        [data-testid="stMetric"] {{ background-color: {card_bg}; padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); }}
        [data-testid="stMetricLabel"] p {{ color: {label_color} !important; }}
        [data-testid="stMetricValue"] div {{ color: {text_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

    if "demo_mode_enabled" not in st.session_state:
        st.session_state["demo_mode_enabled"] = True
    if "demo_mode_type" not in st.session_state:
        st.session_state["demo_mode_type"] = "SINGLE"

    demo_growth_portfolio = pd.DataFrame([
        {'ticker': 'AAPL', 'weight': 0.20, 'asset_class': 'Tech'},
        {'ticker': 'MSFT', 'weight': 0.20, 'asset_class': 'Tech'},
        {'ticker': 'NVDA', 'weight': 0.20, 'asset_class': 'Growth'},
        {'ticker': 'AMZN', 'weight': 0.15, 'asset_class': 'Growth'},
        {'ticker': 'GOOGL', 'weight': 0.15, 'asset_class': 'Growth'},
        {'ticker': 'TSLA', 'weight': 0.10, 'asset_class': 'Growth'}
    ])
    demo_defensive_portfolio = pd.DataFrame([
        {'ticker': 'JNJ', 'weight': 0.15, 'asset_class': 'Health Care'},
        {'ticker': 'PG', 'weight': 0.15, 'asset_class': 'Consumer Staples'},
        {'ticker': 'KO', 'weight': 0.15, 'asset_class': 'Consumer Staples'},
        {'ticker': 'PEP', 'weight': 0.15, 'asset_class': 'Consumer Staples'},
        {'ticker': 'XOM', 'weight': 0.10, 'asset_class': 'Energy'},
        {'ticker': 'TLT', 'weight': 0.20, 'asset_class': 'Bond'},
        {'ticker': 'GLD', 'weight': 0.10, 'asset_class': 'Commodity'}
    ])

    run_analysis = False
    run_scenario_flag = True
    run_backtest_flag = False

    st.sidebar.header("Execution Hub")
    demo_mode_toggle = st.sidebar.toggle("Demo Mode", value=st.session_state["demo_mode_enabled"])

    if demo_mode_toggle != st.session_state["demo_mode_enabled"]:
        st.session_state["demo_mode_enabled"] = demo_mode_toggle
        st.session_state["demo_mode_type"] = "SINGLE" if demo_mode_toggle else None
        st.rerun()

    if st.session_state["demo_mode_enabled"]:
        st.success("⚡ **Demo Mode Active**")
    else:
        st.info("📂 **Custom Mode (Upload Your Portfolio)**")

    st.title("Portfolio Risk & Stress Intelligence")
    st.subheader("Financial Engineering Dashboard")

    st.sidebar.divider()
    uploaded_files = None

    if not st.session_state["demo_mode_enabled"]:
        st.sidebar.header("Upload Portfolio(s)")
        uploaded_files = st.sidebar.file_uploader("Upload Portfolio CSV(s)", type=["csv"], accept_multiple_files=True)
    else:
        st.sidebar.header("Demo Controls")
        d_choice = st.sidebar.selectbox(
            "Run Single Portfolio Demo",
            ["Dynamic Scenario Analysis", "Rolling Backtest"],
            index=0 if st.session_state.get("demo_analysis_type") == "Rolling Backtest" else 0
        )
        current_choice = st.session_state.get("demo_analysis_type")
        if st.session_state["demo_mode_type"] != "SINGLE" or current_choice != d_choice:
            st.session_state["demo_mode_type"] = "SINGLE"
            st.session_state["demo_analysis_type"] = d_choice
            run_analysis = True

        st.sidebar.divider()
        if st.sidebar.button("Run Multi Portfolio Demo", use_container_width=True):
            st.session_state["demo_mode_enabled"] = True
            st.session_state["demo_mode_type"] = "MULTI"
            run_analysis = True

    portfolios = None
    if st.session_state.get("demo_mode_enabled"):
        if st.session_state.get("demo_mode_type") == "SINGLE":
            portfolios = [{"name": "Growth_Strategy_Demo.csv", "df": demo_growth_portfolio}]
        elif st.session_state.get("demo_mode_type") == "MULTI":
            portfolios = [
                {"name": "Growth_Strategy_Demo.csv", "df": demo_growth_portfolio},
                {"name": "Defensive_Yield_Demo.csv", "df": demo_defensive_portfolio}
            ]
    elif uploaded_files:
        portfolios = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

    if portfolios is None:
        exec_mode = None
    elif len(portfolios) == 1:
        exec_mode = "SINGLE"
    else:
        exec_mode = "MULTI"

    if exec_mode:
        st.sidebar.divider()
        st.sidebar.info(f"Execution: **{exec_mode}**")
        st.markdown(f"#### Execution Mode: `{exec_mode}`")

    if not st.session_state.get("demo_mode_enabled"):
        if exec_mode == "SINGLE":
            st.sidebar.subheader("Analysis Components")
            run_scenario_flag = st.sidebar.checkbox("Dynamic Scenario Analysis", value=True)
            run_backtest_flag = st.sidebar.checkbox("Run Rolling Backtest")
        if exec_mode:
            if st.sidebar.button("Run Stress Test", use_container_width=True):
                run_analysis = True

    if exec_mode == "MULTI":
        run_backtest_flag = False
        st.sidebar.caption("💡 Backtesting only available for single portfolio")

    if not run_analysis:
        st.info("Run a demo or upload a portfolio to begin")
        st.divider()
        st.markdown("### Dashboard Preview")
        st.image("https://images.unsplash.com/photo-1611974717482-58284396e8c7?q=80&w=2070&auto=format&fit=crop", caption="Institutional Risk Visualization")
        return

    from utils.config import AppConfig, PortfolioConfig, MarketDataConfig, ModelParametersConfig, \
        ScenarioConfig, DynamicScenariosConfig, ComparisonConfig, BacktestConfig, OutputsConfig, ExcelExportConfig

    config = AppConfig(
        portfolio=PortfolioConfig(file_path="", columns={}),
        market_data=MarketDataConfig(source="yfinance", start_date="2020-01-01", end_date=datetime.now().strftime("%Y-%m-%d"),
                                    factors={"equity": "^GSPC", "rates": "^TNX", "inflation": "TIP", "commodities": "GSG"}),
        model_parameters=ModelParametersConfig(rolling_window_days=252, var_confidence_level=0.95),
        scenarios=[],
        dynamic_scenarios=DynamicScenariosConfig(enable=True, sigma_levels=[1, 2, 3], factors=["equity", "rates", "inflation", "commodities"]),
        comparison=ComparisonConfig(enable=True),
        backtest=BacktestConfig(enabled=True, start_date="2020-01-01", end_date=datetime.now().strftime("%Y-%m-%d")),
        outputs=OutputsConfig(results_dir="outputs", excel_export=ExcelExportConfig(enable=True, file_name="risk_report.xlsx"))
    )

    with st.spinner(f"Executing Analytics for {len(portfolios)} portfolio(s)..."):
        m_loader = MarketDataLoader(config.market_data)
        f_rets = m_loader.fetch_data()
        s_gen = ScenarioGenerator(config.scenarios)
        # Historical scenarios cached for 1hr to avoid repeated yfinance calls
        shocks = _get_cached_shocks(s_gen)
        d_gen = DynamicScenarioGenerator(config.dynamic_scenarios, f_rets)
        shocks.update(d_gen.generate_dynamic_scenarios())

        results_list, temp_files = [], []
        import logging
        logger = logging.getLogger("streamlit_app")

        for target in portfolios:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                if "df" in target: target["df"].to_csv(tmp.name, index=False)
                else: tmp.write(target["bytes"])
                temp_files.append(tmp.name)
            try:
                res = run_portfolio_analysis(tmp.name, config, m_loader, f_rets, shocks, logger)
                res["name"] = target["name"]
                results_list.append(res)
            except Exception as e:
                st.error(f"Analysis Failed for {target['name']}: {e}")
                continue

        if exec_mode == "SINGLE":
            result = results_list[0]
            p_name = Path(result["name"]).stem
            st.divider()

            is_demo = st.session_state.get("demo_mode_enabled")
            single_type = st.session_state.get("demo_analysis_type", "Dynamic Scenario Analysis") if is_demo else None

            if (is_demo and single_type == "Dynamic Scenario Analysis") or (not is_demo and run_scenario_flag):
                st.markdown(f"### Risk Profile: {p_name}")
                st.info("Mode: Dynamic Scenario Stress Testing")
                k1, k2, k3, k4 = st.columns(4)
                scen_data = result['scenario_pnl']
                worst_n = min(scen_data.keys(), key=lambda k: scen_data[k]['portfolio_return'])
                worst = scen_data[worst_n]
                m_val = result['portfolio'].get('market_value', pd.Series([0])).sum()

                k1.metric("Maximum Drawdown", safe_format(result['risk_metrics'].get('max_historical_drawdown')))
                k2.metric("Portfolio VaR (95%)", safe_format(result['risk_metrics'].get('var_percent')))
                k3.metric("Worst Case Return", safe_format(worst.get('portfolio_return')), help=worst_n)
                k4.metric("Market Value", f"${m_val:,.0f}")

                c1, c2 = st.columns([2, 1])
                with c1:
                    plot_df = pd.DataFrame([{'Scenario': n, 'Return': d['portfolio_return']} for n, d in scen_data.items()])
                    st.plotly_chart(px.bar(plot_df, x='Scenario', y='Return', color='Return', color_continuous_scale='RdYlGn', template=plotly_template), use_container_width=True)
                with c2:
                    contrib_df = worst['asset_contributions'].reset_index()
                    st.plotly_chart(px.pie(contrib_df, values=contrib_df.iloc[:,1].abs(), names=contrib_df.columns[0], hole=.4, template=plotly_template), use_container_width=True)

            # CUSTOM SCENARIO BUILDER
            if (is_demo and single_type == "Dynamic Scenario Analysis") or (not is_demo and run_scenario_flag):
                st.divider()
                st.markdown("### Custom Scenario Builder")
                st.caption("Drag the sliders to define a bespoke macro shock and instantly see your portfolio impact.")

                cs_col1, cs_col2 = st.columns(2)
                with cs_col1:
                    cs_market = st.slider("Market shock (%)", min_value=-50, max_value=30, value=0, step=1,
                                          help="Equity market return shock (e.g. -25 = market crash)") / 100.0
                    cs_rates = st.slider("Rate shock (% change in yield)", min_value=-30, max_value=50, value=0, step=1,
                                         help="Percentage change in 10yr yield level") / 100.0
                with cs_col2:
                    cs_inflation = st.slider("Inflation shock (%)", min_value=-10, max_value=20, value=0, step=1,
                                             help="TIPS price return proxy for inflation expectations") / 100.0
                    cs_commodities = st.slider("Commodities shock (%)", min_value=-40, max_value=40, value=0, step=1,
                                               help="Broad commodity index return") / 100.0

                custom_shock = {
                    "market": cs_market,
                    "rates": cs_rates,
                    "inflation": cs_inflation,
                    "commodities": cs_commodities,
                }

                from models.scenario_impact import ScenarioImpactModel
                cs_exposures = result["exposures"]
                cs_portfolio = result["portfolio"].set_index("ticker") if "ticker" in result["portfolio"].columns else result["portfolio"]
                cs_weights = cs_portfolio["weight"]

                cs_impact_model = ScenarioImpactModel(cs_exposures)
                cs_asset_returns = cs_impact_model.propagate_shocks(custom_shock)
                cs_portfolio_return = float((cs_weights.reindex(cs_asset_returns.index).fillna(0) * cs_asset_returns).sum())

                cs_color = "normal" if cs_portfolio_return >= 0 else "inverse"
                cs_m1, cs_m2 = st.columns(2)
                cs_m1.metric("Custom Scenario Portfolio Return", f"{cs_portfolio_return:.2%}", delta=f"{cs_portfolio_return:.2%}", delta_color=cs_color)

                cs_contrib_df = cs_asset_returns.reset_index()
                cs_contrib_df.columns = ["Ticker", "Estimated Return"]
                cs_contrib_df = cs_contrib_df.sort_values("Estimated Return")
                cs_m2.plotly_chart(
                    px.bar(cs_contrib_df, x="Ticker", y="Estimated Return",
                           color="Estimated Return", color_continuous_scale="RdYlGn",
                           title="Asset-Level Impact", template=plotly_template),
                    use_container_width=True,
                )

            if (is_demo and single_type == "Rolling Backtest") or (not is_demo and run_backtest_flag):
                st.divider()
                st.warning("Mode: Historical Rolling Backtest")
                p_df = result['portfolio']
                tickers = p_df['ticker'].tolist() if 'ticker' in p_df.columns else p_df.index.tolist()
                a_rets = m_loader.fetch_asset_returns(tickers)
                backtest_df, metrics = run_rolling_backtest(p_df, a_rets, f_rets, config.backtest, config.model_parameters)

                if backtest_df is not None:
                    b1, b2, b3 = st.columns(3)
                    b1.metric("MAE", f"{metrics['MAE']*100:.2f}%")
                    b2.metric("RMSE", f"{metrics['RMSE']*100:.2f}%")
                    b3.metric("Directional Hit", f"{metrics['Directional Accuracy']*100:.1f}%")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted Return'], name="Predicted"))
                    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual Return'], name="Actual", line=dict(dash='dot')))
                    fig.update_layout(title="Predicted vs Actual Portfolio Returns", template=plotly_template)
                    st.plotly_chart(fig, use_container_width=True)

        elif exec_mode == "MULTI":
            st.divider()
            st.subheader("Multi-Portfolio Risk Comparison")
            comparer = PortfolioComparer(config.comparison)
            for res in results_list:
                comparer.add_portfolio_result(Path(res["name"]).stem, res['scenario_pnl'], res['risk_metrics'], res['portfolio'].get('market_value', pd.Series([0])).sum())
            comp_df = comparer.compare_portfolios()

            if not comp_df.empty:
                if "Resilience Score" not in comp_df.columns:
                    st.error("Resilience Score not computed. Please check simulation input.")
                    st.write(comp_df)
                    return

                st.markdown("### Decision Summary & Resilience Ranking")
                d1, d2, d3 = st.columns(3)
                comp_df["Resilience Score"] = pd.to_numeric(comp_df["Resilience Score"], errors='coerce')
                best_p = comp_df.iloc[0]['Portfolio Name']
                resilient_p = comp_df.loc[comp_df['Resilience Score'].idxmax(), 'Portfolio Name']
                risk_p = comp_df.loc[comp_df['Resilience Score'].idxmin(), 'Portfolio Name']
                d1.success(f"🏆 **Highest Resilience**: {best_p}")
                d2.info(f"🛡️ **Most Stable**: {resilient_p}")
                d3.warning(f"⚠️ **Highest Risk**: {risk_p}")

                st.markdown("### Performance & Risk Comparison")
                num_cols = ['Worst Scenario Return', 'Best Scenario Return', 'Max Drawdown', 'VaR']
                for c in num_cols: comp_df[c] = pd.to_numeric(comp_df[c], errors='coerce')
                fmt = {c: "{:.2%}" for c in num_cols if c in comp_df.columns}
                fmt['Resilience Score'], fmt['Total Value'] = "{:.1f}", "${:,.0f}"
                st.dataframe(comp_df.style.format(fmt, na_rep="N/A").background_gradient(subset=['Resilience Score'], cmap='RdYlGn'), use_container_width=True)

                st.markdown("### Scenario Stress Test Comparison")
                all_scen = []
                for res in results_list:
                    for n, d in res['scenario_pnl'].items():
                        all_scen.append({'Portfolio': Path(res["name"]).stem, 'Scenario': n, 'Return': d['portfolio_return']})
                st.plotly_chart(px.bar(pd.DataFrame(all_scen), x='Scenario', y='Return', color='Portfolio', barmode='group', template=plotly_template), use_container_width=True)

        for f in temp_files:
            if os.path.exists(f): os.unlink(f)

if __name__ == "__main__":
    main()
