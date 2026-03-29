import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import copy
from pathlib import Path
from datetime import datetime

# Helper to safely format numeric KPI values
def safe_format(value, fmt="{:.2f}%"):
    """Return a formatted string for a numeric value or "N/A" if invalid.
    Handles None, NaN, empty Series, or non‑numeric types.
    """
    try:
        # Convert pandas Series to scalar if length 1
        if isinstance(value, pd.Series):
            if len(value) == 1:
                value = value.iloc[0]
            else:
                return "N/A"
        # Handle None
        if value is None:
            return "N/A"
        # Convert to float
        val = float(value)
        # Check for NaN
        if pd.isna(val):
            return "N/A"
        return fmt.format(val)
    except Exception:
        return "N/A"

# Alias for backward compatibility
safe_metric_value = safe_format

from utils.config import load_config
from data.portfolio import PortfolioLoader
from data.market_data import MarketDataLoader
from cli.interface import run_portfolio_analysis
from scenarios.generator import ScenarioGenerator
from scenarios.dynamic_scenarios import DynamicScenarioGenerator
from comparison.portfolio_compare import PortfolioComparer
from outputs.reporting import ReportGenerator
from backtesting.rolling_backtest import run_rolling_backtest

# Page Config
st.set_page_config(
    page_title="Portfolio Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar: Configuration & Upload
    st.sidebar.header("Dashboard Settings")
    theme = st.sidebar.selectbox("UI Theme", ["Dark", "Light"], index=0)
    
    # Dynamic CSS based on Theme
    if theme == "Dark":
        bg_color = "#0e1117"
        card_bg = "#1e1e1e"
        text_color = "#ffffff"
        label_color = "#cccccc"
        plotly_template = "plotly_dark"
    else:
        bg_color = "#f8f9fa"
        card_bg = "#ffffff"
        text_color = "#000000"
        label_color = "#333333"
        plotly_template = "plotly_white"

    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        [data-testid="stMetric"] {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.05);
        }}
        [data-testid="stMetricLabel"] p {{
            color: {label_color} !important;
            font-size: 0.9rem !important;
        }}
        [data-testid="stMetricValue"] div {{
            color: {text_color} !important;
            font-weight: bold !important;
        }}
        .stMarkdown h3, .stMarkdown h2, .stMarkdown h1 {{
            color: {text_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.title("Portfolio Risk & Stress Intelligence")
    st.subheader("Financial Engineering Dashboard")
    
    # Sidebar: Configuration & Upload
    st.sidebar.header("Upload & Configure")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Portfolio CSV(s)", 
        type=["csv"], 
        accept_multiple_files=True
    )
    
    config_path = st.sidebar.text_input("Config Path", "config.yaml")
    
    st.sidebar.divider()
    st.sidebar.header("Execution Modes")
    run_dynamic = st.sidebar.checkbox("Enable Dynamic Scenarios", value=True)
    run_backtest = st.sidebar.checkbox("Run Rolling Backtest (Slow)", value=False)
    
    if uploaded_files:
        num_ports = len(uploaded_files)
        if num_ports == 1:
            if run_backtest:
                exec_mode = "BACKTEST"
            elif run_dynamic:
                exec_mode = "SCENARIO"
            else:
                exec_mode = "BASIC"
        else:
            exec_mode = "MULTI"
        st.sidebar.info(f"Execution Mode: **{exec_mode.upper()}**")
        
        # Load Config
        try:
            config = load_config(config_path)
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return

        if st.sidebar.button("Run Stress Test", use_container_width=True):
            with st.spinner(f"Executing Financial Models for {len(uploaded_files)} portfolio(s)..."):
                # 1. Market Data Ingestion
                m_loader = MarketDataLoader(config.market_data)
                f_rets = m_loader.fetch_data()
                
                # 2. Scenario Generation
                s_gen = ScenarioGenerator(config.scenarios)
                shocks = s_gen.get_shocks()
                if run_dynamic:
                    d_gen = DynamicScenarioGenerator(config.dynamic_scenarios, f_rets)
                    shocks.update(d_gen.generate_dynamic_scenarios())
                
                # 3. Execution Loop
                import logging
                logger = logging.getLogger("streamlit_app")
                
                results_list = []
                temp_files = []
                
                for uploaded_file in uploaded_files:
                    local_config = copy.deepcopy(config)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        temp_files.append(tmp_path)
                    
                    try:
                        res = run_portfolio_analysis(tmp_path, local_config, m_loader, f_rets, shocks, logger)
                        p_res = res.copy()
                        p_res["name"] = uploaded_file.name
                        results_list.append(p_res)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        continue

                # 4. Comparison & Reporting
                comparer = PortfolioComparer(getattr(config, 'comparison', None))
                for res in results_list:
                    p_name = Path(res["name"]).stem
                    m_val = res['portfolio'].get('market_value', pd.Series([0])).sum()
                    comparer.add_portfolio_result(p_name, res['scenario_pnl'], res['risk_metrics'], m_val)

                # 5. UI Presentation
                if exec_mode in ["BASIC", "SCENARIO", "BACKTEST"]:
                    # Single portfolio view
                    result = results_list[0]
                    p_name = Path(result["name"]).stem
                    st.divider()
                    # KPI metrics with safe formatting and unique keys
                    k1, k2, k3, k4 = st.columns(4)
                    scenario_data = result['scenario_pnl']
                    worst_scen_name = min(scenario_data.keys(), key=lambda k: scenario_data[k]['portfolio_return'])
                    worst_scen = scenario_data[worst_scen_name]
                    total_val = result['portfolio'].get('market_value', pd.Series([0])).sum()
                    max_dd = result['risk_metrics'].get('max_historical_drawdown')
                    var_val = result['risk_metrics'].get('var_percent')
                    k1.metric("Maximum Drawdown", safe_format(max_dd, fmt="{:.2f}%"), key=f"Maximum Drawdown*{p_name}*0")
                    k2.metric("Portfolio VaR (95%)", safe_format(var_val, fmt="{:.2f}%"), key=f"Portfolio VaR (95%)*{p_name}*1")
                    k3.metric("Worst Scenario Return", safe_format(worst_scen.get('portfolio_return'), fmt="{:.2f}%"), help=worst_scen_name, key=f"Worst Scenario Return*{p_name}*2")
                    k4.metric("Total Portfolio Value", f"${total_val:,.0f}" if pd.notna(total_val) else "N/A", key=f"Total Portfolio Value*{p_name}*3")
                    # Charts with unique keys
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        plot_df = pd.DataFrame([{'Scenario': n, 'Return': d['portfolio_return']} for n, d in scenario_data.items()])
                        fig = px.bar(plot_df, x='Scenario', y='Return', color='Return', color_continuous_scale='RdYlGn', template=plotly_template)
                        st.plotly_chart(fig, use_container_width=True, key=f"Scenario Performance*{p_name}*0")
                    with c2:
                        contrib_df = worst_scen['asset_contributions'].reset_index()
                        fig_pie = px.pie(contrib_df, values=contrib_df.iloc[:,1].abs(), names=contrib_df.columns[0], hole=.4, template=plotly_template)
                        st.plotly_chart(fig_pie, use_container_width=True, key=f"Worst Scenario Attribution*{p_name}*1")
                    # Backtest if applicable
                    if exec_mode == "BACKTEST":
                        p_loader = PortfolioLoader(config.portfolio)
                        p_df = p_loader.load_portfolio()
                        tickers = p_df['ticker'].tolist() if 'ticker' in p_df.columns else p_df.index.tolist()
                        a_rets = m_loader.fetch_asset_returns(tickers)
                        bt_df, bt_metrics = run_rolling_backtest(p_df, a_rets, f_rets, config.backtest, config.model_parameters)
                        if bt_df is not None and not bt_df.empty:
                            st.divider()
                            st.subheader("Historical Model Validation (Backtest)")
                            b1, b2, b3 = st.columns(3)
                            b1.metric("MAE", f"{bt_metrics['MAE']*100:.2f}%")
                            b2.metric("RMSE", f"{bt_metrics['RMSE']*100:.2f}%")
                            b3.metric("Directional Hit", f"{bt_metrics['Directional Accuracy']*100:.1f}%")
                            fig_bt = go.Figure()
                            fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Predicted Return'], name="Predicted", line=dict(color='royalblue', width=2)))
                            fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Actual Return'], name="Actual", line=dict(color='firebrick', width=2, dash='dot')))
                            fig_bt.update_layout(title="Predicted vs Actual Portfolio Returns", xaxis_title="Date", yaxis_title="Return", template=plotly_template)
                            st.plotly_chart(fig_bt, use_container_width=True, key=f"Backtest Chart*{p_name}*0")
                else:
                    # Multi-portfolio comparison view
                    st.divider()
                    st.subheader("Multi‑Portfolio Risk Comparison")
                    comp_df = comparer.compare_portfolios()
                    if not comp_df.empty:
                        st.dataframe(comp_df.style.format("{:.2%}"), use_container_width=True, key="Comparison Table*MULTI*0")
                        fig = px.bar(comp_df, x='Portfolio Name', y='Worst Scenario Return', color='Worst Scenario Return', color_continuous_scale='RdYlGn', template=plotly_template)
                        st.plotly_chart(fig, use_container_width=True, key="Risk Ranking*MULTI*1")
                    # Deep‑dive sections for each portfolio
                    st.divider()
                    st.subheader("Individual Portfolio Deep‑Dives")
                    for i, res in enumerate(results_list):
                        p_name = Path(res["name"]).stem
                        with st.expander(f"🔍 Deep‑Dive: {p_name}", expanded=False, key=f"Expander*{p_name}*{i}"):
                            k1, k2, k3 = st.columns(3)
                            scen = res['scenario_pnl']
                            worst_name = min(scen.keys(), key=lambda k: scen[k]['portfolio_return'])
                            worst = scen[worst_name]
                            k1.metric("Worst Case", safe_format(worst.get('portfolio_return'), fmt="{:.2f}%"), help=worst_name, key=f"Worst Case*{p_name}*{i}")
                            k2.metric("VaR (95%)", safe_format(res['risk_metrics'].get('var_percent'), fmt="{:.2f}%"), key=f"VaR*{p_name}*{i}")
                            k3.metric("Max Drawdown", safe_format(res['risk_metrics'].get('max_historical_drawdown'), fmt="{:.2f}%"), key=f"Max Drawdown*{p_name}*{i}")
                            plot_df = pd.DataFrame([{'Scenario': n, 'Return': d['portfolio_return']} for n, d in scen.items()])
                            fig = px.bar(plot_df, x='Scenario', y='Return', color='Return', color_continuous_scale='RdYlGn', template=plotly_template)
                            st.plotly_chart(fig, use_container_width=True, key=f"Deep Dive Chart*{p_name}*{i}")

                # Cleanup
                for tmp_path in temp_files:
                    if os.path.exists(tmp_path): os.unlink(tmp_path)

    else:
        st.info("Please upload a portfolio CSV file to begin analysis.")
        st.divider()
        st.markdown("### Dashboard Preview")
        st.image("https://images.unsplash.com/photo-1611974717482-58284396e8c7?q=80&w=2070&auto=format&fit=crop", caption="Professional Risk Intelligence Visualization")

if __name__ == "__main__":
    main()
