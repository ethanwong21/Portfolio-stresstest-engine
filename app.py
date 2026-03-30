import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

# Helper to safely format numeric KPI values

# Helper to safely format numeric KPI values – always returns a string

def safe_format(value, fmt="{:.2f}%"):
    """Return a formatted string for a numeric value or "N/A" if invalid.
    Handles None, NaN, empty pandas Series, numpy arrays, lists, etc.
    """
    # Unwrap pandas Series
    if isinstance(value, pd.Series):
        if len(value) == 1:
            value = value.iloc[0]
        else:
            return "N/A"
    # Reject list, tuple, ndarray
    if isinstance(value, (list, tuple, np.ndarray)):
        return "N/A"
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
        return fmt.format(v)
    except Exception:
        return "N/A"

# Helper to sanitize Streamlit keys – no spaces or asterisks, lower‑case

def _sanitize_key(metric_name: str, portfolio_name: str, idx: int) -> str:
    clean_metric = metric_name.replace(" ", "_").replace("*", "").lower()
    clean_port = portfolio_name.replace(" ", "_").replace("*", "")
    return f"kpi_{clean_metric}*{clean_port}*{idx}"

# Debug logger for KPI values

def _log_kpi(name: str, raw, formatted: str, key: str):
    st.sidebar.info(f"[KPI DEBUG] {name}: raw={raw!r} (type={type(raw)}), formatted={formatted}, key={key}")

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
    
    if uploaded_files:
        num_ports = len(uploaded_files)
        exec_mode = "SINGLE" if num_ports == 1 else "MULTI"
        st.sidebar.info(f"Execution Mode: **{exec_mode}**")
        
        st.sidebar.divider()
        st.sidebar.header("Execution Modes")
        run_dynamic = st.sidebar.checkbox("Dynamic Scenarios", value=True)
        
        # Backtest only for SINGLE mode
        if exec_mode == "SINGLE":
            run_backtest = st.sidebar.checkbox("Run Rolling Backtest (Slow)", value=False)
        else:
            run_backtest = False
            st.sidebar.warning("Backtest disabled in Multi-Portfolio mode")
        
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
                        p_res["name"] = uploaded_files[0].name if num_ports == 1 else uploaded_file.name
                        
                        # Display Ingestion Warnings
                        if res.get('ingestion_warnings'):
                            with st.sidebar.expander(f"⚠️ {uploaded_file.name} Warnings"):
                                for w in res['ingestion_warnings']:
                                    st.write(f"- {w}")
                        
                        # Debug Output (Temporary)
                        if exec_mode == "SINGLE":
                            with st.expander(f"🛠️ Debug: {uploaded_files[0].name if num_ports == 1 else uploaded_file.name} Ingestion"):
                                st.write("**Original Headers:**", res.get('original_cols'))
                                st.write("**Cleaned Headers:**", res.get('cleaned_cols'))
                                st.write("**Detected Mapping:**", res.get('detected_cols'))
                                
                                # Factor Exposures Debug
                                if 'exposures' in res:
                                    st.write("**Factor Exposures (Betas):**")
                                    st.dataframe(res['exposures'])
                                    if not res['exposures'].empty:
                                        st.write(f"**Sample Beta ({res['exposures'].index[0]}):**", res['exposures'].iloc[0].to_dict())

                                st.write("**Cleaned Data Preview:**")
                                st.dataframe(res['portfolio'])

                        results_list.append(p_res)
                    except Exception as e:
                        st.error(f"Ingestion Failed for {uploaded_file.name}: {e}")
                        continue

                # 4. Comparison & Reporting
                comparer = PortfolioComparer(getattr(config, 'comparison', None))
                for res in results_list:
                    p_name = Path(res["name"]).stem
                    m_val = res['portfolio'].get('market_value', pd.Series([0])).sum()
                    comparer.add_portfolio_result(p_name, res['scenario_pnl'], res['risk_metrics'], m_val)

                # 5. UI Presentation
                if exec_mode == "SINGLE":
                    # Single portfolio view
                    result = results_list[0]
                    p_name = Path(result["name"]).stem
                    st.divider()
                    
                    # Scenario Analysis Section
                    with st.container():
                        st.subheader("Scenario Analysis")
                        k1, k2, k3, k4 = st.columns(4)
                        scenario_data = result['scenario_pnl']
                        worst_scen_name = min(scenario_data.keys(), key=lambda k: scenario_data[k]['portfolio_return'])
                        worst_scen = scenario_data[worst_scen_name]
                        total_val = result['portfolio'].get('market_value', pd.Series([0])).sum()
                        
                        max_dd_raw = result['risk_metrics'].get('max_historical_drawdown')
                        var_raw = result['risk_metrics'].get('var_percent')
                        worst_ret_raw = worst_scen.get('portfolio_return')
                        
                        max_dd_fmt = safe_format(max_dd_raw)
                        var_fmt = safe_format(var_raw)
                        worst_fmt = safe_format(worst_ret_raw)
                        total_fmt = f"${total_val:,.0f}" if pd.notna(total_val) else "N/A"
                        
                        k1.metric("Maximum Drawdown", max_dd_fmt)
                        k2.metric("Portfolio VaR (95%)", var_fmt)
                        k3.metric("Worst Scenario Return", worst_fmt, help=worst_scen_name)
                        k4.metric("Total Portfolio Value", total_fmt)
                        
                        scen_c1, scen_c2 = st.columns([2, 1])
                        with scen_c1:
                            scenario_plot_df = pd.DataFrame([{'Scenario': n, 'Return': d['portfolio_return']} for n, d in scenario_data.items()])
                            scenario_fig = px.bar(scenario_plot_df, x='Scenario', y='Return', color='Return', color_continuous_scale='RdYlGn', template=plotly_template)
                            st.plotly_chart(scenario_fig, use_container_width=True, key=f"Scenario Performance*{p_name}*0")
                        with scen_c2:
                            scenario_contrib_df = worst_scen['asset_contributions'].reset_index()
                            scenario_pie_fig = px.pie(scenario_contrib_df, values=scenario_contrib_df.iloc[:,1].abs(), names=scenario_contrib_df.columns[0], hole=.4, template=plotly_template)
                            st.plotly_chart(scenario_pie_fig, use_container_width=True, key=f"Worst Scenario Attribution*{p_name}*1")

                    # Backtest Section
                    if run_backtest:
                        with st.container():
                            st.divider()
                            st.subheader("Historical Model Validation (Backtest)")
                            p_loader = PortfolioLoader(config.portfolio)
                            p_df = p_loader.load_portfolio()
                            tickers = p_df['ticker'].tolist() if 'ticker' in p_df.columns else p_df.index.tolist()
                            a_rets = m_loader.fetch_asset_returns(tickers)
                            backtest_df, backtest_metrics = run_rolling_backtest(p_df, a_rets, f_rets, config.backtest, config.model_parameters)
                            
                            if backtest_df is not None and not backtest_df.empty:
                                b1, b2, b3 = st.columns(3)
                                b1.metric("MAE", f"{backtest_metrics['MAE']*100:.2f}%")
                                b2.metric("RMSE", f"{backtest_metrics['RMSE']*100:.2f}%")
                                b3.metric("Directional Hit", f"{backtest_metrics['Directional Accuracy']*100:.1f}%")
                                
                                backtest_fig = go.Figure()
                                backtest_fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted Return'], name="Predicted", line=dict(color='royalblue', width=2)))
                                backtest_fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual Return'], name="Actual", line=dict(color='firebrick', width=2, dash='dot')))
                                backtest_fig.update_layout(title="Predicted vs Actual Portfolio Returns", xaxis_title="Date", yaxis_title="Return", template=plotly_template)
                                st.plotly_chart(backtest_fig, use_container_width=True, key=f"Backtest Chart*{p_name}*0")

                else:
                    # Multi-portfolio comparison view
                    st.divider()
                    st.subheader("Multi‑Portfolio Risk Intelligence")
                    comp_df = comparer.compare_portfolios()
                    
                    if not comp_df.empty:
                        # 1. Enforce numeric types
                        numeric_cols = ['Worst Scenario Return', 'Best Scenario Return', 'Max Drawdown', 'VaR']
                        for col in numeric_cols:
                            if col in comp_df.columns:
                                comp_df[col] = pd.to_numeric(comp_df[col], errors='coerce')

                        # 2. Support for Resilience Scoring
                        # The Resilience Score (0-100) and Rank are already calculated in comparer.compare_portfolios()
                        # Weights: 50% Worst Case, 30% Max Drawdown, 20% VaR

                        # 3. Decision Output
                        st.markdown("### Decision Summary")
                        d_col1, d_col2, d_col3 = st.columns(3)
                        
                        # Best overall is the one with Rank 1 (highest Resilience Score)
                        best_p = comp_df.iloc[0]['Portfolio Name']
                        
                        # Most resilient is the one with the highest Resilience Score
                        resilient_p = comp_df.loc[comp_df['Resilience Score'].idxmax(), 'Portfolio Name']
                        
                        # Highest risk is the one with the lowest Resilience Score
                        risk_p = comp_df.loc[comp_df['Resilience Score'].idxmin(), 'Portfolio Name']

                        with d_col1:
                            st.success(f"🏆 **Best Overall**: {best_p}")
                        with d_col2:
                            st.info(f"🛡️ **Most Resilient**: {resilient_p}")
                        with d_col3:
                            st.warning(f"⚠️ **Highest Risk**: {risk_p}")

                        # 4. Updated Comparison Table
                        st.markdown("### Performance & Resilience Comparison")
                        format_dict = {col: "{:.2%}" for col in numeric_cols if col in comp_df.columns}
                        format_dict['Resilience Score'] = "{:.1f}"
                        if 'Total Value' in comp_df.columns:
                            format_dict['Total Value'] = "${:,.0f}"

                        st.dataframe(
                            comp_df.style.format(format_dict, na_rep="N/A").background_gradient(subset=['Resilience Score'], cmap='RdYlGn'),
                            use_container_width=True, 
                            key="Comparison Table*MULTI*0"
                        )
                        
                        # 5. Grouped Scenario Visualization
                        st.markdown("### Scenario Stress Test Comparison")
                        all_scen_data = []
                        for res in results_list:
                            name = Path(res["name"]).stem
                            for s_name, s_vals in res['scenario_pnl'].items():
                                all_scen_data.append({
                                    'Portfolio': name,
                                    'Scenario': s_name,
                                    'Return': s_vals['portfolio_return']
                                })
                        grouped_df = pd.DataFrame(all_scen_data)
                        fig_grouped = px.bar(
                            grouped_df, 
                            x='Scenario', 
                            y='Return', 
                            color='Portfolio', 
                            barmode='group',
                            template=plotly_template,
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        st.plotly_chart(fig_grouped, use_container_width=True, key="Grouped Scenarios*MULTI*1")
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
                            k1.metric("Worst Case", safe_format(worst.get('portfolio_return'), fmt="{:.2f}%"), help=worst_name)
                            k2.metric("VaR (95%)", safe_format(res['risk_metrics'].get('var_percent'), fmt="{:.2f}%"))
                            k3.metric("Max Drawdown", safe_format(res['risk_metrics'].get('max_historical_drawdown'), fmt="{:.2f}%"))
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
