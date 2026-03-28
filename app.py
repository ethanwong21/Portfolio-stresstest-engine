import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
from pathlib import Path
from datetime import datetime

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
        mode = "comparison" if len(uploaded_files) > 1 else "single"
        st.sidebar.info(f"Execution Mode: **{mode.upper()}**")
        
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
                
                comparer = PortfolioComparer(getattr(config, 'comparison', None))
                all_results = {}
                temp_files = []
                
                for uploaded_file in uploaded_files:
                    # Save to temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        temp_files.append(tmp_path)
                    
                    # Run Analysis
                    res = run_portfolio_analysis(
                        tmp_path, config, m_loader, f_rets, shocks, logger
                    )
                    
                    # Accumulate for Comparison
                    p_name = Path(uploaded_file.name).stem
                    m_val = res['portfolio'].get('market_value', pd.Series([0])).sum()
                    comparer.add_portfolio_result(p_name, res['scenario_pnl'], res['risk_metrics'], m_val)
                    
                    all_results[p_name] = res

                # 4. Backtest (Optional - only for the first portfolio in UI for now to avoid clutter)
                bt_data = None
                if run_backtest and mode == "single":
                    first_p_name = list(all_results.keys())[0]
                    res = all_results[first_p_name]
                    p_loader = PortfolioLoader(config.portfolio) 
                    p_df = p_loader.load_portfolio()
                    tickers = p_df['ticker'].tolist() if 'ticker' in p_df.columns else p_df.index.tolist()
                    a_rets = m_loader.fetch_asset_returns(tickers)
                    bt_df, bt_metrics = run_rolling_backtest(p_df, a_rets, f_rets, config.backtest, config.model_parameters)
                    bt_data = (bt_df, bt_metrics)

                # --- UI PRESENTATION ---
                
                if mode == "single":
                    # --- SINGLE PORTFOLIO DASHBOARD ---
                    p_name = list(all_results.keys())[0]
                    results = all_results[p_name]
                    
                    # 1. KPI Strip
                    st.divider()
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    
                    scenario_data = results['scenario_pnl']
                    worst_scen_name = min(scenario_data.keys(), key=lambda k: scenario_data[k]['portfolio_return'])
                    worst_scen = scenario_data[worst_scen_name]
                    
                    total_val = results['portfolio'].get('market_value', pd.Series([0])).sum()
                    var_val = results['risk_metrics'].get('var_percent', 0)
                    max_dd = results['risk_metrics'].get('max_historical_drawdown', 0)
                    
                    kpi1.metric("Maximum Drawdown", f"{max_dd*100:.2f}%")
                    kpi2.metric("Portfolio VaR (95%)", f"{var_val*100:.2f}%")
                    kpi3.metric("Worst Scenario Return", f"{worst_scen['portfolio_return']*100:.2f}%", help=worst_scen_name)
                    kpi4.metric("Total Portfolio Value", f"${total_val:,.0f}")
                    
                    # 2. Charts Row
                    c1, c2 = st.columns([2, 1])
                    
                    with c1:
                        st.markdown(f"### Stress Scenario Performance: {p_name}")
                        plot_df = pd.DataFrame([
                            {'Scenario': name, 'Return': data['portfolio_return']} 
                            for name, data in scenario_data.items()
                        ]).sort_values(by='Return')
                        
                        fig = px.bar(
                            plot_df, x='Scenario', y='Return', 
                            color='Return', color_continuous_scale='RdYlGn',
                            text_auto='.2%',
                            template=plotly_template
                        )
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with c2:
                        st.markdown("### Worst Scenario Attribution")
                        contrib_df = worst_scen['asset_contributions'].reset_index()
                        contrib_df.columns = ['Asset', 'PnL Contribution']
                        fig_pie = px.pie(
                            contrib_df, values=contrib_df['PnL Contribution'].abs(), 
                            names='Asset', hole=.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            template=plotly_template
                        )
                        fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    # 3. Backtest (if run)
                    if bt_data:
                        st.divider()
                        st.markdown("### Historical Model Validation (Backtest)")
                        bt_df, bt_metrics = bt_data
                        
                        if bt_df is not None and not bt_df.empty:
                            bt_df = bt_df.dropna()
                            required_cols = ["Predicted Return", "Actual Return"]
                            if all(col in bt_df.columns for col in required_cols):
                                b1, b2, b3 = st.columns(3)
                                b1.metric("MAE", f"{bt_metrics['MAE']*100:.2f}%")
                                b2.metric("RMSE", f"{bt_metrics['RMSE']*100:.2f}%")
                                b3.metric("Directional Hit", f"{bt_metrics['Directional Accuracy']*100:.1f}%")
                                
                                fig_bt = go.Figure()
                                fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Predicted Return'], name="Predicted", line=dict(color='royalblue', width=2)))
                                fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Actual Return'], name="Actual", line=dict(color='firebrick', width=2, dash='dot')))
                                fig_bt.update_layout(
                                    title="Predicted vs Actual Portfolio Returns", 
                                    xaxis_title="Date", yaxis_title="Return",
                                    template=plotly_template,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                try:
                                    st.plotly_chart(fig_bt, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Chart rendering failed: {e}")
                
                else:
                    # --- MULTI-PORTFOLIO COMPARISON VIEW ---
                    st.divider()
                    st.markdown("### Multi-Portfolio Risk Comparison")
                    
                    comp_df = comparer.compare_portfolios()
                    
                    if comp_df.empty:
                        st.warning("No portfolios analyzed for comparison.")
                    else:
                        # 1. Comparison Metrics Table
                        # Dynamically format all numeric columns
                        num_cols = comp_df.select_dtypes(include=['number']).columns
                        pct_cols = [c for c in num_cols if "Value" not in c]
                        val_cols = [c for c in num_cols if "Value" in c]
                        
                        styled_df = comp_df.style.format({
                            **{c: "{:.2%}" for c in pct_cols},
                            **{c: "${:,.0f}" for c in val_cols}
                        })
                        
                        # Add gradient to worst-case column if it exists
                        wc_col = 'Worst Scenario Return'
                        if wc_col in comp_df.columns:
                            styled_df = styled_df.background_gradient(subset=[wc_col], cmap='RdYlGn')
                            
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # 2. Risk Rank Chart
                        st.markdown("#### Portfolio Risk Ranking (Worst Case Scenario)")
                        fig_comp = px.bar(
                            comp_df, 
                            x='Portfolio Name', y='Worst Scenario Return',
                            color='Worst Scenario Return',
                            color_continuous_scale='RdYlGn',
                            template=plotly_template,
                            text_auto='.2%'
                        )
                        fig_comp.update_layout(yaxis_title="Expected Loss (%)", showlegend=False)
                        st.plotly_chart(fig_comp, use_container_width=True)

                    # 3. Individual Portfolio Deep-Dives (requested enhancement)
                    st.divider()
                    st.markdown("### Individual Portfolio Analysis Details")
                    for p_name, results in all_results.items():
                        with st.expander(f"🔍 Deep-Dive: {p_name}", expanded=False):
                            i_kpi1, i_kpi2, i_kpi3 = st.columns(3)
                            
                            i_scenario_data = results['scenario_pnl']
                            i_worst_scen_name = min(i_scenario_data.keys(), key=lambda k: i_scenario_data[k]['portfolio_return'])
                            i_worst_scen = i_scenario_data[i_worst_scen_name]
                            
                            i_kpi1.metric("Worst Case", f"{i_worst_scen['portfolio_return']*100:.2f}%", help=i_worst_scen_name)
                            i_kpi2.metric("VaR (95%)", f"{results['risk_metrics'].get('var_percent', 0)*100:.2f}%")
                            i_kpi3.metric("Max Drawdown", f"{results['risk_metrics'].get('max_historical_drawdown', 0)*100:.2f}%")
                            
                            # Simple Viz for the expander
                            i_plot_df = pd.DataFrame([
                                {'Scenario': n, 'Return': d['portfolio_return']} 
                                for n, d in i_scenario_data.items()
                            ]).sort_values(by='Return')
                            
                            fig_i = px.bar(
                                i_plot_df, x='Scenario', y='Return', 
                                color='Return', color_continuous_scale='RdYlGn',
                                template=plotly_template
                            )
                            fig_i.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                            st.plotly_chart(fig_i, use_container_width=True)

                # 5. Download Unified Report
                st.divider()
                st.markdown("### Institutional Reports")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"risk_report_{timestamp}.xlsx"
                report_path = os.path.join(config.outputs.results_dir, report_name)
                
                config.outputs.excel_export.file_name = report_name
                config.outputs.excel_export.timestamp = False
                
                report_gen = ReportGenerator(config.outputs)
                
                # Use the first portfolio's results as the primary master_res
                first_p_name = list(all_results.keys())[0]
                master_res = all_results[first_p_name]
                master_res['shocks'] = shocks
                
                report_gen.export_unified_report(
                    master_res, 
                    comp_df=comparer.compare_portfolios() if mode == "comparison" else None,
                    bt_df=bt_data[0] if bt_data else None, 
                    bt_metrics=bt_data[1] if bt_data else None
                )
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="Download High-Fidelity Excel Dashboard",
                        data=f,
                        file_name=report_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
        # Clean up temp files
        for tmp_path in locals().get('temp_files', []):
            try:
                os.unlink(tmp_path)
            except:
                pass

    else:
        st.info("Please upload a portfolio CSV file to begin analysis.")
        
        # Sample Visual Preview
        st.divider()
        st.markdown("### Dashboard Preview")
        st.image("https://images.unsplash.com/photo-1611974717482-58284396e8c7?q=80&w=2070&auto=format&fit=crop", caption="Professional Risk Intelligence Visualization")

if __name__ == "__main__":
    main()
