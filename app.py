import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

def safe_format(value, fmt="{:.2f}%"):
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
    page_title="Portfolio Stress-Test Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Historical crisis scenario names (used to split chart tabs) ──────────────
HISTORICAL_SCENARIO_KEYS = {
    "2008 GFC (Sep-Nov)",
    "2020 COVID Crash (Feb-Mar)",
    "2022 Rate Hike Cycle",
    "2001 Dot-Com Bust (Mar-Sep)",
}

@st.cache_data(ttl=3600, show_spinner=False)
def _get_cached_shocks(_s_gen):
    return _s_gen.get_shocks(include_historical=True)


def _metric_card(col, label, value, sublabel):
    col.metric(label, value)
    col.caption(sublabel)


def _beta_heatmap(exposures_df, plotly_template):
    """Render a colour-coded heatmap of OLS factor betas per ticker."""
    fig = go.Figure(data=go.Heatmap(
        z=exposures_df.values,
        x=exposures_df.columns.tolist(),
        y=exposures_df.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(exposures_df.values, 2),
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Beta"),
    ))
    fig.update_layout(
        title="OLS Factor Betas — how much each asset moves per unit factor shock",
        xaxis_title="Factor",
        yaxis_title="Ticker",
        template=plotly_template,
        height=max(300, 60 + 40 * len(exposures_df)),
        margin=dict(l=80, r=40, t=60, b=40),
    )
    return fig


def _landing_page():
    """Hero landing page shown before any analysis is run."""
    st.markdown("## Portfolio Stress-Test Engine")
    st.markdown(
        "A quantitative factor-model engine that measures how your portfolio holds up "
        "under real and hypothetical macro shocks."
    )

    c1, c2, c3 = st.columns(3)
    c1.info("**OLS Factor Betas**\nRegression-estimated sensitivities to market, rates, inflation & commodities — not lookup tables.")
    c2.info("**Calibrated Crisis Scenarios**\nShock vectors derived from real 2008 GFC, 2020 COVID crash, 2022 rate hike cycle & 2001 dot-com bust data.")
    c3.info("**Custom Scenario Builder**\nDrag sliders to define any macro environment and see per-asset impact in real time.")

    st.divider()
    st.markdown("#### Try it now")
    col_a, col_b, _ = st.columns([1, 1, 3])
    if col_a.button("Run Single Portfolio Demo", use_container_width=True, type="primary"):
        st.session_state["demo_mode_enabled"] = True
        st.session_state["demo_mode_type"] = "SINGLE"
        st.session_state["demo_analysis_type"] = "Dynamic Scenario Analysis"
        st.rerun()
    if col_b.button("Run Multi-Portfolio Demo", use_container_width=True):
        st.session_state["demo_mode_enabled"] = True
        st.session_state["demo_mode_type"] = "MULTI"
        st.rerun()


def main():
    # ── Session state defaults ────────────────────────────────────────────────
    if "demo_mode_enabled" not in st.session_state:
        st.session_state["demo_mode_enabled"] = False
    if "demo_mode_type" not in st.session_state:
        st.session_state["demo_mode_type"] = None

    # ── Demo portfolios ───────────────────────────────────────────────────────
    demo_growth_portfolio = pd.DataFrame([
        {'ticker': 'AAPL',  'weight': 0.20, 'asset_class': 'Tech'},
        {'ticker': 'MSFT',  'weight': 0.20, 'asset_class': 'Tech'},
        {'ticker': 'NVDA',  'weight': 0.20, 'asset_class': 'Growth'},
        {'ticker': 'AMZN',  'weight': 0.15, 'asset_class': 'Growth'},
        {'ticker': 'GOOGL', 'weight': 0.15, 'asset_class': 'Growth'},
        {'ticker': 'TSLA',  'weight': 0.10, 'asset_class': 'Growth'},
    ])
    demo_defensive_portfolio = pd.DataFrame([
        {'ticker': 'JNJ', 'weight': 0.15, 'asset_class': 'Health Care'},
        {'ticker': 'PG',  'weight': 0.15, 'asset_class': 'Consumer Staples'},
        {'ticker': 'KO',  'weight': 0.15, 'asset_class': 'Consumer Staples'},
        {'ticker': 'PEP', 'weight': 0.15, 'asset_class': 'Consumer Staples'},
        {'ticker': 'XOM', 'weight': 0.10, 'asset_class': 'Energy'},
        {'ticker': 'TLT', 'weight': 0.20, 'asset_class': 'Bond'},
        {'ticker': 'GLD', 'weight': 0.10, 'asset_class': 'Commodity'},
    ])

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
        st.divider()

        mode_label = "Demo Mode" if st.session_state["demo_mode_enabled"] else "Upload Mode"
        demo_toggle = st.toggle(f"{mode_label}", value=st.session_state["demo_mode_enabled"])
        if demo_toggle != st.session_state["demo_mode_enabled"]:
            st.session_state["demo_mode_enabled"] = demo_toggle
            st.session_state["demo_mode_type"] = "SINGLE" if demo_toggle else None
            st.rerun()

        st.divider()
        uploaded_files = None
        run_analysis = False
        run_scenario_flag = True
        run_backtest_flag = False

        if st.session_state["demo_mode_enabled"]:
            st.markdown("**Demo Portfolio**")
            d_choice = st.selectbox(
                "Analysis type",
                ["Dynamic Scenario Analysis", "Rolling Backtest"],
            )
            current_choice = st.session_state.get("demo_analysis_type")
            if current_choice != d_choice or st.session_state["demo_mode_type"] != "SINGLE":
                st.session_state["demo_mode_type"] = "SINGLE"
                st.session_state["demo_analysis_type"] = d_choice
                run_analysis = True

            if st.button("Compare Growth vs Defensive", use_container_width=True):
                st.session_state["demo_mode_type"] = "MULTI"
                run_analysis = True
        else:
            st.markdown("**Upload Portfolio CSV(s)**")
            st.caption("Columns: `ticker`, `weight`, `asset_class` (optional)")
            uploaded_files = st.file_uploader(
                "Drop files here", type=["csv"], accept_multiple_files=True, label_visibility="collapsed"
            )
            if uploaded_files:
                st.markdown("**Analysis options**")
                run_scenario_flag = st.checkbox("Scenario Analysis", value=True)
                run_backtest_flag = st.checkbox("Rolling Backtest")
                if st.button("Run Stress Test", use_container_width=True, type="primary"):
                    run_analysis = True

    # ── Theme CSS ─────────────────────────────────────────────────────────────
    if theme == "Dark":
        bg, card, text, label, plotly_t = "#0e1117", "#1a1a2e", "#ffffff", "#aaaaaa", "plotly_dark"
    else:
        bg, card, text, label, plotly_t = "#f8f9fa", "#ffffff", "#111111", "#555555", "plotly_white"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        [data-testid="stMetric"] {{
            background-color: {card};
            padding: 18px 20px 12px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.07);
        }}
        [data-testid="stMetricLabel"] p {{ color: {label} !important; font-size: 0.8rem; }}
        [data-testid="stMetricValue"] div {{ color: {text} !important; font-size: 1.4rem; font-weight: 700; }}
        .section-header {{ font-size: 1.1rem; font-weight: 600; color: {label}; text-transform: uppercase;
                           letter-spacing: 0.08em; margin: 1.5rem 0 0.5rem; }}
    </style>
    """, unsafe_allow_html=True)

    # ── Resolve portfolios ────────────────────────────────────────────────────
    portfolios = None
    if st.session_state.get("demo_mode_enabled"):
        if st.session_state.get("demo_mode_type") == "SINGLE":
            portfolios = [{"name": "Growth_Strategy", "df": demo_growth_portfolio}]
        elif st.session_state.get("demo_mode_type") == "MULTI":
            portfolios = [
                {"name": "Growth_Strategy",   "df": demo_growth_portfolio},
                {"name": "Defensive_Yield",   "df": demo_defensive_portfolio},
            ]
    elif uploaded_files:
        portfolios = [{"name": Path(f.name).stem, "bytes": f.getvalue()} for f in uploaded_files]

    exec_mode = None if portfolios is None else ("SINGLE" if len(portfolios) == 1 else "MULTI")
    if exec_mode == "MULTI":
        run_backtest_flag = False

    # ── Landing page ──────────────────────────────────────────────────────────
    if not run_analysis and exec_mode is None and not st.session_state.get("demo_mode_enabled"):
        _landing_page()
        return

    if not run_analysis and not st.session_state.get("demo_mode_enabled"):
        st.info("Upload a portfolio CSV and click **Run Stress Test** to begin.")
        return

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("# Portfolio Stress-Test Engine")
    st.caption("Factor-model driven scenario analysis · OLS-estimated betas · Real crisis calibration")
    st.divider()

    # ── Config ────────────────────────────────────────────────────────────────
    from utils.config import (AppConfig, PortfolioConfig, MarketDataConfig,
                               ModelParametersConfig, ScenarioConfig,
                               DynamicScenariosConfig, ComparisonConfig,
                               BacktestConfig, OutputsConfig, ExcelExportConfig)

    config = AppConfig(
        portfolio=PortfolioConfig(file_path="", columns={}),
        market_data=MarketDataConfig(
            source="yfinance",
            start_date="2020-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            factors={"equity": "^GSPC", "rates": "^TNX", "inflation": "TIP", "commodities": "GSG"},
        ),
        model_parameters=ModelParametersConfig(rolling_window_days=252, var_confidence_level=0.95),
        scenarios=[],
        dynamic_scenarios=DynamicScenariosConfig(
            enable=True, sigma_levels=[1, 2, 3],
            factors=["equity", "rates", "inflation", "commodities"]
        ),
        comparison=ComparisonConfig(enable=True),
        backtest=BacktestConfig(enabled=True, start_date="2020-01-01",
                                end_date=datetime.now().strftime("%Y-%m-%d")),
        outputs=OutputsConfig(
            results_dir="outputs",
            excel_export=ExcelExportConfig(enable=True, file_name="risk_report.xlsx")
        ),
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Fetching market data and running analysis…"):
        import logging
        logger = logging.getLogger("streamlit_app")

        m_loader = MarketDataLoader(config.market_data)
        f_rets   = m_loader.fetch_data()
        s_gen    = ScenarioGenerator(config.scenarios)
        shocks   = _get_cached_shocks(s_gen)
        d_gen    = DynamicScenarioGenerator(config.dynamic_scenarios, f_rets)
        shocks.update(d_gen.generate_dynamic_scenarios())

        results_list, temp_files = [], []
        for target in portfolios:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                if "df" in target:
                    target["df"].to_csv(tmp.name, index=False)
                else:
                    tmp.write(target["bytes"])
                temp_files.append(tmp.name)
            try:
                res = run_portfolio_analysis(tmp.name, config, m_loader, f_rets, shocks, logger)
                res["name"] = target["name"]
                results_list.append(res)
            except Exception as e:
                st.error(f"Analysis failed for **{target['name']}**: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE PORTFOLIO VIEW
    # ══════════════════════════════════════════════════════════════════════════
    if exec_mode == "SINGLE" and results_list:
        result      = results_list[0]
        p_name      = result["name"]
        single_type = st.session_state.get("demo_analysis_type", "Dynamic Scenario Analysis")
        is_demo     = st.session_state.get("demo_mode_enabled", False)
        show_scen   = (is_demo and single_type == "Dynamic Scenario Analysis") or (not is_demo and run_scenario_flag)
        show_bt     = (is_demo and single_type == "Rolling Backtest")          or (not is_demo and run_backtest_flag)

        # ── Scenario Analysis ─────────────────────────────────────────────────
        if show_scen:
            scen_data = result["scenario_pnl"]
            worst_n   = min(scen_data, key=lambda k: scen_data[k]["portfolio_return"])
            worst     = scen_data[worst_n]
            m_val     = result["portfolio"].get("market_value", pd.Series([0])).sum()

            # KPI row
            st.markdown(f'<p class="section-header">Risk Summary — {p_name}</p>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            _metric_card(k1, "Max Drawdown",
                         safe_format(result["risk_metrics"].get("max_historical_drawdown")),
                         "Peak-to-trough decline in historical returns")
            _metric_card(k2, "VaR (95%, 1-day)",
                         safe_format(result["risk_metrics"].get("var_percent")),
                         "Daily loss not exceeded 95% of the time")
            _metric_card(k3, "Worst Scenario Return",
                         safe_format(worst.get("portfolio_return")),
                         f"Under: {worst_n}")
            _metric_card(k4, "Portfolio Value",
                         f"${m_val:,.0f}",
                         "Sum of market values in portfolio")

            st.divider()

            # ── Scenario charts split into tabs ───────────────────────────────
            st.markdown('<p class="section-header">Scenario Stress Test Results</p>', unsafe_allow_html=True)

            hist_scens    = {n: d for n, d in scen_data.items() if n in HISTORICAL_SCENARIO_KEYS}
            synth_scens   = {n: d for n, d in scen_data.items() if n not in HISTORICAL_SCENARIO_KEYS}

            tab_hist, tab_synth, tab_pie = st.tabs([
                f"Historical Crises ({len(hist_scens)})",
                f"Stress Scenarios ({len(synth_scens)})",
                "Worst-Case Breakdown",
            ])

            def _scenario_bar(data_dict, title):
                df = pd.DataFrame([
                    {"Scenario": n, "Return": d["portfolio_return"]}
                    for n, d in data_dict.items()
                ]).sort_values("Return")
                fig = px.bar(
                    df, x="Return", y="Scenario", orientation="h",
                    color="Return", color_continuous_scale="RdYlGn",
                    title=title, template=plotly_t,
                    text=df["Return"].map(lambda v: f"{v:.1%}"),
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    xaxis_tickformat=".0%",
                    height=max(300, 60 + 35 * len(df)),
                    margin=dict(l=20, r=60, t=50, b=20),
                )
                return fig

            with tab_hist:
                if hist_scens:
                    st.caption("Shock vectors calibrated from actual factor returns during each named crisis period.")
                    st.plotly_chart(_scenario_bar(hist_scens, "Portfolio Return Under Historical Crises"),
                                    use_container_width=True)
                else:
                    st.info("Historical scenarios failed to load — check network access.")

            with tab_synth:
                st.caption("Forward-looking hypothetical scenarios plus data-driven sigma shocks.")
                st.plotly_chart(_scenario_bar(synth_scens, "Portfolio Return Under Stress Scenarios"),
                                use_container_width=True)

            with tab_pie:
                st.caption(f"Asset-level contribution to loss under the worst scenario: **{worst_n}**")
                contrib_df = worst["asset_contributions"].reset_index()
                contrib_df.columns = ["Ticker", "PnL"]
                fig_pie = px.pie(
                    contrib_df, values=contrib_df["PnL"].abs(),
                    names="Ticker", hole=0.45, template=plotly_t,
                    title=f"Loss Attribution — {worst_n}",
                )
                fig_pie.update_traces(textinfo="label+percent")
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            # ── Factor Beta Heatmap ───────────────────────────────────────────
            st.markdown('<p class="section-header">OLS Factor Exposures (Betas)</p>', unsafe_allow_html=True)
            st.caption(
                "Regression-estimated sensitivities of each holding to the four macro factors. "
                "Green = positive exposure, Red = negative. Computed from 2+ years of daily returns."
            )
            exposures_df = result["exposures"]
            st.plotly_chart(_beta_heatmap(exposures_df, plotly_t), use_container_width=True)

            st.divider()

            # ── Custom Scenario Builder ───────────────────────────────────────
            with st.expander("🔧 Build Your Own Scenario", expanded=False):
                st.caption(
                    "Define any combination of macro shocks below. "
                    "The impact is computed instantly from the OLS betas above — no re-run needed."
                )
                cs_col1, cs_col2 = st.columns(2)
                with cs_col1:
                    cs_market = st.slider("Market shock (%)", -50, 30, 0, 1,
                                          help="Equity index return (e.g. −25 = severe crash)") / 100.0
                    cs_rates  = st.slider("Rate shock (% Δ in yield level)", -30, 50, 0, 1,
                                          help="% change in 10yr yield (e.g. +20 = yield rises 20%)") / 100.0
                with cs_col2:
                    cs_infl   = st.slider("Inflation shock (%)", -10, 20, 0, 1,
                                          help="TIPS ETF return as inflation expectations proxy") / 100.0
                    cs_commod = st.slider("Commodities shock (%)", -40, 40, 0, 1,
                                          help="Broad commodity index return") / 100.0

                from models.scenario_impact import ScenarioImpactModel
                cs_portfolio = (result["portfolio"].set_index("ticker")
                                if "ticker" in result["portfolio"].columns
                                else result["portfolio"])
                cs_weights  = cs_portfolio["weight"]
                cs_model    = ScenarioImpactModel(result["exposures"])
                cs_returns  = cs_model.propagate_shocks({
                    "market": cs_market, "rates": cs_rates,
                    "inflation": cs_infl, "commodities": cs_commod,
                })
                cs_port_ret = float((cs_weights.reindex(cs_returns.index).fillna(0) * cs_returns).sum())

                r1, r2 = st.columns(2)
                delta_color = "normal" if cs_port_ret >= 0 else "inverse"
                r1.metric("Estimated Portfolio Return", f"{cs_port_ret:.2%}",
                          delta=f"{cs_port_ret:.2%}", delta_color=delta_color)

                cs_df = cs_returns.reset_index()
                cs_df.columns = ["Ticker", "Return"]
                cs_df = cs_df.sort_values("Return")
                fig_cs = px.bar(
                    cs_df, x="Ticker", y="Return",
                    color="Return", color_continuous_scale="RdYlGn",
                    text=cs_df["Return"].map(lambda v: f"{v:.1%}"),
                    title="Per-Asset Impact", template=plotly_t,
                )
                fig_cs.update_traces(textposition="outside")
                fig_cs.update_layout(coloraxis_showscale=False, showlegend=False,
                                     yaxis_tickformat=".0%")
                r2.plotly_chart(fig_cs, use_container_width=True)

        # ── Rolling Backtest ──────────────────────────────────────────────────
        if show_bt:
            st.markdown('<p class="section-header">Historical Rolling Backtest</p>', unsafe_allow_html=True)
            st.caption(
                "The model is re-fitted on a rolling 252-day window with no look-ahead. "
                "Predicted vs actual portfolio returns validate how well the factor model tracks reality."
            )
            p_df    = result["portfolio"]
            tickers = p_df["ticker"].tolist() if "ticker" in p_df.columns else p_df.index.tolist()
            a_rets  = m_loader.fetch_asset_returns(tickers)
            bt_df, metrics = run_rolling_backtest(p_df, a_rets, f_rets, config.backtest, config.model_parameters)

            if bt_df is not None and not bt_df.empty:
                b1, b2, b3 = st.columns(3)
                _metric_card(b1, "MAE",  f"{metrics['MAE']*100:.2f}%",
                             "Mean absolute prediction error")
                _metric_card(b2, "RMSE", f"{metrics['RMSE']*100:.2f}%",
                             "Root mean squared error — penalises large misses")
                _metric_card(b3, "Directional Accuracy", f"{metrics['Directional Accuracy']*100:.1f}%",
                             "% of periods where model predicted the correct sign")

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df["Predicted Return"],
                    name="Model Predicted", line=dict(color="#4fc3f7"),
                ))
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df["Actual Return"],
                    name="Actual Realised", line=dict(dash="dot", color="#ef9a9a"),
                ))
                fig_bt.update_layout(
                    title="Model Prediction vs Realised Portfolio Return",
                    yaxis_tickformat=".1%", template=plotly_t,
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.warning("Insufficient data for backtest over the configured date range.")

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-PORTFOLIO VIEW
    # ══════════════════════════════════════════════════════════════════════════
    elif exec_mode == "MULTI" and results_list:
        st.markdown('<p class="section-header">Multi-Portfolio Resilience Ranking</p>', unsafe_allow_html=True)

        comparer = PortfolioComparer(config.comparison)
        for res in results_list:
            m_val = res["portfolio"].get("market_value", pd.Series([0])).sum()
            comparer.add_portfolio_result(res["name"], res["scenario_pnl"], res["risk_metrics"], m_val)
        comp_df = comparer.compare_portfolios()

        if comp_df.empty or "Resilience Score" not in comp_df.columns:
            st.error("Resilience Score not computed — check simulation input.")
            st.dataframe(comp_df)
        else:
            comp_df["Resilience Score"] = pd.to_numeric(comp_df["Resilience Score"], errors="coerce")

            best_p      = comp_df.iloc[0]["Portfolio Name"]
            resilient_p = comp_df.loc[comp_df["Resilience Score"].idxmax(), "Portfolio Name"]
            risk_p      = comp_df.loc[comp_df["Resilience Score"].idxmin(), "Portfolio Name"]

            d1, d2, d3 = st.columns(3)
            d1.success(f"🏆 **Best Overall**: {best_p}")
            d2.info(f"🛡️ **Most Resilient**: {resilient_p}")
            d3.warning(f"⚠️ **Highest Risk**: {risk_p}")

            st.divider()

            # Summary table
            st.markdown('<p class="section-header">Performance & Risk Summary</p>', unsafe_allow_html=True)
            num_cols = ["Worst Scenario Return", "Best Scenario Return", "Max Drawdown", "VaR"]
            for c in num_cols:
                comp_df[c] = pd.to_numeric(comp_df[c], errors="coerce")
            fmt = {c: "{:.2%}" for c in num_cols if c in comp_df.columns}
            fmt["Resilience Score"] = "{:.1f}"
            fmt["Total Value"]      = "${:,.0f}"
            st.dataframe(
                comp_df.style.format(fmt, na_rep="N/A")
                             .background_gradient(subset=["Resilience Score"], cmap="RdYlGn"),
                use_container_width=True,
            )

            st.divider()

            # Scenario comparison split into tabs
            st.markdown('<p class="section-header">Scenario Comparison</p>', unsafe_allow_html=True)
            all_scen = []
            for res in results_list:
                for n, d in res["scenario_pnl"].items():
                    all_scen.append({
                        "Portfolio": res["name"], "Scenario": n,
                        "Return": d["portfolio_return"],
                        "Type": "Historical" if n in HISTORICAL_SCENARIO_KEYS else "Stress",
                    })
            all_df = pd.DataFrame(all_scen)

            tab_h, tab_s = st.tabs(["Historical Crises", "Stress Scenarios"])
            for tab, label in [(tab_h, "Historical"), (tab_s, "Stress")]:
                with tab:
                    sub = all_df[all_df["Type"] == label]
                    if not sub.empty:
                        fig = px.bar(
                            sub, x="Scenario", y="Return", color="Portfolio",
                            barmode="group", template=plotly_t,
                            text=sub["Return"].map(lambda v: f"{v:.1%}"),
                        )
                        fig.update_traces(textposition="outside")
                        fig.update_layout(yaxis_tickformat=".0%", height=420)
                        st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Beta heatmaps per portfolio
            st.markdown('<p class="section-header">OLS Factor Betas by Portfolio</p>', unsafe_allow_html=True)
            st.caption("Each portfolio's regression-estimated factor sensitivities.")
            cols = st.columns(min(len(results_list), 2))
            for i, res in enumerate(results_list):
                with cols[i % 2]:
                    st.markdown(f"**{res['name']}**")
                    st.plotly_chart(_beta_heatmap(res["exposures"], plotly_t), use_container_width=True)

    # Cleanup temp files
    for f in temp_files:
        if os.path.exists(f):
            os.unlink(f)


if __name__ == "__main__":
    main()
