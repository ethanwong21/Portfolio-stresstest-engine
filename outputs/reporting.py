import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, config):
        """
        Initialize report generator.
        
        Args:
            config: Output config specifying the results directory.
        """
        self.results_dir = config.results_dir
        self.excel_config = getattr(config, 'excel_export', None)
        self.csv_export = getattr(config, 'csv_export', False)
        self.png_export = getattr(config, 'png_export', False)
        
        os.makedirs(self.results_dir, exist_ok=True)
        # Use a non-interactive backend for server-side plot generation
        plt.switch_backend('Agg')
        sns.set_theme(style="whitegrid")

    def export_scenario_results(self, scenario_pnl: dict):
        if not self.csv_export: return
        summary_data = []
        for scenario_name, metrics in scenario_pnl.items():
            summary_data.append({
                'Scenario': scenario_name,
                'Portfolio Return (%)': metrics['portfolio_return'] * 100,
                'Dollar PnL': metrics['portfolio_dollar_pnl']
            })
        summary_df = pd.DataFrame(summary_data)
        out_path = os.path.join(self.results_dir, "scenario_summary.csv")
        summary_df.to_csv(out_path, index=False)
        logger.info(f"Exported scenario summary to {out_path}")
        
    def export_asset_contributions(self, scenario_pnl: dict):
        if not self.csv_export: return
        frames = []
        for scenario_name, metrics in scenario_pnl.items():
            df = metrics['asset_contributions'].reset_index()
            df.columns = ['Ticker', 'Dollar PnL Contribution']
            df['Scenario'] = scenario_name
            frames.append(df)
        final_df = pd.concat(frames)
        out_path = os.path.join(self.results_dir, "asset_contributions.csv")
        final_df.to_csv(out_path, index=False)
        logger.info(f"Exported asset contributions to {out_path}")

    def plot_scenario_impacts(self, scenario_pnl: dict):
        if not self.png_export: return
        scenarios = list(scenario_pnl.keys())
        returns = [scenario_pnl[s]['portfolio_return'] * 100 for s in scenarios]
        plt.figure(figsize=(10, 6))
        colors = ['red' if r < 0 else 'green' for r in returns]
        bars = plt.bar(scenarios, returns, color=colors)
        plt.title('Portfolio Impact by Stress Scenario')
        plt.ylabel('Expected Return (%)')
        plt.xlabel('Scenario')
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15), textcoords="offset points", ha='center', va='bottom')
        plt.tight_layout()
        out_path = os.path.join(self.results_dir, "scenario_impact_chart.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved scenario impact chart to {out_path}")

    def export_risk_metrics(self, risk_metrics: dict):
        if not self.csv_export: return
        df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in risk_metrics.items()])
        out_path = os.path.join(self.results_dir, "risk_metrics.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Exported base risk metrics to {out_path}")

    def export_to_excel(self, results_dict: dict):
        """
        Consolidates all metrics into a single multi-sheet professional Excel Dashboard.
        Uses separate helper methods for data prep, formatting, and charting.
        """
        if not self.excel_config or not getattr(self.excel_config, 'enable', False):
            return
            
        out_path = os.path.join(self.results_dir, self.excel_config.file_name)
        
        scenario_pnl = results_dict['scenario_pnl']
        portfolio = results_dict['portfolio']
        exposures = results_dict['exposures']
        shocks = results_dict['shocks']
        risk_metrics = results_dict['risk_metrics']
        
        # --- 1. DATA PREPARATION ---
        df_summary, df_dash_stats, worst_scenario_name = self._prepare_summary_data(scenario_pnl, shocks, risk_metrics, portfolio)
        df_risk = pd.DataFrame([{'Risk Metric': str(k).replace('_', ' ').title(), 'Value': v} for k, v in risk_metrics.items()])
        df_contrib = self._prepare_asset_contrib_data(scenario_pnl, portfolio)
        df_portfolio = self._prepare_portfolio_data(portfolio)
        df_exposures = self._prepare_exposure_data(exposures, portfolio)
        
        # --- 2. EXCEL WRITING & FORMATTING ---
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
            self._write_and_format_sheets(writer, df_summary, df_dash_stats, df_risk, df_contrib, df_portfolio, df_exposures)
            
            # --- 3. CHART CREATION ---
            self._create_charts(writer, df_summary, df_portfolio, worst_scenario_name)
            
        logger.info(f"Exported professional institutional Excel dashboard to {out_path}")
        
    def export_comparison_excel(self, all_results: dict, comp_df: pd.DataFrame, shocks: dict):
        """
        Generates a comparative analysis workbook evaluating multiple portfolios simultaneously.
        """
        if not self.excel_config or not getattr(self.excel_config, 'enable', False):
            return
            
        out_path = os.path.join(self.results_dir, "portfolio_comparison.xlsx")
        
        # Inject Ranking Metric
        comp_df = comp_df.copy()
        if 'Worst Scenario Return' in comp_df.columns:
            comp_df.insert(1, 'Risk Rank', range(1, len(comp_df) + 1))
        
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            pct_fmt = workbook.add_format({'num_format': '0.00%', 'align': 'center'})
            money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            header_fmt = workbook.add_format({'bold': True, 'bottom': 2, 'bg_color': '#2C3E50', 'font_color': 'white', 'align': 'center'})
            rank_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#F2F2F2', 'border': 1})
            
            red_cond = {'type': 'cell', 'criteria': '<', 'value': 0, 'format': workbook.add_format({'font_color': '#9C0006', 'bg_color': '#FFC7CE'})}
            green_cond = {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': workbook.add_format({'font_color': '#006100', 'bg_color': '#C6EFCE'})}
            
            # 1. Comparison Dashboard
            comp_df.to_excel(writer, sheet_name='Portfolio Comparison', index=False)
            ws_comp = writer.sheets['Portfolio Comparison']
            
            for col_num, value in enumerate(comp_df.columns.values):
                ws_comp.write(0, col_num, value, header_fmt)
                
            ws_comp.set_column(0, 0, 25)           # Portfolio Name
            ws_comp.set_column(1, 1, 12, rank_fmt) # Rank
            ws_comp.set_column(2, 5, 20, pct_fmt)  # Returns and risk metrics
            ws_comp.set_column(6, 6, 20, money_fmt)# Total Value
            
            # Conditional formatting for Returns
            max_r = len(comp_df)
            ws_comp.conditional_format(f'C2:E{max_r+1}', red_cond)
            ws_comp.conditional_format(f'C2:E{max_r+1}', green_cond)
            
            # Aggregate Bar Chart
            chart_comp = workbook.add_chart({'type': 'column'})
            col_idx_worst = comp_df.columns.get_loc('Worst Scenario Return')
            
            chart_comp.add_series({
                'name': 'Worst Scenario Return',
                'categories': ['Portfolio Comparison', 1, 0, max_r, 0],
                'values':     ['Portfolio Comparison', 1, col_idx_worst, max_r, col_idx_worst],
                'fill':       {'color': '#C0504D'},
                'data_labels': {'value': True, 'num_format': '0.00%'}
            })
            chart_comp.set_title({'name': 'Portfolio Risk Comparison (Max Drawdown Impacts)'})
            chart_comp.set_y_axis({'name': 'Return', 'num_format': '0.00%'})
            chart_comp.set_size({'width': 750, 'height': 350})
            
            # Ensure it clears any columns to layout cleanly
            ws_comp.insert_chart('I2', chart_comp)
            
            # 2. Append sub-sheets for specific portfolio summaries
            for name, data in all_results.items():
                df_summ, _, _ = self._prepare_summary_data(data['scenario_pnl'], shocks, data['risk_metrics'], pd.DataFrame())
                sh_name = f"{name[:25].capitalize()} Config"
                df_summ.to_excel(writer, sheet_name=sh_name, index=False)
                ws = writer.sheets[sh_name]
                for c_n, val in enumerate(df_summ.columns.values):
                    ws.write(0, c_n, val, header_fmt)
                ws.set_column(0, 1, 20)
                ws.set_column(2, 4, 15, pct_fmt)
                ws.conditional_format(f'C2:E{len(df_summ)+1}', red_cond)

        logger.info(f"Exported multi-portfolio comparison to {out_path}")

    def _prepare_summary_data(self, scenario_pnl, shocks, risk_metrics, portfolio):
        summary_data = []
        worst_scenario_name = "N/A"
        worst_scenario_return = float('inf')
        
        for scenario_name, metrics in scenario_pnl.items():
            port_ret = metrics['portfolio_return']
            if port_ret < worst_scenario_return:
                worst_scenario_return = port_ret
                worst_scenario_name = scenario_name
                
            shock_params = ", ".join([f"{k}: {v*100:.1f}%" for k, v in shocks.get(scenario_name, {}).items() if v != 0])
            drawdown = abs(port_ret) if port_ret < 0 else 0.0
            
            summary_data.append({
                'Scenario Name': scenario_name,
                'Shock Parameters': shock_params,
                'Portfolio Return': port_ret,
                'Portfolio PnL': metrics['portfolio_dollar_pnl'],
                'Drawdown Base': drawdown # Will format to % in excel
            })
            
        df_summary = pd.DataFrame(summary_data)
        # Sort scenarios from worst to best
        df_summary = df_summary.sort_values(by='Portfolio Return', ascending=True)
        
        # Add Rank based on worst return
        df_summary['Rank'] = df_summary['Portfolio Return'].rank(ascending=True).astype(int)
        
        # Add Risk Level
        def get_risk_level(ret):
            if ret <= -0.15: return 'Severe'
            elif ret <= -0.05: return 'Moderate'
            else: return 'Mild'
        df_summary['Risk Level'] = df_summary['Portfolio Return'].apply(get_risk_level)
        
        # Reorder columns matching prompt exactly
        df_summary = df_summary[['Scenario Name', 'Shock Parameters', 'Portfolio Return', 'Portfolio PnL', 'Drawdown Base', 'Rank', 'Risk Level']]
        
        # Dashboard top section
        port_total_val = portfolio.get('market_value', pd.Series([0])).sum()
        df_dash_stats = pd.DataFrame([
            {'KPI': 'Worst Scenario Impact', 'Value': f"{worst_scenario_name} ({worst_scenario_return*100:.2f}%)"},
            {'KPI': 'Max Historical Drawdown', 'Value': f"{risk_metrics.get('max_historical_drawdown', 0)*100:.2f}%"},
            {'KPI': 'Value-at-Risk (VaR)', 'Value': f"${risk_metrics.get('var_dollar', 0):,.2f}"},
            {'KPI': 'Total Portfolio Value', 'Value': f"${port_total_val:,.2f}"}
        ])
        return df_summary, df_dash_stats, worst_scenario_name
        
    def _prepare_asset_contrib_data(self, scenario_pnl, portfolio):
        frames = []
        for scenario_name, metrics in scenario_pnl.items():
            ac = metrics['asset_contributions']
            df = ac.reset_index()
            df.columns = ['Asset', 'Contribution to PnL']
            df['Scenario'] = scenario_name
            port_weights = portfolio[['weight']].reset_index()
            port_weights.columns = ['Asset', 'Weight']
            df = df.merge(port_weights, on='Asset', how='left')
            frames.append(df[['Asset', 'Weight', 'Contribution to PnL', 'Scenario']])
            
        df_contrib = pd.concat(frames).sort_values(by='Contribution to PnL', ascending=True)
        
        scenario_sums = df_contrib.groupby('Scenario')['Contribution to PnL'].transform('sum')
        df_contrib['% Contribution'] = df_contrib['Contribution to PnL'] / scenario_sums.replace(0, float('nan'))
        df_contrib['% Contribution'] = df_contrib['% Contribution'].fillna(0.0)
        
        df_contrib = df_contrib[['Asset', 'Weight', 'Contribution to PnL', '% Contribution', 'Scenario']]
        return df_contrib
        
    def _prepare_portfolio_data(self, portfolio):
        df_portfolio = portfolio.reset_index()
        for c in ['ticker', 'index']:
            if c in df_portfolio.columns:
                df_portfolio.rename(columns={c:'Asset'}, inplace=True)
                
        if 'asset_class' in df_portfolio.columns:
            df_portfolio.rename(columns={'asset_class':'Asset Class'}, inplace=True)
            
        keep_cols = ['Asset', 'weight']
        if 'Asset Class' in df_portfolio.columns: keep_cols.insert(1, 'Asset Class')
        if 'market_value' in df_portfolio.columns: keep_cols.append('market_value')
            
        df_portfolio = df_portfolio[keep_cols].rename(columns={'weight': 'Weight', 'market_value': 'Market Value'})
        return df_portfolio

    def _prepare_exposure_data(self, exposures, portfolio):
        port_exposures = exposures.mul(portfolio['weight'], axis=0).sum()
        df_exposures = port_exposures.reset_index()
        df_exposures.columns = ['Factor', 'Portfolio Exposure']
        return df_exposures

    def _write_and_format_sheets(self, writer, df_summary, df_dash_stats, df_risk, df_contrib, df_portfolio, df_exposures):
        workbook = writer.book
        
        # Define formats
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        val_fmt = workbook.add_format({'num_format': '#,##0.00'})
        
        header_fmt = workbook.add_format({'bold': True, 'bottom': 2, 'bg_color': '#2C3E50', 'font_color': 'white'})
        kpi_label_fmt = workbook.add_format({'bold': True, 'font_size': 11, 'bg_color': '#D9D9D9', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        kpi_val_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        red_cond_fmt = {'type': 'cell', 'criteria': '<', 'value': 0, 'format': workbook.add_format({'font_color': '#9C0006', 'bg_color': '#FFC7CE'})}
        green_cond_fmt = {'type': 'cell', 'criteria': '>', 'value': 0, 'format': workbook.add_format({'font_color': '#006100', 'bg_color': '#C6EFCE'})}

        def write_table(df, sheet_name, col_widths, col_formats):
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col_num, value in enumerate(df.columns.values):
                ws.write(0, col_num, value, header_fmt)
            for i, (width, fmt) in enumerate(zip(col_widths, col_formats)):
                if fmt: ws.set_column(i, i, width, fmt)
                else: ws.set_column(i, i, width)
            return ws

        # 1. Dashboard
        ws_dash = workbook.add_worksheet('Dashboard')
        # We manually write the KPIs
        ws_dash.set_column('B:E', 25)
        ws_dash.set_row(2, 30) # Make value row taller
        
        ws_dash.write('B2', 'Worst Scenario', kpi_label_fmt)
        ws_dash.write('C2', 'Max Drawdown', kpi_label_fmt)
        ws_dash.write('D2', 'VaR', kpi_label_fmt)
        ws_dash.write('E2', 'Portfolio Value', kpi_label_fmt)
        
        ws_dash.write('B3', df_dash_stats.iloc[0]['Value'], kpi_val_fmt)
        ws_dash.write('C3', df_dash_stats.iloc[1]['Value'], kpi_val_fmt)
        ws_dash.write('D3', df_dash_stats.iloc[2]['Value'], kpi_val_fmt)
        ws_dash.write('E3', df_dash_stats.iloc[3]['Value'], kpi_val_fmt)
        
        # 2. Scenario Summary
        ws_summary = write_table(df_summary, 'Scenario Summary', [22, 35, 18, 18, 15, 10, 15], [None, None, pct_fmt, money_fmt, pct_fmt, None, None])
        ws_summary.conditional_format('C2:E100', red_cond_fmt)
        ws_summary.conditional_format('C2:E100', green_cond_fmt)
        
        # 3. Risk Metrics
        ws_risk = write_table(df_risk, 'Risk Metrics', [25, 20], [None, val_fmt])
        
        # 4. Asset Contributions
        ws_contrib = write_table(df_contrib, 'Asset Contributions', [15, 15, 20, 15, 20], [None, pct_fmt, money_fmt, pct_fmt, None])
        ws_contrib.conditional_format('C2:D1000', red_cond_fmt)
        ws_contrib.conditional_format('C2:D1000', green_cond_fmt)

        # 5. Factor Exposure
        ws_expo = write_table(df_exposures, 'Factor Exposure', [20, 20], [None, val_fmt])
        ws_expo.conditional_format('B2:B100', red_cond_fmt)
        ws_expo.conditional_format('B2:B100', green_cond_fmt)

        # 6. Portfolio Breakdown
        num_cols = len(df_portfolio.columns)
        ws_port_widths = [15, 20, 15, 20][:num_cols]
        ws_port_fmts = [None, None, pct_fmt, money_fmt] if 'Asset Class' in df_portfolio.columns else [None, pct_fmt, money_fmt]
        ws_port_fmts = ws_port_fmts[:num_cols]
        ws_port = write_table(df_portfolio, 'Portfolio Breakdown', ws_port_widths, ws_port_fmts)

    def _create_charts(self, writer, df_summary, df_portfolio, worst_scenario_name):
        workbook = writer.book
        ws_dash = writer.sheets['Dashboard']
        ws_port = writer.sheets['Portfolio Breakdown']
        
        # Dashboard Chart
        chart_loss = workbook.add_chart({'type': 'column'})
        max_s_row = len(df_summary) + 1
        
        chart_loss.add_series({
            'name': 'Portfolio Return',
            'categories': ['Scenario Summary', 1, 0, max_s_row-1, 0],
            'values':     ['Scenario Summary', 1, 2, max_s_row-1, 2],
            'fill':       {'color': '#C0504D'},
            'data_labels': {'value': True, 'num_format': '0.00%'}
        })
        chart_loss.set_title({'name': 'Portfolio Performance Under Stress Scenarios'})
        chart_loss.set_y_axis({'name': 'Return', 'num_format': '0.00%'})
        chart_loss.set_legend({'none': True})
        chart_loss.set_size({'width': 700, 'height': 350})
        ws_dash.insert_chart('B6', chart_loss)
        
        # Insights text
        insight_f = workbook.add_format({'italic': True, 'font_size': 11, 'color': '#595959'})
        ws_dash.merge_range('B25:H26', f"Insight: The portfolio's worst stress impact comes from '{worst_scenario_name}'. " 
                                       "Continuous monitoring of associated factor exposures is recommended.", insight_f)

        # Portfolio Pie Chart
        chart_pie = workbook.add_chart({'type': 'pie'})
        max_p_row = len(df_portfolio) + 1
        weight_col_idx = df_portfolio.columns.get_loc('Weight')
        chart_pie.add_series({
            'name': 'Asset Allocation',
            'categories': ['Portfolio Breakdown', 1, 0, max_p_row-1, 0],
            'values':     ['Portfolio Breakdown', 1, weight_col_idx, max_p_row-1, weight_col_idx],
            'data_labels': {'percentage': True, 'leader_lines': True}
        })
        chart_pie.set_title({'name': 'Portfolio Allocation'})
        chart_pie.set_size({'width': 500, 'height': 350})
        ws_port.insert_chart('F2', chart_pie)
