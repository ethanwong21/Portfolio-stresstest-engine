import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, config):
        """
        Initialize unified report generator.
        
        Args:
            config: Output config specifying the results directory.
        """
        self.results_dir = config.results_dir
        self.excel_config = getattr(config, 'excel_export', None)
        os.makedirs(self.results_dir, exist_ok=True)

    def export_unified_report(self, results_dict: dict, comp_df: pd.DataFrame = None, bt_df: pd.DataFrame = None, bt_metrics: dict = None):
        """
        Consolidates ALL multi-variate modeling outputs into a single Microsoft Excel dashboard.
        Conditionally triggers Comparison and Backtest sheets natively if arrays are provided.
        """
        if not self.excel_config or not getattr(self.excel_config, 'enable', False):
            return
            
        out_path = os.path.join(self.results_dir, self.excel_config.file_name)
        
        scenario_pnl = results_dict['scenario_pnl']
        portfolio = results_dict['portfolio']
        exposures = results_dict['exposures']
        shocks = results_dict['shocks']
        risk_metrics = results_dict['risk_metrics']
        
        # Data preparations
        df_summary, df_dash_stats, worst_scenario_name = self._prepare_summary_data(scenario_pnl, shocks, risk_metrics, portfolio)
        df_risk = pd.DataFrame([{'Risk Metric': str(k).replace('_', ' ').title(), 'Value': v} for k, v in risk_metrics.items()])
        df_contrib = self._prepare_asset_contrib_data(scenario_pnl, portfolio)
        df_exposures = self._prepare_exposure_data(exposures, portfolio)
        
        with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # --- GLOBAL FORMATS ---
            fmt_header = workbook.add_format({'bold': True, 'bottom': 2, 'bg_color': '#2C3E50', 'font_color': 'white', 'align': 'center'})
            fmt_pct = workbook.add_format({'num_format': '0.00%', 'align': 'center'})
            fmt_money = workbook.add_format({'num_format': '$#,##0.00'})
            fmt_kpi_label = workbook.add_format({'bold': True, 'font_size': 11, 'bg_color': '#D9D9D9', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
            fmt_kpi_val = workbook.add_format({'bold': True, 'font_size': 14, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
            
            cond_red = {'type': 'cell', 'criteria': '<', 'value': 0, 'format': workbook.add_format({'font_color': '#9C0006', 'bg_color': '#FFC7CE'})}
            cond_green = {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': workbook.add_format({'font_color': '#006100', 'bg_color': '#C6EFCE'})}
            
            # --- SHEET 1: DASHBOARD ---
            ws_dash = workbook.add_worksheet('Dashboard')
            ws_dash.set_column('B:E', 25)
            ws_dash.set_row(2, 30)
            
            # Write KPIs
            ws_dash.write('B2', 'Worst Scenario', fmt_kpi_label)
            ws_dash.write('C2', 'Max Drawdown', fmt_kpi_label)
            ws_dash.write('D2', 'VaR', fmt_kpi_label)
            ws_dash.write('E2', 'Portfolio Value', fmt_kpi_label)
            
            ws_dash.write('B3', df_dash_stats.iloc[0]['Value'], fmt_kpi_val)
            ws_dash.write('C3', df_dash_stats.iloc[1]['Value'], fmt_kpi_val)
            ws_dash.write('D3', df_dash_stats.iloc[2]['Value'], fmt_kpi_val)
            ws_dash.write('E3', df_dash_stats.iloc[3]['Value'], fmt_kpi_val)
            
            # Temporarily write df_summary to a hidden space or just use list arrays to chart?
            # It's cleaner to chart directly from the Scenario Summary sheet!
            
            chart_dash = workbook.add_chart({'type': 'column'})
            s_rows = len(df_summary)
            # Chart reading off Sheet 2
            chart_dash.add_series({
                'name': 'Portfolio Return',
                'categories': ['Scenario Summary', 1, 0, s_rows, 0],
                'values':     ['Scenario Summary', 1, 2, s_rows, 2],
                'fill':       {'color': '#C0504D'},
                'data_labels': {'value': True, 'num_format': '0.00%'}
            })
            chart_dash.set_title({'name': 'Portfolio Performance Under Stress Scenarios'})
            chart_dash.set_y_axis({'name': 'Return', 'num_format': '0.00%'})
            chart_dash.set_legend({'none': True})
            chart_dash.set_size({'width': 700, 'height': 350})
            
            ws_dash.insert_chart('B6', chart_dash)
            
            fmt_insight = workbook.add_format({'italic': True, 'font_size': 11, 'color': '#595959'})
            ws_dash.merge_range('B25:H26', f"Insight: The portfolio's worst stress impact comes from '{worst_scenario_name}'. " 
                                           "Continuous monitoring of associated factor exposures is explicitly recommended.", fmt_insight)

            # --- SHEET 2: SCENARIO SUMMARY ---
            df_summary.to_excel(writer, sheet_name='Scenario Summary', index=False)
            ws_summ = writer.sheets['Scenario Summary']
            for c_n, val in enumerate(df_summary.columns.values):
                ws_summ.write(0, c_n, val, fmt_header)
            ws_summ.set_column(0, 0, 22)
            ws_summ.set_column(1, 1, 35)
            ws_summ.set_column(2, 2, 18, fmt_pct)
            ws_summ.set_column(3, 3, 18, fmt_money)
            ws_summ.set_column(4, 4, 15, fmt_pct)
            ws_summ.set_column(5, 6, 15, workbook.add_format({'align': 'center'}))
            ws_summ.conditional_format(f'C2:E{len(df_summary)+1}', cond_red)
            ws_summ.conditional_format(f'C2:C{len(df_summary)+1}', cond_green) # Only Return really needs green

            # --- SHEET 3: RISK & ATTRIBUTION ---
            ws_attr = workbook.add_worksheet('Risk & Attribution')
            
            # Section 1: Risk Metrics
            ws_attr.write('A1', 'Section 1: Risk Metrics', workbook.add_format({'bold': True, 'font_size': 14, 'color': '#2C3E50'}))
            df_risk.to_excel(writer, sheet_name='Risk & Attribution', index=False, startrow=2)
            for c_n, val in enumerate(df_risk.columns.values):
                ws_attr.write(2, c_n, val, fmt_header)
            
            r_end = 2 + len(df_risk) + 1
            
            # Section 2: Asset Contributions
            ws_attr.write(r_end + 2, 0, 'Section 2: Asset Contributions', workbook.add_format({'bold': True, 'font_size': 14, 'color': '#2C3E50'}))
            df_contrib.to_excel(writer, sheet_name='Risk & Attribution', index=False, startrow=r_end + 4)
            for c_n, val in enumerate(df_contrib.columns.values):
                ws_attr.write(r_end + 4, c_n, val, fmt_header)
                
            c_end = r_end + 4 + len(df_contrib) + 1
            
            # Section 3: Factor Exposure
            ws_attr.write(c_end + 2, 0, 'Section 3: Factor Exposure', workbook.add_format({'bold': True, 'font_size': 14, 'color': '#2C3E50'}))
            df_exposures.to_excel(writer, sheet_name='Risk & Attribution', index=False, startrow=c_end + 4)
            for c_n, val in enumerate(df_exposures.columns.values):
                ws_attr.write(c_end + 4, c_n, val, fmt_header)
                
            ws_attr.set_column(0, 0, 25)
            ws_attr.set_column(1, 1, 15, fmt_pct)  # Assumes weight or value, let's just make it broad
            ws_attr.set_column('C:C', 20, fmt_money) # Money contribution
            ws_attr.set_column('D:D', 20, fmt_pct) # Pct contribution
            ws_attr.set_column('E:E', 20)
            
            ws_attr.conditional_format(f'B{r_end+5}:D{c_end}', cond_red)
            ws_attr.conditional_format(f'B{r_end+5}:D{c_end}', cond_green)

            # --- SHEET 4: PORTFOLIO COMPARISON (Conditional) ---
            if comp_df is not None and not comp_df.empty:
                comp_df = comp_df.copy()
                if 'Worst Scenario Return' in comp_df.columns and 'Rank' not in comp_df.columns:
                    comp_df.insert(1, 'Rank', range(1, len(comp_df) + 1))
                    
                comp_df.to_excel(writer, sheet_name='Portfolio Comparison', index=False)
                ws_comp = writer.sheets['Portfolio Comparison']
                for c_n, val in enumerate(comp_df.columns.values):
                    ws_comp.write(0, c_n, val, fmt_header)
                    
                ws_comp.set_column(0, 0, 25)
                ws_comp.set_column(1, 1, 12, workbook.add_format({'align': 'center', 'bold': True}))
                ws_comp.set_column(2, 4, 18, fmt_pct)
                ws_comp.set_column(5, 5, 20, fmt_pct)
                ws_comp.set_column(6, 6, 20, fmt_money) if len(comp_df.columns) > 6 else None
                
                max_c = len(comp_df)
                ws_comp.conditional_format(f'C2:E{max_c+1}', cond_red)
                ws_comp.conditional_format(f'C2:E{max_c+1}', cond_green)
                
                chart_comp = workbook.add_chart({'type': 'column'})
                col_i_worst = comp_df.columns.get_loc('Worst Scenario Return')
                chart_comp.add_series({
                    'name': 'Worst Scenario Return',
                    'categories': ['Portfolio Comparison', 1, 0, max_c, 0],
                    'values':     ['Portfolio Comparison', 1, col_i_worst, max_c, col_i_worst],
                    'fill':       {'color': '#C0504D'},
                    'data_labels': {'value': True, 'num_format': '0.00%'}
                })
                chart_comp.set_title({'name': 'Portfolio Risk Comparison'})
                chart_comp.set_y_axis({'name': 'Return', 'num_format': '0.00%'})
                chart_comp.set_size({'width': 750, 'height': 350})
                ws_comp.insert_chart('J2', chart_comp)

            # --- SHEET 5: MODEL PERFORMANCE (Conditional) ---
            if bt_df is not None and bt_metrics is not None:
                bt_df_out = bt_df.reset_index()
                if np.issubdtype(bt_df_out['Date'].dtype, np.datetime64):
                    bt_df_out['Date'] = bt_df_out['Date'].dt.strftime('%Y-%m-%d')
                    
                bt_df_out.to_excel(writer, sheet_name='Model Performance', index=False, startrow=5)
                ws_bt = writer.sheets['Model Performance']
                
                ws_bt.write('B2', 'Mean Absolute Error (MAE)', fmt_kpi_label)
                ws_bt.write('D2', 'Root Mean Squared (RMSE)', fmt_kpi_label)
                ws_bt.write('F2', 'Directional Accuracy', fmt_kpi_label)
                
                ws_bt.merge_range('B3:C3', bt_metrics.get('MAE', 0), fmt_kpi_val)
                ws_bt.merge_range('D3:E3', bt_metrics.get('RMSE', 0), fmt_kpi_val)
                ws_bt.merge_range('F3:G3', f"{bt_metrics.get('Directional Accuracy', 0)*100:.1f}%", fmt_kpi_val)
                
                for c_n, val in enumerate(bt_df_out.columns.values):
                    ws_bt.write(5, c_n, val, fmt_header)
                    
                ws_bt.set_column(0, 0, 15, workbook.add_format({'align': 'center'}))
                ws_bt.set_column(1, 3, 18, fmt_pct)
                
                max_b = len(bt_df_out) + 5
                ws_bt.conditional_format(f'B7:D{max_b+1}', cond_red)
                ws_bt.conditional_format(f'B7:D{max_b+1}', cond_green)
                
                # Chart 1
                chart_ret = workbook.add_chart({'type': 'line'})
                chart_ret.add_series({
                    'name': 'Predicted Return',
                    'categories': ['Model Performance', 6, 0, max_b, 0],
                    'values':     ['Model Performance', 6, 1, max_b, 1],
                    'line':       {'color': '#4F81BD', 'width': 2}
                })
                chart_ret.add_series({
                    'name': 'Actual Return',
                    'categories': ['Model Performance', 6, 0, max_b, 0],
                    'values':     ['Model Performance', 6, 2, max_b, 2],
                    'line':       {'color': '#9BBB59', 'width': 2}
                })
                chart_ret.set_title({'name': 'Predicted vs Actual Portfolio Returns'})
                chart_ret.set_y_axis({'name': 'Return', 'num_format': '0.00%'})
                chart_ret.set_size({'width': 700, 'height': 350})
                ws_bt.insert_chart('F7', chart_ret)
                
                # Chart 2
                chart_err = workbook.add_chart({'type': 'line'})
                chart_err.add_series({
                    'name': 'Prediction Error',
                    'categories': ['Model Performance', 6, 0, max_b, 0],
                    'values':     ['Model Performance', 6, 3, max_b, 3],
                    'line':       {'color': '#C0504D', 'width': 2}
                })
                chart_err.set_title({'name': 'Model Prediction Error Over Time'})
                chart_err.set_y_axis({'name': 'Error', 'num_format': '0.00%'})
                chart_err.set_size({'width': 700, 'height': 350})
                ws_bt.insert_chart('F25', chart_err)

        logger.info(f"Compiled and exported master Unified Institutional Report successfully to {out_path}!")

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
                'Drawdown Base': drawdown
            })
            
        df_summary = pd.DataFrame(summary_data)
        if df_summary.empty:
            return df_summary, pd.DataFrame(), "N/A"
            
        df_summary = df_summary.sort_values(by='Portfolio Return', ascending=True)
        df_summary['Rank'] = df_summary['Portfolio Return'].rank(ascending=True).astype(int)
        
        def get_risk_level(ret):
            if ret <= -0.15: return 'Severe'
            elif ret <= -0.05: return 'Moderate'
            else: return 'Mild'
        df_summary['Risk Level'] = df_summary['Portfolio Return'].apply(get_risk_level)
        df_summary = df_summary[['Scenario Name', 'Shock Parameters', 'Portfolio Return', 'Portfolio PnL', 'Drawdown Base', 'Rank', 'Risk Level']]
        
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
            
        if not frames:
            return pd.DataFrame()
            
        df_contrib = pd.concat(frames).sort_values(by='Contribution to PnL', ascending=True)
        scenario_sums = df_contrib.groupby('Scenario')['Contribution to PnL'].transform('sum')
        df_contrib['% Contribution'] = df_contrib['Contribution to PnL'] / scenario_sums.replace(0, float('nan'))
        df_contrib['% Contribution'] = df_contrib['% Contribution'].fillna(0.0)
        
        df_contrib = df_contrib[['Asset', 'Weight', 'Contribution to PnL', '% Contribution', 'Scenario']]
        return df_contrib

    def _prepare_exposure_data(self, exposures, portfolio):
        if exposures.empty:
            return pd.DataFrame()
        port_exposures = exposures.mul(portfolio['weight'], axis=0).sum()
        df_exposures = port_exposures.reset_index()
        df_exposures.columns = ['Factor', 'Portfolio Exposure']
        return df_exposures
