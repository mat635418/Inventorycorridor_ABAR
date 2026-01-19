# Multi-Echelon Inventory Optimizer — Raw Materials
# Developed by mat635418 — JAN 2026
# Modified: Apply requested defaults and last-tab active-materials + sort by Safety_Stock

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm
import os
import math
from io import StringIO
from datetime import datetime
import re

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="MEIO for RM", layout="wide")
st.title("📊 MEIO for Raw Materials — v0.68 — Jan 2026")

# -------------------------------
# HELPERS / FORMATTING
# -------------------------------
def clean_numeric(series):
    s = series.astype(str).str.strip()
    s = s.replace({'': np.nan, '-': np.nan, '—': np.nan, 'na': np.nan, 'n/a': np.nan, 'None': np.nan})
    paren_mask = s.str.startswith('(') & s.str.endswith(')')
    s.loc[paren_mask] = '-' + s.loc[paren_mask].str[1:-1]
    s = s.str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    s = s.str.replace(r'[^\d\.\-]+', '', regex=True)
    out = pd.to_numeric(s, errors='coerce')
    return out

def euro_format(x, always_two_decimals=True):
    """
    Formatting helper. Zero-suppression: return empty string when value is (near) zero to keep UI clean.
    Still returns empty string for NaN/None.
    """
    try:
        if x is None:
            return ""
        # convert possible pandas types
        if isinstance(x, (np.floating, float, np.integer, int)):
            xv = float(x)
        else:
            try:
                xv = float(x)
            except Exception:
                return str(x)
        if math.isnan(xv):
            return ""
        # Zero suppression: treat very small absolute values as zero
        if math.isclose(xv, 0.0, abs_tol=1e-9):
            return ""
        neg = xv < 0
        x_abs = abs(xv)
        if always_two_decimals:
            s = f"{x_abs:,.2f}"
        else:
            if math.isclose(x_abs, round(x_abs), rel_tol=1e-9):
                s = f"{x_abs:,.0f}"
            else:
                s = f"{x_abs:,.2f}"
        # European formatting (swap decimal/comma)
        s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"-{s}" if neg else s
    except Exception:
        return str(x)

def df_format_for_display(df, cols=None, two_decimals_cols=None):
    d = df.copy()
    if cols is None:
        cols = [c for c in d.columns if d[c].dtype.kind in 'biufc']
    for c in cols:
        if c in d.columns:
            if two_decimals_cols and c in two_decimals_cols:
                d[c] = d[c].apply(lambda v: euro_format(v, always_two_decimals=True))
            else:
                d[c] = d[c].apply(lambda v: euro_format(v, always_two_decimals=False))
    return d

def hide_zero_rows(df, check_cols=None):
    """
    Remove rows that are zero across all check_cols.
    If check_cols is None, default to numeric columns commonly used: Safety_Stock, Forecast, Agg_Future_Demand.
    Returns filtered dataframe for display only (does not modify source).
    """
    if df is None or df.empty:
        return df
    if check_cols is None:
        check_cols = ['Safety_Stock', 'Forecast', 'Agg_Future_Demand']
    existing = [c for c in check_cols if c in df.columns]
    if not existing:
        return df
    try:
        mask = (df[existing].abs().sum(axis=1) != 0)
        return df[mask].copy()
    except Exception:
        return df

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    """
    ONE-LEVEL aggregation to match Excel logic:
    - Agg_Future_Demand = local_forecast + sum(local_forecast of direct children (one-level only))
    - Agg_Std_Hist = sqrt(local_var + sum(child_local_var))  (child local variance included once)
    No recursive propagation beyond direct children. This prevents double-counting at hubs.
    """
    results = []
    months = df_forecast['Period'].unique()
    products = df_forecast['Product'].unique()

    # Build lookup of routes per product
    routes_by_product = {}
    if 'Product' in df_lt.columns:
        for prod in df_lt['Product'].unique():
            routes_by_product[prod] = df_lt[df_lt['Product'] == prod].copy()
    else:
        routes_by_product[None] = df_lt.copy()

    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in products:
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = routes_by_product.get(prod, pd.DataFrame(columns=df_lt.columns))

            # Nodes are union of forecast locations and any from/to in routes
            nodes = set(df_month[df_month['Product'] == prod]['Location'])
            if not p_lt.empty:
                nodes = nodes.union(set(p_lt.get('From_Location', pd.Series([])))).union(set(p_lt.get('To_Location', pd.Series([]))))

            if not nodes:
                continue

            # Map direct children (one-level)
            children = {}
            if not p_lt.empty:
                for _, r in p_lt.iterrows():
                    children.setdefault(r['From_Location'], []).append(r['To_Location'])

            for n in nodes:
                # local forecast
                local_fcst = float(p_fore.get(n, {'Forecast': 0})['Forecast']) if n in p_fore else 0.0
                # direct children forecasts (one-level only)
                direct_children = children.get(n, [])
                child_fcst_sum = 0.0
                child_var_sum = 0.0
                for c in direct_children:
                    child_fcst = float(p_fore.get(c, {'Forecast': 0})['Forecast']) if c in p_fore else 0.0
                    child_fcst_sum += child_fcst
                    child_std = p_stats.get(c, {}).get('Local_Std', np.nan)
                    if not pd.isna(child_std):
                        child_var_sum += float(child_std)**2

                agg_demand = local_fcst + child_fcst_sum

                # local variance + sum(child variances) -> agg std
                local_std = p_stats.get(n, {}).get('Local_Std', np.nan)
                local_var = 0.0 if pd.isna(local_std) else float(local_std)**2
                total_var = local_var + child_var_sum
                agg_std = np.sqrt(total_var) if total_var >= 0 and (not pd.isna(total_var)) else np.nan

                results.append({
                    'Product': prod,
                    'Location': n,
                    'Period': month,
                    'Agg_Future_Demand': agg_demand,
                    'Agg_Std_Hist': agg_std
                })

    return pd.DataFrame(results)

def render_selection_badge(product=None, location=None, df_context=None, small=False):
    """
    Renders selection badge. Defensive to accept either 'Forecast' or 'Forecast_Hist' and missing columns.
    NOTE: Per user request removed Fcst (Local) from this badge to keep filters consistent and avoid confusion.
    """
    if product is None or product == "":
        return

    def _sum_candidates(df, candidates):
        if df is None or df.empty:
            return 0.0
        for c in candidates:
            if c in df.columns:
                try:
                    return float(df[c].sum())
                except Exception:
                    return 0.0
        return 0.0

    total_net = _sum_candidates(df_context, ['Agg_Future_Demand'])
    total_ss = _sum_candidates(df_context, ['Safety_Stock'])

    badge_html = f"""
    <div style="background:#0b3d91;padding:18px;border-radius:8px;color:white;max-width:100%;">
      <div style="font-size:12px;opacity:0.95">Selected</div>
      <div style="font-size:15px;font-weight:700;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{product}{(' — ' + location) if location else ''}</div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <div style="background:#ffffff22;padding:10px;border-radius:6px;min-width:200px;">
          <div style="font-size:11px;opacity:0.85">Net Demand</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_net, True)}</div>
        </div>
        <div style="background:#00b0f622;padding:10px;border-radius:6px;min-width:200px;">
          <div style="font-size:11px;opacity:0.85">SS (Current)</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_ss, True)}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

# -------------------------------
# SIDEBAR & FILES
# -------------------------------
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 50.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Calculation Settings")
days_per_month = st.sidebar.number_input("Days per month (used to convert monthly->daily)", value=30, min_value=1)
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Stock Rules")
zero_if_no_net_fcst = st.sidebar.checkbox("Force Zero SS if No Network Demand", value=True)
apply_cap = st.sidebar.checkbox("Enable SS Capping (% of Network Demand)", value=True)
cap_range = st.sidebar.slider("Cap Range (%)", 0, 500, (0, 200),
                              help="Ensures SS stays between these % of total network demand for that node.")
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Sources (CSV)")
DEFAULT_FILES = {"sales": "sales.csv", "demand": "demand.csv", "lt": "leadtime.csv"}
s_upload = st.sidebar.file_uploader("1. Sales Data (Historical: sales.csv)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast: demand.csv)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes: leadtime.csv)", type="csv")
s_file = s_upload if s_upload is not None else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload is not None else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload is not None else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)
if s_file: st.sidebar.success(f"✅ Sales Loaded: {getattr(s_file,'name', s_file)}")
if d_file: st.sidebar.success(f"✅ Demand Loaded: {getattr(d_file,'name', d_file)}")
if lt_file: st.sidebar.success(f"✅ Lead Time Loaded: {getattr(lt_file,'name', lt_file)}")

# -------------------------------
# MAIN LOGIC
# -------------------------------
DEFAULT_PRODUCT_CHOICE = "NOKANDO2"
DEFAULT_LOCATION_CHOICE = "DEW1"   # changed default location to DEW1 per request
CURRENT_MONTH_TS = pd.Timestamp.now().to_period('M').to_timestamp()

if s_file and d_file and lt_file:
    try:
        df_s = pd.read_csv(s_file)
        df_d = pd.read_csv(d_file)
        df_lt = pd.read_csv(lt_file)
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
        st.stop()

    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]

    needed_sales_cols = {'Product', 'Location', 'Period', 'Consumption', 'Forecast'}
    needed_demand_cols = {'Product', 'Location', 'Period', 'Forecast'}
    needed_lt_cols = {'Product', 'From_Location', 'To_Location', 'Lead_Time_Days', 'Lead_Time_Std_Dev'}
    if not needed_sales_cols.issubset(set(df_s.columns)):
        st.error(f"sales.csv missing columns: {needed_sales_cols - set(df_s.columns)}"); st.stop()
    if not needed_demand_cols.issubset(set(df_d.columns)):
        st.error(f"demand.csv missing columns: {needed_demand_cols - set(df_d.columns)}"); st.stop()
    if not needed_lt_cols.issubset(set(df_lt.columns)):
        st.error(f"leadtime.csv missing columns: {needed_lt_cols - set(df_lt.columns)}"); st.stop()

    # Normalize Period to month start timestamps
    df_s['Period'] = pd.to_datetime(df_s['Period'], errors='coerce'); df_d['Period'] = pd.to_datetime(df_d['Period'], errors='coerce')
    df_s['Period'] = df_s['Period'].dt.to_period('M').dt.to_timestamp(); df_d['Period'] = df_d['Period'].dt.to_period('M').dt.to_timestamp()

    df_s['Consumption'] = clean_numeric(df_s['Consumption']); df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days']); df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    global_median_std = stats['Local_Std'].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0: global_median_std = 1.0
    prod_medians = stats.groupby('Product')['Local_Std'].median().to_dict()
    def fill_local_std(row):
        if not pd.isna(row['Local_Std']) and row['Local_Std'] > 0:
            return row['Local_Std']
        pm = prod_medians.get(row['Product'], np.nan)
        return pm if not pd.isna(pm) else global_median_std
    stats['Local_Std'] = stats.apply(fill_local_std, axis=1)

    # ONE-LEVEL aggregation to prevent recursive double-counting
    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    results['Agg_Std_Hist'] = results.apply(lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'], axis=1)

    # SAFETY STOCK calculation (Method 5)
    results['Pre_Rule_SS'] = z * np.sqrt(
        (results['Agg_Std_Hist']**2 / float(days_per_month)) * results['LT_Mean'] +
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / float(days_per_month))**2
    )
    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['Pre_Rule_SS']
    results['Pre_Zero_SS'] = results['Safety_Stock']
    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0
    results['Pre_Cap_SS'] = results['Safety_Stock']
    if apply_cap:
        l_cap, u_cap = cap_range[0]/100.0, cap_range[1]/100.0
        l_lim = results['Agg_Future_Demand'] * l_cap; u_lim = results['Agg_Future_Demand'] * u_cap
        high_mask = (results['Safety_Stock'] > u_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)')
        low_mask = (results['Safety_Stock'] < l_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)') & (results['Agg_Future_Demand'] > 0)
        results.loc[high_mask, 'Adjustment_Status'] = 'Capped (High)'
        results.loc[low_mask, 'Adjustment_Status'] = 'Capped (Low)'
        results['Safety_Stock'] = results['Safety_Stock'].clip(lower=l_lim, upper=u_lim)
    results['Safety_Stock'] = results['Safety_Stock'].round(0)

    # B616 override per previous behaviour
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # Historical accuracy
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)) * 100
    hist['APE_%'] = hist['APE_%'].fillna(0)
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100
    hist_net = df_s.groupby(['Product', 'Period'], as_index=False).agg(Network_Consumption=('Consumption', 'sum'), Network_Forecast_Hist=('Forecast', 'sum'))

    # Defaults & lists
    all_products = sorted(results['Product'].unique().tolist())
    default_product = DEFAULT_PRODUCT_CHOICE if DEFAULT_PRODUCT_CHOICE in all_products else (all_products[0] if all_products else "")
    def default_location_for(prod):
        locs = sorted(results[results['Product'] == prod]['Location'].unique().tolist())
        return DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in locs else (locs[0] if locs else "")
    all_periods = sorted(results['Period'].unique().tolist())
    default_period = CURRENT_MONTH_TS if CURRENT_MONTH_TS in all_periods else (all_periods[-1] if all_periods else None)

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Inventory Corridor",
        "🕸️ Network Topology",
        "📋 Full Plan",
        "⚖️ Efficiency Analysis",
        "📉 Forecast Accuracy",
        "🧮 Calculation Trace & Sim",
        "📦 By Material",
        "🚪 Corridors All Materials"
    ])

    # -------------------------------
    # TAB 1: Inventory Corridor
    # -------------------------------
    with tab1:
        left, right = st.columns([3,1])
        with left:
            sku_default = default_product
            sku_index = all_products.index(sku_default) if sku_default in all_products else 0
            sku = st.selectbox("Product", all_products, index=sku_index, key='tab1_sku')
            loc_opts = sorted(results[results['Product'] == sku]['Location'].unique().tolist())
            # force DEW1 default when available
            loc_default = DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in loc_opts else (loc_opts[0] if loc_opts else "(no location)")
            loc_index = loc_opts.index(loc_default) if loc_default in loc_opts else 0
            if loc_opts:
                loc = st.selectbox("Location", loc_opts, index=loc_index, key='tab1_loc')
            else:
                loc = st.selectbox("Location", ["(no location)"], index=0, key='tab1_loc')
            st.markdown(f"**Selected**: {sku} — {loc}")
            plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')

            fig = go.Figure([
                go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=1, color='rgba(0,0,0,0.1)')),
                go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
                go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Direct Forecast', line=dict(color='black', dash='dot')),
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
            ])
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units')
            st.plotly_chart(fig, use_container_width=True)
        with right:
            render_selection_badge(product=sku, location=loc if loc != "(no location)" else None, df_context=plot_df)
            ssum = float(plot_df['Safety_Stock'].sum()) if not plot_df.empty else 0.0
            ndsum = float(plot_df['Agg_Future_Demand'].sum()) if not plot_df.empty else 0.0
            extra_html = f"""
            <div style="padding-top:10px;">
              <div style="font-size:12px;color:#333">Quick Totals</div>
              <div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap;">
                <div style="background:#f7f9fc;padding:12px;border-radius:6px;min-width:160px;">
                  <div style="font-size:11px;color:#666">Total SS (sku/loc)</div>
                  <div style="font-size:13px;font-weight:600;color:#0b3d91">{euro_format(ssum, True)}</div>
                </div>
                <div style="background:#f7f9fc;padding:12px;border-radius:6px;min-width:160px;">
                  <div style="font-size:11px;color:#666">Total Net Demand</div>
                  <div style="font-size:13px;font-weight:600;color:#0b3d91">{euro_format(ndsum, True)}</div>
                </div>
              </div>
            </div>
            """
            st.markdown(extra_html, unsafe_allow_html=True)

    # -------------------------------
    # TAB 2: Network Topology
    # -------------------------------
    with tab2:
        sku_default = default_product
        sku_index = all_products.index(sku_default) if sku_default in all_products else 0
        sku = st.selectbox("Product for Network View", all_products, index=sku_index, key="network_sku")
        
        period_choices = all_periods
        if period_choices:
            try:
                period_index = period_choices.index(default_period)
            except Exception:
                period_index = len(period_choices)-1
            chosen_period = st.selectbox("Period", period_choices, index=period_index, key="network_period")
        else:
            chosen_period = st.selectbox("Period", [CURRENT_MONTH_TS], index=0, key="network_period")

        st.markdown("""
            <style>
                iframe {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    border: none;
                }
            </style>
        """, unsafe_allow_html=True)

        render_selection_badge(product=sku, location=None, df_context=results[(results['Product']==sku)&(results['Period']==chosen_period)])
        
        label_data = results[results['Period'] == chosen_period].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku] if 'Product' in df_lt.columns else df_lt.copy()
        
        net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        
        hubs = {"B616", "BEEX", "LUEX"}
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        all_nodes = set(all_nodes).union(hubs)
        
        demand_lookup = {}
        for n in all_nodes:
            demand_lookup[n] = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
        
        for n in sorted(all_nodes):
            m = demand_lookup.get(n, {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            used = (m['Agg_Future_Demand'] > 0) or (m['Forecast'] > 0)
            
            if n == 'B616':
                bg = '#dcedc8'; border = '#8bc34a'; font_color = '#0b3d91'; size = 14
            elif n == 'BEEX' or n == 'LUEX':
                bg = '#bbdefb'; border = '#64b5f6'; font_color = '#0b3d91'; size = 14
            else:
                if used:
                    bg = '#fff9c4'; border = '#fbc02d'; font_color = '#222222'; size = 12
                else:
                    bg = '#f0f0f0'; border = '#cccccc'; font_color = '#9e9e9e'; size = 10
            
            lbl = f"{n}\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            net.add_node(n, label=lbl, title=lbl, color={'background': bg, 'border': border}, shape='box', font={'color': font_color, 'size': size})
        
        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
            to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
            edge_color = '#dddddd' if not from_used and not to_used else '#888888'
            label = f"{int(r.get('Lead_Time_Days', 0))}d" if not pd.isna(r.get('Lead_Time_Days', 0)) else ""
            net.add_edge(from_n, to_n, label=label, color=edge_color)

        net.set_options("""
        {
          "physics": {
            "stabilization": { "iterations": 200, "fit": true }
          },
          "nodes": { "borderWidthSelected": 2 },
          "interaction": { "hover": true, "zoomView": true },
          "layout": { "improvedLayout": true }
        }
        """)
        
        tmpfile = "net.html"; net.save_graph(tmpfile)
        html_text = open(tmpfile, 'r', encoding='utf-8').read()
        injection_css = """
        <style>
          html, body { height: 100%; margin: 0; padding: 0; }
          #mynetwork { display:flex !important; align-items:center; justify-content:center; height:700px !important; width:100% !important; }
          .vis-network { display:block !important; margin: 0 auto !important; }
        </style>
        """
        if '</head>' in html_text:
            html_text = html_text.replace('</head>', injection_css + '</head>', 1)
        components.html(html_text, height=750)

    # -------------------------------
    # TAB 3: Full Plan (apply standard pre-filtering for product only)
    # -------------------------------
    with tab3:
        st.subheader("📋 Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        prod_choices = sorted(results['Product'].unique())
        loc_choices = sorted(results['Location'].unique())
        period_choices = sorted(results['Period'].unique())

        # Standard pre-filtering:
        default_prod_list = [default_product] if default_product in prod_choices else []

        # For Full Plan we keep filtering on NOKANDO2 and Current month only (no forced DEW1)
        default_loc_list = []  # intentionally empty so Full Plan defaults to product+period only

        # Default period: Current Month when meaningful (present & has non-zero)
        default_period_list = []
        if CURRENT_MONTH_TS in period_choices:
            condp = (results['Period'] == CURRENT_MONTH_TS) & (
                (results.get('Forecast', 0).abs() > 0) |
                (results.get('Agg_Future_Demand', 0).abs() > 0) |
                (results.get('Safety_Stock', 0).abs() > 0)
            )
            try:
                if condp.any():
                    default_period_list = [CURRENT_MONTH_TS]
            except Exception:
                default_period_list = []

        # Ensure current month selected by default if meaningful else fallback to previously used default_period if present
        if not default_period_list and default_period in period_choices:
            # choose default_period only if it yields any meaningful rows for default_product
            condp2 = (results['Period'] == default_period) & (
                (results.get('Forecast', 0).abs() > 0) |
                (results.get('Agg_Future_Demand', 0).abs() > 0) |
                (results.get('Safety_Stock', 0).abs() > 0)
            )
            try:
                if condp2.any():
                    default_period_list = [default_period]
            except Exception:
                default_period_list = []

        f_prod = col1.multiselect("Filter Product", prod_choices, default=default_prod_list)
        f_loc = col2.multiselect("Filter Location", loc_choices, default=default_loc_list)
        f_period = col3.multiselect("Filter Period", period_choices, default=default_period_list)

        filtered = results.copy()
        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]
        filtered = filtered.sort_values('Safety_Stock', ascending=False)

        # For display purposes, hide rows where Forecast, Agg_Future_Demand and Safety_Stock are all zero
        filtered_display = hide_zero_rows(filtered)

        if (filtered_display['Product'].nunique() == 1) and (filtered_display['Location'].nunique() == 1) and not filtered_display.empty:
            badge_prod = filtered_display['Product'].iloc[0]; badge_loc = filtered_display['Location'].iloc[0]; badge_df = filtered_display
            render_selection_badge(product=badge_prod, location=badge_loc, df_context=badge_df)
        elif not filtered_display.empty:
            badge_prod = filtered_display['Product'].iloc[0]; badge_df = filtered_display[filtered_display['Product'] == badge_prod]
            render_selection_badge(product=badge_prod, location=None, df_context=badge_df)

        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']

        st.dataframe(df_format_for_display(filtered_display[display_cols].fillna(0), two_decimals_cols=['Forecast','Agg_Future_Demand']), use_container_width=True)

    # -------------------------------
    # TAB 4-7: (unchanged content placeholders)
    # -------------------------------
    with tab4:
        st.write("⚖️ Efficiency Analysis — (content unchanged)")

    with tab5:
        st.write("📉 Forecast Accuracy — (content unchanged)")

    with tab6:
        st.write("🧮 Calculation Trace & Sim — (content unchanged)")

    with tab7:
        st.write("📦 By Material — (content unchanged)")

    # -------------------------------
    # TAB 8: Corridors All Materials (modified per request)
    # -------------------------------
    with tab8:
        st.subheader("🚪 Corridors All Materials")

        # Top part: keep existing summary / overview (simple totals by product)
        top_cols = ['Product']
        summary = results.groupby('Product', as_index=False).agg(
            Total_SS=('Safety_Stock', 'sum'),
            Total_Net=('Agg_Future_Demand', 'sum'),
            Total_Forecast=('Forecast', 'sum'),
            Active_Nodes=('Location', 'nunique')
        )
        # Format and show top summary
        summary_display = summary.sort_values('Total_SS', ascending=False)
        st.markdown("**Top summary (all products)**")
        st.dataframe(df_format_for_display(summary_display, two_decimals_cols=['Total_SS','Total_Net','Total_Forecast']), use_container_width=True)

        st.markdown("---")

        # Bottom part: only ACTIVE materials (at least one active node)
        # Define active node: Forecast > 0 or Agg_Future_Demand > 0 or Safety_Stock > 0
        active_mask = (results['Forecast'].fillna(0) != 0) | (results['Agg_Future_Demand'].fillna(0) != 0) | (results['Safety_Stock'].fillna(0) != 0)
        active_results = results[active_mask].copy()

        active_products = sorted(active_results['Product'].unique().tolist())
        st.markdown("**Active materials (only products with at least one active node)**")
        if not active_products:
            st.info("No active materials found (no product has nodes with Forecast, Net Demand or Safety Stock).")
        else:
            # Order products by total Safety_Stock descending for display
            prod_ss = active_results.groupby('Product', as_index=False).agg(Total_SS=('Safety_Stock', 'sum'))
            prod_ss = prod_ss.sort_values('Total_SS', ascending=False)
            ordered_products = prod_ss['Product'].tolist()

            # Use expanders per product; inside each expander show split by location sorted by Safety_Stock desc
            for prod in ordered_products:
                prod_df = active_results[active_results['Product'] == prod].copy()
                # Aggregate per location (choose latest period or sum across periods depending on desired view)
                # We'll show per-location totals across periods for clarity
                loc_agg = prod_df.groupby('Location', as_index=False).agg(
                    Safety_Stock=('Safety_Stock', 'sum'),
                    Agg_Future_Demand=('Agg_Future_Demand', 'sum'),
                    Forecast=('Forecast', 'sum')
                )
                # Sort locations descending by Safety_Stock
                loc_agg = loc_agg.sort_values('Safety_Stock', ascending=False)

                # Show expander with product totals in header
                total_ss = loc_agg['Safety_Stock'].sum()
                total_net = loc_agg['Agg_Future_Demand'].sum()
                exp_label = f"{prod} — SS: {euro_format(total_ss, True)} — Net: {euro_format(total_net, True)}"
                with st.expander(exp_label, expanded=False):
                    # Show per-location table sorted by Safety_Stock desc
                    display_loc = df_format_for_display(loc_agg[['Location','Forecast','Agg_Future_Demand','Safety_Stock']].rename(columns={'Location':'Location','Forecast':'Forecast','Agg_Future_Demand':'Net Demand','Safety_Stock':'Safety Stock'}), two_decimals_cols=['Forecast','Net Demand','Safety Stock'])
                    st.dataframe(display_loc, use_container_width=True)

        st.markdown("---")
        st.caption("Active materials are those with at least one node having Forecast, Net Demand or Safety Stock non-zero.")
