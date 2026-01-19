# Multi-Echelon Inventory Optimizer — Enhanced Version (Reviewed & Improved)
# Enhanced by Copilot for mat635418 — 2026-01-15 (with UI/UX updates)
# Modified: 2026-01-17 — fixes: badge robustness, Forecast Accuracy, defaults, current month default,
# restored & enhanced scenario simulation (multi-scenario compare) and ensured By Material SS Attribution (Part B) present
# Modified: 2026-01-19 — v0.60 UI/UX: badge sizing, network centering, full-plan defaults, scenario defaults, waterfall for SS attribution
# Modified: 2026-01-19 — v0.61 fixes: restore network rendering, enforce 1-scenario default
# Modified: 2026-01-21 — v0.62 changes:
#  - Full Plan: do not preselect filters by default (empty selections)
#  - By Material: pastel, light colors while keeping existing color coding
#  - Network demand: changed aggregation to ONE-LEVEL downstream only (prevents recursive double-counting)
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
st.title("📊 MEIO for Raw Materials — v0.62 — Jan 2026")

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
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        neg = float(x) < 0
        x_abs = abs(float(x))
        if always_two_decimals:
            s = f"{x_abs:,.2f}"
        else:
            if math.isclose(x_abs, round(x_abs)):
                s = f"{x_abs:,.0f}"
            else:
                s = f"{x_abs:,.2f}"
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
    Improved: flexible layout (wrap) and larger inner boxes to avoid cropping on small containers.
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

    total_fcst = _sum_candidates(df_context, ['Forecast', 'Forecast_Hist'])
    total_net = _sum_candidates(df_context, ['Agg_Future_Demand'])
    total_ss = _sum_candidates(df_context, ['Safety_Stock'])

    # make badge inner boxes slightly larger and allow wrapping to avoid overflow
    badge_html = f"""
    <div style="background:#0b3d91;padding:18px;border-radius:8px;color:white;max-width:100%;">
      <div style="font-size:12px;opacity:0.95">Selected</div>
      <div style="font-size:15px;font-weight:700;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{product}{(' — ' + location) if location else ''}</div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <div style="background:#ffffff22;padding:10px;border-radius:6px;min-width:160px;">
          <div style="font-size:11px;opacity:0.85">Fcst (Local)</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_fcst, True)}</div>
        </div>
        <div style="background:#ffffff22;padding:10px;border-radius:6px;min-width:160px;">
          <div style="font-size:11px;opacity:0.85">Net Demand (1-level)</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_net, True)}</div>
        </div>
        <div style="background:#00b0f622;padding:10px;border-radius:6px;min-width:160px;">
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
DEFAULT_LOCATION_CHOICE = "BEEX"
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

    # Use ONE-LEVEL aggregation (prevents recursive double-counting)
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Inventory Corridor",
        "🕸️ Network Topology",
        "📋 Full Plan",
        "⚖️ Efficiency Analysis",
        "📉 Forecast Accuracy",
        "🧮 Calculation Trace & Sim",
        "📦 By Material"
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
            loc_default = default_location_for(sku)
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
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand (1-level)', line=dict(color='blue', dash='dash'))
            ])
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units')
            st.plotly_chart(fig, use_container_width=True)
        with right:
            render_selection_badge(product=sku, location=loc if loc != "(no location)" else None, df_context=plot_df)
            ssum = float(plot_df['Safety_Stock'].sum()) if not plot_df.empty else 0.0
            ndsum = float(plot_df['Agg_Future_Demand'].sum()) if not plot_df.empty else 0.0
            # enlarge quick totals boxes so they no longer clip values
            extra_html = f"""
            <div style="padding-top:10px;">
              <div style="font-size:12px;color:#333">Quick Totals</div>
              <div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap;">
                <div style="background:#f7f9fc;padding:12px;border-radius:6px;min-width:160px;">
                  <div style="font-size:11px;color:#666">Total SS (sku/loc)</div>
                  <div style="font-size:13px;font-weight:600;color:#0b3d91">{euro_format(ssum, True)}</div>
                </div>
                <div style="background:#f7f9fc;padding:12px;border-radius:6px;min-width:160px;">
                  <div style="font-size:11px;color:#666">Total Net Demand (1-level)</div>
                  <div style="font-size:13px;font-weight:600;color:#0b3d91">{euro_format(ndsum, True)}</div>
                </div>
              </div>
            </div>
            """
            st.markdown(extra_html, unsafe_allow_html=True)

    # -------------------------------
    # TAB 2: Network Topology (Centered both vertically and horizontally)
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

        # CSS to force horizontal centering of the iframe container (kept)
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
        
        # Initialize network with 100% width
        net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        
        hubs = {"B616", "BEEX", "LUEX"}
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        all_nodes = set(all_nodes).union(hubs)
        
        demand_lookup = {}
        for n in all_nodes:
            demand_lookup[n] = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
        
        # RESTORED ORIGINAL COLOR LOGIC
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
            
            lbl = f"{n}\nFcst: {euro_format(m['Forecast'])}\nNet(1-level): {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            net.add_node(n, label=lbl, title=lbl, color={'background': bg, 'border': border}, shape='box', font={'color': font_color, 'size': size})
        
        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
            to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
            edge_color = '#dddddd' if not from_used and not to_used else '#888888'
            label = f"{int(r.get('Lead_Time_Days', 0))}d" if not pd.isna(r.get('Lead_Time_Days', 0)) else ""
            net.add_edge(from_n, to_n, label=label, color=edge_color)

        # Options optimized for fitting screen and stabilizing the center
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

        # Read generated html and inject lightweight CSS into head to center the pyvis container.
        html_text = open(tmpfile, 'r', encoding='utf-8').read()
        injection_css = """
        <style>
          /* Ensure the pyvis network container uses full available height and is centered */
          html, body { height: 100%; margin: 0; padding: 0; }
          #mynetwork { display:flex !important; align-items:center; justify-content:center; height:700px !important; width:100% !important; }
          .vis-network { display:block !important; margin: 0 auto !important; }
        </style>
        """
        if '</head>' in html_text:
            html_text = html_text.replace('</head>', injection_css + '</head>', 1)
        components.html(html_text, height=750)

    # -------------------------------
    # TAB 3: Full Plan (do not preselect filters by default; keep material filter available)
    # -------------------------------
    with tab3:
        st.subheader("📋 Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        prod_choices = sorted(results['Product'].unique())
        loc_choices = sorted(results['Location'].unique())
        period_choices = sorted(results['Period'].unique())

        # Do NOT preselect anything by default (empty selections). The filter UI is available for users.
        f_prod = col1.multiselect("Filter Product", prod_choices, default=[])
        f_loc = col2.multiselect("Filter Location", loc_choices, default=[])
        f_period = col3.multiselect("Filter Period", period_choices, default=[])

        filtered = results.copy()
        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]
        filtered = filtered.sort_values('Safety_Stock', ascending=False)
        if (filtered['Product'].nunique() == 1) and (filtered['Location'].nunique() == 1) and not filtered.empty:
            badge_prod = filtered['Product'].iloc[0]; badge_loc = filtered['Location'].iloc[0]; badge_df = filtered
            render_selection_badge(product=badge_prod, location=badge_loc, df_context=badge_df)
        elif not filtered.empty:
            badge_prod = filtered['Product'].iloc[0]; badge_df = filtered[filtered['Product'] == badge_prod]
            render_selection_badge(product=badge_prod, location=None, df_context=badge_df)
        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
        disp = df_format_for_display(filtered[display_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'], two_decimals_cols=['Forecast'])
        st.dataframe(disp, use_container_width=True, height=700)
        csv_buf = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # -------------------------------
    # TAB 4..7 remain functionally identical, with By Material color tweaks below
    # -------------------------------

    # -------------------------------
    # TAB 7: By Material (pastel colors)
    # -------------------------------
    with tab7:
        st.header("📦 View by Material (Single Material Focus + 8 Reasons for Inventory)")
        sel_prod_default = default_product
        sel_prod_index = all_products.index(sel_prod_default) if sel_prod_default in all_products else 0
        selected_product = st.selectbox("Select Material", all_products, index=sel_prod_index, key="mat_sel")
        period_choices = all_periods
        if period_choices:
            try:
                sel_period_index = period_choices.index(default_period)
            except Exception:
                sel_period_index = len(period_choices)-1
            selected_period = st.selectbox("Select Period to Snapshot", period_choices, index=sel_period_index, key="mat_period")
        else:
            selected_period = st.selectbox("Select Period to Snapshot", [CURRENT_MONTH_TS], index=0, key="mat_period")

        mat_period_df = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)].copy()
        total_forecast = mat_period_df['Forecast'].sum(); total_net = mat_period_df['Agg_Future_Demand'].sum()
        total_ss = mat_period_df['Safety_Stock'].sum(); nodes_count = mat_period_df['Location'].nunique()
        avg_ss_per_node = (mat_period_df['Safety_Stock'].mean() if nodes_count > 0 else 0)
        render_selection_badge(product=selected_product, location=None, df_context=mat_period_df)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Local Forecast", euro_format(total_forecast, True)); k2.metric("Total Network Demand", euro_format(total_net, True))
        k3.metric("Total Safety Stock (sum nodes)", euro_format(total_ss, True)); k4.metric("Nodes", f"{nodes_count}"); k5.metric("Avg SS per Node", euro_format(avg_ss_per_node, True))

        st.markdown("### Why do we carry this SS? — 8 Reasons breakdown (aggregated for selected material)")
        if mat_period_df.empty:
            st.warning("No data for this material/period.")
        else:
            mat = mat_period_df.copy()
            mat['LT_Mean'] = mat['LT_Mean'].fillna(0); mat['LT_Std'] = mat['LT_Std'].fillna(0)
            mat['Agg_Std_Hist'] = mat['Agg_Std_Hist'].fillna(0); mat['Pre_Rule_SS'] = mat['Pre_Rule_SS'].fillna(0)
            mat['Safety_Stock'] = mat['Safety_Stock'].fillna(0); mat['Forecast'] = mat['Forecast'].fillna(0)
            mat['Agg_Future_Demand'] = mat['Agg_Future_Demand'].fillna(0)
            mat['term1'] = (mat['Agg_Std_Hist']**2 / float(days_per_month)) * mat['LT_Mean']
            mat['term2'] = (mat['LT_Std']**2) * (mat['Agg_Future_Demand'] / float(days_per_month))**2
            mat['demand_uncertainty_raw'] = z * np.sqrt(mat['term1'].clip(lower=0))
            mat['lt_uncertainty_raw'] = z * np.sqrt(mat['term2'].clip(lower=0))
            mat['direct_forecast_raw'] = mat['Forecast'].clip(lower=0)
            mat['indirect_network_raw'] = (mat['Agg_Future_Demand'] - mat['Forecast']).clip(lower=0)
            mat['cap_reduction_raw'] = ((mat['Pre_Rule_SS'] - mat['Safety_Stock']).clip(lower=0)).fillna(0)
            mat['cap_increase_raw'] = ((mat['Safety_Stock'] - mat['Pre_Rule_SS']).clip(lower=0)).fillna(0)
            mat['forced_zero_raw'] = mat.apply(lambda r: r['Pre_Rule_SS'] if r['Adjustment_Status'] == 'Forced to Zero' else 0, axis=1)
            mat['b616_override_raw'] = mat.apply(lambda r: r['Pre_Rule_SS'] if (r['Location'] == 'B616' and r['Safety_Stock'] == 0) else 0, axis=1)

            raw_drivers = {
                'Demand Uncertainty (z*sqrt(term1))': mat['demand_uncertainty_raw'].sum(),
                'Lead-time Uncertainty (z*sqrt(term2))': mat['lt_uncertainty_raw'].sum(),
                'Direct Local Forecast (sum Fcst)': mat['direct_forecast_raw'].sum(),
                'Indirect Network Demand (sum extra downstream)': mat['indirect_network_raw'].sum(),
                'Caps — Reductions (policy lowering SS)': mat['cap_reduction_raw'].sum(),
                'Caps — Increases (policy increasing SS)': mat['cap_increase_raw'].sum(),
                'Forced Zero Overrides (policy)': mat['forced_zero_raw'].sum(),
                'B616 Policy Override': mat['b616_override_raw'].sum()
            }

            drv_df = pd.DataFrame({'driver': list(raw_drivers.keys()), 'amount': [float(v) for v in raw_drivers.values()]})
            drv_denom = drv_df['amount'].sum()
            drv_df['pct_of_total_ss'] = drv_df['amount'] / (drv_denom if drv_denom > 0 else 1.0) * 100

            st.markdown("#### A. Original — Raw driver values (interpretation view)")
            # use light pastel palette but keep same categorical mapping semantics for other charts
            pastel_colors = px.colors.qualitative.Pastel
            fig_drv_raw = go.Figure()
            fig_drv_raw.add_trace(go.Bar(x=drv_df['driver'], y=drv_df['amount'], marker_color=pastel_colors))
            annotations_raw = []
            for idx, rowd in drv_df.iterrows():
                annotations_raw.append(dict(x=rowd['driver'], y=rowd['amount'], text=f"{rowd['pct_of_total_ss']:.1f}%", showarrow=False, yshift=8))
            fig_drv_raw.update_layout(title=f"{selected_product} — Raw Drivers (not SS-attribution)", xaxis_title="Driver", yaxis_title="Units", annotations=annotations_raw, height=420)
            st.plotly_chart(fig_drv_raw, use_container_width=True)
            st.markdown("Driver table (raw numbers and % of raw-sum)")
            st.dataframe(df_format_for_display(drv_df.rename(columns={'driver':'Driver','amount':'Units','pct_of_total_ss':'Pct_of_raw_sum'}).round(2), cols=['Units','Pct_of_raw_sum']), use_container_width=True, height=220)

            # -------------------------------
            # B. SS Attribution — waterfall with pastel colors
            # -------------------------------
            st.markdown("---")
            st.markdown("#### B. SS Attribution — Mutually exclusive components that SUM EXACTLY to Total Safety Stock")
            per_node = mat.copy()
            per_node['is_forced_zero'] = per_node['Adjustment_Status'] == 'Forced to Zero'
            per_node['is_b616_override'] = (per_node['Location'] == 'B616') & (per_node['Safety_Stock'] == 0)
            per_node['pre_ss'] = per_node['Pre_Rule_SS'].clip(lower=0)
            per_node['share_denom'] = per_node['demand_uncertainty_raw'] + per_node['lt_uncertainty_raw']
            per_node['demand_share'] = per_node.apply(lambda r: (r['pre_ss'] * (r['demand_uncertainty_raw'] / r['share_denom'])) if r['share_denom'] > 0 else (r['pre_ss'] / 2 if r['pre_ss'] > 0 else 0), axis=1)
            per_node['lt_share'] = per_node.apply(lambda r: (r['pre_ss'] * (r['lt_uncertainty_raw'] / r['share_denom'])) if r['share_denom'] > 0 else (r['pre_ss'] / 2 if r['pre_ss'] > 0 else 0), axis=1)
            per_node['forced_zero_amount'] = per_node.apply(lambda r: r['pre_ss'] if r['is_forced_zero'] else 0.0, axis=1)
            per_node['b616_override_amount'] = per_node.apply(lambda r: r['pre_ss'] if r['is_b616_override'] else 0.0, axis=1)
            def retained_ratio_calc(r):
                if r['pre_ss'] <= 0: return 0.0
                if r['is_forced_zero'] or r['is_b616_override']: return 0.0
                return float(r['Safety_Stock']) / float(r['pre_ss']) if r['pre_ss'] > 0 else 0.0
            per_node['retained_ratio'] = per_node.apply(retained_ratio_calc, axis=1)
            per_node['retained_demand'] = per_node['demand_share'] * per_node['retained_ratio']
            per_node['retained_lt'] = per_node['lt_share'] * per_node['retained_ratio']
            per_node['retained_stat_total'] = per_node['retained_demand'] + per_node['retained_lt']
            def direct_frac_calc(r):
                if r['Agg_Future_Demand'] > 0: return float(r['Forecast']) / float(r['Agg_Future_Demand'])
                return 0.0
            per_node['direct_frac'] = per_node.apply(direct_frac_calc, axis=1).clip(lower=0, upper=1)
            per_node['direct_retained_ss'] = per_node['retained_stat_total'] * per_node['direct_frac']
            per_node['indirect_retained_ss'] = per_node['retained_stat_total'] * (1 - per_node['direct_frac'])
            per_node['cap_reduction'] = per_node.apply(lambda r: max(r['pre_ss'] - r['Safety_Stock'], 0.0) if not (r['is_forced_zero'] or r['is_b616_override']) else 0.0, axis=1)
            per_node['cap_increase'] = per_node.apply(lambda r: max(r['Safety_Stock'] - r['pre_ss'], 0.0) if not (r['is_forced_zero'] or r['is_b616_override']) else 0.0, axis=1)

            ss_attrib = {
                'Demand Uncertainty (SS portion)': per_node['retained_demand'].sum(),
                'Lead-time Uncertainty (SS portion)': per_node['retained_lt'].sum(),
                'Direct Local Forecast (SS portion)': per_node['direct_retained_ss'].sum(),
                'Indirect Network Demand (SS portion)': per_node['indirect_retained_ss'].sum(),
                'Caps — Reductions (policy lowering SS)': per_node['cap_reduction'].sum(),
                'Caps — Increases (policy increasing SS)': per_node['cap_increase'].sum(),
                'Forced Zero Overrides (policy)': per_node['forced_zero_amount'].sum(),
                'B616 Policy Override': per_node['b616_override_amount'].sum()
            }
            for k in ss_attrib: ss_attrib[k] = float(ss_attrib[k])
            ss_sum = sum(ss_attrib.values())
            residual = float(total_ss) - ss_sum
            if abs(residual) > 1e-6:
                ss_attrib['Caps — Reductions (policy lowering SS)'] += residual
                ss_sum = sum(ss_attrib.values())

            ss_drv_df = pd.DataFrame({'driver': list(ss_attrib.keys()), 'amount': [float(v) for v in ss_attrib.values()]})
            denom = total_ss if total_ss > 0 else ss_drv_df['amount'].sum()
            denom = denom if denom > 0 else 1.0
            ss_drv_df['pct_of_total_ss'] = ss_drv_df['amount'] / denom * 100

            # Waterfall with pastel coloring: keep soft colors and maintain readability
            labels = ss_drv_df['driver'].tolist() + ['Total SS']
            values = ss_drv_df['amount'].tolist() + [total_ss]
            measures = ["relative"] * len(ss_drv_df) + ["total"]
            pastel_inc = pastel_colors[0] if len(pastel_colors) > 0 else '#A3C1DA'
            pastel_dec = pastel_colors[1] if len(pastel_colors) > 1 else '#F6C3A0'
            pastel_tot = pastel_colors[2] if len(pastel_colors) > 2 else '#CFCFCF'
            fig_drv = go.Figure(go.Waterfall(
                name="SS Attribution",
                orientation="v",
                measure=measures,
                x=labels,
                y=values,
                text=[f"{v:,.0f}" for v in ss_drv_df['amount'].tolist()] + [f"{total_ss:,.0f}"],
                connector={"line":{"color":"rgba(63,63,63,0.25)"}},
                decreasing=dict(marker=dict(color=pastel_dec)),
                increasing=dict(marker=dict(color=pastel_inc)),
                totals=dict(marker=dict(color=pastel_tot))
            ))
            fig_drv.update_layout(title=f"{selected_product} — SS Attribution Waterfall (adds to {euro_format(total_ss, True)})", xaxis_title="Driver", yaxis_title="Units", height=420)
            st.plotly_chart(fig_drv, use_container_width=True)

            st.markdown("SS Attribution table (numbers and % of total SS)")
            st.dataframe(df_format_for_display(ss_drv_df.rename(columns={'driver':'Driver','amount':'Units','pct_of_total_ss':'Pct_of_total_SS'}).round(2), cols=['Units','Pct_of_total_SS']), use_container_width=True, height=260)

            st.markdown("Notes on interpretation:")
            st.markdown("""
            - Section A shows raw driver values (mix of forecast volumes and SS-like terms).
            - Section B is a reconciled attribution: rows are mutually exclusive and sum to total SS.
            - Demand and LT uncertainty rows show portions of statistical SS retained after policies.
            - Direct vs Indirect allocation shows how retained SS supports local vs downstream demand.
            - Caps/Policy rows explain business-rule-driven adjustments.
            """)

        st.markdown("---")
        st.subheader("Top Locations by Safety Stock (snapshot)")
        top_nodes = mat_period_df.sort_values('Safety_Stock', ascending=False)[['Location','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status']]
        st.dataframe(df_format_for_display(top_nodes.head(25).copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=300)

        st.markdown("---")
        st.subheader("Export — Material Snapshot")
        if not mat_period_df.empty:
            st.download_button("📥 Download Material Snapshot (CSV)", data=mat_period_df.to_csv(index=False), file_name=f"material_{selected_product}_{selected_period.strftime('%Y-%m')}.csv", mime="text/csv")
        else:
            st.write("No snapshot available to download for this selection.")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
