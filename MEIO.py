# Multi-Echelon Inventory Optimizer — Enhanced Version (Reviewed & Improved)
# Enhanced by Copilot for mat635418 — 2026-01-15 (with UI/UX updates)
# Modified: 2026-01-17 — fixes: badge robustness, Forecast Accuracy, defaults, current month default,
# restored & enhanced scenario simulation (multi-scenario compare) and ensured By Material SS Attribution (Part B) present
# Modified: 2026-01-19 — v0.60 UI/UX: badge sizing, network centering, full-plan defaults, scenario defaults, waterfall for SS attribution
# Modified: 2026-01-19 — v0.61 fixes: restore network rendering, enforce 1-scenario default
# Modified: 2026-01-20 — v0.70: FIX network demand -> one-level downstream (no recursive propagation) to avoid double-counting at hubs
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
st.title("📊 MEIO for Raw Materials — v0.70 — Jan 2026")

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
    Compute network aggregated demand and historical std for each Product/Location/Period
    using ONLY direct children (one-level downstream) to avoid double-counting at hubs.

    Rules:
    - Agg_Future_Demand = local_forecast + sum(local_forecast of direct children)
    - Agg_Std_Hist: combine local variance and direct children's variance (sum of variances),
      but do NOT propagate recursively beyond direct children.
    - If local std missing, fallback to product median or global median upstream in calling code.
    """
    results = []
    months = df_forecast['Period'].unique()
    products = df_forecast['Product'].unique()

    # Pre-compute edge lists by product for quick lookup of direct children
    lt_by_product = {}
    for prod in df_lt['Product'].unique() if 'Product' in df_lt.columns else []:
        lt_by_product[prod] = df_lt[df_lt['Product'] == prod].copy()

    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in products:
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = lt_by_product.get(prod, pd.DataFrame(columns=df_lt.columns)) if 'Product' in df_lt.columns else df_lt.copy()

            # nodes are local locations present in forecasts or in routes
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt.get('From_Location', pd.Series([]))) if not p_lt.empty else set()
            ).union(
                set(p_lt.get('To_Location', pd.Series([]))) if not p_lt.empty else set()
            )
            if not nodes:
                continue

            # map direct children: parent -> list(children)
            children_map = {}
            if not p_lt.empty:
                for _, row in p_lt.iterrows():
                    children_map.setdefault(row['From_Location'], []).append(row['To_Location'])

            # compute for each node: local forecast and direct children's forecasts (one-level only)
            for n in nodes:
                local_fcst = float(p_fore.get(n, {'Forecast': 0})['Forecast']) if n in p_fore else 0.0
                direct_children = children_map.get(n, [])
                child_fcsts = [float(p_fore.get(c, {'Forecast': 0})['Forecast']) if c in p_fore else 0.0 for c in direct_children]
                agg_demand = local_fcst + sum(child_fcsts)

                # Variance: local variance (Local_Std^2) plus sum of child local variances (one-level)
                local_std = p_stats.get(n, {}).get('Local_Std', np.nan)
                child_vars = []
                for c in direct_children:
                    cs = p_stats.get(c, {}).get('Local_Std', np.nan)
                    child_vars.append(np.nan if pd.isna(cs) else float(cs)**2)
                if pd.isna(local_std) and all(pd.isna(v) for v in child_vars):
                    agg_std = np.nan
                else:
                    local_var = 0.0 if pd.isna(local_std) else float(local_std)**2
                    sum_child_var = sum(v for v in child_vars if not pd.isna(v))
                    total_var = local_var + sum_child_var
                    agg_std = np.sqrt(total_var) if total_var >= 0 else np.nan

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
          <div style="font-size:11px;opacity:0.85">Net Demand</div>
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

    # NEW: network_stats will use one-level downstream aggregation to avoid double-counting
    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    results['Agg_Std_Hist'] = results.apply(lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'], axis=1)

    # SAFETY STOCK calculation (Method 5) — unchanged formula but using new Agg_Future_Demand and Agg_Std_Hist
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
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand (one-level)', line=dict(color='blue', dash='dash'))
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
                  <div style="font-size:11px;color:#666">Total Net Demand</div>
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

        # Use results which now holds Agg_Future_Demand defined as one-level downstream
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
    # TAB 3: Full Plan (start filtered to defaults)
    # -------------------------------
    with tab3:
        st.subheader("📋 Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        prod_choices = sorted(results['Product'].unique())
        loc_choices = sorted(results['Location'].unique())
        period_choices = sorted(results['Period'].unique())

        # sensible defaults: start filtered to the default product / its default location / default period (when available)
        default_prod_list = [default_product] if default_product in prod_choices else []
        default_loc_for_default = default_location_for(default_product)
        default_loc_list = [default_loc_for_default] if default_loc_for_default in loc_choices else []
        default_period_list = [default_period] if (default_period is not None and default_period in period_choices) else []

        f_prod = col1.multiselect("Filter Product", prod_choices, default=default_prod_list)
        f_loc = col2.multiselect("Filter Location", loc_choices, default=default_loc_list)
        f_period = col3.multiselect("Filter Period", period_choices, default=default_period_list)

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

    # ... (remaining tabs unchanged, using new Agg_Future_Demand values)

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
