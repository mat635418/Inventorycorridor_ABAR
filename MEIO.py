# Multi-Echelon Inventory Optimizer — Enhanced Version (Reviewed & Improved)
# Enhanced by Copilot for mat635418 — 2026-01-15 (with UI/UX updates)
# Modified: 2026-01-16/2026-01-17 — removed global filtering, fixed historical FC vs Actuals robustness (badge + tab5)
# Added: default selection NOKANDO2 / BEEX across tabs and default period = current month when available
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

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="MEIO for Raws", layout="wide")
st.title("📊 MEIO for Raw Materials — v0.55 — Jan 2026 (standard filtering)")

# -------------------------------
# HELPERS / FORMATTING
# -------------------------------
def clean_numeric(series):
    """
    Robust numeric cleaning:
    - Accepts numbers with thousand separators and decimals.
    - Converts "(123)" -> -123
    - Converts empty strings or '-' to NaN (not 0).
    - Leaves negative signs intact.
    Returns numeric (float) with NaN for unparsable values.
    """
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
    Formats numbers with '.' as thousand separator and ',' as decimal separator.
    Examples: 1234.5 -> '1.234,50' (if always_two_decimals True)
    """
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
    """
    Returns a copy of df with selected numeric columns formatted to euro_format strings.
    If cols is None, attempt to format common numeric columns.
    """
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
    Propagates demand and variance up the supply chain network.
    """
    results = []
    months = df_forecast['Period'].unique()
    products = df_forecast['Product'].unique()
    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in products:
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod] if 'Product' in df_lt.columns else df_lt.copy()

            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt.get('From_Location', pd.Series([]))) if not p_lt.empty else set()
            ).union(
                set(p_lt.get('To_Location', pd.Series([]))) if not p_lt.empty else set()
            )
            if not nodes:
                continue

            agg_demand = {n: float(p_fore.get(n, {'Forecast': 0})['Forecast']) for n in nodes}
            agg_var = {}
            for n in nodes:
                local_std = p_stats.get(n, {}).get('Local_Std', np.nan)
                if pd.isna(local_std):
                    agg_var[n] = np.nan
                else:
                    agg_var[n] = float(local_std)**2

            children = {}
            if not p_lt.empty:
                for _, row in p_lt.iterrows():
                    children.setdefault(row['From_Location'], []).append(row['To_Location'])

            for _ in range(30):
                changed = False
                for parent in nodes:
                    child_list = children.get(parent, [])
                    if child_list:
                        new_d = float(p_fore.get(parent, {'Forecast': 0})['Forecast']) + sum(agg_demand.get(c, 0) for c in child_list)
                        child_vars = [agg_var.get(c, np.nan) for c in child_list]
                        if any(pd.isna(v) for v in child_vars) or pd.isna(agg_var.get(parent, np.nan)):
                            new_v = np.nan
                        else:
                            new_v = float(p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(child_vars)
                        if abs(new_d - agg_demand[parent]) > 0.0001 or (pd.isna(agg_var[parent]) and not pd.isna(new_v)):
                            agg_demand[parent] = new_d
                            agg_var[parent] = new_v
                            changed = True
                if not changed:
                    break

            for n in nodes:
                results.append({
                    'Product': prod,
                    'Location': n,
                    'Period': month,
                    'Agg_Future_Demand': agg_demand.get(n, 0.0),
                    'Agg_Std_Hist': np.sqrt(agg_var[n]) if (n in agg_var and not pd.isna(agg_var[n])) else np.nan
                })
    return pd.DataFrame(results)

def render_selection_badge(product=None, location=None, df_context=None, small=False):
    """
    Renders the consistent blue badge used in multiple tabs.
    - df_context may be a slice from `results` (preferred) or from `hist`.
    The function is defensive and accepts alternative column names (Forecast_Hist).
    """
    if product is None or product == "":
        return

    def _sum_candidates(df, candidates):
        """Return sum of the first existing candidate column in df, else 0.0"""
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
    total_net = _sum_candidates(df_context, ['Agg_Future_Demand', 'Agg_Future_Demand'])  # second duplicate just for clarity
    total_ss = _sum_candidates(df_context, ['Safety_Stock'])

    badge_html = f"""
    <div style="background:#0b3d91;padding:14px;border-radius:8px;color:white;">
      <div style="font-size:12px;opacity:0.85">Selected</div>
      <div style="font-size:16px;font-weight:700;margin-bottom:6px">{product}{(' — ' + location) if location else ''}</div>
      <div style="display:flex;gap:8px;align-items:center;">
        <div style="background:#ffffff22;padding:8px;border-radius:6px;min-width:140px;">
          <div style="font-size:11px;opacity:0.85">Fcst (Local)</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_fcst, True)}</div>
        </div>
        <div style="background:#ffffff22;padding:8px;border-radius:6px;min-width:140px;">
          <div style="font-size:11px;opacity:0.85">Net Demand</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_net, True)}</div>
        </div>
        <div style="background:#00b0f622;padding:8px;border-radius:6px;min-width:140px;">
          <div style="font-size:11px;opacity:0.85">SS (Current)</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_ss, True)}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

# -------------------------------
# SIDEBAR & FILE LOADING LOGIC
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

if s_file:
    st.sidebar.success(f"✅ Sales Loaded: {getattr(s_file,'name', s_file)}")
if d_file:
    st.sidebar.success(f"✅ Demand Loaded: {getattr(d_file,'name', d_file)}")
if lt_file:
    st.sidebar.success(f"✅ Lead Time Loaded: {getattr(lt_file,'name', lt_file)}")

# -------------------------------
# MAIN LOGIC
# -------------------------------
# Defaults requested by user
DEFAULT_PRODUCT_CHOICE = "NOKANDO2"
DEFAULT_LOCATION_CHOICE = "BEEX"
# current month as period timestamp (month start)
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
        st.error(f"sales.csv missing columns: {needed_sales_cols - set(df_s.columns)}")
        st.stop()
    if not needed_demand_cols.issubset(set(df_d.columns)):
        st.error(f"demand.csv missing columns: {needed_demand_cols - set(df_d.columns)}")
        st.stop()
    if not needed_lt_cols.issubset(set(df_lt.columns)):
        st.error(f"leadtime.csv missing columns: {needed_lt_cols - set(df_lt.columns)}")
        st.stop()

    # Normalize period columns to month-start timestamps
    df_s['Period'] = pd.to_datetime(df_s['Period'], errors='coerce')
    df_d['Period'] = pd.to_datetime(df_d['Period'], errors='coerce')
    df_s['Period'] = df_s['Period'].dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = df_d['Period'].dt.to_period('M').dt.to_timestamp()

    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    global_median_std = stats['Local_Std'].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0

    prod_medians = stats.groupby('Product')['Local_Std'].median().to_dict()
    def fill_local_std(row):
        if not pd.isna(row['Local_Std']) and row['Local_Std'] > 0:
            return row['Local_Std']
        pm = prod_medians.get(row['Product'], np.nan)
        return pm if not pd.isna(pm) else global_median_std
    stats['Local_Std'] = stats.apply(fill_local_std, axis=1)

    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)

    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']],
                       on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')

    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    results['Agg_Std_Hist'] = results.apply(
        lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'],
        axis=1
    )

    # SAFETY STOCK — SS METHOD 5 (vectorized)
    results['Pre_Rule_SS'] = z * np.sqrt(
        (results['Agg_Std_Hist']**2 / float(days_per_month)) * results['LT_Mean'] +
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / float(days_per_month))**2
    )

    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['Pre_Rule_SS']

    # Rule: Zero if no NETWORK demand
    results['Pre_Zero_SS'] = results['Safety_Stock']
    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0

    # Rule: Capping based on NETWORK demand
    results['Pre_Cap_SS'] = results['Safety_Stock']
    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100.0, cap_range[1] / 100.0
        l_lim = results['Agg_Future_Demand'] * l_cap
        u_lim = results['Agg_Future_Demand'] * u_cap

        high_mask = (results['Safety_Stock'] > u_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)')
        results.loc[high_mask, 'Adjustment_Status'] = 'Capped (High)'
        low_mask = (results['Safety_Stock'] < l_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)') & (results['Agg_Future_Demand'] > 0)
        results.loc[low_mask, 'Adjustment_Status'] = 'Capped (Low)'

        results['Safety_Stock'] = results['Safety_Stock'].clip(lower=l_lim, upper=u_lim)

    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # ACCURACY DATA (LOCAL)
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)) * 100
    hist['APE_%'] = hist['APE_%'].fillna(0)
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    hist_net = (
        df_s.groupby(['Product', 'Period'], as_index=False)
            .agg(Network_Consumption=('Consumption', 'sum'),
                 Network_Forecast_Hist=('Forecast', 'sum'))
    )

    # -------------------------------
    # Prepare lists + defaults
    # -------------------------------
    all_products = sorted(results['Product'].unique().tolist())
    # Determine defaults: prefer provided default names, else fallback to first available
    default_product = DEFAULT_PRODUCT_CHOICE if DEFAULT_PRODUCT_CHOICE in all_products else (all_products[0] if all_products else "")
    # We'll set default location per product when needed
    def default_location_for(prod):
        locs = sorted(results[results['Product'] == prod]['Location'].unique().tolist())
        return DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in locs else (locs[0] if locs else "")
    # Period choices and default as CURRENT_MONTH_TS when present
    all_periods = sorted(results['Period'].unique().tolist())
    default_period = CURRENT_MONTH_TS if CURRENT_MONTH_TS in all_periods else (all_periods[-1] if all_periods else None)

    # -------------------------------
    # TABS
    # -------------------------------
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
        left, right = st.columns([3, 1])
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
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
            ])
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units')
            st.plotly_chart(fig, use_container_width=True)

        with right:
            render_selection_badge(product=sku, location=loc if loc != "(no location)" else None, df_context=plot_df)
            ssum = float(plot_df['Safety_Stock'].sum()) if not plot_df.empty else 0.0
            ndsum = float(plot_df['Agg_Future_Demand'].sum()) if not plot_df.empty else 0.0
            extra_html = f"""
            <div style="padding-top:8px;">
              <div style="font-size:12px;color:#333">Quick Totals</div>
              <div style="display:flex;gap:8px;margin-top:6px;">
                <div style="background:#f7f9fc;padding:8px;border-radius:6px;min-width:120px;">
                  <div style="font-size:11px;color:#666">Total SS (sku/loc)</div>
                  <div style="font-size:13px;font-weight:600;color:#0b3d91">{euro_format(ssum, True)}</div>
                </div>
                <div style="background:#f7f9fc;padding:8px;border-radius:6px;min-width:120px;">
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
            except ValueError:
                period_index = len(period_choices)-1
            chosen_period = st.selectbox("Period", period_choices, index=period_index, key="network_period")
        else:
            chosen_period = st.selectbox("Period", [CURRENT_MONTH_TS], index=0, key="network_period")

        render_selection_badge(product=sku, location=None, df_context=results[(results['Product']==sku)&(results['Period']==chosen_period)])

        label_data = results[results['Period'] == chosen_period].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku] if 'Product' in df_lt.columns else df_lt.copy()

        net = Network(height="1200px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")

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
                bg = '#dcedc8'
                border = '#8bc34a'
                font_color = '#0b3d91'
                size = 14
            elif n == 'BEEX' or n == 'LUEX':
                bg = '#bbdefb'
                border = '#64b5f6'
                font_color = '#0b3d91'
                size = 14
            else:
                if used:
                    bg = '#fff9c4'
                    border = '#fbc02d'
                    font_color = '#222222'
                    size = 12
                else:
                    bg = '#f0f0f0'
                    border = '#cccccc'
                    font_color = '#9e9e9e'
                    size = 10
            lbl = f"{n}\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            net.add_node(n, label=lbl, title=lbl,
                         color={'background': bg, 'border': border},
                         shape='box', font={'color': font_color, 'size': size})

        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
            to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
            if not from_used and not to_used:
                edge_color = '#dddddd'
            else:
                edge_color = '#888888'
            label = f"{int(r.get('Lead_Time_Days', 0))}d" if not pd.isna(r.get('Lead_Time_Days', 0)) else ""
            net.add_edge(from_n, to_n, label=label, color=edge_color)

        net.set_options("""
        var options = {
          "physics": {"stabilization": {"iterations": 200}},
          "nodes": {"borderWidthSelected":2},
          "interaction": {"hover":true}
        }
        """)
        tmpfile = "net.html"
        net.save_graph(tmpfile)
        with open(tmpfile, 'r', encoding='utf-8') as f:
            html = f.read()
        components.html(html, height=900, scrolling=True)

    # -------------------------------
    # TAB 3: Full Plan
    # -------------------------------
    with tab3:
        st.subheader("📋 Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        f_prod = col1.multiselect("Filter Product", sorted(results['Product'].unique()))
        f_loc = col2.multiselect("Filter Location", sorted(results['Location'].unique()))
        f_period = col3.multiselect("Filter Period", sorted(results['Period'].unique()))

        filtered = results.copy()
        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]

        filtered = filtered.sort_values('Safety_Stock', ascending=False)

        if (filtered['Product'].nunique() == 1) and (filtered['Location'].nunique() == 1) and not filtered.empty:
            badge_prod = filtered['Product'].iloc[0]
            badge_loc = filtered['Location'].iloc[0]
            badge_df = filtered
            render_selection_badge(product=badge_prod, location=badge_loc, df_context=badge_df)
        elif not filtered.empty:
            badge_prod = filtered['Product'].iloc[0]
            badge_df = filtered[filtered['Product'] == badge_prod]
            render_selection_badge(product=badge_prod, location=None, df_context=badge_df)

        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
        disp = df_format_for_display(filtered[display_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'], two_decimals_cols=['Forecast'])
        st.dataframe(disp, use_container_width=True, height=700)

        csv_buf = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # -------------------------------
    # TAB 4: Efficiency Analysis
    # -------------------------------
    with tab4:
        st.subheader("⚖️ Efficiency & Policy Analysis")
        sku_default = default_product
        sku_index = all_products.index(sku_default) if sku_default in all_products else 0
        sku = st.selectbox("Material", all_products, index=sku_index, key="eff_sku")
        # choose snapshot period = current month if available
        snapshot_period = default_period if default_period in all_periods else (all_periods[-1] if all_periods else None)
        if snapshot_period is None:
            st.warning("No period data available for Efficiency Analysis.")
            eff = results[(results['Product'] == sku)].copy()
        else:
            eff = results[(results['Product'] == sku) & (results['Period'] == snapshot_period)].copy()

        eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Agg_Future_Demand'].replace(0, np.nan)).fillna(0)
        total_ss_sku = eff['Safety_Stock'].sum()
        total_net_demand_sku = eff['Agg_Future_Demand'].sum()
        sku_ratio = total_ss_sku / total_net_demand_sku if total_net_demand_sku > 0 else 0
        all_res = results[results['Period'] == snapshot_period] if snapshot_period is not None else results
        global_ratio = all_res['Safety_Stock'].sum() / all_res['Agg_Future_Demand'].replace(0, np.nan).sum() if not all_res.empty else 0

        render_selection_badge(product=sku, location=None, df_context=eff)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Network Ratio ({sku})", f"{sku_ratio:.2f}")
        m2.metric("Global Network Ratio (All Items)", f"{global_ratio:.2f}")
        m3.metric("Total SS for Material", euro_format(int(total_ss_sku), True))

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_eff = px.scatter(
                eff, x="Agg_Future_Demand", y="Safety_Stock", color="Adjustment_Status",
                size="SS_to_FCST_Ratio", hover_name="Location",
                color_discrete_map={'Optimal (Statistical)': '#00CC96', 'Capped (High)': '#EF553B','Capped (Low)': '#636EFA', 'Forced to Zero': '#AB63FA'},
                title="Policy Impact & Efficiency Ratio (Bubble Size = SS_to_FCST_Ratio)"
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with c2:
            st.markdown("**Status Breakdown**")
            st.table(eff['Adjustment_Status'].value_counts())

            st.markdown("**Top Nodes by Safety Stock (snapshot)**")
            eff_top = eff.sort_values('Safety_Stock', ascending=False)
            st.dataframe(
                df_format_for_display(
                    eff_top[['Location', 'Adjustment_Status', 'Safety_Stock', 'SS_to_FCST_Ratio']],
                    cols=['Safety_Stock'],
                    two_decimals_cols=['Safety_Stock']
                ).head(10),
                use_container_width=True
            )

    # -------------------------------
    # TAB 5: Forecast Accuracy (robust + use results for badge)
    # -------------------------------
    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")
        h_sku_default = default_product
        h_sku_index = all_products.index(h_sku_default) if h_sku_default in all_products else 0
        h_sku = st.selectbox("Select Product", all_products, index=h_sku_index, key="h1")
        h_loc_opts = sorted(results[results['Product'] == h_sku]['Location'].unique().tolist())
        # if no locations found in results for selected product, fall back to historical locs
        if not h_loc_opts:
            h_loc_opts = sorted(hist[hist['Product'] == h_sku]['Location'].unique().tolist())
        if not h_loc_opts:
            h_loc_opts = ["(no location)"]
        # prefer default location
        h_loc_default = DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in h_loc_opts else (h_loc_opts[0] if h_loc_opts else "(no location)")
        h_loc_index = h_loc_opts.index(h_loc_default) if h_loc_default in h_loc_opts else 0
        h_loc = st.selectbox("Select Location", h_loc_opts, index=h_loc_index, key="h2")

        # show badge: use the planning 'results' table as context so columns Forecast/Agg_Future_Demand/Safety_Stock exist
        if h_loc != "(no location)":
            badge_df = results[(results['Product'] == h_sku) & (results['Location'] == h_loc)]
        else:
            badge_df = results[results['Product'] == h_sku]
        render_selection_badge(product=h_sku, location=(h_loc if h_loc != "(no location)" else None), df_context=badge_df)

        # Prepare historical dataframe selection
        hdf = hist.copy()
        if h_loc != "(no location)":
            hdf = hdf[(hdf['Product'] == h_sku) & (hdf['Location'] == h_loc)].sort_values('Period')
        else:
            hdf = hdf[hdf['Product'] == h_sku].sort_values('Period')

        if not hdf.empty:
            k1, k2, k3 = st.columns(3)
            denom_consumption = hdf['Consumption'].replace(0, np.nan).sum()
            if denom_consumption > 0:
                wape_val = (hdf['Abs_Error'].sum() / denom_consumption * 100)
                bias_val = (hdf['Deviation'].sum() / denom_consumption * 100)
                k1.metric("WAPE (%)", f"{wape_val:.1f}")
                k2.metric("Bias (%)", f"{bias_val:.1f}")
            else:
                k1.metric("WAPE (%)", "N/A")
                k2.metric("Bias (%)", "N/A")
            avg_acc = hdf['Accuracy_%'].mean() if not hdf['Accuracy_%'].isna().all() else np.nan
            k3.metric("Avg Accuracy (%)", f"{avg_acc:.1f}" if not np.isnan(avg_acc) else "N/A")

            fig_hist = go.Figure([
                go.Scatter(x=hdf['Period'], y=hdf['Consumption'], name='Actuals', line=dict(color='black')),
                go.Scatter(x=hdf['Period'], y=hdf['Forecast_Hist'], name='Forecast', line=dict(color='blue', dash='dot')),
            ])
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("🌐 Aggregated Network History (Selected Product)")
            net_table = (
                hist_net[hist_net['Product'] == h_sku]
                        .merge(hdf[['Period']].drop_duplicates(), on='Period', how='inner')
                        .sort_values('Period')
                        .drop(columns=['Product'])
            )

            if not net_table.empty:
                net_table['Net_Abs_Error'] = (net_table['Network_Consumption'] - net_table['Network_Forecast_Hist']).abs()
                denom_net = net_table['Network_Consumption'].replace(0, np.nan).sum()
                net_wape = (net_table['Net_Abs_Error'].sum() / denom_net * 100) if denom_net > 0 else np.nan
            else:
                net_wape = np.nan

            c_net1, c_net2 = st.columns([3, 1])
            with c_net1:
                if not net_table.empty:
                    st.dataframe(df_format_for_display(net_table[['Period', 'Network_Consumption', 'Network_Forecast_Hist']].copy(),
                                                       cols=['Network_Consumption','Network_Forecast_Hist'], two_decimals_cols=['Network_Consumption']), use_container_width=True, height=500)
                else:
                    st.write("No aggregated network history available for the chosen selection.")
            with c_net2:
                c_val = f"{net_wape:.1f}" if not np.isnan(net_wape) else "N/A"
                st.metric("Network WAPE (%)", c_val)

            st.subheader("📊 Detailed Accuracy by Month")
            st.dataframe(df_format_for_display(hdf[['Period','Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']].copy(),
                                              cols=['Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']), use_container_width=True, height=500)
        else:
            st.warning("⚠️ No historical sales data found for this selection. Accuracy metrics cannot be calculated.")

    # --------------------------------
    # TAB 6: Calculation Trace & Simulation
    # --------------------------------
    with tab6:
        st.header("🧮 Transparent Calculation Engine")
        st.write("Select a specific node and period to see exactly how the Safety Stock number was derived and simulate impacts interactively.")

        c1, c2, c3 = st.columns(3)
        calc_sku_default = default_product
        calc_sku_index = all_products.index(calc_sku_default) if calc_sku_default in all_products else 0
        calc_sku = c1.selectbox("Select Product", all_products, index=calc_sku_index, key="c_sku")
        avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique().tolist())
        if not avail_locs:
            avail_locs = ["(no location)"]
        calc_loc_default = DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in avail_locs else (avail_locs[0] if avail_locs else "(no location)")
        calc_loc_index = avail_locs.index(calc_loc_default) if calc_loc_default in avail_locs else 0
        calc_loc = c2.selectbox("Select Location", avail_locs, index=calc_loc_index, key="c_loc")
        avail_periods = all_periods
        if avail_periods:
            try:
                calc_period_index = avail_periods.index(default_period)
            except ValueError:
                calc_period_index = len(avail_periods)-1
            calc_period = c3.selectbox("Select Period", avail_periods, index=calc_period_index, key="c_period")
        else:
            calc_period = c3.selectbox("Select Period", [CURRENT_MONTH_TS], index=0, key="c_period")

        row = results[
            (results['Product'] == calc_sku) &
            (results['Location'] == calc_loc) &
            (results['Period'] == calc_period)
        ]
        if row.empty:
            st.warning("Selection not found in results.")
        else:
            row = row.iloc[0]
            render_selection_badge(product=calc_sku, location=calc_loc if calc_loc != "(no location)" else None, df_context=results[(results['Product']==calc_sku)&(results['Location']==calc_loc)&(results['Period']==calc_period)])

            st.markdown("---")
            st.subheader("1. Actual Inputs (Frozen)")
            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("Service Level", f"{service_level*100:.2f}%", help=f"Z-Score: {z:.4f}")
            i2.metric("Network Demand (D, monthly)", euro_format(row['Agg_Future_Demand'], True), help="Aggregated Future Demand (monthly)")
            i3.metric("Network Std Dev (σ_D, monthly)", euro_format(row['Agg_Std_Hist'], True), help="Aggregated Historical Std Dev (monthly totals)")
            i4.metric("Avg Lead Time (L)", f"{row['LT_Mean']} days")
            i5.metric("LT Std Dev (σ_L)", f"{row['LT_Std']} days")

            st.subheader("2. Statistical Calculation (Actual)")
            term1_demand_var = (row['Agg_Std_Hist']**2 / float(days_per_month)) * row['LT_Mean']
            term2_supply_var = (row['LT_Std']**2) * ((row['Agg_Future_Demand'] / float(days_per_month))**2)
            combined_sd = np.sqrt(term1_demand_var + term2_supply_var)
            raw_ss_calc = z * combined_sd

            st.markdown("Using Safety Stock Method 5 (daily form):")
            st.latex(r"SS_{\text{raw}} = Z \times \sqrt{\,\sigma_D^2 \times L \;+\; \sigma_L^2 \times D^2\,}")
            st.markdown("Where σ_D and D are daily values (converted from monthly inputs in the dataset).")

            st.markdown("**Step-by-Step Substitution (values used):**")
            st.code(f"""
1. σ_D_daily^2 (from monthly agg std) = ({euro_format(row['Agg_Std_Hist'], True)})^2 / {days_per_month}
   Demand Component = σ_D_daily^2 * L = {euro_format(term1_demand_var, True)}
2. Supply Component = σ_L^2 * D_daily^2 = ({euro_format(row['LT_Std'], True)})^2 * ({euro_format(row['Agg_Future_Demand'], True)} / {days_per_month})^2
   = {euro_format(term2_supply_var, True)}
3. Combined Variance = {euro_format(term1_demand_var, True)} + {euro_format(term2_supply_var, True)}
   = {euro_format(term1_demand_var + term2_supply_var, True)}
4. Combined Std Dev = sqrt(Combined Variance)
   = {euro_format(combined_sd, True)}
5. Raw SS = {z:.4f} (Z-Score) * {euro_format(combined_sd, True)}
   = {euro_format(raw_ss_calc, True)} units
""")
            st.info(f"🧮 **Resulting Statistical SS (Method 5):** {euro_format(raw_ss_calc, True)} units")

            # SIMULATION (interactive visuals)
            st.markdown("---")
            st.subheader("3. What-If Simulation (interactive visuals)")
            sim_cols = st.columns(3)
            sim_sl = sim_cols[0].slider(
                "Simulated Service Level (%)",
                min_value=50.0, max_value=99.9,
                value=service_level*100,
                key=f"sim_sl_{calc_sku}_{calc_loc}"
            )
            sim_lt = sim_cols[1].slider(
                "Simulated Avg Lead Time (Days)",
                min_value=0.0, max_value=max(30.0, row['LT_Mean']*2),
                value=float(row['LT_Mean']),
                key=f"sim_lt_{calc_sku}_{calc_loc}"
            )
            sim_lt_std = sim_cols[2].slider(
                "Simulated LT Variability (Days)",
                min_value=0.0, max_value=max(10.0, row['LT_Std']*2),
                value=float(row['LT_Std']),
                key=f"sim_lt_std_{calc_sku}_{calc_loc}"
            )

            sim_z = norm.ppf(sim_sl / 100.0)
            sim_ss = sim_z * np.sqrt(
                (row['Agg_Std_Hist']**2 / float(days_per_month)) * sim_lt +
                (sim_lt_std**2) * (row['Agg_Future_Demand'] / float(days_per_month))**2
            )

            bar_df = pd.DataFrame({
                'label': ['Pre_Rule_SS', 'Implemented SS', 'Simulated SS'],
                'value': [row['Pre_Rule_SS'], row['Safety_Stock'], sim_ss]
            })
            fig_bar = go.Figure()
            colors = ['#636EFA', '#00CC96', '#EF553B']
            fig_bar.add_trace(go.Bar(x=bar_df['label'], y=bar_df['value'], marker_color=colors))
            fig_bar.update_layout(title="SS Comparison: Statistical vs Implemented vs Simulated", yaxis_title="Units")

            sl_range = np.linspace(50.0, 99.9, 50)
            ss_curve = []
            for slev in sl_range:
                zz = norm.ppf(slev/100.0)
                val = zz * np.sqrt(
                    (row['Agg_Std_Hist']**2 / float(days_per_month)) * sim_lt +
                    (sim_lt_std**2) * (row['Agg_Future_Demand'] / float(days_per_month))**2
                )
                ss_curve.append(val)
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=sl_range, y=ss_curve, mode='lines', line=dict(color='#0b3d91')))
            fig_curve.add_vline(x=sim_sl, line_dash="dash", line_color="red", annotation_text=f"Selected SL {sim_sl:.1f}%", annotation_position="top right")
            fig_curve.update_layout(title="Simulated SS Sensitivity to Service Level (other inputs fixed)", xaxis_title="Service Level (%)", yaxis_title="Simulated SS (units)")

            chart_col1, chart_col2 = st.columns([1,1])
            with chart_col1:
                st.plotly_chart(fig_bar, use_container_width=True)
            with chart_col2:
                st.plotly_chart(fig_curve, use_container_width=True)

            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Original SS (Actual)", euro_format(row['Pre_Rule_SS'], True))
            res_col2.metric(
                "Simulated SS (New)",
                euro_format(sim_ss, True),
                delta=euro_format(sim_ss - row['Pre_Rule_SS'], True),
                delta_color="inverse"
            )
            if sim_ss < row['Pre_Rule_SS']:
                st.success(f"📉 Reducing uncertainty could lower inventory by **{euro_format(row['Pre_Rule_SS'] - sim_ss, True)}** units.")
            elif sim_ss > row['Pre_Rule_SS']:
                st.warning(f"📈 Increasing service or lead time requires **{euro_format(sim_ss - row['Pre_Rule_SS'], True)}** more units.")

            # BUSINESS RULES
            st.markdown("---")
            st.subheader("4. Business Rules Application")
            col_rule_1, col_rule_2 = st.columns(2)
            with col_rule_1:
                st.markdown("**Check 1: Zero Demand Rule**")
                if zero_if_no_net_fcst and row['Agg_Future_Demand'] <= 0:
                    st.error("❌ Network Demand is 0. SS Forced to 0.")
                else:
                    st.success("✅ Network Demand exists. Keep Statistical SS.")
            with col_rule_2:
                st.markdown("**Check 2: Capping (Min/Max)**")
                if apply_cap:
                    lower_limit = row['Agg_Future_Demand'] * (cap_range[0]/100)
                    upper_limit = row['Agg_Future_Demand'] * (cap_range[1]/100)
                    st.write(f"Constraint: {int(cap_range[0])}% to {int(cap_range[1])}% of Demand")
                    st.write(f"Range: [{euro_format(lower_limit, True)}, {euro_format(upper_limit, True)}]")
                    if raw_ss_calc > upper_limit:
                        st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) > Max Cap ({euro_format(upper_limit, True)}). Capping downwards.")
                    elif raw_ss_calc < lower_limit and row['Agg_Future_Demand'] > 0:
                        st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) < Min Cap ({euro_format(lower_limit, True)}). Buffering upwards.")
                    else:
                        st.success("✅ Raw SS is within efficient boundaries.")
                else:
                    st.write("Capping logic disabled.")

    # -------------------------------
    # TAB 7: By Material
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
            except ValueError:
                sel_period_index = len(period_choices)-1
            selected_period = st.selectbox("Select Period to Snapshot", period_choices, index=sel_period_index, key="mat_period")
        else:
            selected_period = st.selectbox("Select Period to Snapshot", [CURRENT_MONTH_TS], index=0, key="mat_period")

        mat_period_df = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)].copy()
        total_forecast = mat_period_df['Forecast'].sum()
        total_net = mat_period_df['Agg_Future_Demand'].sum()
        total_ss = mat_period_df['Safety_Stock'].sum()
        nodes_count = mat_period_df['Location'].nunique()
        avg_ss_per_node = (mat_period_df['Safety_Stock'].mean() if nodes_count > 0 else 0)

        render_selection_badge(product=selected_product, location=None, df_context=mat_period_df)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Local Forecast", euro_format(total_forecast, True))
        k2.metric("Total Network Demand", euro_format(total_net, True))
        k3.metric("Total Safety Stock (sum nodes)", euro_format(total_ss, True))
        k4.metric("Nodes", f"{nodes_count}")
        k5.metric("Avg SS per Node", euro_format(avg_ss_per_node, True))

        st.markdown("### Why do we carry this SS? — 8 Reasons breakdown (aggregated for selected material)")
        if mat_period_df.empty:
            st.warning("No data for this material/period.")
        else:
            mat = mat_period_df.copy()
            mat['LT_Mean'] = mat['LT_Mean'].fillna(0)
            mat['LT_Std'] = mat['LT_Std'].fillna(0)
            mat['Agg_Std_Hist'] = mat['Agg_Std_Hist'].fillna(0)
            mat['Pre_Rule_SS'] = mat['Pre_Rule_SS'].fillna(0)
            mat['Safety_Stock'] = mat['Safety_Stock'].fillna(0)
            mat['Forecast'] = mat['Forecast'].fillna(0)
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

            drv_df = pd.DataFrame({
                'driver': list(raw_drivers.keys()),
                'amount': [float(v) for v in raw_drivers.values()]
            })
            drv_denom = drv_df['amount'].sum()
            drv_df['pct_of_total_ss'] = drv_df['amount'] / (drv_denom if drv_denom > 0 else 1.0) * 100

            st.markdown("#### A. Original — Raw driver values (interpretation view)")
            fig_drv_raw = go.Figure()
            fig_drv_raw.add_trace(go.Bar(
                x=drv_df['driver'],
                y=drv_df['amount'],
                marker_color=px.colors.qualitative.Pastel
            ))
            annotations_raw = []
            for idx, rowd in drv_df.iterrows():
                annotations_raw.append(dict(x=rowd['driver'], y=rowd['amount'], text=f"{rowd['pct_of_total_ss']:.1f}%", showarrow=False, yshift=8))
            fig_drv_raw.update_layout(title=f"{selected_product} — Raw Drivers (not SS-attribution)", xaxis_title="Driver", yaxis_title="Units", annotations=annotations_raw, height=420)
            st.plotly_chart(fig_drv_raw, use_container_width=True)

            st.markdown("Driver table (raw numbers and % of raw-sum)")
            st.dataframe(df_format_for_display(drv_df.rename(columns={'driver':'Driver','amount':'Units','pct_of_total_ss':'Pct_of_raw_sum'}).round(2), cols=['Units','Pct_of_raw_sum']), use_container_width=True, height=260)

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
                if r['pre_ss'] <= 0:
                    return 0.0
                if r['is_forced_zero'] or r['is_b616_override']:
                    return 0.0
                return float(r['Safety_Stock']) / float(r['pre_ss']) if r['pre_ss'] > 0 else 0.0

            per_node['retained_ratio'] = per_node.apply(retained_ratio_calc, axis=1)
            per_node['retained_demand'] = per_node['demand_share'] * per_node['retained_ratio']
            per_node['retained_lt'] = per_node['lt_share'] * per_node['retained_ratio']
            per_node['retained_stat_total'] = per_node['retained_demand'] + per_node['retained_lt']

            def direct_frac_calc(r):
                if r['Agg_Future_Demand'] > 0:
                    return float(r['Forecast']) / float(r['Agg_Future_Demand'])
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

            for k in ss_attrib:
                ss_attrib[k] = float(ss_attrib[k])

            ss_sum = sum(ss_attrib.values())
            residual = float(total_ss) - ss_sum
            if abs(residual) > 1e-6:
                ss_attrib['Caps — Reductions (policy lowering SS)'] += residual
                ss_sum = sum(ss_attrib.values())

            ss_drv_df = pd.DataFrame({
                'driver': list(ss_attrib.keys()),
                'amount': [float(v) for v in ss_attrib.values()]
            })
            denom = total_ss if total_ss > 0 else ss_drv_df['amount'].sum()
            denom = denom if denom > 0 else 1.0
            ss_drv_df['pct_of_total_ss'] = ss_drv_df['amount'] / denom * 100

            fig_drv = go.Figure()
            fig_drv.add_trace(go.Bar(
                x=ss_drv_df['driver'],
                y=ss_drv_df['amount'],
                marker_color=px.colors.qualitative.Pastel
            ))
            annotations = []
            for idx, rowd in ss_drv_df.iterrows():
                annotations.append(dict(x=rowd['driver'], y=rowd['amount'], text=f"{rowd['pct_of_total_ss']:.1f}%", showarrow=False, yshift=8))
            fig_drv.update_layout(title=f"{selected_product} — SS Attribution (adds to {euro_format(total_ss, True)})", xaxis_title="Driver", yaxis_title="Units", annotations=annotations, height=420)
            st.plotly_chart(fig_drv, use_container_width=True)

            st.markdown("SS Attribution table (numbers and % of total SS)")
            st.dataframe(df_format_for_display(ss_drv_df.rename(columns={'driver':'Driver','amount':'Units','pct_of_total_ss':'Pct_of_total_SS'}).round(2), cols=['Units','Pct_of_total_SS']), use_container_width=True, height=260)

            st.markdown("Notes on interpretation:")
            st.markdown("""
            - The first section (A) shows the raw driver values as originally computed (these mix SS-like terms and forecast volumes for interpretation).
            - The second section (B) is a reconciled, mutually exclusive SS attribution: each row is an amount of Safety Stock and the rows sum exactly to the Total Safety Stock for the selected material and snapshot.
            - Demand Uncertainty and Lead-time Uncertainty here represent the portions of the statistical SS that remain after policy adjustments.
            - Direct vs Indirect in the attribution table are SS-allocations (how much of the retained statistical SS supports local vs downstream demand).
            - Caps/Policy rows explain how business rules changed the statistical SS into the final implemented Safety Stock.
            """)

        st.markdown("---")
        st.subheader("Top Locations by Safety Stock (snapshot)")
        top_nodes = mat_period_df.sort_values('Safety_Stock', ascending=False)[['Location','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status']]
        st.dataframe(df_format_for_display(top_nodes.head(25).copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=300)

        st.markdown("---")
        st.subheader("Actionable Insights (simple heuristics based on current data)")
        if not mat_period_df.empty:
            insights = []
            mat_period_df['SS_to_FCST_Ratio'] = (mat_period_df['Safety_Stock'] / mat_period_df['Forecast'].replace(0, np.nan)).fillna(np.inf)
            high_ratio = mat_period_df[mat_period_df['SS_to_FCST_Ratio'] > 1.0].sort_values('SS_to_FCST_Ratio', ascending=False)
            if not high_ratio.empty:
                insights.append(f"- Nodes with SS > Forecast (ratio>1): {len(high_ratio)} (top examples below)")
                st.dataframe(df_format_for_display(high_ratio[['Location','Forecast','Safety_Stock','SS_to_FCST_Ratio']].head(10), cols=['Forecast','Safety_Stock','SS_to_FCST_Ratio']), use_container_width=True)
            else:
                insights.append("- No nodes found with SS > Forecast (good sign).")

            policy_nodes = mat_period_df[mat_period_df['Adjustment_Status'] != 'Optimal (Statistical)']
            if not policy_nodes.empty:
                insights.append(f"- Nodes with business-rule adjustments: {len(policy_nodes)} (forced zeros, caps).")
                st.dataframe(df_format_for_display(policy_nodes[['Location','Adjustment_Status','Safety_Stock']], cols=['Safety_Stock']), use_container_width=True)
            else:
                insights.append("- No nodes currently modified by policy rules.")

            long_lt = mat_period_df.sort_values('LT_Mean', ascending=False).head(5)
            insights.append(f"- Top lead time nodes (highest avg LT): {', '.join(long_lt['Location'].tolist())}")
            for s in insights:
                st.markdown(s)
        else:
            st.write("No actionable insights — dataset empty for this material/period.")

        st.markdown("---")
        st.subheader("Export — Material Snapshot")
        if not mat_period_df.empty:
            st.download_button("📥 Download Material Snapshot (CSV)", data=mat_period_df.to_csv(index=False), file_name=f"material_{selected_product}_{selected_period.strftime('%Y-%m')}.csv", mime="text/csv")
        else:
            st.write("No snapshot available to download for this selection.")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
