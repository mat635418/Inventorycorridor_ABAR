# Multi-Echelon Inventory Optimizer — Enhanced Version (Reviewed & Improved)
# Enhanced by Copilot for mat635418 — 2026-01-15
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

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer (Method 5 SS)", layout="wide")
st.title("�� Multi-Echelon Network Inventory Optimizer — SS Method 5 (σD & σLT) — Reviewed")

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
    # Convert empty or dash-like placeholders to NaN
    s = s.replace({'': np.nan, '-': np.nan, '—': np.nan, 'na': np.nan, 'n/a': np.nan, 'None': np.nan})
    # Handle parentheses indicating negatives: (123) -> -123
    paren_mask = s.str.startswith('(') & s.str.endswith(')')
    s.loc[paren_mask] = '-' + s.loc[paren_mask].str[1:-1]
    # Remove common thousand separators and non-numeric characters but preserve '-' and '.' and digits
    # Replace commas (thousand separators), spaces
    s = s.str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    # Drop any trailing non-numeric characters (e.g., currency), and keep minus and dot and digits
    s = s.str.replace(r'[^\d\.\-]+', '', regex=True)
    # Convert to numeric
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
            s = f"{x_abs:,.2f}"  # 1,234,567.89
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

    Notes:
    - Expects df_forecast with columns ['Product','Location','Period','Forecast'] (Forecast is local direct forecast monthly).
    - df_stats should contain ['Product','Location','Local_Std'] where Local_Std is monthly std of local consumption.
    - df_lt contains network routes with 'From_Location' -> 'To_Location' mapping for the product.
    - Algorithm aggregated demand (sum of downstream children) upwards to parents iteratively.
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
                # If missing local std, we will set as NaN for now and handle later
                if pd.isna(local_std):
                    agg_var[n] = np.nan
                else:
                    agg_var[n] = float(local_std)**2

            # Build children map (parent -> list(children))
            children = {}
            if not p_lt.empty:
                for _, row in p_lt.iterrows():
                    children.setdefault(row['From_Location'], []).append(row['To_Location'])

            # Propagate children demand up to parents iteratively
            for _ in range(30):
                changed = False
                for parent in nodes:
                    child_list = children.get(parent, [])
                    if child_list:
                        new_d = float(p_fore.get(parent, {'Forecast': 0})['Forecast']) + sum(agg_demand.get(c, 0) for c in child_list)
                        # For variance: sum variances of children + local variance
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

            # Finalize: replace NaN variances with np.nan (to be handled by caller with fallback)
            for n in nodes:
                results.append({
                    'Product': prod,
                    'Location': n,
                    'Period': month,
                    'Agg_Future_Demand': agg_demand.get(n, 0.0),
                    'Agg_Std_Hist': np.sqrt(agg_var[n]) if (n in agg_var and not pd.isna(agg_var[n])) else np.nan
                })
    return pd.DataFrame(results)

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
# Default local filenames
DEFAULT_FILES = {"sales": "sales.csv", "demand": "demand.csv", "lt": "leadtime.csv"}

# File Uploaders
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
if s_file and d_file and lt_file:
    try:
        df_s = pd.read_csv(s_file)
        df_d = pd.read_csv(d_file)
        df_lt = pd.read_csv(lt_file)
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
        st.stop()

    # Trim column names
    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]

    # Validate required columns (minimal set)
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

    # Parse Periods to month start timestamps
    df_s['Period'] = pd.to_datetime(df_s['Period'], errors='coerce')
    df_d['Period'] = pd.to_datetime(df_d['Period'], errors='coerce')
    df_s['Period'] = df_s['Period'].dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = df_d['Period'].dt.to_period('M').dt.to_timestamp()

    # Clean numeric columns robustly
    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # HISTORICAL VARIABILITY (monthly, per product-location)
    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    # Where std is NaN (insufficient data), we'll keep NaN and apply fallback below
    # Provide a better fallback: global median std for same product or global if not available
    global_median_std = stats['Local_Std'].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0  # conservative fallback if everything is zero/missing

    # For any missing Local_Std use product median or global median
    prod_medians = stats.groupby('Product')['Local_Std'].median().to_dict()
    def fill_local_std(row):
        if not pd.isna(row['Local_Std']) and row['Local_Std'] > 0:
            return row['Local_Std']
        pm = prod_medians.get(row['Product'], np.nan)
        return pm if not pd.isna(pm) else global_median_std
    stats['Local_Std'] = stats.apply(fill_local_std, axis=1)

    # NETWORK AGGREGATION
    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)

    # LEAD TIME RECEIVING NODES: mean lead times for the receiving location (To_Location)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # MERGE
    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']],
                       on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')

    # Fill defaults
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # If Agg_Std_Hist is NaN, replace with product/global fallback
    # We'll use the stats table where available: if a location hasn't a measured agg std, fall back to product median or global_median_std
    # Note: stats' Local_Std is local monthly std; we will use product median of Local_Std to fill missing Agg_Std_Hist
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    results['Agg_Std_Hist'] = results.apply(
        lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'],
        axis=1
    )

    # --------------------------------
    # SAFETY STOCK — SS METHOD 5 (vectorized)
    # --------------------------------
    # Keep raw calculation trace columns
    # Convert monthly -> daily using days_per_month carefully. Assumption: Agg_Std_Hist is monthly std of monthly totals.
    # Daily variance approximation: var_daily = var_monthly / days_per_month -> std_daily = std_monthly / sqrt(days_per_month)
    # SS_Raw = z * sqrt( (σ_D_monthly^2 / days_per_month) * LT_Mean + (LT_Std^2) * (Agg_Future_Demand / days_per_month)^2 )
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

    # Final rounding & additional derived columns
    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    # domain-specific override preserved, but highlight as a special policy
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # ACCURACY DATA (LOCAL)
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)).fillna(0) * 100
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    # Aggregated historical network view (per Product, Period)
    hist_net = (
        df_s.groupby(['Product', 'Period'], as_index=False)
            .agg(Network_Consumption=('Consumption', 'sum'),
                 Network_Forecast_Hist=('Forecast', 'sum'))
    )

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
            sku = st.selectbox("Product", sorted(results['Product'].unique()), key='tab1_sku')
            loc = st.selectbox("Location", sorted(results[results['Product'] == sku]['Location'].unique()), key='tab1_loc')
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
            badge_html = f"""
            <div style="background:#0b3d91;padding:18px;border-radius:8px;color:white;text-align:right;">
                <div style="font-size:14px;opacity:0.8">Selected</div>
                <div style="font-size:18px;font-weight:700">{sku} — {loc}</div>
                <div style="margin-top:8px;font-size:13px;opacity:0.95">
                    Fcst (Local): <strong>{euro_format(float(plot_df['Forecast'].sum()))}</strong><br>
                    Net Demand: <strong>{euro_format(float(plot_df['Agg_Future_Demand'].sum()))}</strong><br>
                    SS (Current): <strong>{euro_format(float(plot_df['Safety_Stock'].sum()), True)}</strong>
                </div>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            s1.metric("Total SS (sku/loc)", euro_format(float(plot_df['Safety_Stock'].sum()), True))
            s2.metric("Total Net Demand", euro_format(float(plot_df['Agg_Future_Demand'].sum()), True))

    # -------------------------------
    # TAB 2: Network Topology (improved per-request)
    # -------------------------------
    with tab2:
        sku = st.selectbox("Product for Network View", sorted(results['Product'].unique()), key="network_sku")
        # default to latest period
        period_choices = sorted(results['Period'].unique())
        default_period = period_choices[-1] if period_choices else None
        chosen_period = st.selectbox("Period", period_choices, index=len(period_choices)-1 if period_choices else 0, key="network_period")

        label_data = results[results['Period'] == chosen_period].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku] if 'Product' in df_lt.columns else df_lt.copy()

        net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")

        # Nodes to emphasise as hubs
        hubs = {"B616", "BEEX", "LUEX"}

        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        # Ensure hubs are present in the node set so they always render
        all_nodes = set(all_nodes).union(hubs)

        # Build demand lookup for the chosen period (for the product)
        demand_lookup = {}
        for n in all_nodes:
            demand_lookup[n] = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})

        # Fixed positions: B616 left, hubs center-left, others right
        # Simple layout grid
        left_x = -400
        hub_x = -150
        right_x = 300
        y_step = 80
        # Prepare ordering
        hubs_present = [h for h in ["B616", "BEEX", "LUEX"] if h in all_nodes]
        others = sorted([n for n in all_nodes if n not in hubs_present])

        # Add hub nodes first with emphasized color
        y_cursor = - (len(hubs_present) - 1) * y_step / 2
        for h in hubs_present:
            m = demand_lookup.get(h, {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label = f"{h}\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            net.add_node(h, label=label, title=label,
                         color={'background': '#2E7D32', 'border': '#144d14'},
                         shape='box', font={'color': 'white', 'size': 14},
                         x=hub_x, y=y_cursor, fixed=True, physics=False)
            y_cursor += y_step

        # Ensure B616 specifically on the far left
        if "B616" in all_nodes and "B616" not in hubs_present:
            # if B616 wasn't in hubs_present due to ordering, we still want it left
            m = demand_lookup.get("B616", {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label = f"B616\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            net.add_node("B616", label=label, title=label,
                         color={'background': '#2E7D32', 'border': '#144d14'},
                         shape='box', font={'color': 'white', 'size': 14},
                         x=left_x, y=0, fixed=True, physics=False)

        # Add other nodes as muted grey
        y_cursor = - (len(others) - 1) * (y_step / 2)
        for i, n in enumerate(others):
            m = demand_lookup.get(n, {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            used = (m['Agg_Future_Demand'] > 0) or (m['Forecast'] > 0)
            label = f"{n}\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            # If this node is actually a hub (maybe we missed it), ensure it's colored green
            if n in hubs:
                bg = '#2E7D32'
                font_color = 'white'
                size = 12
            else:
                bg = '#f0f0f0'
                font_color = '#888888'
                size = 10
            net.add_node(n, label=label, title=label,
                         color={'background': bg, 'border': '#cccccc' if bg == '#f0f0f0' else '#222222'},
                         shape='box', font={'color': font_color, 'size': size},
                         x=right_x, y=y_cursor, fixed=True, physics=False)
            y_cursor += y_step/1.5

        # Add edges, greyed out for non-relevant nodes (for this sku)
        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            # Color edges dark only if at least one endpoint is a hub (emphasize hub connectivity), else very light grey
            if from_n in hubs or to_n in hubs:
                edge_color = '#666666'
            else:
                edge_color = '#dddddd'
            label = f"{int(r.get('Lead_Time_Days', 0))}d" if not pd.isna(r.get('Lead_Time_Days', 0)) else ""
            net.add_edge(from_n, to_n, label=label, color=edge_color, arrows='to')

        net.set_options("""
        var options = {
          "physics": {"enabled": false},
          "nodes": {"borderWidthSelected":2},
          "interaction": {"hover":true}
        }
        """)
        tmpfile = "net.html"
        net.save_graph(tmpfile)
        # read and embed
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
        sku = st.selectbox("Material", sorted(results['Product'].unique()), key="eff_sku")
        # default to latest period
        next_month = sorted(results['Period'].unique())[-1]
        eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()

        eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Agg_Future_Demand'].replace(0, np.nan)).fillna(0)
        total_ss_sku = eff['Safety_Stock'].sum()
        total_net_demand_sku = eff['Agg_Future_Demand'].sum()
        sku_ratio = total_ss_sku / total_net_demand_sku if total_net_demand_sku > 0 else 0
        all_res = results[results['Period'] == next_month]
        global_ratio = all_res['Safety_Stock'].sum() / all_res['Agg_Future_Demand'].replace(0, np.nan).sum()

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

            st.markdown("**Top Nodes by Efficiency Gap**")
            eff['Gap'] = (eff['Safety_Stock'] - eff['Pre_Rule_SS']).abs()
            st.dataframe(
                df_format_for_display(
                    eff.sort_values('Gap', ascending=False)[
                        ['Location', 'Adjustment_Status', 'Safety_Stock', 'SS_to_FCST_Ratio']
                    ],
                    cols=['Safety_Stock'],
                    two_decimals_cols=['Safety_Stock']
                ).head(10),
                use_container_width=True
            )

    # -------------------------------
    # TAB 5: Forecast Accuracy
    # -------------------------------
    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")
        h_sku = st.selectbox("Select Product", sorted(results['Product'].unique()), key="h1")
        h_loc = st.selectbox("Select Location", sorted(results[results['Product'] == h_sku]['Location'].unique()), key="h2")

        hdf = hist[(hist['Product'] == h_sku) & (hist['Location'] == h_loc)].sort_values('Period')
        if not hdf.empty:
            k1, k2, k3 = st.columns(3)
            wape_val = (hdf['Abs_Error'].sum() / hdf['Consumption'].replace(0, np.nan).sum() * 100)
            bias_val = (hdf['Deviation'].sum() / hdf['Consumption'].replace(0, np.nan).sum() * 100)
            avg_acc = hdf['Accuracy_%'].mean()
            k1.metric("WAPE (%)", f"{wape_val:.1f}")
            k2.metric("Bias (%)", f"{bias_val:.1f}")
            k3.metric("Avg Accuracy (%)", f"{avg_acc:.1f}")

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
                net_wape = (net_table['Net_Abs_Error'].sum() / net_table['Network_Consumption'].replace(0, np.nan).sum() * 100)
            else:
                net_wape = 0.0

            c_net1, c_net2 = st.columns([3, 1])
            with c_net1:
                st.dataframe(df_format_for_display(net_table[['Period', 'Network_Consumption', 'Network_Forecast_Hist']].copy(),
                                                   cols=['Network_Consumption','Network_Forecast_Hist'], two_decimals_cols=['Network_Consumption']), use_container_width=True, height=500)
            with c_net2:
                st.metric("Network WAPE (%)", f"{net_wape:.1f}")

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
        st.write("Select a specific node and period to see exactly how the Safety Stock number was derived.")

        c1, c2, c3 = st.columns(3)
        calc_sku = c1.selectbox("Select Product", sorted(results['Product'].unique()), key="c_sku")
        avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique())
        calc_loc = c2.selectbox("Select Location", avail_locs, key="c_loc")
        avail_periods = sorted(results['Period'].unique())
        calc_period = c3.selectbox("Select Period", avail_periods, index=len(avail_periods)-1 if avail_periods else 0, key="c_period")

        row = results[
            (results['Product'] == calc_sku) &
            (results['Location'] == calc_loc) &
            (results['Period'] == calc_period)
        ]
        if row.empty:
            st.warning("Selection not found in results.")
        else:
            row = row.iloc[0]
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

            st.subheader("3. Business Rules Application")
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

            st.markdown("---")
            st.subheader("4. What-If Simulation")
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

    # -------------------------------
    # TAB 7: By Material (reworked — focus single material)
    # -------------------------------
    with tab7:
        st.header("📦 View by Material (Single Material Focus)")
        # Select single product
        selected_product = st.selectbox("Select Material", sorted(results['Product'].unique()), key="mat_sel")
        # Choose period to view aggregated snapshot, default latest
        period_choices = sorted(results['Period'].unique())
        selected_period = st.selectbox("Select Period to Snapshot", period_choices, index=len(period_choices)-1 if period_choices else 0, key="mat_period")

        # Aggregated material-level metrics for chosen period
        mat_period_df = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)]
        total_forecast = mat_period_df['Forecast'].sum()
        total_net = mat_period_df['Agg_Future_Demand'].sum()
        total_ss = mat_period_df['Safety_Stock'].sum()
        nodes_count = mat_period_df['Location'].nunique()
        avg_ss_per_node = (mat_period_df['Safety_Stock'].mean() if nodes_count > 0 else 0)

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Local Forecast", euro_format(total_forecast, True))
        k2.metric("Total Network Demand", euro_format(total_net, True))
        k3.metric("Total Safety Stock (sum nodes)", euro_format(total_ss, True))
        k4.metric("Nodes", f"{nodes_count}")
        k5.metric("Avg SS per Node", euro_format(avg_ss_per_node, True))

        # Time series for this material across periods (aggregated)
        ts = (
            results[results['Product'] == selected_product]
            .groupby('Period', as_index=False)
            .agg(Forecast=('Forecast', 'sum'), NetDemand=('Agg_Future_Demand', 'sum'), SafetyStock=('Safety_Stock', 'sum'))
            .sort_values('Period')
        )
        if not ts.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(x=ts['Period'], y=ts['Forecast'], name='Local Forecast'))
            fig_ts.add_trace(go.Bar(x=ts['Period'], y=ts['NetDemand'], name='Network Demand'))
            fig_ts.add_trace(go.Line(x=ts['Period'], y=ts['SafetyStock'], name='Total SS', line=dict(color='black', width=3)))
            fig_ts.update_layout(barmode='group', title=f"{selected_product} — Period Totals", xaxis_title='Period', yaxis_title='Units')
            st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")
        st.subheader("Top Locations by Safety Stock (snapshot)")
        top_nodes = mat_period_df.sort_values('Safety_Stock', ascending=False)[['Location','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status']]
        st.dataframe(df_format_for_display(top_nodes.head(25).copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=400)

        st.markdown("**Distribution of SS across Hubs vs Others**")
        hubs = ['B616','BEEX','LUEX']
        mat_period_df['Is_Hub'] = mat_period_df['Location'].isin(hubs)
        dist = mat_period_df.groupby('Is_Hub', as_index=False).agg(SS=('Safety_Stock','sum'), Nodes=('Location', lambda s: s.nunique()))
        if not dist.empty:
            fig_pie = px.pie(dist, names=dist['Is_Hub'].map({True:'Hubs', False:'Other Nodes'}), values='SS', title='SS Distribution: Hubs vs Others')
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("Export — Material Snapshot")
        st.download_button("📥 Download Material Snapshot (CSV)", data=mat_period_df.to_csv(index=False), file_name=f"material_{selected_product}_{selected_period.strftime('%Y-%m')}.csv", mime="text/csv")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
