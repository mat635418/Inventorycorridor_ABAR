# Multi-Echelon Inventory Optimizer — Enhanced Version (Final fixes)
# For mat635418 — 2026-01-15
# - Removed global persistent editor UI; replaced with shared selection that syncs across tabs:
#   when you change product/location in any tab it updates the shared selection and other tabs
#   render their selectboxes defaulted to the shared selection (stays until the user changes it).
# - Fixed Historical Forecast vs Actuals tab: ensured Product/Location/Period types trimmed and aligned,
#   provided robust fallbacks when historical rows are missing, and made the selection sync with other tabs.
# - Kept UI improvements (compact selection badge, simulation charting).
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

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer (Method 5 SS)", layout="wide")
st.title("📊 MEIO for Raw Materials — v0.5 — Jan 2026 (Final)")

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

def render_selection_badge(product=None, location=None, df_context=None):
    if product is None or product == "":
        return
    if df_context is not None and not df_context.empty:
        total_fcst = float(df_context['Forecast'].sum())
        total_net = float(df_context['Agg_Future_Demand'].sum())
        total_ss = float(df_context['Safety_Stock'].sum())
    else:
        total_fcst = total_net = total_ss = 0.0
    badge_html = f"""
    <div style="background:#0b3d91;padding:12px;border-radius:8px;color:white;">
      <div style="font-size:12px;opacity:0.85">Selected</div>
      <div style="font-size:15px;font-weight:700;margin-bottom:6px">{product}{(' — ' + location) if location else ''}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
        <div style="background:#ffffff22;padding:8px;border-radius:6px;min-width:120px;">
          <div style="font-size:11px;opacity:0.85">Fcst (Local)</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_fcst, True)}</div>
        </div>
        <div style="background:#ffffff22;padding:8px;border-radius:6px;min-width:120px;">
          <div style="font-size:11px;opacity:0.85">Net Demand</div>
          <div style="font-size:13px;font-weight:700">{euro_format(total_net, True)}</div>
        </div>
        <div style="background:#00b0f622;padding:8px;border-radius:6px;min-width:120px;">
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
# Shared selection state: product & location that should remain consistent across tabs.
if 'shared_product' not in st.session_state:
    st.session_state['shared_product'] = ""
if 'shared_location' not in st.session_state:
    st.session_state['shared_location'] = ""

# Callbacks to sync shared selection when any tab selectbox changes.
def on_product_change(widget_key):
    # When user changes a product in any widget, update the shared product and clear shared_location
    st.session_state['shared_product'] = st.session_state.get(widget_key, "")
    st.session_state['shared_location'] = ""  # clear location to force re-selection relevant to product

def on_location_change(widget_key):
    st.session_state['shared_location'] = st.session_state.get(widget_key, "")

if s_file and d_file and lt_file:
    try:
        df_s = pd.read_csv(s_file)
        df_d = pd.read_csv(d_file)
        df_lt = pd.read_csv(lt_file)
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
        st.stop()

    # Trim columns & normalize product/location columns to strings without trailing spaces
    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]
        if 'Product' in df.columns:
            df['Product'] = df['Product'].astype(str).str.strip()
        if 'Location' in df.columns:
            df['Location'] = df['Location'].astype(str).str.strip()
        # For leadtime file ensure From/To columns are strings
        if 'From_Location' in df.columns:
            df['From_Location'] = df['From_Location'].astype(str).str.strip()
        if 'To_Location' in df.columns:
            df['To_Location'] = df['To_Location'].astype(str).str.strip()

    # Validate required columns
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

    # Normalize Periods (month start)
    df_s['Period'] = pd.to_datetime(df_s['Period'], errors='coerce')
    df_d['Period'] = pd.to_datetime(df_d['Period'], errors='coerce')
    df_s['Period'] = df_s['Period'].dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = df_d['Period'].dt.to_period('M').dt.to_timestamp()

    # Clean numeric columns
    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # HISTORICAL VARS
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

    # NETWORK AGGREGATION
    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)

    # LEAD TIME per receiving node
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # MERGE results
    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']],
                       on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    results['Agg_Std_Hist'] = results.apply(
        lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'],
        axis=1
    )

    # SAFETY STOCK calculation
    results['Pre_Rule_SS'] = z * np.sqrt(
        (results['Agg_Std_Hist']**2 / float(days_per_month)) * results['LT_Mean'] +
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / float(days_per_month))**2
    )
    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['Pre_Rule_SS']

    # Business rules
    results['Pre_Zero_SS'] = results['Safety_Stock']
    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0

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

    # ACCURACY DATA (LOCAL) — prepare historical DataFrame used by Tab 5
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Location'] = hist['Location'].astype(str).str.strip()
    hist['Product'] = hist['Product'].astype(str).str.strip()
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)).fillna(0) * 100
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    hist_net = (
        df_s.groupby(['Product', 'Period'], as_index=False)
            .agg(Network_Consumption=('Consumption', 'sum'),
                 Network_Forecast_Hist=('Forecast', 'sum'))
    )

    # -------------------------------
    # GLOBAL/SHARED SELECTION UI
    # (Removed persistent global editor — instead we use a shared selection that syncs)
    st.markdown("## Shared selection — change product/location in any tab and the choice will propagate to others")
    # Show current shared selection
    cur_prod = st.session_state.get('shared_product', "")
    cur_loc = st.session_state.get('shared_location', "")
    if cur_prod:
        st.info(f"Shared selection: Product = {cur_prod}  •  Location = {cur_loc if cur_loc else '(not selected)'}")
    else:
        st.info("Shared selection: (not set) — pick a product + location in any tab to start")

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

    # Helper for computing default index given options and shared value
    def default_index_for(options, shared_value):
        try:
            return options.index(shared_value) if shared_value in options else 0
        except Exception:
            return 0

    # -------------------------------
    # TAB 1: Inventory Corridor
    # -------------------------------
    with tab1:
        left, right = st.columns([3, 1])
        with left:
            prod_opts = sorted(results['Product'].unique())
            if not prod_opts:
                st.warning("No products available in results.")
                prod = ""
                loc = ""
                plot_df = pd.DataFrame(columns=results.columns)
            else:
                # selectbox key is unique; on_change sets shared product and clears shared location
                sku = st.selectbox("Product", prod_opts, index=default_index_for(prod_opts, st.session_state['shared_product']), key='tab1_sku', on_change=on_product_change, args=('tab1_sku',))
                loc_opts = sorted(results[results['Product'] == sku]['Location'].unique())
                if loc_opts:
                    loc = st.selectbox("Location", loc_opts, index=default_index_for(loc_opts, st.session_state['shared_location']), key='tab1_loc', on_change=on_location_change, args=('tab1_loc',))
                else:
                    st.warning("No locations available for this product.")
                    loc = ""
                plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period') if (sku and loc) else pd.DataFrame(columns=results.columns)

            st.markdown(f"**Selected (shared/default)**: {st.session_state.get('shared_product','')} — {st.session_state.get('shared_location','')}")

            fig = go.Figure()
            if not plot_df.empty:
                fig.add_trace(go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=1, color='rgba(0,0,0,0.1)')))
                fig.add_trace(go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'))
                fig.add_trace(go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Direct Forecast', line=dict(color='black', dash='dot')))
                fig.add_trace(go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash')))
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units')
            st.plotly_chart(fig, use_container_width=True)

        with right:
            # Badge uses plot_df (if empty shows zeros)
            render_selection_badge(product=st.session_state.get('shared_product',''), location=st.session_state.get('shared_location',''), df_context=plot_df)
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
        prod_opts = sorted(results['Product'].unique())
        if not prod_opts:
            st.warning("No products available.")
        else:
            sku = st.selectbox("Product for Network View", prod_opts, index=default_index_for(prod_opts, st.session_state['shared_product']), key='network_sku', on_change=on_product_change, args=('network_sku',))
            period_choices = sorted(results['Period'].unique())
            chosen_period = st.selectbox("Period", period_choices, index=len(period_choices)-1 if period_choices else 0, key="network_period")
            # location not required for topology, but badge shows shared_location if set
            render_selection_badge(product=st.session_state.get('shared_product','') or sku, location=st.session_state.get('shared_location',''), df_context=results[(results['Product']==sku)&(results['Period']==chosen_period)])

            label_data = results[results['Period'] == chosen_period].set_index(['Product', 'Location']).to_dict('index')
            sku_lt = df_lt[df_lt['Product'] == sku] if 'Product' in df_lt.columns else df_lt.copy()

            net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
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
                elif n in ('BEEX','LUEX'):
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
                edge_color = '#dddddd' if (not from_used and not to_used) else '#888888'
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
            components.html(html, height=800, scrolling=True)

    # -------------------------------
    # TAB 3: Full Plan
    # -------------------------------
    with tab3:
        st.subheader("📋 Global Inventory Plan")
        col1, col2, col3 = st.columns([3,3,2])
        # filters are optional; but shared selection will act as default filter if user doesn't pick filters
        f_prod = col1.multiselect("Filter Product", sorted(results['Product'].unique()))
        f_loc = col2.multiselect("Filter Location", sorted(results['Location'].unique()))
        f_period = col3.multiselect("Filter Period", sorted(results['Period'].unique()))

        filtered = results.copy()
        # apply shared selection as implicit default only when user didn't filter explicitly
        if st.session_state.get('shared_product') and not f_prod:
            filtered = filtered[filtered['Product'] == st.session_state['shared_product']]
        if st.session_state.get('shared_location') and not f_loc:
            filtered = filtered[filtered['Location'] == st.session_state['shared_location']]

        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]

        # Sort descending by Safety Stock
        filtered = filtered.sort_values('Safety_Stock', ascending=False)

        # Show badge if a single product is in the filtered result (for context)
        badge_prod = filtered['Product'].iloc[0] if filtered['Product'].nunique() == 1 else st.session_state.get('shared_product', "")
        badge_loc = filtered['Location'].iloc[0] if filtered['Location'].nunique() == 1 else st.session_state.get('shared_location', "")
        render_selection_badge(product=badge_prod, location=badge_loc, df_context=filtered if not filtered.empty else None)

        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
        disp = df_format_for_display(filtered[display_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'], two_decimals_cols=['Forecast'])
        st.dataframe(disp, use_container_width=True, height=650)

        csv_buf = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # -------------------------------
    # TAB 4: Efficiency Analysis
    # -------------------------------
    with tab4:
        st.subheader("⚖️ Efficiency & Policy Analysis")
        prod_opts = sorted(results['Product'].unique())
        if not prod_opts:
            st.warning("No products.")
        else:
            sku = st.selectbox("Material", prod_opts, index=default_index_for(prod_opts, st.session_state['shared_product']), key="eff_sku", on_change=on_product_change, args=('eff_sku',))
            next_month = sorted(results['Period'].unique())[-1]
            eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()

            eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Agg_Future_Demand'].replace(0, np.nan)).fillna(0)
            total_ss_sku = eff['Safety_Stock'].sum()
            total_net_demand_sku = eff['Agg_Future_Demand'].sum()
            sku_ratio = total_ss_sku / total_net_demand_sku if total_net_demand_sku > 0 else 0
            all_res = results[results['Period'] == next_month]
            global_ratio = all_res['Safety_Stock'].sum() / all_res['Agg_Future_Demand'].replace(0, np.nan).sum()

            # product-level badge
            render_selection_badge(product=sku, location=st.session_state.get('shared_location',''), df_context=eff)

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
    # TAB 5: Forecast Accuracy (fixed)
    # -------------------------------
    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals (per product/location)")
        prod_opts = sorted(results['Product'].unique())
        if not prod_opts:
            st.warning("No products available.")
        else:
            # Product selectbox - updates shared_product when changed
            h_sku = st.selectbox("Select Product", prod_opts, index=default_index_for(prod_opts, st.session_state['shared_product']), key="h1", on_change=on_product_change, args=('h1',))
            # Find locations from the historical dataframe first (preferred), else fall back to results
            h_loc_opts_hist = sorted(hist[hist['Product'] == h_sku]['Location'].unique())
            h_loc_opts_results = sorted(results[results['Product'] == h_sku]['Location'].unique())
            # Prefer historical locations (where accuracy can be computed). If none, fall back to locations present in results
            h_loc_opts = h_loc_opts_hist if len(h_loc_opts_hist) > 0 else h_loc_opts_results

            if not h_loc_opts:
                st.warning("No locations available for this product in historical or forecast data.")
            else:
                # location selectbox - updates shared_location when changed
                h_loc = st.selectbox("Select Location", h_loc_opts, index=default_index_for(h_loc_opts, st.session_state['shared_location']), key="h2", on_change=on_location_change, args=('h2',))

                # sync badge/context
                render_selection_badge(product=h_sku, location=h_loc, df_context=hist[(hist['Product']==h_sku)&(hist['Location']==h_loc)])

                # Pull historical rows for this product/location; if none in hist (we fell back), show a helpful message and use aggregated network history
                hdf = hist[(hist['Product'] == h_sku) & (hist['Location'] == h_loc)].sort_values('Period')
                if not hdf.empty:
                    total_consumption = hdf['Consumption'].replace(0, np.nan).sum()
                    wape_val = (hdf['Abs_Error'].sum() / total_consumption * 100) if total_consumption and not np.isnan(total_consumption) else 0.0
                    bias_val = (hdf['Deviation'].sum() / total_consumption * 100) if total_consumption and not np.isnan(total_consumption) else 0.0
                    avg_acc = hdf['Accuracy_%'].mean()

                    k1, k2, k3 = st.columns(3)
                    k1.metric("WAPE (%)", f"{wape_val:.1f}")
                    k2.metric("Bias (%)", f"{bias_val:.1f}")
                    k3.metric("Avg Accuracy (%)", f"{avg_acc:.1f}")

                    fig_hist = go.Figure([
                        go.Scatter(x=hdf['Period'], y=hdf['Consumption'], name='Actuals', line=dict(color='black')),
                        go.Scatter(x=hdf['Period'], y=hdf['Forecast_Hist'], name='Forecast', line=dict(color='blue', dash='dot')),
                    ])
                    fig_hist.update_layout(title=f"Historical Actuals vs Forecast — {h_sku} / {h_loc}", xaxis_title="Period", yaxis_title="Units")
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.subheader("📊 Detailed Accuracy by Month")
                    st.dataframe(df_format_for_display(hdf[['Period','Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']].copy(),
                                                      cols=['Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']), use_container_width=True, height=400)

                else:
                    # No historical rows for this exact product/location — give fallback view and guidance
                    st.warning("No historical sales rows found for this exact product/location selection.")
                    # Show aggregated network history for product (if any)
                    net_table = hist_net[hist_net['Product'] == h_sku].sort_values('Period')
                    if not net_table.empty:
                        net_table['Net_Abs_Error'] = (net_table['Network_Consumption'] - net_table['Network_Forecast_Hist']).abs()
                        net_wape = (net_table['Net_Abs_Error'].sum() / net_table['Network_Consumption'].replace(0, np.nan).sum() * 100)
                        st.subheader("🌐 Aggregated Network History (Selected Product)")
                        c_net1, c_net2 = st.columns([3,1])
                        with c_net1:
                            st.dataframe(df_format_for_display(net_table[['Period','Network_Consumption','Network_Forecast_Hist']].copy(),
                                                               cols=['Network_Consumption','Network_Forecast_Hist'], two_decimals_cols=['Network_Consumption']), use_container_width=True, height=400)
                        with c_net2:
                            st.metric("Network WAPE (%)", f"{net_wape:.1f}")
                    else:
                        st.info("No aggregated network history available for this product either.")

    # --------------------------------
    # TAB 6: Calculation Trace & Simulation
    # --------------------------------
    with tab6:
        st.header("🧮 Transparent Calculation Engine")
        st.write("Select a specific node and period to see exactly how the Safety Stock number was derived and simulate impacts interactively.")
        prod_opts = sorted(results['Product'].unique())
        if not prod_opts:
            st.warning("No products.")
        else:
            c1, c2, c3 = st.columns(3)
            calc_sku = c1.selectbox("Select Product", prod_opts, index=default_index_for(prod_opts, st.session_state['shared_product']), key="c_sku", on_change=on_product_change, args=('c_sku',))
            avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique())
            if avail_locs:
                calc_loc = c2.selectbox("Select Location", avail_locs, index=default_index_for(avail_locs, st.session_state['shared_location']), key="c_loc", on_change=on_location_change, args=('c_loc',))
            else:
                st.warning("No locations available for this product.")
                calc_loc = ""
            avail_periods = sorted(results['Period'].unique())
            calc_period = c3.selectbox("Select Period", avail_periods, index=len(avail_periods)-1 if avail_periods else 0, key="c_period")

            row_df = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)] if calc_loc else pd.DataFrame()
            if row_df.empty:
                st.warning("Selection not found in results.")
            else:
                row = row_df.iloc[0]
                render_selection_badge(product=calc_sku, location=calc_loc, df_context=row_df)

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

                # Simulation moved up and interactive visuals
                st.subheader("3. What-If Simulation (interactive visuals)")
                sim_cols = st.columns(3)
                sim_sl = sim_cols[0].slider("Simulated Service Level (%)", min_value=50.0, max_value=99.9, value=service_level*100, key=f"sim_sl_{calc_sku}_{calc_loc}")
                sim_lt = sim_cols[1].slider("Simulated Avg Lead Time (Days)", min_value=0.0, max_value=max(30.0, row['LT_Mean']*2), value=float(row['LT_Mean']), key=f"sim_lt_{calc_sku}_{calc_loc}")
                sim_lt_std = sim_cols[2].slider("Simulated LT Variability (Days)", min_value=0.0, max_value=max(10.0, row['LT_Std']*2), value=float(row['LT_Std']), key=f"sim_lt_std_{calc_sku}_{calc_loc}")

                sim_z = norm.ppf(sim_sl / 100.0)
                sim_ss = sim_z * np.sqrt(
                    (row['Agg_Std_Hist']**2 / float(days_per_month)) * sim_lt +
                    (sim_lt_std**2) * (row['Agg_Future_Demand'] / float(days_per_month))**2
                )

                bar_df = pd.DataFrame({'label': ['Pre_Rule_SS','Implemented SS','Simulated SS'],'value':[row['Pre_Rule_SS'], row['Safety_Stock'], sim_ss]})
                fig_bar = go.Figure()
                colors = ['#636EFA','#00CC96','#EF553B']
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
                res_col2.metric("Simulated SS (New)", euro_format(sim_ss, True), delta=euro_format(sim_ss - row['Pre_Rule_SS'], True), delta_color="inverse")
                if sim_ss < row['Pre_Rule_SS']:
                    st.success(f"📉 Reducing uncertainty could lower inventory by **{euro_format(row['Pre_Rule_SS'] - sim_ss, True)}** units.")
                elif sim_ss > row['Pre_Rule_SS']:
                    st.warning(f"📈 Increasing service or lead time requires **{euro_format(sim_ss - row['Pre_Rule_SS'], True)}** more units.")

                st.subheader("4. Business Rules Application")
                col_rule_1, col_rule_2 = st.columns(2)
                with col_rule_1:
                    if zero_if_no_net_fcst and row['Agg_Future_Demand'] <= 0:
                        st.error("❌ Network Demand is 0. SS Forced to 0.")
                    else:
                        st.success("✅ Network Demand exists. Keep Statistical SS.")
                with col_rule_2:
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
        prod_opts = sorted(results['Product'].unique())
        if not prod_opts:
            st.warning("No products available.")
        else:
            selected_product = st.selectbox("Select Material", prod_opts, index=default_index_for(prod_opts, st.session_state['shared_product']), key="mat_sel", on_change=on_product_change, args=('mat_sel',))
            period_choices = sorted(results['Period'].unique())
            selected_period = st.selectbox("Select Period to Snapshot", period_choices, index=len(period_choices)-1 if period_choices else 0, key="mat_period")

            mat_period_df = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)].copy()
            total_forecast = mat_period_df['Forecast'].sum()
            total_net = mat_period_df['Agg_Future_Demand'].sum()
            total_ss = mat_period_df['Safety_Stock'].sum()
            nodes_count = mat_period_df['Location'].nunique()
            avg_ss_per_node = (mat_period_df['Safety_Stock'].mean() if nodes_count > 0 else 0)

            render_selection_badge(product=selected_product, location=st.session_state.get('shared_location',''), df_context=mat_period_df)

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

                drv_df = pd.DataFrame({'driver': list(raw_drivers.keys()), 'amount': [float(v) for v in raw_drivers.values()]})
                drv_denom = drv_df['amount'].sum()
                drv_df['pct_of_total_ss'] = drv_df['amount'] / (drv_denom if drv_denom > 0 else 1.0) * 100

                st.markdown("#### A. Original — Raw driver values (interpretation view)")
                fig_drv_raw = go.Figure()
                fig_drv_raw.add_trace(go.Bar(x=drv_df['driver'], y=drv_df['amount'], marker_color=px.colors.qualitative.Pastel))
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

                ss_drv_df = pd.DataFrame({'driver': list(ss_attrib.keys()), 'amount': [float(v) for v in ss_attrib.values()]})
                denom = total_ss if total_ss > 0 else ss_drv_df['amount'].sum()
                denom = denom if denom > 0 else 1.0
                ss_drv_df['pct_of_total_ss'] = ss_drv_df['amount'] / denom * 100

                fig_drv = go.Figure()
                fig_drv.add_trace(go.Bar(x=ss_drv_df['driver'], y=ss_drv_df['amount'], marker_color=px.colors.qualitative.Pastel))
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
                mat_period_df['SS_to_FCST_Ratio'] = (mat_period_df['Safety_Stock'] / mat_period_df['Forecast'].replace(0, np.nan)).fillna(np.inf)
                high_ratio = mat_period_df[mat_period_df['SS_to_FCST_Ratio'] > 1.0].sort_values('SS_to_FCST_Ratio', ascending=False)
                if not high_ratio.empty:
                    st.markdown(f"- Nodes with SS > Forecast (ratio>1): {len(high_ratio)} (top examples below)")
                    st.dataframe(df_format_for_display(high_ratio[['Location','Forecast','Safety_Stock','SS_to_FCST_Ratio']].head(10), cols=['Forecast','Safety_Stock','SS_to_FCST_Ratio']), use_container_width=True)
                else:
                    st.markdown("- No nodes found with SS > Forecast (good sign).")

                policy_nodes = mat_period_df[mat_period_df['Adjustment_Status'] != 'Optimal (Statistical)']
                if not policy_nodes.empty:
                    st.markdown(f"- Nodes with business-rule adjustments: {len(policy_nodes)} (forced zeros, caps).")
                    st.dataframe(df_format_for_display(policy_nodes[['Location','Adjustment_Status','Safety_Stock']], cols=['Safety_Stock']), use_container_width=True)
                else:
                    st.markdown("- No nodes currently modified by policy rules.")

                long_lt = mat_period_df.sort_values('LT_Mean', ascending=False).head(5)
                st.markdown(f"- Top lead time nodes (highest avg LT): {', '.join(long_lt['Location'].tolist())}")
            else:
                st.write("No actionable insights — dataset empty for this material/period.")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
