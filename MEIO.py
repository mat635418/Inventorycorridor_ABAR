# Multi-Echelon Inventory Optimizer — Raw Materials
# Developed by mat635418 — Jan 2026

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

st.set_page_config(page_title="MEIO for RM", layout="wide")

# Logo configuration
LOGO_FILENAME = "GY_logo.jpg"
LOGO_BASE_WIDTH = 160

# Fixed conversion (30 days/month)
days_per_month = 30

st.markdown("<h1 style='margin:0; padding-top:6px;'>MEIO for Raw Materials — v0.79 — Jan 2026</h1>", unsafe_allow_html=True)

# Small UI styling tweak to make selected multiselect chips match app theme.
st.markdown(
    """
    <style>
      div[data-baseweb="tag"], .stMultiSelect div[data-baseweb="tag"], .stSelectbox div[data-baseweb="tag"] {
        background: #e3f2fd !important;
        color: #0b3d91 !important;
        border: 1px solid #90caf9 !important;
        border-radius: 8px !important;
        padding: 2px 8px !important;
        font-weight: 600 !important;
      }
      div[data-baseweb="tag"] span, .stMultiSelect div[data-baseweb="tag"] span, .stSelectbox div[data-baseweb="tag"] span {
        color: #0b3d91 !important;
      }
      div[data-baseweb="tag"] svg, .stMultiSelect div[data-baseweb="tag"] svg, .stSelectbox div[data-baseweb="tag"] svg { fill: #0b3d91 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helpers / formatting utilities
def clean_numeric(series):
    s = series.astype(str).str.strip()
    s = s.replace({'': np.nan, '-': np.nan, '—': np.nan, 'na': np.nan, 'n/a': np.nan, 'None': np.nan})
    paren_mask = s.str.startswith('(') & s.str.endswith(')')
    try:
        s.loc[paren_mask] = '-' + s.loc[paren_mask].str[1:-1]
    except Exception:
        s = s.apply(lambda v: ('-' + v[1:-1]) if isinstance(v, str) and v.startswith('(') and v.endswith(')') else v)
    s = s.str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    s = s.str.replace(r'[^\d\.\-]+', '', regex=True)
    out = pd.to_numeric(s, errors='coerce')
    return out

def euro_format(x, always_two_decimals=True, show_zero=False):
    """
    Format numeric values using '.' as thousands separator (e.g. 1.234)
    Default behaviour: hide zeros (return empty string) to preserve prior app UI.
    If show_zero=True, zeros will be shown as '0' (or '0' formatted).
    """
    try:
        if x is None:
            return ""
        if isinstance(x, (np.floating, float, np.integer, int)):
            xv = float(x)
        else:
            try:
                xv = float(x)
            except Exception:
                return str(x)
        if math.isnan(xv):
            return ""
        if math.isclose(xv, 0.0, abs_tol=1e-9):
            if not show_zero:
                return ""
            # if show_zero, continue and format zero as '0' (or '0.00' if requested)
        neg = xv < 0
        rounded = int(round(abs(xv)))
        s = f"{rounded:,}"
        s = s.replace(',', '.')
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

def aggregate_network_stats(df_forecast, df_stats, df_lt, transitive=True, rho=1.0):
    """
    Aggregate monthly forecast and variance through the network.

    - transitive: include full downstream subtree (recursive) when True;
                  include only immediate children when False.
    - rho: a damping factor in [0,1] used to scale how much downstream variance
           contributes to the upstream node's variance.

    Returns:
    - DataFrame with Product, Location, Period, Agg_Future_Demand, Agg_Std_Hist
    - reachable_map: mapping (product, start_location, period) -> set(reachable locations)
    """
    results = []
    months = df_forecast['Period'].unique()
    products = df_forecast['Product'].unique()
    routes_by_product = {}
    if 'Product' in df_lt.columns:
        for prod in df_lt['Product'].unique():
            routes_by_product[prod] = df_lt[df_lt['Product'] == prod].copy()
    else:
        routes_by_product[None] = df_lt.copy()

    reachable_map = {}

    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in products:
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = routes_by_product.get(prod, pd.DataFrame(columns=df_lt.columns))
            nodes = set(df_month[df_month['Product'] == prod]['Location'])
            if not p_lt.empty:
                froms = set([v for v in p_lt['From_Location'].tolist() if pd.notna(v)])
                tos = set([v for v in p_lt['To_Location'].tolist() if pd.notna(v)])
                nodes = nodes.union(froms).union(tos)
            if not nodes:
                continue
            children = {}
            if not p_lt.empty:
                for _, r in p_lt.iterrows():
                    f = r.get('From_Location', None)
                    t = r.get('To_Location', None)
                    if pd.isna(f) or pd.isna(t):
                        continue
                    children.setdefault(f, set()).add(t)
            reachable_cache = {}

            def get_reachable(start):
                # If not transitive, return only direct children plus the start node.
                if not transitive:
                    direct = children.get(start, set())
                    outset = set(direct)
                    outset.add(start)
                    return outset

                if start in reachable_cache:
                    return reachable_cache[start]
                visited = set()
                stack = [start]
                while stack:
                    cur = stack.pop()
                    kids = children.get(cur, set())
                    for k in kids:
                        if k not in visited and k != start:
                            visited.add(k)
                            stack.append(k)
                visited_with_start = set(visited)
                visited_with_start.add(start)
                reachable_cache[start] = visited_with_start
                return visited_with_start

            for n in nodes:
                reachable = get_reachable(n)
                reachable_map[(prod, n, month)] = reachable

                local_fcst = float(p_fore.get(n, {'Forecast': 0})['Forecast']) if n in p_fore else 0.0

                # Aggregate demand: local + downstream reachable nodes (depends on transitive flag)
                child_fcst_sum = 0.0
                child_var_sum = 0.0
                for c in reachable:
                    if c == n:
                        continue
                    child_fcst = float(p_fore.get(c, {'Forecast': 0})['Forecast']) if c in p_fore else 0.0
                    child_fcst_sum += child_fcst
                    child_std = p_stats.get(c, {}).get('Local_Std', np.nan)
                    if not pd.isna(child_std):
                        child_var_sum += (float(child_std) ** 2) * float(rho)

                agg_demand = local_fcst + child_fcst_sum
                local_std = p_stats.get(n, {}).get('Local_Std', np.nan)
                local_var = 0.0 if pd.isna(local_std) else float(local_std) ** 2
                total_var = local_var + child_var_sum
                agg_std = np.sqrt(total_var) if total_var >= 0 and (not pd.isna(total_var)) else np.nan
                results.append({
                    'Product': prod,
                    'Location': n,
                    'Period': month,
                    'Agg_Future_Demand': agg_demand,
                    'Agg_Std_Hist': agg_std
                })
    return pd.DataFrame(results), reachable_map

# Render logo helper
def render_logo_above_parameters(scale=1.5):
    if LOGO_FILENAME and os.path.exists(LOGO_FILENAME):
        try:
            width = int(LOGO_BASE_WIDTH * float(scale))
            st.image(LOGO_FILENAME, width=width)
        except Exception:
            pass

def render_selection_badge(product=None, location=None, df_context=None, small=False, period=None):
    """
    Renders the compact selection badge.

    Behavior change:
    - If df_context contains a 'Period' column and contains multiple periods, we default to using
      the current month (CURRENT_MONTH_TS) for the badge numbers unless an explicit period is provided.
    - Badge calls euro_format(..., show_zero=True) so zeros are shown explicitly in the badge (blue box),
      while the rest of the app keeps the original zero-suppression behavior.
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

    # If df_context has Periods and no explicit period requested, prefer current month if present,
    # otherwise show the latest period available.
    df_for_badge = df_context
    try:
        if df_for_badge is not None and 'Period' in df_for_badge.columns:
            if period is None:
                # CURRENT_MONTH_TS is defined later in the script but exists at call time.
                preferred = CURRENT_MONTH_TS
            else:
                preferred = period
            if preferred in df_for_badge['Period'].values:
                df_for_badge = df_for_badge[df_for_badge['Period'] == preferred]
            else:
                # fallback to the most recent period available in df_for_badge
                try:
                    max_p = df_for_badge['Period'].max()
                    df_for_badge = df_for_badge[df_for_badge['Period'] == max_p]
                except Exception:
                    pass
    except Exception:
        df_for_badge = df_context

    local_demand = _sum_candidates(df_for_badge, ['Forecast', 'Forecast_Hist'])
    total_demand = _sum_candidates(df_for_badge, ['Agg_Future_Demand'])
    total_ss = _sum_candidates(df_for_badge, ['Safety_Stock'])
    title = f"{product}{(' — ' + location) if location else ''}"

    badge_html = f"""
    <div style="background:#0b3d91;padding:14px;border-radius:8px;color:white;max-width:100%;font-family:inherit;">
      <div style="font-size:11px;opacity:0.95;margin-bottom:6px;">Selected</div>
      <div style="font-size:13px;font-weight:700;margin-bottom:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</div>
      <div style="background:#FFC107;color:#0b3d91;padding:10px;border-radius:6px;width:100%;box-sizing:border-box;margin-bottom:10px;display:block;">
        <div style="font-size:11px;font-weight:600;margin-bottom:4px;">Safety Stock</div>
        <div style="font-size:14px;font-weight:700;text-align:right;">{euro_format(total_ss, show_zero=True)}</div>
      </div>
      <div style="background:#e3f2fd;color:#0b3d91;padding:10px;border-radius:6px;width:100%;box-sizing:border-box;margin-bottom:8px;display:block;">
        <div style="font-size:10px;opacity:0.85;">Local Direct Demand</div>
        <div style="font-size:13px;font-weight:700;text-align:right;">{euro_format(local_demand, show_zero=True)}</div>
      </div>
      <div style="background:#90caf9;color:#0b3d91;padding:10px;border-radius:6px;width:100%;box-sizing:border-box;display:block;">
        <div style="font-size:10px;opacity:0.85;">Total Network Demand</div>
        <div style="font-size:13px;font-weight:700;text-align:right;">{euro_format(total_demand, show_zero=True)}</div>
      </div>
    </div>
    """

    st.markdown(badge_html, unsafe_allow_html=True)

# small utility to format Period labels: "JAN 2026"
def period_label(ts):
    try:
        return pd.to_datetime(ts).strftime('%b %Y').upper()
    except Exception:
        return str(ts)

# ----------------------
# SIDEBAR: collapsible sections (all collapsed by default except Data Sources)
# ----------------------

with st.sidebar.expander("⚙️ Service Level Configuration", expanded=False):
    service_level = st.slider(
        "Service Level (%)",
        50.0,
        99.9,
        99.0,
        help="Target probability of not stocking out. Higher values increase the Z-score and therefore Safety Stock — reduces stockouts but raises inventory holdings."
    ) / 100
    z = norm.ppf(service_level)

with st.sidebar.expander("⚙️ Safety Stock Rules", expanded=False):
    # Extended explanation:
    # - zero_if_no_net_fcst: when enabled, nodes with zero aggregated network demand are forced to have 0 Safety Stock.
    #   This avoids retaining inventory at inactive or decommissioned nodes.
    # - apply_cap + cap_range: allow business to limit extreme statistical SS values by clipping the computed SS
    #   within a configurable percentage of node's total network demand (e.g., 0–200%).
    zero_if_no_net_fcst = st.checkbox(
        "Force Zero SS if No Network Demand",
        value=True,
        help="When enabled, nodes with zero aggregated network demand will have Safety Stock forced to zero."
    )
    apply_cap = st.checkbox(
        "Enable SS Capping (% of Network Demand)",
        value=True,
        help="Enable clipping of calculated Safety Stock within a percentage range of the node's total network demand."
    )
    cap_range = st.slider(
        "Cap Range (%)",
        0,
        500,
        (0, 200),
        help="Lower and upper bounds (as % of total network demand) applied to Safety Stock. For example, 0–200% allows SS up to twice the node's network demand."
    )

with st.sidebar.expander("⚙️ Aggregation & Uncertainty", expanded=False):
    # Controls for how downstream demand/variance and lead-time variance are handled.
    agg_mode = st.selectbox(
        "Network Aggregation Mode",
        ["Transitive (full downstream)", "Direct children only"],
        index=0,
        help="Choose how downstream demand is aggregated: 'Transitive' includes all downstream nodes recursively; 'Direct' uses only immediate children."
    )
    use_transitive = True if agg_mode.startswith("Transitive") else False

    var_rho = st.slider(
        "Variance damping factor (ρ)",
        0.0,
        1.0,
        1.0,
        help="Scales how much downstream nodes' variance contributes to a parent's variance (0 = ignore downstream variance; 1 = full add)."
    )

    lt_mode = st.selectbox(
        "Lead-time variance handling",
        ["Apply LT variance", "Ignore LT variance", "Average LT Std across downstream"],
        index=0,
        help="How lead-time uncertainty is included: 'Apply LT variance' uses each node's lead-time variance; 'Ignore LT variance' omits lead-time uncertainty from the SS calculation; 'Average LT Std across downstream' uses the mean downstream LT std to smooth local LT uncertainty."
    )

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

# ----------------------
# MAIN LOGIC
# ----------------------
DEFAULT_PRODUCT_CHOICE = "NOKANDO2"
DEFAULT_LOCATION_CHOICE = "DEW1"
CURRENT_MONTH_TS = pd.Timestamp.now().to_period('M').to_timestamp()

def run_pipeline(transitive, rho, lt_mode_param):
    """
    Run aggregation -> stats -> safety-stock pipeline.

    Steps (high level):
    1. Aggregate downstream demand and historical variance via aggregate_network_stats.
    2. Map per-node lead time statistics (mean and std).
    3. Merge aggregates with forecasts and LT info, ensuring sensible defaults.
    4. Convert monthly aggregates into daily equivalents for the statistical model.
    5. Apply low-demand floor logic to avoid underestimating variance at very low volumes.
    6. Compute the demand-component variance and lead-time component variance (modes supported).
    7. Combine variances, compute Z * sqrt(variance) as statistical SS, apply minimum floor.
    8. Apply business rules: zero suppression, capping, and location-specific overrides.

    Parameters:
    - transitive: bool controlling subtree inclusion
    - rho: downstream variance damping factor
    - lt_mode_param: string controlling LT handling ('Ignore LT variance', 'Apply LT variance', 'Average LT Std across downstream')
    """
    # 1) aggregate demand & historical variance through the network
    network_stats, reachable_map = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt, transitive=transitive, rho=rho)

    # 2) per-node lead time averages (map To_Location -> node LT)
    node_lt_local = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt_local.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # 3) merge aggregated stats with forecasts and LT info
    res = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    res = pd.merge(res, node_lt_local, on=['Product', 'Location'], how='left')

    # sensible defaults to avoid NaNs in downstream calculations
    res = res.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # if Agg_Std_Hist is missing, fall back to product median, then global median
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    res['Agg_Std_Hist'] = res.apply(lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'], axis=1)

    # Ensure numeric types for calculations
    res['Agg_Std_Hist'] = res['Agg_Std_Hist'].astype(float)
    res['LT_Mean'] = res['LT_Mean'].astype(float)
    res['LT_Std'] = res['LT_Std'].astype(float)
    res['Agg_Future_Demand'] = res['Agg_Future_Demand'].astype(float)
    res['Forecast'] = res['Forecast'].astype(float)

    # 4) Convert monthly to daily values for demand variance modelling
    res['Sigma_D_Day'] = res['Agg_Std_Hist'] / np.sqrt(float(days_per_month))
    res['D_day'] = res['Agg_Future_Demand'] / float(days_per_month)
    res['Var_D_Day'] = res['Sigma_D_Day']**2

    # 5) Low-demand guard: at very small demand, ensure variance is not unrealistically tiny
    low_demand_monthly_threshold = 20.0
    low_mask = res['Agg_Future_Demand'] < low_demand_monthly_threshold
    res.loc[low_mask, 'Var_D_Day'] = res.loc[low_mask, 'Var_D_Day'].where(res.loc[low_mask, 'Var_D_Day'] >= res.loc[low_mask, 'D_day'], res.loc[low_mask, 'D_day'])

    # demand component (per-day variance scaled by lead time)
    demand_component = res['Var_D_Day'] * res['LT_Mean']

    # 6) Lead-time component — support different handling modes.
    lt_component_list = []
    for idx, row in res.iterrows():
        d_day = float(row['D_day'])
        D_for_LT = d_day  # daily conversion fixed
        if lt_mode_param == 'Ignore LT variance':
            lt_comp = 0.0
        elif lt_mode_param == 'Apply LT variance':
            # classical term: Var(L) * D^2
            lt_comp = (float(row['LT_Std'])**2) * (D_for_LT**2)
        elif lt_mode_param == 'Average LT Std across downstream':
            # Use reachable_map to compute the average downstream LT Std then apply Var(L_avg) * D^2
            reachable = reachable_map.get((row['Product'], row['Location'], row['Period']), set())
            if not reachable:
                lt_used = float(row['LT_Std'])
            else:
                vals = []
                for rn in reachable:
                    match = node_lt_local[(node_lt_local['Product'] == row['Product']) & (node_lt_local['Location'] == rn)]
                    if not match.empty:
                        vals.append(float(match['LT_Std'].iloc[0]))
                lt_used = float(np.mean(vals)) if len(vals) > 0 else float(row['LT_Std'])
            lt_comp = (lt_used**2) * (D_for_LT**2)
        else:
            lt_comp = (float(row['LT_Std'])**2) * (D_for_LT**2)
        lt_component_list.append(lt_comp)

    res['lt_component'] = np.array(lt_component_list)

    # 7) Combine variance components; ensure non-negative, compute statistical safety stock
    combined_variance = demand_component + res['lt_component']
    combined_variance = combined_variance.clip(lower=0)
    res['SS_stat'] = z * np.sqrt(combined_variance)

    # Minimum floor: 1% of mean LT demand (small safeguard)
    min_floor_fraction_of_LT_demand = 0.01
    res['Mean_Demand_LT'] = res['D_day'] * res['LT_Mean']
    res['SS_floor'] = res['Mean_Demand_LT'] * min_floor_fraction_of_LT_demand
    res['Pre_Rule_SS'] = res[['SS_stat', 'SS_floor']].max(axis=1)
    res['Adjustment_Status'] = 'Optimal (Statistical)'
    res['Safety_Stock'] = res['Pre_Rule_SS']

    # 8) Business rules
    if zero_if_no_net_fcst:
        zero_mask = (res['Agg_Future_Demand'] <= 0)
        res.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        res.loc[zero_mask, 'Safety_Stock'] = 0.0

    res['Pre_Cap_SS'] = res['Safety_Stock']
    if apply_cap:
        l_cap, u_cap = cap_range[0]/100.0, cap_range[1]/100.0
        l_lim = res['Agg_Future_Demand'] * l_cap
        u_lim = res['Agg_Future_Demand'] * u_cap
        high_mask = (res['Safety_Stock'] > u_lim) & (res['Adjustment_Status'] == 'Optimal (Statistical)')
        low_mask = (res['Safety_Stock'] < l_lim) & (res['Adjustment_Status'] == 'Optimal (Statistical)') & (res['Agg_Future_Demand'] > 0)
        res.loc[high_mask, 'Adjustment_Status'] = 'Capped (High)'
        res.loc[low_mask, 'Adjustment_Status'] = 'Capped (Low)'
        res['Safety_Stock'] = res['Safety_Stock'].clip(lower=l_lim, upper=u_lim)

    # Final rounding & overrides
    res['Safety_Stock'] = res['Safety_Stock'].round(0)
    res.loc[res['Location'] == 'B616', 'Safety_Stock'] = 0
    res['Max_Corridor'] = res['Safety_Stock'] + res['Forecast']

    return res, reachable_map

# Data loading and validation
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

    # Clean numeric columns
    df_s['Consumption'] = clean_numeric(df_s['Consumption']); df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days']); df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # Historical stats: per product/location
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

    # Run the main pipeline
    results, reachable_map = run_pipeline(transitive=use_transitive, rho=var_rho, lt_mode_param=lt_mode)

    # Historical accuracy table
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)) * 100
    hist['APE_%'] = hist['APE_%'].fillna(0)
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100
    hist_net = df_s.groupby(['Product', 'Period'], as_index=False).agg(Network_Consumption=('Consumption', 'sum'), Network_Forecast_Hist=('Forecast', 'sum'))

    # Filters: show only meaningful (non-zero) rows where appropriate
    meaningful_mask = results[['Agg_Future_Demand', 'Forecast', 'Safety_Stock', 'Pre_Rule_SS']].fillna(0).abs().sum(axis=1) > 0
    meaningful_results = results[meaningful_mask].copy()

    # Defaults & lists
    all_products = sorted(meaningful_results['Product'].unique().tolist())
    if not all_products:
        all_products = sorted(results['Product'].unique().tolist())
    default_product = DEFAULT_PRODUCT_CHOICE if DEFAULT_PRODUCT_CHOICE in all_products else (all_products[0] if all_products else "")
    def default_location_for(prod):
        locs = sorted(meaningful_results[meaningful_results['Product'] == prod]['Location'].unique().tolist())
        if not locs:
            locs = sorted(results[results['Product'] == prod]['Location'].unique().tolist())
        return DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in locs else (locs[0] if locs else "")
    all_periods = sorted(results['Period'].unique().tolist())
    default_period = CURRENT_MONTH_TS if CURRENT_MONTH_TS in all_periods else (all_periods[-1] if all_periods else None)

    # Prepare human-friendly period labels mapping for right-side filters (e.g., "JAN 2026")
    period_label_map = {period_label(p): p for p in all_periods}
    period_labels = list(period_label_map.keys())

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

    # TAB 1: Inventory Corridor
    with tab1:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            sku_default = default_product
            sku_index = all_products.index(sku_default) if sku_default in all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key='tab1_sku')

            # Prefer locations that are relevant in the CURRENT month
            loc_opts = sorted(meaningful_results[(meaningful_results['Product'] == sku) & (meaningful_results['Period'] == CURRENT_MONTH_TS)]['Location'].unique().tolist())
            if not loc_opts:
                loc_opts = sorted(results[(results['Product'] == sku) & (results['Period'] == CURRENT_MONTH_TS)]['Location'].unique().tolist())
            if not loc_opts:
                loc_opts = sorted(meaningful_results[meaningful_results['Product'] == sku]['Location'].unique().tolist())
            if not loc_opts:
                loc_opts = sorted(results[results['Product'] == sku]['Location'].unique().tolist())
            if not loc_opts:
                loc_opts = ["(no location)"]

            loc_default = DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in loc_opts else (loc_opts[0] if loc_opts else "(no location)")
            loc_index = loc_opts.index(loc_default) if loc_default in loc_opts else 0
            if loc_opts:
                loc = st.selectbox("LOCATION", loc_opts, index=loc_index, key='tab1_loc')
            else:
                loc = st.selectbox("LOCATION", ["(no location)"], index=0, key='tab1_loc')

            # Render badge using current month context (badge will internally filter to current month)
            render_selection_badge(product=sku, location=loc if loc != "(no location)" else None, df_context=results[(results['Product'] == sku) & (results['Location'] == loc)])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            st.markdown(f"**Selected**: {sku} — {loc}")
            # show all months in the graph: create a Period axis covering all_periods and merge selection data onto it
            plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')
            # create DataFrame covering all periods (even if zeros)
            df_all_periods = pd.DataFrame({'Period': all_periods})
            plot_full = pd.merge(df_all_periods, plot_df[['Period','Max_Corridor','Safety_Stock','Forecast','Agg_Future_Demand']], on='Period', how='left')
            # fill missing with zeros so months with no data are shown explicitly
            plot_full[['Max_Corridor','Safety_Stock','Forecast','Agg_Future_Demand']] = plot_full[['Max_Corridor','Safety_Stock','Forecast','Agg_Future_Demand']].fillna(0)

            # New: Allow toggling Max Corridor visibility (default OFF)
            show_max_corridor = st.checkbox("Show Max Corridor", value=False, key="show_max_corridor")

            traces = []
            if show_max_corridor:
                traces.append(go.Scatter(x=plot_full['Period'], y=plot_full['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=1, color='rgba(0,0,0,0.1)')))
            traces.extend([
                go.Scatter(x=plot_full['Period'], y=plot_full['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
                go.Scatter(x=plot_full['Period'], y=plot_full['Forecast'], name='Local Direct Demand', line=dict(color='black', dash='dot')),
                go.Scatter(x=plot_full['Period'], y=plot_full['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
            ])

            fig = go.Figure(traces)
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units', xaxis=dict(tickformat="%b\n%Y"))
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Network Topology
    with tab2:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            sku_default = default_product
            sku_index = all_products.index(sku_default) if sku_default in all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="network_sku")

            # use human-readable period labels
            if period_labels:
                try:
                    default_label = period_label(default_period) if default_period is not None else period_labels[-1]
                    period_index = period_labels.index(default_label) if default_label in period_labels else len(period_labels)-1
                except Exception:
                    period_index = len(period_labels)-1
                chosen_label = st.selectbox("PERIOD", period_labels, index=period_index, key="network_period")
                chosen_period = period_label_map.get(chosen_label, default_period)
            else:
                chosen_period = CURRENT_MONTH_TS

            render_selection_badge(product=sku, location=None, df_context=results[(results['Product']==sku)&(results['Period']==chosen_period)])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
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
                # Now show zeros explicitly in labels by forcing show_zero=True
                # Node considered 'used' visually if any of the metrics are > 0
                used = (float(m.get('Agg_Future_Demand', 0)) > 0) or (float(m.get('Forecast', 0)) > 0)

                if n == 'B616':
                    bg = '#dcedc8'; border = '#8bc34a'; font_color = '#0b3d91'; size = 14
                elif n == 'BEEX' or n == 'LUEX':
                    bg = '#bbdefb'; border = '#64b5f6'; font_color = '#0b3d91'; size = 14
                else:
                    if used:
                        bg = '#fff9c4'; border = '#fbc02d'; font_color = '#222222'; size = 12
                    else:
                        bg = '#f0f0f0'; border = '#cccccc'; font_color = '#9e9e9e'; size = 10

                # consistent naming: LDD (Local Direct Demand), TND (Total Network Demand), SS (Safety Stock)
                lbl = f"{n}\\nLDD: {euro_format(m.get('Forecast', 0), show_zero=True)}\\nTND: {euro_format(m.get('Agg_Future_Demand', 0), show_zero=True)}\\nSS: {euro_format(m.get('Safety_Stock', 0), show_zero=True)}"
                # pyvis expects newline as '\n'
                lbl = lbl.replace("\\n", "\n")
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
            injection_js = """
            <script>
              // Delay to ensure the network object is available, then fit + center.
              setTimeout(function(){
                try {
                  if (typeof network !== 'undefined') {
                    network.fit();
                    var bounds = network.getBoundingBox();
                    if (bounds) {
                      var cx = (bounds.right + bounds.left) / 2;
                      var cy = (bounds.top + bounds.bottom) / 2;
                      network.moveTo({position: {x: cx, y: cy}});
                    }
                  }
                } catch (e) { console.warn("Network fit/center failed:", e); }
              }, 700);
            </script>
            """
            if '</head>' in html_text:
                html_text = html_text.replace('</head>', injection_css + '</head>', 1)
            if '</body>' in html_text:
                html_text = html_text.replace('</body>', injection_js + '</body>', 1)
            else:
                html_text = html_text + injection_js
            components.html(html_text, height=750)

            # Move legend/agenda below graph and center it (previously above the graph)
            st.markdown("""
                <div style="text-align:center; font-size:12px; padding:8px 0;">
                  <div style="display:inline-block; background:#f7f9fc; padding:8px 12px; border-radius:8px;">
                    <strong>Legend:</strong><br/>
                    LDD = Local Direct Demand (local forecast) &nbsp;&nbsp;|&nbsp;&nbsp;
                    TND = Total Network Demand (aggregated downstream + local) &nbsp;&nbsp;|&nbsp;&nbsp;
                    SS  = Safety Stock (final policy value)
                  </div>
                </div>
            """, unsafe_allow_html=True)

    # TAB 3: Full Plan
    with tab3:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            st.markdown("<div style='padding:6px 0;'></div>", unsafe_allow_html=True)
            prod_choices = sorted(meaningful_results['Product'].unique()) if not meaningful_results.empty else sorted(results['Product'].unique())
            loc_choices = sorted(meaningful_results['Location'].unique()) if not meaningful_results.empty else sorted(results['Location'].unique())

            # human-friendly period labels for multiselect
            period_choices_labels = period_labels
            period_choices_ts = [period_label_map[lbl] for lbl in period_choices_labels]

            # Ensure current month is selected by default (if present)
            default_prod_list = [default_product] if default_product in prod_choices else []
            default_period_list = []
            cur_label = period_label(CURRENT_MONTH_TS)
            if cur_label in period_choices_labels:
                default_period_list = [cur_label]
            else:
                if default_period is not None:
                    dp_label = period_label(default_period)
                    if dp_label in period_choices_labels:
                        default_period_list = [dp_label]

            # Additional CSS override here to remove any red-highlighted selected chips (user request)
            st.markdown(
                """
                <style>
                /* remove red highlight for selected multiselect/selectbox chips */
                div[data-baseweb="tag"], .stMultiSelect div[data-baseweb="tag"], .stSelectbox div[data-baseweb="tag"] {
                    background: #e3f2fd !important;
                    color: #0b3d91 !important;
                    border: 1px solid #90caf9 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            f_prod = st.multiselect("MATERIAL", prod_choices, default=default_prod_list, key="full_f_prod")
            f_loc = st.multiselect("LOCATION", loc_choices, default=[], key="full_f_loc")
            f_period_labels = st.multiselect("PERIOD", period_choices_labels, default=default_period_list, key="full_f_period")

            # Map selected labels back to timestamps for filtering
            f_period = [period_label_map[lbl] for lbl in f_period_labels] if f_period_labels else []

            badge_product = f_prod[0] if f_prod else (default_product if default_product in all_products else (all_products[0] if all_products else ""))
            badge_df = results[results['Product'] == badge_product] if badge_product else None
            render_selection_badge(product=badge_product, location=None, df_context=badge_df)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            st.subheader("📋 Global Inventory Plan")
            filtered = results.copy()
            if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
            if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
            if f_period: filtered = filtered[filtered['Period'].isin(f_period)]
            filtered = filtered.sort_values('Safety_Stock', ascending=False)

            filtered_display = hide_zero_rows(filtered)

            display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
            fmt_cols = [c for c in ['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'] if c in filtered_display.columns]
            disp = df_format_for_display(filtered_display[display_cols].copy(), cols=fmt_cols, two_decimals_cols=fmt_cols)
            st.dataframe(disp, use_container_width=True, height=700)

            csv_buf = filtered[display_cols].to_csv(index=False)
            st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # TAB 4: Efficiency Analysis
    with tab4:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            sku_default = default_product
            sku_index = all_products.index(sku_default) if sku_default in all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="eff_sku")

            # period selection with labels
            if period_labels:
                try:
                    default_label = period_label(default_period) if default_period is not None else period_labels[-1]
                    period_index = period_labels.index(default_label) if default_label in period_labels else len(period_labels)-1
                except Exception:
                    period_index = len(period_labels)-1
                chosen_label = st.selectbox("PERIOD", period_labels, index=period_index, key="eff_period")
                eff_period = period_label_map.get(chosen_label, default_period)
            else:
                eff_period = CURRENT_MONTH_TS

            render_selection_badge(product=sku, location=None, df_context=results[(results['Product'] == sku)&(results['Period'] == eff_period)])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            st.subheader("⚖️ Efficiency & Policy Analysis")
            snapshot_period = eff_period if eff_period in all_periods else (all_periods[-1] if all_periods else None)
            if snapshot_period is None:
                st.warning("No period data available for Efficiency Analysis.")
                eff = results[(results['Product'] == sku)].copy()
            else:
                eff = results[(results['Product'] == sku) & (results['Period'] == snapshot_period)].copy()
            eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Agg_Future_Demand'].replace(0, np.nan)).fillna(0)
            eff_display = hide_zero_rows(eff)
            total_ss_sku = eff['Safety_Stock'].sum(); total_net_demand_sku = eff['Agg_Future_Demand'].sum()
            sku_ratio = total_ss_sku / total_net_demand_sku if total_net_demand_sku > 0 else 0
            all_res = results[results['Period'] == snapshot_period] if snapshot_period is not None else results
            global_ratio = all_res['Safety_Stock'].sum() / all_res['Agg_Future_Demand'].replace(0, np.nan).sum() if not all_res.empty else 0

            m1, m2, m3 = st.columns(3)
            m1.metric(f"Network Ratio ({sku})", f"{sku_ratio:.2f}"); m2.metric("Global Network Ratio (All Items)", f"{global_ratio:.2f}")
            m3.metric("Total SS for Material", euro_format(int(total_ss_sku), True))
            st.markdown("---")
            c1, c2 = st.columns([2,1])
            with c1:
                fig_eff = px.scatter(eff_display, x="Agg_Future_Demand", y="Safety_Stock", color="Adjustment_Status",
                                     size="SS_to_FCST_Ratio", hover_name="Location",
                                     color_discrete_map={'Optimal (Statistical)': '#00CC96', 'Capped (High)': '#EF553B','Capped (Low)': '#636EFA', 'Forced to Zero': '#AB63FA'},
                                     title="Policy Impact & Efficiency Ratio (Bubble Size = SS_to_FCST_Ratio)")
                st.plotly_chart(fig_eff, use_container_width=True)
            with c2:
                st.markdown("**Status Breakdown**")
                st.table(eff_display['Adjustment_Status'].value_counts())
                st.markdown("**Top Nodes by Safety Stock (snapshot)**")
                eff_top = eff_display.sort_values('Safety_Stock', ascending=False)
                st.dataframe(
                    df_format_for_display(
                        eff_top[['Location', 'Adjustment_Status', 'Safety_Stock', 'SS_to_FCST_Ratio']].head(10),
                        cols=['Safety_Stock'],
                        two_decimals_cols=['Safety_Stock']
                    ),
                    use_container_width=True
                )

    # TAB 5: Forecast Accuracy
    with tab5:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            h_sku_default = default_product
            h_sku_index = all_products.index(h_sku_default) if h_sku_default in all_products else 0
            h_sku = st.selectbox("MATERIAL", all_products, index=h_sku_index, key="h1")

            h_loc_opts = sorted(results[results['Product'] == h_sku]['Location'].unique().tolist())
            if not h_loc_opts:
                h_loc_opts = sorted(hist[hist['Product'] == h_sku]['Location'].unique().tolist())
            if not h_loc_opts:
                h_loc_opts = ["(no location)"]
            h_loc_default = DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in h_loc_opts else (h_loc_opts[0] if h_loc_opts else "(no location)")
            h_loc_index = h_loc_opts.index(h_loc_default) if h_loc_default in h_loc_opts else 0
            h_loc = st.selectbox("LOCATION", h_loc_opts, index=h_loc_index, key="h2")

            if h_loc != "(no location)":
                badge_df = results[(results['Product'] == h_sku) & (results['Location'] == h_loc)]
            else:
                badge_df = results[results['Product'] == h_sku]
            render_selection_badge(product=h_sku, location=(h_loc if h_loc != "(no location)" else None), df_context=badge_df)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            st.subheader("📉 Historical Forecast vs Actuals")
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
                    k1.metric("WAPE (%)", f"{wape_val:.1f}"); k2.metric("Bias (%)", f"{bias_val:.1f}")
                else:
                    k1.metric("WAPE (%)", "N/A"); k2.metric("Bias (%)", "N/A")
                avg_acc = hdf['Accuracy_%'].mean() if not hdf['Accuracy_%'].isna().all() else np.nan
                k3.metric("Avg Accuracy (%)", f"{avg_acc:.1f}" if not np.isnan(avg_acc) else "N/A")

                fig_hist = go.Figure([go.Scatter(x=hdf['Period'], y=hdf['Consumption'], name='Actuals', line=dict(color='black')),
                                      go.Scatter(x=hdf['Period'], y=hdf['Forecast_Hist'], name='Forecast', line=dict(color='blue', dash='dot'))])
                st.plotly_chart(fig_hist, use_container_width=True)

                st.subheader("🌐 Aggregated Network History (Selected Product)")
                net_table = (hist_net[hist_net['Product'] == h_sku].merge(hdf[['Period']].drop_duplicates(), on='Period', how='inner').sort_values('Period').drop(columns=['Product']))
                if not net_table.empty:
                    net_table['Net_Abs_Error'] = (net_table['Network_Consumption'] - net_table['Network_Forecast_Hist']).abs()
                    denom_net = net_table['Network_Consumption'].replace(0, np.nan).sum()
                    net_wape = (net_table['Net_Abs_Error'].sum() / denom_net * 100) if denom_net > 0 else np.nan
                else:
                    net_wape = np.nan
                c_net1, c_net2 = st.columns([3,1])
                with c_net1:
                    if not net_table.empty:
                        st.dataframe(df_format_for_display(net_table[['Period', 'Network_Consumption', 'Network_Forecast_Hist']].copy(),
                                                          cols=['Network_Consumption','Network_Forecast_Hist'],
                                                          two_decimals_cols=['Network_Consumption','Network_Forecast_Hist']),
                                     use_container_width=True)
                    else:
                        st.write("No aggregated network history available for the chosen selection.")
                with c_net2:
                    c_val = f"{net_wape:.1f}" if not np.isnan(net_wape) else "N/A"
                    st.metric("Network WAPE (%)", c_val)

    # TAB 6: Calculation Trace & Simulation
    with tab6:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            calc_sku_default = default_product
            calc_sku_index = all_products.index(calc_sku_default) if calc_sku_default in all_products else 0
            calc_sku = st.selectbox("MATERIAL", all_products, index=calc_sku_index, key="c_sku")

            avail_locs = sorted(meaningful_results[meaningful_results['Product'] == calc_sku]['Location'].unique().tolist())
            if not avail_locs:
                avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique().tolist())
            if not avail_locs: avail_locs = ["(no location)"]
            calc_loc_default = DEFAULT_LOCATION_CHOICE if DEFAULT_LOCATION_CHOICE in avail_locs else (avail_locs[0] if avail_locs else "(no location)")
            calc_loc_index = avail_locs.index(calc_loc_default) if calc_loc_default in avail_locs else 0
            calc_loc = st.selectbox("LOCATION", avail_locs, index=calc_loc_index, key="c_loc")

            # period selection with labels
            if period_labels:
                try:
                    default_label = period_label(default_period) if default_period is not None else period_labels[-1]
                    calc_period_index = period_labels.index(default_label) if default_label in period_labels else len(period_labels)-1
                except Exception:
                    calc_period_index = len(period_labels)-1
                chosen_label = st.selectbox("PERIOD", period_labels, index=calc_period_index, key="c_period")
                calc_period = period_label_map.get(chosen_label, default_period)
            else:
                calc_period = CURRENT_MONTH_TS

            row_df_small = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)]
            render_selection_badge(product=calc_sku, location=calc_loc if calc_loc != "(no location)" else None, df_context=row_df_small)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            st.header("🧮 Transparent Calculation Engine & Scenario Simulation")
            st.write("See how changing service level or lead-time assumptions affects safety stock. The scenario planning area below is collapsed by default.")

            row_df = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)]
            if row_df.empty:
                st.warning("Selection not found in results.")
            else:
                row = row_df.iloc[0]

                st.markdown("---")
                st.subheader("1. Frozen Inputs (current)")
                i1, i2, i3, i4, i5 = st.columns(5)
                i1.metric("Service Level", f"{service_level*100:.2f}%", help=f"Z-Score: {z:.4f}")
                i2.metric("Network Demand (D, monthly)", euro_format(row['Agg_Future_Demand'], True), help="Aggregated Future Demand (monthly)")
                i3.metric("Network Std Dev (σ_D, monthly)", euro_format(row['Agg_Std_Hist'], True), help="Aggregated Historical Std Dev (monthly totals)")
                i4.metric("Avg Lead Time (L)", f"{row['LT_Mean']} days"); i5.metric("LT Std Dev (σ_L)", f"{row['LT_Std']} days")

                st.subheader("2. Statistical Calculation (Actual)")
                sigma_d_day = float(row['Agg_Std_Hist']) / math.sqrt(float(days_per_month))
                d_day = float(row['Agg_Future_Demand']) / float(days_per_month)
                demand_var_day = sigma_d_day**2
                if row['Agg_Future_Demand'] < 20.0:
                    demand_var_day = max(demand_var_day, d_day)
                term1_demand_var = demand_var_day * float(row['LT_Mean'])
                term2_supply_var = (float(row['LT_Std'])**2) * (d_day**2)
                combined_sd = math.sqrt(max(term1_demand_var + term2_supply_var, 0.0))
                raw_ss_calc = float(norm.ppf(service_level)) * combined_sd
                ss_floor = (d_day * float(row['LT_Mean'])) * 0.01

                st.latex(r"SS = Z \times \sqrt{\sigma_D^2 \cdot L \;+\; \sigma_L^2 \cdot D^2}")
                st.markdown("Where σ_D and D are daily values (converted from monthly inputs in the dataset).")
                st.markdown("**Step-by-Step Substitution (values used):**")
                st.code(f"""
    1. σ_D_daily = {sigma_d_day:.4f} (monthly σ / sqrt(days_per_month))
       Demand component variance = σ_D_daily^2 * L = {term1_demand_var:.4f}
    2. Supply component variance = σ_L^2 * D_daily^2 = ({row['LT_Std']:.4f})^2 * ({d_day:.4f})^2
       = {term2_supply_var:.4f}
    3. Combined variance = {term1_demand_var:.4f} + {term2_supply_var:.4f} = {term1_demand_var + term2_supply_var:.4f}
    4. Combined Std Dev = sqrt(Combined Variance) = {combined_sd:.4f}
    5. Raw SS = Z({service_level*100:.2f}%) * {combined_sd:.4f} = {raw_ss_calc:.2f} units
    6. Floor applied (1% of mean LT demand) = {ss_floor:.2f} units
    7. Pre-rule SS (max of raw vs floor) = {max(raw_ss_calc, ss_floor):.2f} units
    """)
                st.info(f"🧮 **Resulting Pre-rule SS:** {euro_format(max(raw_ss_calc, ss_floor), True)} units")

                # Scenario planning (analysis-only, does not alter implemented policy)
                with st.expander("Scenario Planning (expand to configure scenarios)", expanded=False):
                    st.write("Use scenarios to test sensitivity to Service Level or Lead Time. Scenarios do not change implemented policy — they are analysis-only.")
                    if 'n_scen' not in st.session_state:
                        st.session_state['n_scen'] = 1
                    options = [1,2,3]
                    default_index = options.index(st.session_state.get('n_scen',1)) if st.session_state.get('n_scen',1) in options else 0
                    n_scen = st.selectbox("Number of Scenarios to compare", options, index=default_index, key="n_scen")
                    scenarios = []
                    for s in range(n_scen):
                        with st.expander(f"Scenario {s+1} inputs", expanded=False):
                            sc_sl_default = float(service_level*100) if s==0 else min(99.9, float(service_level*100) + 0.5*s)
                            sc_sl = st.slider(f"Scenario {s+1} Service Level (%)", 50.0, 99.9, sc_sl_default, key=f"sc_sl_{s}")
                            sc_lt = st.slider(f"Scenario {s+1} Service Level Avg Lead Time (Days)", 0.0, max(30.0, float(row['LT_Mean'])*2), value=float(row['LT_Mean'] if s==0 else row['LT_Mean']), key=f"sc_lt_{s}")
                            sc_lt_std = st.slider(f"Scenario {s+1} LT Std Dev (Days)", 0.0, max(10.0, float(row['LT_Std'])*2), value=float(row['LT_Std'] if s==0 else row['LT_Std']), key=f"sc_lt_std_{s}")
                            scenarios.append({'SL_pct': sc_sl, 'LT_mean': sc_lt, 'LT_std': sc_lt_std})

                    scen_rows = []
                    for idx, sc in enumerate(scenarios):
                        sc_z = norm.ppf(sc['SL_pct']/100.0)
                        d_day = float(row['Agg_Future_Demand']) / float(days_per_month)
                        sigma_d_day = float(row['Agg_Std_Hist']) / math.sqrt(float(days_per_month))
                        var_d = sigma_d_day**2
                        if row['Agg_Future_Demand'] < 20.0:
                            var_d = max(var_d, d_day)
                        sc_ss = sc_z * math.sqrt(var_d * sc['LT_mean'] + (sc['LT_std']**2) * (d_day**2))
                        sc_floor = (d_day * sc['LT_mean']) * 0.01
                        sc_ss = max(sc_ss, sc_floor)
                        scen_rows.append({
                            'Scenario': f"S{idx+1}",
                            'Service_Level_%': sc['SL_pct'],
                            'LT_mean_days': sc['LT_mean'],
                            'LT_std_days': sc['LT_std'],
                            'Simulated_SS': sc_ss
                        })
                    scen_df = pd.DataFrame(scen_rows)

                    base_row = {'Scenario': 'Base (Stat)', 'Service_Level_%': service_level*100, 'LT_mean_days': row['LT_Mean'], 'LT_std_days': row['LT_Std'], 'Simulated_SS': row['Pre_Rule_SS']}
                    impl_row = {'Scenario': 'Implemented', 'Service_Level_%': np.nan, 'LT_mean_days': np.nan, 'LT_std_days': np.nan, 'Simulated_SS': row['Safety_Stock']}
                    compare_df = pd.concat([pd.DataFrame([base_row, impl_row]), scen_df], ignore_index=True, sort=False)
                    display_comp = compare_df.copy()
                    display_comp['Simulated_SS'] = display_comp['Simulated_SS'].astype(float)
                    st.markdown("Scenario comparison (Simulated SS). 'Implemented' shows the final Safety_Stock after rules.")
                    st.dataframe(df_format_for_display(display_comp[['Scenario','Service_Level_%','LT_mean_days','LT_std_days','Simulated_SS']].copy(),
                                                      cols=['Service_Level_%','LT_mean_days','LT_std_days','Simulated_SS'],
                                                      two_decimals_cols=['Service_Level_%','Simulated_SS']),
                                 use_container_width=True)

                    fig_bar = go.Figure()
                    colors = px.colors.qualitative.Pastel
                    fig_bar.add_trace(go.Bar(x=display_comp['Scenario'], y=display_comp['Simulated_SS'], marker_color=colors[:len(display_comp)]))
                    fig_bar.update_layout(title="Scenario SS Comparison", yaxis_title="SS (units)")
                    st.plotly_chart(fig_bar, use_container_width=True)

                    sel_lt = scenarios[0]['LT_mean'] if len(scenarios)>0 else row['LT_Mean']
                    sel_lt_std = scenarios[0]['LT_std'] if len(scenarios)>0 else row['LT_Std']
                    sl_range = np.linspace(50.0, 99.9, 100)
                    ss_curve = []
                    for slev in sl_range:
                        zz = norm.ppf(slev/100.0)
                        sigma_d_day = float(row['Agg_Std_Hist']) / math.sqrt(float(days_per_month))
                        d_day = float(row['Agg_Future_Demand']) / float(days_per_month)
                        var_d = sigma_d_day**2
                        if row['Agg_Future_Demand'] < 20.0:
                            var_d = max(var_d, d_day)
                        val = zz * math.sqrt(var_d * sel_lt + (sel_lt_std**2) * (d_day**2))
                        ss_curve.append(val)
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(x=sl_range, y=ss_curve, mode='lines', line=dict(color='#0b3d91')))
                    if len(scenarios)>0:
                        fig_curve.add_vline(x=scenarios[0]['SL_pct'], line_dash="dash", line_color="red", annotation_text=f"Scenario SL {scenarios[0]['SL_pct']:.1f}%", annotation_position="top right")
                    fig_curve.update_layout(title="SS Sensitivity to Service Level (Scenario 1 LT assumptions)", xaxis_title="Service Level (%)", yaxis_title="Simulated SS (units)")
                    st.plotly_chart(fig_curve, use_container_width=True)

                st.subheader("4. Business Rules & Diagnostics — explanation & diagnostics")
                st.markdown(r"""
                Background & explanation of the calculation steps and business-rule adjustments.
                """)
                st.latex(r"SS = Z \cdot \sqrt{\mathrm{Var}(D_{\text{day}})\cdot L \;+\; (\bar{D}_{\text{day}})^2 \cdot \mathrm{Var}(L)}")
                st.markdown(r"""
                Capping policy (if enabled):
                """)
                st.latex(r"\text{lower\_limit} = D \times \text{cap\_min\_pct}")
                st.latex(r"\text{upper\_limit} = D \times \text{cap\_max\_pct}")
                st.markdown(r"""
                Then: Safety_Stock = clip(SS_{pre}, lower_limit, upper_limit)
                with exceptions:
                - If D == 0 and zero suppression rule is enabled -> Safety_Stock = 0
                - Explicit overrides (e.g., specific locations such as B616) may force 0
                """)
                # add some vertical space for clarity before the checks
                st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

                # Shadowed separator
                st.markdown("""<div style="height:12px;background:#f0f0f2;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.07);margin:12px 0;"></div>""", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Zero Demand Rule**")
                    if zero_if_no_net_fcst and row['Agg_Future_Demand'] <= 0:
                        st.error("❌ Network Demand is 0. SS Forced to 0.")
                    else:
                        st.success("✅ Network Demand exists. Keep Statistical SS.")
                with c2:
                    st.markdown("**Capping (Min/Max) Diagnostics**")
                    if apply_cap:
                        lower_limit = row['Agg_Future_Demand'] * (cap_range[0]/100)
                        upper_limit = row['Agg_Future_Demand'] * (cap_range[1]/100)
                        st.write(f"Constraint: {int(cap_range[0])}% to {int(cap_range[1])}% of Demand")
                        st.write(f"Lower limit = Agg_Future_Demand * {cap_range[0]}% = {euro_format(lower_limit, True)}")
                        st.write(f"Upper limit = Agg_Future_Demand * {cap_range[1]}% = {euro_format(upper_limit, True)}")
                        if raw_ss_calc > upper_limit:
                            st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) > Max Cap ({euro_format(upper_limit, True)}). Capping downwards to {euro_format(upper_limit, True)}.")
                        elif raw_ss_calc < lower_limit and row['Agg_Future_Demand'] > 0:
                            st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) < Min Cap ({euro_format(lower_limit, True)}). Raising to {euro_format(lower_limit, True)}.")
                        else:
                            st.success("✅ Raw SS is within caps (no capping applied).")
                    else:
                        st.write("Capping logic disabled.")

    
    # TAB 7: By Material
    with tab7:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            sel_prod_default = default_product
            sel_prod_index = all_products.index(sel_prod_default) if sel_prod_default in all_products else 0
            selected_product = st.selectbox("MATERIAL", all_products, index=sel_prod_index, key="mat_sel")

            # period selection with labels
            if period_labels:
                try:
                    sel_label = period_label(default_period) if default_period is not None else period_labels[-1]
                    sel_period_index = period_labels.index(sel_label) if sel_label in period_labels else len(period_labels)-1
                except Exception:
                    sel_period_index = len(period_labels)-1
                chosen_label = st.selectbox("PERIOD", period_labels, index=sel_period_index, key="mat_period")
                selected_period = period_label_map.get(chosen_label, default_period)
            else:
                selected_period = CURRENT_MONTH_TS

            render_selection_badge(product=selected_product, location=None, df_context=results[(results['Product']==selected_product)&(results['Period']==selected_period)])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            st.header("📦 View by Material (Single Material Focus + 8 Reasons for Inventory)")
            mat_period_df = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)].copy()
            mat_period_df_display = hide_zero_rows(mat_period_df)
            total_forecast = mat_period_df['Forecast'].sum(); total_net = mat_period_df['Agg_Future_Demand'].sum()
            total_ss = mat_period_df['Safety_Stock'].sum(); nodes_count = mat_period_df['Location'].nunique()
            avg_ss_per_node = (mat_period_df['Safety_Stock'].mean() if nodes_count > 0 else 0)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Local Forecast", euro_format(total_forecast, True)); k2.metric("Total Network Demand", euro_format(total_net, True))
            k3.metric("Total Safety Stock (sum nodes)", euro_format(total_ss, True)); k4.metric("Nodes", f"{nodes_count}"); k5.metric("Avg SS per Node", euro_format(avg_ss_per_node, True))

            st.markdown("### Why do we carry this SS? — 8 Reasons breakdown (aggregated for selected material)")
            if mat_period_df_display.empty:
                st.warning("No data for this material/period (non-zero rows filtered).")
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
                drv_df_display = drv_df[drv_df['amount'] != 0].copy()
                drv_denom = drv_df['amount'].sum()
                drv_df_display['pct_of_total_ss'] = drv_df_display['amount'] / (drv_denom if drv_denom > 0 else 1.0) * 100

                st.markdown("#### A. Original — Raw driver values (interpretation view)")
                pastel_colors = px.colors.qualitative.Pastel
                fig_drv_raw = go.Figure()
                color_slice = pastel_colors[:len(drv_df_display)] if len(drv_df_display)>0 else pastel_colors
                fig_drv_raw.add_trace(go.Bar(x=drv_df_display['driver'], y=drv_df_display['amount'], marker_color=color_slice))
                annotations_raw = []
                for idx, rowd in drv_df_display.iterrows():
                    annotations_raw.append(dict(x=rowd['driver'], y=rowd['amount'], text=f"{rowd['pct_of_total_ss']:.1f}%", showarrow=False, yshift=8))
                fig_drv_raw.update_layout(title=f"{selected_product} — Raw Drivers (not SS-attribution)", xaxis_title="Driver", yaxis_title="Units", annotations=annotations_raw, height=420)
                st.plotly_chart(fig_drv_raw, use_container_width=True)
                st.markdown("Driver table (raw numbers and % of raw-sum)")
                st.dataframe(df_format_for_display(drv_df_display.rename(columns={'driver':'Driver','amount':'Units','pct_of_total_ss':'Pct_of_raw_sum'}).round(2),
                                                  cols=['Units','Pct_of_raw_sum'],
                                                  two_decimals_cols=['Pct_of_raw_sum']),
                             use_container_width=True)

                # B. SS Attribution — mutually exclusive components that sum to total SS
                st.markdown("---")
                st.markdown("#### B. SS Attribution — Mutually exclusive components that SUM EXACTLY to Total Safety Stock")
                per_node = mat.copy()
                per_node['is_forced_zero'] = per_node['Adjustment_Status'] == 'Forced to Zero'
                per_node['is_b616_override'] = (per_node['Location'] == 'B616') & (per_node['Safety_Stock'] == 0)
                per_node['pre_ss'] = per_node['Pre_Rule_SS'].clip(lower=0)
                per_node['share_denom'] = per_node['demand_uncertainty_raw'] + per_node['lt_uncertainty_raw']
                def demand_share_calc(r):
                    if r['share_denom'] > 0:
                        return r['pre_ss'] * (r['demand_uncertainty_raw'] / r['share_denom'])
                    else:
                        return (r['pre_ss'] / 2) if r['pre_ss'] > 0 else 0.0
                per_node['demand_share'] = per_node.apply(demand_share_calc, axis=1)
                def lt_share_calc(r):
                    if r['share_denom'] > 0:
                        return r['pre_ss'] * (r['lt_uncertainty_raw'] / r['share_denom'])
                    else:
                        return (r['pre_ss'] / 2) if r['pre_ss'] > 0 else 0.0
                per_node['lt_share'] = per_node.apply(lt_share_calc, axis=1)
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
                ss_drv_df_display = ss_drv_df[ss_drv_df['amount'] != 0].copy()
                denom = total_ss if total_ss > 0 else ss_drv_df['amount'].sum()
                denom = denom if denom > 0 else 1.0
                ss_drv_df_display['pct_of_total_ss'] = ss_drv_df_display['amount'] / denom * 100

                labels = ss_drv_df_display['driver'].tolist() + ['Total SS']
                values = ss_drv_df_display['amount'].tolist() + [total_ss]
                measures = ["relative"] * len(ss_drv_df_display) + ["total"]
                pastel_inc = pastel_colors[0] if len(pastel_colors) > 0 else '#A3C1DA'
                pastel_dec = pastel_colors[1] if len(pastel_colors) > 1 else '#F6C3A0'
                pastel_tot = pastel_colors[2] if len(pastel_colors) > 2 else '#CFCFCF'
                fig_drv = go.Figure(go.Waterfall(
                    name="SS Attribution",
                    orientation="v",
                    measure=measures,
                    x=labels,
                    y=values,
                    text=[f"{v:,.0f}" for v in ss_drv_df_display['amount'].tolist()] + [f"{total_ss:,.0f}"],
                    connector={"line":{"color":"rgba(63,63,63,0.25)"}},
                    decreasing=dict(marker=dict(color=pastel_dec)),
                    increasing=dict(marker=dict(color=pastel_inc)),
                    totals=dict(marker=dict(color=pastel_tot))
                ))
                fig_drv.update_layout(title=f"{selected_product} — SS Attribution Waterfall (adds to {euro_format(total_ss, True)})", xaxis_title="Driver", yaxis_title="Units", height=420)
                st.plotly_chart(fig_drv, use_container_width=True)

                st.markdown("SS Attribution table (numbers and % of total SS)")
                st.dataframe(df_format_for_display(ss_drv_df_display.rename(columns={'driver':'Driver','amount':'Units','pct_of_total_ss':'Pct_of_total_SS'}).round(2),
                                                  cols=['Units','Pct_of_total_SS'],
                                                  two_decimals_cols=['Pct_of_total_SS']),
                             use_container_width=True)

                grand_forecast = mat_period_df['Forecast'].sum()
                grand_net = mat_period_df['Agg_Future_Demand'].sum()
                grand_ss = mat_period_df['Safety_Stock'].sum()
                summary_html = f"""
                <div style="margin-top:12px;">
                  <table style="border-collapse:collapse;">
                    <tr>
                      <td style="padding:2px 12px;font-size:12px;">Grand Totals</td>
                      <td style="padding:2px 12px;font-size:12px;">Local Demand</td>
                      <td style="padding:2px 12px;font-size:12px;">Total Network Demand</td>
                      <td style="padding:2px 12px;font-size:12px;">Safety Stock</td>                      
                    </tr>
                    <tr style="color:#666">
                      <td style="padding:6px 12px;"><strong>Grand Totals</strong></td>
                      <td style="padding:6px 12px;"><strong>{euro_format(grand_forecast, True)}</strong></td>
                      <td style="padding:6px 12px;"><strong>{euro_format(grand_net, True)}</strong></td>
                      <td style="padding:6px 12px;"><strong>{euro_format(grand_ss, True)}</strong></td>
                    </tr>
                  </table>
                </div>
                """
                st.markdown(summary_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Top Locations by Safety Stock (snapshot)")
        top_nodes = mat_period_df.sort_values('Safety_Stock', ascending=False)[['Location','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status']]
        top_nodes_display = hide_zero_rows(top_nodes)
        st.dataframe(
            df_format_for_display(
                top_nodes_display.head(25).copy(),
                cols=['Forecast','Agg_Future_Demand','Safety_Stock'],
                two_decimals_cols=['Forecast']
            ),
            use_container_width=True,
            height=400
        )

        st.markdown("---")
        st.subheader("Export — Material Snapshot")
        if not mat_period_df.empty:
            try:
                filename = f"material_{selected_product}_{pd.to_datetime(selected_period).strftime('%Y-%m')}.csv"
            except Exception:
                filename = f"material_{selected_product}_snapshot.csv"
            st.download_button("📥 Download Material Snapshot (CSV)", data=mat_period_df.to_csv(index=False), file_name=filename, mime="text/csv")
        else:
            st.write("No snapshot available to download for this selection.")

else:
    st.info("Please upload sales.csv, demand.csv and leadtime.csv in the sidebar to run the optimizer.")
