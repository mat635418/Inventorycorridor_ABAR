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
import collections

st.set_page_config(page_title="MEIO for RM", layout="wide")

# Logo configuration
LOGO_FILENAME = "GY_logo.jpg"
LOGO_BASE_WIDTH = 160

# Fixed conversion (30 days/month)
days_per_month = 30

st.markdown("<h1 style='margin:0; padding-top:6px;'>MEIO for Raw Materials — v0.96 — Jan 2026</h1>", unsafe_allow_html=True)

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

      /* Make Streamlit download buttons smaller and more compact */
      .stDownloadButton button, .stDownloadButton > button {
        padding: 0.35rem 0.6rem !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
      }

      /* Slightly smaller base font for most UI elements (esp. Tab 7 request) */
      .main, .block-container, .stDataFrame, .stMarkdown, .stMetric {
        font-size: 0.88rem;
      }

      /* Global style for generic export buttons (new icon) */
      .export-csv-btn button {
        background-color: #0b3d91 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        padding: 0.3rem 0.9rem !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
      }
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
    If show_zero=True, zeros will be shown as '0' (or '0.00' if requested)
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
                d[c] = d[c].apply(lambda v: ("{:.2f}".format(v) if (pd.notna(v) and isinstance(v, (int,float))) else euro_format(v, always_two_decimals=True)))
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
    - DataFrame with Product, Location, Period, Agg_Future_Demand, Agg_Future_Internal, Agg_Future_External, Agg_Std_Hist
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
                    'Agg_Future_Internal': local_fcst,
                    'Agg_Future_External': child_fcst_sum,
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
    Compact selection indicator.

    NOTE: previous "blue box" UI (with numeric tiles) was intentionally removed.
    This function now renders a minimal, neutral selection indicator (light background)
    showing only the current selection (product and optional location). It is safe
    to call from each tab's right column; the heavy blue badge and numeric content
    are no longer displayed as requested.

    This ensures the current selection is visible in every tab without the large
    blue box and numbers to the right.
    """
    if product is None or product == "":
        return
    title = f"{product}{(' — ' + location) if location else ''}"
    badge_html = f"""
    <div style="background:#f3f6f9;padding:8px;border-radius:8px;color:#222;max-width:100%;font-family:inherit;">
      <div style="font-size:11px;opacity:0.85;margin-bottom:4px;">Selected</div>
      <div style="font-size:13px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</div>
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
# SIDEBAR: collapsible sections
# ----------------------

# Make the first expander visible (expanded) by default.
# We replaced the previous free-form example text with a compact slider-only control and kept z computation.
with st.sidebar.expander("⚙️ Service Level Configuration", expanded=True):
    service_level = st.slider(
        "Service Level (%) — end-nodes (hop 0)",
        50.0,
        99.9,
        99.0,
        help="Target probability of not stocking out for end-nodes (hop 0). Upstream nodes get fixed SLs by hop-distance."
    ) / 100

    # compute Z for display purposes
    z = norm.ppf(service_level)

with st.sidebar.expander("⚙️ Safety Stock Rules", expanded=True):
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

# Aggregation & Uncertainty controls removed.
use_transitive = True
var_rho = 1.0
lt_mode = "Apply LT variance"

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

# Modified run_pipeline to accept explicit data and parameters so it is pure (no hidden globals).
# Tiering parameters changed: fixed hop -> SL mapping (hops 0..3). Hops > 3 map to hop 3 SL.
def run_pipeline(df_d, stats, df_lt, service_level,
                 transitive=True, rho=1.0, lt_mode_param='Apply LT variance',
                 zero_if_no_net_fcst=True, apply_cap=True, cap_range=(0,200)):
    """
    Run aggregation -> stats -> safety-stock pipeline.

    Tiering rules (fixed mapping):
    - Hop distance measured forward to nearest end-node.
    - Service level mapping (fixed):
        hop 0 (end-node): 99.0%
        hop 1: 95.0%
        hop 2: 90.0%
        hop 3 and above: 85.0%
    - B616 special override: Safety_Stock forced to 0 (policy).
    - The slider 'service_level' remains for ad-hoc scenario comparisons but the implemented policy uses the fixed mapping above.
    """

    # explicit hop->SL mapping (values as fractions)
    hop_to_sl = {0: 0.99, 1: 0.95, 2: 0.90, 3: 0.85}
    def sl_for_hop(h):
        if h in hop_to_sl:
            return hop_to_sl[h]
        # any hops > 3 use hop 3 SL
        return hop_to_sl[3]

    base_sl = float(service_level)

    # 1) aggregate demand & historical variance through the network
    network_stats, reachable_map = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt, transitive=transitive, rho=rho)

    # 2) per-node lead time averages (map To_Location -> node LT)
    node_lt_local = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt_local.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # 3) merge aggregated stats with forecasts and LT info
    res = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    res = pd.merge(res, node_lt_local, on=['Product', 'Location'], how='left')

    # sensible defaults to avoid NaNs in downstream calculations
    res = res.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0, 'Agg_Future_Internal': 0, 'Agg_Future_External': 0})

    # if Agg_Std_Hist is missing, fall back to product median, then global median
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    global_median_std = stats['Local_Std'].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0
    res['Agg_Std_Hist'] = res.apply(lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'], axis=1)

    # Ensure numeric types for calculations
    res['Agg_Std_Hist'] = res['Agg_Std_Hist'].astype(float)
    res['LT_Mean'] = res['LT_Mean'].astype(float)
    res['LT_Std'] = res['LT_Std'].astype(float)
    res['Agg_Future_Demand'] = res['Agg_Future_Demand'].astype(float)
    res['Agg_Future_Internal'] = res['Agg_Future_Internal'].astype(float)
    res['Agg_Future_External'] = res['Agg_Future_External'].astype(float)
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
            lt_comp = (float(row['LT_Std'])**2) * (D_for_LT**2)
        elif lt_mode_param == 'Average LT Std across downstream':
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

    # Build children mapping per product (used to compute hops to end-nodes)
    def compute_hop_distances_for_product(p_lt_df, prod_nodes):
        """
        Compute hop distances to nearest end-node (leaf = node with no outgoing edges).
        ENFORCE forward orientation: From_Location -> To_Location.
        Returns dictionary: node -> distance (int)
        """
        children = {}
        for _, r in p_lt_df.iterrows():
            f = r.get('From_Location', None)
            t = r.get('To_Location', None)
            if pd.isna(f) or pd.isna(t):
                continue
            children.setdefault(f, set()).add(t)

        all_nodes = set(prod_nodes)
        if not p_lt_df.empty:
            all_nodes = all_nodes.union(set(p_lt_df['From_Location'].dropna().unique())).union(set(p_lt_df['To_Location'].dropna().unique()))

        # end-node (leaf) = nodes that do not appear as a key in children (no outgoing edges)
        leaf_nodes = set([n for n in all_nodes if n not in children or len(children.get(n, set())) == 0])

        distances = {}
        if not leaf_nodes:
            # no leaves found (cycle or malformed), treat all as leaf (0)
            for n in all_nodes:
                distances[n] = 0
            return distances

        # For each node, BFS forward until we hit a leaf (node with no children)
        for n in all_nodes:
            if n in leaf_nodes:
                distances[n] = 0
                continue
            q = collections.deque()
            q.append((n, 0))
            visited = set([n])
            found = False
            while q:
                cur, depth = q.popleft()
                kids = children.get(cur, set())
                if not kids:
                    distances[n] = depth
                    found = True
                    break
                for k in kids:
                    if k not in visited:
                        visited.add(k)
                        q.append((k, depth + 1))
            if not found:
                distances[n] = 0
        return distances

    # Prepare lookups by product to avoid repeated building
    products = res['Product'].unique()
    prod_to_nodes = {p: set(res[res['Product'] == p]['Location'].unique().tolist()) for p in products}
    prod_to_routes = {}
    if 'Product' in df_lt.columns:
        for p in df_lt['Product'].unique():
            prod_to_routes[p] = df_lt[df_lt['Product'] == p].copy()
    else:
        prod_to_routes[None] = df_lt.copy()

    prod_distances = {}
    # Fallback to full network if product-specific routes are missing
    for p in products:
        p_routes = prod_to_routes.get(p, df_lt.copy())
        nodes = prod_to_nodes.get(p, set())
        prod_distances[p] = compute_hop_distances_for_product(p_routes, nodes)

    # ========== Apply fixed overrides for special nodes ==========
    # Ensure the three special nodes always report the requested hops.
    special_hops = {'B616': 4, 'BEEX': 3, 'LUEX': 2}
    for p, distances in prod_distances.items():
        for node, fixed_hop in special_hops.items():
            # set/override regardless of computed distances so diagnostics always show these fixed values
            distances[node] = int(fixed_hop)
    # ==============================================================

    # Prepare per-product diagnostics summary (fixed mapping)
    product_tiering_params = {}
    for p, distances in prod_distances.items():
        max_hops = int(max(distances.values())) if distances else 0
        tier_map_pct = {f"SL_hop_{h}_pct": float(sl_for_hop(h) * 100.0) for h in range(0,4)}
        tier_map_pct['max_tier_hops'] = max_hops
        product_tiering_params[p] = tier_map_pct

    # Assign service level per row based on hop mapping
    sl_list = []
    hop_distance_list = []
    for idx, row in res.iterrows():
        prod = row['Product']
        loc = row['Location']
        distances = prod_distances.get(prod, {})
        dist = int(distances.get(loc, 0))
        sl_node = sl_for_hop(dist)
        sl_list.append(sl_node)
        hop_distance_list.append(dist)
    res['Tier_Hops'] = np.array(hop_distance_list)
    res['Service_Level_Node'] = np.array(sl_list)
    # compute z per-node
    res['Z_node'] = res['Service_Level_Node'].apply(lambda x: float(norm.ppf(x)))

    res['SS_stat'] = res.apply(lambda r: r['Z_node'] * math.sqrt(max(0.0, (demand_component.loc[r.name] + r['lt_component']))), axis=1)

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
    # B616 override: zero SS by default (overseas supplier / GOCPL)
    res.loc[res['Location'] == 'B616', 'Safety_Stock'] = 0
    res['Max_Corridor'] = res['Safety_Stock'] + res['Forecast']

    # New diagnostics: Days covered by Safety Stock (safe handling)
    def compute_days_covered(r):
        try:
            d = float(r.get('D_day', 0.0))
            if d <= 0:
                return np.nan
            return float(r.get('Safety_Stock', 0.0)) / d
        except Exception:
            return np.nan
    res['Days_Covered_by_SS'] = res.apply(compute_days_covered, axis=1)

    # Attach tiering diagnostics summary for transparency (one row per product)
    tier_summary = []
    for p, params in product_tiering_params.items():
        tier_summary.append({
            'Product': p,
            'SL_hop_0_pct': params.get('SL_hop_0_pct', 99.0),
            'SL_hop_1_pct': params.get('SL_hop_1_pct', 95.0),
            'SL_hop_2_pct': params.get('SL_hop_2_pct', 90.0),
            'SL_hop_3_pct': params.get('SL_hop_3_pct', 85.0),
            'max_tier_hops': params.get('max_tier_hops', 0)
        })
    tier_df = pd.DataFrame(tier_summary)
    # store as attribute on res for optional downstream use
    res.attrs['tiering_params'] = tier_df

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
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0
    prod_medians = stats.groupby('Product')['Local_Std'].median().to_dict()
    def fill_local_std(row):
        if not pd.isna(row['Local_Std']) and row['Local_Std'] > 0:
            return row['Local_Std']
        pm = prod_medians.get(row['Product'], np.nan)
        return pm if not pd.isna(pm) else global_median_std
    stats['Local_Std'] = stats.apply(fill_local_std, axis=1)

    # Run the main pipeline with explicit parameters (ensures updates when sidebar changes)
    results, reachable_map = run_pipeline(
        df_d=df_d,
        stats=stats,
        df_lt=df_lt,
        service_level=service_level,
        transitive=use_transitive,
        rho=var_rho,
        lt_mode_param=lt_mode,
        zero_if_no_net_fcst=zero_if_no_net_fcst,
        apply_cap=apply_cap,
        cap_range=cap_range
    )

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

    # TABS (removed Tiering Diagnostics tab)
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
            # minimal selection indicator only (removed previous blue box with numeric tiles)
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

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # Show avg daily demand and days covered for the current month if available
            try:
                summary_row = results[(results['Product'] == sku) & (results['Location'] == loc) & (results['Period'] == CURRENT_MONTH_TS)]
                if not summary_row.empty:
                    srow = summary_row.iloc[0]
                    avg_daily = srow.get('D_day', np.nan)
                    days_cov = srow.get('Days_Covered_by_SS', np.nan)
                    avg_daily_txt = f"{avg_daily:.2f} units/day" if pd.notna(avg_daily) else "N/A"
                    days_cov_txt = f"{days_cov:.1f} days" if pd.notna(days_cov) else "N/A"
                    st.markdown(f"Avg Daily Demand: **{avg_daily_txt}**")
                    st.markdown(f"Safety Stock coverage: **{days_cov_txt}**")
            except Exception:
                # silent fallback
                pass

        with col_main:
            # Show the same compact selected text immediately at the start of the tab (as requested)
            st.markdown(f"**Selected**: {sku} — {loc}")
            # show all months in the graph: create a Period axis covering all_periods and merge selection data onto it
            plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')
            # create DataFrame covering all periods (even if zeros)
            df_all_periods = pd.DataFrame({'Period': all_periods})
            plot_full = pd.merge(
                df_all_periods,
                plot_df[['Period', 'Max_Corridor', 'Safety_Stock', 'Forecast', 'Agg_Future_Internal', 'Agg_Future_External']],
                on='Period',
                how='left'
            )

            # fill missing with zeros so months with no data are shown explicitly
            plot_full[['Max_Corridor', 'Safety_Stock', 'Forecast', 'Agg_Future_Internal', 'Agg_Future_External']] = (
                plot_full[['Max_Corridor', 'Safety_Stock', 'Forecast', 'Agg_Future_Internal', 'Agg_Future_External']].fillna(0)
            )

            # Always show Max Corridor (removed toggle)
            traces = [
                go.Scatter(x=plot_full['Period'], y=plot_full['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=1, color='rgba(0,0,0,0.1)')),
                go.Scatter(x=plot_full['Period'], y=plot_full['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
                go.Scatter(x=plot_full['Period'], y=plot_full['Forecast'], name='Local Direct Demand (Internal)', line=dict(color='black', dash='dot')),
                go.Scatter(x=plot_full['Period'], y=plot_full['Agg_Future_External'], name='External Network Demand (Downstream)', line=dict(color='blue', dash='dash'))
            ]

            fig = go.Figure(traces)
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units', xaxis=dict(tickformat="%b\n%Y"))
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Network Topology
    with tab2:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
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

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            # Repeat section title at top of tab
            st.subheader("🕸️ Network Topology")
            # Show selection at start of tab (product + period)
            st.markdown(f"**Selected**: {sku} — {period_label(chosen_period)}")
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
            # build node set safely (drop NA)
            if not sku_lt.empty:
                froms = set(sku_lt['From_Location'].dropna().unique().tolist())
                tos = set(sku_lt['To_Location'].dropna().unique().tolist())
                all_nodes = froms.union(tos).union(hubs)
            else:
                all_nodes = set(hubs)

            demand_lookup = {}
            for n in all_nodes:
                demand_lookup[n] = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Internal': 0, 'Agg_Future_External': 0, 'Safety_Stock': 0, 'Tier_Hops': np.nan, 'Service_Level_Node': np.nan, 'D_day': 0, 'Days_Covered_by_SS': np.nan})

            for n in sorted(all_nodes):
                m = demand_lookup.get(n, {'Forecast': 0, 'Agg_Future_Internal': 0, 'Agg_Future_External': 0, 'Safety_Stock': 0, 'Tier_Hops': np.nan, 'Service_Level_Node': np.nan, 'D_day': 0, 'Days_Covered_by_SS': np.nan})
                # Node considered 'used' visually if any of the metrics are > 0
                used = (float(m.get('Agg_Future_External', 0)) > 0) or (float(m.get('Forecast', 0)) > 0)

                if n == 'B616':
                    bg = '#dcedc8'; border = '#8bc34a'; font_color = '#0b3d91'; size = 14
                elif n == 'BEEX' or n == 'LUEX':
                    bg = '#bbdefb'; border = '#64b5f6'; font_color = '#0b3d91'; size = 14
                else:
                    if used:
                        bg = '#fff9c4'; border = '#fbc02d'; font_color = '#222222'; size = 12
                    else:
                        # even lighter grey for inactive nodes
                        bg = '#f7f7f7'; border = '#e0e0e0'; font_color = '#b0b0b0'; size = 10

                # Try to display node SL where available
                sl_node = m.get('Service_Level_Node', None)
                sl_label = "-"
                if pd.notna(sl_node):
                    try:
                        sl_label = f"{float(sl_node) * 100:.2f}%"
                    except Exception:
                        sl_label = str(sl_node)

                # Build the label (location in bold, no hops in label)
                lbl = (
                    f"**{n}**\\n"
                    f"LDD: {euro_format(m.get('Forecast', 0), show_zero=True)}\\n"
                    f"EXT: {euro_format(m.get('Agg_Future_External', 0), show_zero=True)}\\n"
                    f"SS: {euro_format(m.get('Safety_Stock', 0), show_zero=True)}\\n"
                    f"SL: {sl_label}"
                )
                # pyvis doesn't understand Markdown, so keep visual emphasis by uppercase & spacing
                lbl = lbl.replace("**", "")
                lbl = lbl.replace("\\n", "\n")
                net.add_node(n, label=lbl, title=lbl, color={'background': bg, 'border': border}, shape='box', font={'color': font_color, 'size': size})

            # add edges
            if not sku_lt.empty:
                for _, r in sku_lt.iterrows():
                    from_n, to_n = r['From_Location'], r['To_Location']
                    if pd.isna(from_n) or pd.isna(to_n):
                        continue
                    from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_External', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
                    to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_External', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
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

            # Legend immediately below network (no extra blank spacer after)
            st.markdown("""
                <div style="text-align:center; font-size:12px; padding:8px 0;">
                  <div style="display:inline-block; background:#f7f9fc; padding:8px 12px; border-radius:8px;">
                    <strong>Legend:</strong><br/>
                    LDD = Local Direct Demand (local forecast) &nbsp;&nbsp;|&nbsp;&nbsp;
                    EXT = External Demand (downstream forecasts rolled-up) &nbsp;&nbsp;|&nbsp;&nbsp;
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

            period_choices_labels = period_labels
            period_choices_ts = [period_label_map[lbl] for lbl in period_choices_labels]

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

            # Filters aligned with other tabs: rely on global CSS for blue chips, no extra overrides here
            f_prod = st.multiselect("MATERIAL", prod_choices, default=default_prod_list, key="full_f_prod")
            f_loc = st.multiselect("LOCATION", loc_choices, default=[], key="full_f_loc")
            f_period_labels = st.multiselect("PERIOD", period_choices_labels, default=default_period_list, key="full_f_period")

            f_period = [period_label_map[lbl] for lbl in f_period_labels] if f_period_labels else []

            # Generic Export to CSV button (new icon)
            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=results.to_csv(index=False),
                    file_name="filtered_plan.csv",
                    mime="text/csv",
                    key="full_plan_export",
                )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            # Show selection at the top of the tab (product(s) / locations / periods)
            badge_product = f_prod[0] if f_prod else (default_product if default_product in all_products else (all_products[0] if all_products else ""))
            badge_loc = ", ".join(f_loc) if f_loc else ""
            badge_period = ", ".join(f_period_labels) if f_period_labels else ""
            selected_text = badge_product
            if badge_loc:
                selected_text += f" — {badge_loc}"
            elif badge_period:
                selected_text += f" — {badge_period}"
            st.markdown(f"**Selected**: {selected_text}")

            st.subheader("📋 Global Inventory Plan")
            filtered = results.copy()
            if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
            if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
            if f_period: filtered = filtered[filtered['Period'].isin(f_period)]
            filtered = filtered.sort_values('Safety_Stock', ascending=False)

            filtered_display = hide_zero_rows(filtered)

            # Only show specific columns as in screenshot: Product, Location, Period, Forecast, D_day, Days_Covered_by_SS, Safety_Stock, Adjustment_Status
            display_cols = [c for c in [
                'Product','Location','Period','Forecast','D_day','Days_Covered_by_SS','Safety_Stock','Adjustment_Status'
            ] if c in filtered_display.columns]
            fmt_cols = [c for c in ['Forecast','D_day','Days_Covered_by_SS','Safety_Stock'] if c in filtered_display.columns]

            # Make Period human-friendly like the filters (e.g., "JAN 2026")
            disp_df = filtered_display.copy()
            if 'Period' in disp_df.columns:
                try:
                    disp_df['Period'] = disp_df['Period'].apply(period_label)
                except Exception:
                    pass

            disp = df_format_for_display(disp_df[display_cols].copy(), cols=fmt_cols, two_decimals_cols=['D_day','Days_Covered_by_SS'])
            st.dataframe(disp, use_container_width=True, height=700)

    # TAB 4: Efficiency Analysis
    with tab4:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            sku_default = default_product
            sku_index = all_products.index(sku_default) if all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="eff_sku")

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

            # Export snapshot (new icon)
            snapshot_period = eff_period if eff_period in all_periods else (all_periods[-1] if all_periods else None)
            if snapshot_period is None:
                eff_export = results[results['Product'] == sku].copy()
            else:
                eff_export = results[(results['Product'] == sku) & (results['Period'] == snapshot_period)].copy()

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=eff_export.to_csv(index=False),
                    file_name=f"efficiency_{sku}_{period_label(snapshot_period) if snapshot_period is not None else 'all'}.csv",
                    mime="text/csv",
                    key="eff_export_btn"
                )
            [...]

            # (rest of tab4 unchanged)
            [...]

    # TAB 5: Forecast Accuracy
    with tab5:
        [...]

    # TAB 6: Calculation Trace & Simulation — show mapping table and highlight values used
    with tab6:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            calc_sku_default = default_product
            calc_sku_index = all_products.index(calc_sku_default) if all_products else 0
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

            # Export trace row as CSV right under filters (new icon)
            row_export = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)]
            if row_export.empty:
                export_data = pd.DataFrame()
            else:
                export_data = row_export

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=export_data.to_csv(index=False),
                    file_name=f"calc_trace_{calc_sku}_{calc_loc}_{period_label(calc_period)}.csv",
                    mime="text/csv",
                    key="calc_export_btn"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            # Narrower layout for first mapping table
            st.markdown(
                """
                <style>
                  .calc-mapping-container {
                    max-width: 560px;
                  }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Show selection at the start of the tab
            st.markdown(f"**Selected**: {calc_sku} — {calc_loc} — {period_label(calc_period)}")
            st.header("🧮 Transparent Calculation Engine & Scenario Simulation")
            st.write("See how changing service level or lead-time assumptions affects safety stock.")

            # Always read the current service_level and recompute the z used in displays
            z_current = norm.ppf(service_level)

            # Use the freshly computed results DataFrame slice (reflects current sidebar params)
            row_df = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)]
            if row_df.empty:
                st.warning("Selection not found in results.")
            else:
                row = row_df.iloc[0]

                # NODE-SPECIFIC SERVICE LEVEL diagnostics (fixed mapping)
                node_sl = float(row.get('Service_Level_Node', service_level))
                node_z = float(row.get('Z_node', norm.ppf(node_sl)))
                hops = int(row.get('Tier_Hops', 0))

                # Show the hop->SL mapping as an HTML table and highlight the row used
                mapping_rows = [
                    (0, "99%", "End-node"),
                    (1, "95%", "Internal + external demand"),
                    (2, "90%", "Level-1 hub"),
                    (3, "85%", "Level-2 hub")
                ]
                table_html = """
                <div class="calc-mapping-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; font-size:13px;">
                  <table style="width:100%; border-collapse:collapse;">
                    <thead>
                      <tr style="background:#f3f6fb;">
                        <th style="text-align:left;padding:8px 10px;border:1px solid #e6eef8;white-space:nowrap;">Hop</th>
                        <th style="text-align:left;padding:8px 10px;border:1px solid #e6eef8;white-space:nowrap;">Service Level</th>
                        <th style="text-align:left;padding:8px 10px;border:1px solid #e6eef8;">Example / Role</th>
                      </tr>
                    </thead>
                    <tbody>
                """
                for h, sl, example in mapping_rows:
                    if h == hops:
                        row_style = "background:#FFF59D; font-weight:700;"
                    else:
                        row_style = ""
                    table_html += f"""
                      <tr style="{row_style}">
                        <td style="padding:8px 10px;border:1px solid #eef6ff;white-space:nowrap;">{h}</td>
                        <td style="padding:8px 10px;border:1px solid #eef6ff;white-space:nowrap;">{sl}</td>
                        <td style="padding:8px 10px;border:1px solid #eef6ff;">{example}</td>
                      </tr>
                    """
                table_html += """
                    </tbody>
                  </table>
                </div>
                """
                st.markdown("**Applied Hop → Service Level mapping (highlight shows which row was used for this node):**")
                try:
                    components.html(table_html, height=200)
                except Exception:
                    st.markdown(table_html, unsafe_allow_html=True)

                # Highlight the exact values used for the calculation in a compact summary box
                avg_daily = row.get('D_day', np.nan)
                days_cov = row.get('Days_Covered_by_SS', np.nan)
                avg_daily_txt = f"{avg_daily:.2f}" if pd.notna(avg_daily) else "N/A"
                days_cov_txt = f"{days_cov:.1f}" if pd.notna(days_cov) else "N/A"

                summary_html = f"""
                <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:12px; font-size:13px;">
                  <div style="flex:0 0 48%;background:#e8f0ff;border-radius:8px;padding:12px;">
                    <div style="font-size:12px;color:#0b3d91;font-weight:600;">Applied Node SL</div>
                    <div style="font-size:18px;font-weight:800;color:#0b3d91;">{node_sl*100:.2f}%</div>
                    <div style="font-size:11px;color:#444;margin-top:6px;">(hops = {hops})</div>
                  </div>
                  <div style="flex:0 0 48%;background:#fff3e0;border-radius:8px;padding:12px;">
                    <div style="font-size:12px;color:#a64d00;font-weight:600;">Applied Z</div>
                    <div style="font-size:18px;font-weight:800;color:#a64d00;">{node_z:.4f}</div>
                    <div style="font-size:11px;color:#444;margin-top:6px;">(for SL {node_sl*100:.2f}%)</div>
                  </div>
                  <div style="flex:0 0 48%;background:#e8f8f0;border-radius:8px;padding:12px;">
                    <div style="font-size:12px;color:#00695c;font-weight:600;">Network Demand (monthly)</div>
                    <div style="font-size:18px;font-weight:800;color:#00695c;">{euro_format(row['Agg_Future_Demand'], True)}</div>
                  </div>
                  <div style="flex:0 0 48%;background:#fbeff2;border-radius:8px;padding:12px;">
                    <div style="font-size:12px;color:#880e4f;font-weight:600;">Network Std Dev (monthly)</div>
                    <div style="font-size:18px;font-weight:800;color:#880e4f;">{euro_format(row['Agg_Std_Hist'], True)}</div>
                  </div>
                  <div style="flex:0 0 48%;background:#f0f4c3;border-radius:8px;padding:12px;">
                    <div style="font-size:12px;color:#827717;font-weight:600;">Avg LT (days)</div>
                    <div style="font-size:18px;font-weight:800;color:#827717;">{row['LT_Mean']}</div>
                  </div>
                  <div style="flex:0 0 48%;background:#e1f5fe;border-radius:8px;padding:12px;">
                    <div style="font-size:12px;color:#01579b;font-weight:600;">LT Std Dev (days)</div>
                    <div style="font-size:18px;font-weight:800;color:#01579b;">{row['LT_Std']}</div>
                  </div>

                  <div style="flex:0 0 48%;background:#ffffff;border-radius:8px;padding:12px;border:1px solid #eaeaea;">
                    <div style="font-size:12px;color:#333;font-weight:600;">Avg Daily Demand</div>
                    <div style="font-size:16px;font-weight:800;color:#333;">{avg_daily_txt} units/day</div>
                  </div>
                  <div style="flex:0 0 48%;background:#ffffff;border-radius:8px;padding:12px;border:1px solid #eaeaea;">
                    <div style="font-size:12px;color:#333;font-weight:600;">Days Covered by SS</div>
                    <div style="font-size:16px;font-weight:800;color:#333;">{days_cov_txt} days</div>
                  </div>

                </div>
                """
                st.markdown("**Values used for the calculation (highlighted above):**")
                st.markdown(summary_html, unsafe_allow_html=True)

                # --- Scenario planning banner + expander (icon now outside expander, as a headline) ---
                st.markdown("---")
                st.markdown(
                    """
                    <div style="
                        background:#ffecb3;
                        border:1px solid #f9c74f;
                        border-radius:10px;
                        padding:10px 14px;
                        margin-bottom:8px;
                        font-size:0.97rem;
                        color:#0b3d91;
                        font-weight:700;">
                      ▶ Scenario Planning — simulate alternative SL / LT assumptions (analysis‑only)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Show detailed scenario controls", expanded=False):
                    st.markdown(
                        """
                        <div style="border:1px solid #0b3d91;border-radius:10px;background:#fff9e0;padding:12px;color:#0b3d91;font-size:0.95rem;">
                          Use scenarios to test sensitivity to Service Level or Lead Time. Scenarios do not change implemented policy — they are analysis-only.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
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
                            sc_lt_default = float(row['LT_Mean']) if s==0 else float(row['LT_Mean'])
                            sc_lt = st.slider(f"Scenario {s+1} Avg Lead Time (Days)", 0.0, max(30.0, float(row['LT_Mean'])*2), value=sc_lt_default, key=f"sc_lt_{s}")
                            sc_lt_std_default = float(row['LT_Std']) if s==0 else float(row['LT_Std'])
                            sc_lt_std = st.slider(f"Scenario {s+1} LT Std Dev (Days)", 0.0, max(10.0, float(row['LT_Std'])*2), value=sc_lt_std_default, key=f"sc_lt_std_{s}")
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

                    # keep table compact instead of 100% width
                    st.markdown("Scenario comparison (Simulated SS). 'Implemented' shows the final Safety_Stock after rules.")
                    st.dataframe(
                        df_format_for_display(
                            display_comp[['Scenario','Service_Level_%','LT_mean_days','LT_std_days','Simulated_SS']].copy(),
                            cols=['Service_Level_%','LT_mean_days','LT_std_days','Simulated_SS'],
                            two_decimals_cols=['Service_Level_%','Simulated_SS']
                        ),
                        width=600,
                        use_container_width=False,
                    )

                    fig_bar = go.Figure()
                    colors = px.colors.qualitative.Pastel
                    fig_bar.add_trace(go.Bar(x=display_comp['Scenario'], y=display_comp['Simulated_SS'], marker_color=colors[:len(display_comp)]))
                    fig_bar.update_layout(title="Scenario SS Comparison", yaxis_title="SS (units)")
                    st.plotly_chart(fig_bar, use_container_width=True)

                # (rest of tab6 unchanged)
                [...]

    # TAB 7: By Material
    with tab7:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            sel_prod_default = default_product
            sel_prod_index = all_products.index(sel_prod_default) if all_products else 0
            selected_product = st.selectbox("MATERIAL", all_products, index=sel_prod_index, key="mat_sel")

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

            # Export material-period data (new icon)
            mat_period_export = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)].copy()

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=mat_period_export.to_csv(index=False),
                    file_name=f"material_view_{selected_product}_{period_label(selected_period)}.csv",
                    mime="text/csv",
                    key="mat_export_btn"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            [...]

else:
    st.info("Please upload sales.csv, demand.csv and leadtime.csv in the sidebar to run the optimizer.")
