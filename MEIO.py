# Multi-Echelon Inventory Optimizer — Raw Materials

import os
import math
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from scipy.stats import norm

st.set_page_config(page_title="MEIO for RM", layout="wide")

LOGO_FILENAME = "GY_logo.jpg"
LOGO_BASE_WIDTH = 160
days_per_month = 30

st.markdown(
    "<h1 style='margin:0; padding-top:6px;'>MEIO for Raw Materials — v0.973 — Jan 2026</h1>",
    unsafe_allow_html=True,
)

# Global CSS
st.markdown(
    """
    <style>
      div[data-baseweb="tag"],
      .stMultiSelect div[data-baseweb="tag"],
      .stSelectbox div[data-baseweb="tag"] {
        background: #e3f2fd !important;
        color: #0b3d91 !important;
        border: 1px solid #90caf9 !important;
        border-radius: 8px !important;
        padding: 2px 8px !important;
        font-weight: 600 !important;
      }
      div[data-baseweb="tag"] span,
      .stMultiSelect div[data-baseweb="tag"] span,
      .stSelectbox div[data-baseweb="tag"] span {
        color: #0b3d91 !important;
      }
      div[data-baseweb="tag"] svg,
      .stMultiSelect div[data-baseweb="tag"] svg,
      .stSelectbox div[data-baseweb="tag"] svg {
        fill: #0b3d91 !important;
      }
      .stDownloadButton button,
      .stDownloadButton > button {
        padding: 0.35rem 0.6rem !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
      }
      .main,
      .block-container,
      .stDataFrame,
      .stMarkdown,
      .stMetric {
        font-size: 0.88rem;
      }
      .export-csv-btn button {
        background-color: #0b3d91 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        padding: 0.3rem 0.9rem !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
      }
      .scenario-table-container {
        max-width: 620px;
        margin-left: 0;
        margin-right: auto;
      }
      button[data-baseweb="tab"] span {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
      }
      .selection-line {
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
      }
      .selection-label {
        color: #444444;
        font-weight: 500;
      }
      .selection-value {
        color: #0b3d91;
        font-weight: 700;
      }
      .ss-top-table table thead tr th {
        white-space: normal !important;
        word-break: break-word !important;
        max-width: 120px;
      }
      .ss-top-table table {
        table-layout: fixed;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------- Helpers / formatting --------
def clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace(
        {
            "": np.nan,
            "-": np.nan,
            "—": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "None": np.nan,
        }
    )
    paren_mask = s.str.startswith("(") & s.str.endswith(")")
    try:
        s.loc[paren_mask] = "-" + s.loc[paren_mask].str[1:-1]
    except Exception:
        s = s.apply(
            lambda v: ("-" + v[1:-1])
            if isinstance(v, str) and v.startswith("(") and v.endswith(")")
            else v
        )
    s = s.str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(r"[^\d\.\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def euro_format(x, always_two_decimals: bool = True, show_zero: bool = False) -> str:
    """Format numeric values using '.' as thousands separator; hide zeros unless show_zero is True."""
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
        if math.isclose(xv, 0.0, abs_tol=1e-9) and not show_zero:
            return ""
        neg = xv < 0
        rounded = int(round(abs(xv)))
        s = f"{rounded:,}".replace(",", ".")
        return f"-{s}" if neg else s
    except Exception:
        return str(x)


def df_format_for_display(df: pd.DataFrame, cols=None, two_decimals_cols=None) -> pd.DataFrame:
    """Apply euro_format to numeric columns; optionally keep two decimals for specific columns."""
    d = df.copy()
    if cols is None:
        cols = [c for c in d.columns if d[c].dtype.kind in "biufc"]
    for c in cols:
        if c in d.columns:
            if two_decimals_cols and c in two_decimals_cols:
                d[c] = d[c].apply(
                    lambda v: (
                        "{:.2f}".format(v)
                        if (pd.notna(v) and isinstance(v, (int, float)))
                        else euro_format(v, always_two_decimals=True)
                    )
                )
            else:
                d[c] = d[c].apply(lambda v: euro_format(v, always_two_decimals=False))
    return d


def hide_zero_rows(df: pd.DataFrame, check_cols=None) -> pd.DataFrame:
    """Drop rows where the sum of selected columns equals zero."""
    if df is None or df.empty:
        return df
    if check_cols is None:
        check_cols = ["Safety_Stock", "Forecast", "Agg_Future_Demand"]
    existing = [c for c in check_cols if c in df.columns]
    if not existing:
        return df
    try:
        mask = df[existing].abs().sum(axis=1) != 0
        return df[mask].copy()
    except Exception:
        return df


def aggregate_network_stats(df_forecast, df_stats, df_lt, transitive: bool = True, rho: float = 1.0):
    """Aggregate monthly forecast and variance through the network."""
    results = []
    months = df_forecast["Period"].unique()
    products = df_forecast["Product"].unique()

    routes_by_product = {}
    if "Product" in df_lt.columns:
        for prod in df_lt["Product"].unique():
            routes_by_product[prod] = df_lt[df_lt["Product"] == prod].copy()
    else:
        routes_by_product[None] = df_lt.copy()

    reachable_map = {}

    for month in months:
        df_month = df_forecast[df_forecast["Period"] == month]
        for prod in products:
            p_stats = df_stats[df_stats["Product"] == prod].set_index("Location").to_dict("index")
            p_fore = df_month[df_month["Product"] == prod].set_index("Location").to_dict("index")
            p_lt = routes_by_product.get(prod, pd.DataFrame(columns=df_lt.columns))

            nodes = set(df_month[df_month["Product"] == prod]["Location"])
            if not p_lt.empty:
                froms = set(v for v in p_lt["From_Location"].tolist() if pd.notna(v))
                tos = set(v for v in p_lt["To_Location"].tolist() if pd.notna(v))
                nodes = nodes.union(froms).union(tos)
            if not nodes:
                continue

            children = {}
            if not p_lt.empty:
                for _, r in p_lt.iterrows():
                    f = r.get("From_Location", None)
                    t = r.get("To_Location", None)
                    if pd.isna(f) or pd.isna(t):
                        continue
                    children.setdefault(f, set()).add(t)

            reachable_cache = {}

            def get_reachable(start):
                if not transitive:
                    direct = children.get(start, set())
                    out = set(direct)
                    out.add(start)
                    return out
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

                local_fcst = float(p_fore.get(n, {"Forecast": 0}).get("Forecast", 0.0))
                child_fcst_sum = 0.0
                child_var_sum = 0.0

                for c in reachable:
                    if c == n:
                        continue
                    child_fcst = float(p_fore.get(c, {"Forecast": 0}).get("Forecast", 0.0))
                    child_fcst_sum += child_fcst
                    child_std = p_stats.get(c, {}).get("Local_Std", np.nan)
                    if not pd.isna(child_std):
                        child_var_sum += float(child_std) ** 2 * float(rho)

                agg_demand = local_fcst + child_fcst_sum
                local_std = p_stats.get(n, {}).get("Local_Std", np.nan)
                local_var = 0.0 if pd.isna(local_std) else float(local_std) ** 2
                total_var = local_var + child_var_sum
                agg_std = np.sqrt(total_var) if total_var >= 0 and not pd.isna(total_var) else np.nan

                results.append(
                    {
                        "Product": prod,
                        "Location": n,
                        "Period": month,
                        "Agg_Future_Demand": agg_demand,
                        "Agg_Future_Internal": local_fcst,
                        "Agg_Future_External": child_fcst_sum,
                        "Agg_Std_Hist": agg_std,
                    }
                )

    return pd.DataFrame(results), reachable_map


def render_logo_above_parameters(scale: float = 1.5) -> None:
    if LOGO_FILENAME and os.path.exists(LOGO_FILENAME):
        try:
            width = int(LOGO_BASE_WIDTH * float(scale))
            st.image(LOGO_FILENAME, width=width)
        except Exception:
            pass


def render_selection_line(label, product=None, location=None, period_text=None) -> None:
    """Thin selection line used above titles in each tab."""
    if not product and not location and not period_text:
        return
    parts = []
    if product:
        parts.append(f"<span class='selection-value'>{product}</span>")
    if location:
        parts.append(f"<span class='selection-value'>{location}</span>")
    if period_text:
        parts.append(f"<span class='selection-value'>{period_text}</span>")
    values = " — ".join(parts)
    html = f"""
    <div class="selection-line">
      <span class="selection-label">{label}</span>
      {values}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(
        "<hr style='margin:4px 0 10px 0; border: none; border-top: 1px solid #e0e0e0;'/>",
        unsafe_allow_html=True,
    )


def period_label(ts) -> str:
    try:
        return pd.to_datetime(ts).strftime("%b %Y").upper()
    except Exception:
        return str(ts)


# -------- Sidebar configuration --------
with st.sidebar.expander("⚙️ Service Level Configuration", expanded=True):
    service_level = st.slider(
        "Service Level (%) — end-nodes (hop 0)",
        50.0,
        99.9,
        99.0,
        help="Target probability of not stocking out for end-nodes (hop 0). Upstream nodes get fixed SLs by hop-distance.",
    ) / 100
    z = norm.ppf(service_level)

with st.sidebar.expander("⚙️ Safety Stock Rules", expanded=True):
    zero_if_no_net_fcst = st.checkbox(
        "Force Zero SS if No Network Demand",
        value=True,
        help="When enabled, nodes with zero aggregated network demand will have Safety Stock forced to zero.",
    )
    apply_cap = st.checkbox(
        "Enable SS Capping (% of Network Demand)",
        value=True,
        help="Clip Safety Stock within a percentage range of total network demand.",
    )
    cap_range = st.slider(
        "Cap Range (%)",
        0,
        500,
        (0, 200),
        help="Lower and upper bounds (as % of total network demand) applied to Safety Stock.",
    )

use_transitive = True
var_rho = 1.0
lt_mode = "Apply LT variance"

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Sources (CSV)")
DEFAULT_FILES = {
    "sales": "sales.csv",
    "demand": "demand.csv",
    "lt": "leadtime.csv",
}
s_upload = st.sidebar.file_uploader("1. Sales Data (Historical: sales.csv)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast: demand.csv)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes: leadtime.csv)", type="csv")

s_file = s_upload if s_upload is not None else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload is not None else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload is not None else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)

if s_file:
    st.sidebar.success(f"✅ Sales Loaded: {getattr(s_file, 'name', s_file)}")
if d_file:
    st.sidebar.success(f"✅ Demand Loaded: {getattr(d_file, 'name', d_file)}")
if lt_file:
    st.sidebar.success(f"✅ Lead Time Loaded: {getattr(lt_file, 'name', lt_file)}")

# -------- Core pipeline --------
DEFAULT_PRODUCT_CHOICE = "NOKANDO2"
DEFAULT_LOCATION_CHOICE = "DEW1"
CURRENT_MONTH_TS = pd.Timestamp.now().to_period("M").to_timestamp()


def run_pipeline(
    df_d,
    stats,
    df_lt,
    service_level,
    transitive: bool = True,
    rho: float = 1.0,
    lt_mode_param: str = "Apply LT variance",
    zero_if_no_net_fcst: bool = True,
    apply_cap: bool = True,
    cap_range=(0, 200),
):
    """Aggregation → stats → safety-stock pipeline with fixed hop→SL mapping."""
    hop_to_sl = {0: 0.99, 1: 0.95, 2: 0.90, 3: 0.85}

    def sl_for_hop(h: int) -> float:
        return hop_to_sl.get(h, hop_to_sl[3])

    network_stats, reachable_map = aggregate_network_stats(
        df_forecast=df_d,
        df_stats=stats,
        df_lt=df_lt,
        transitive=transitive,
        rho=rho,
    )

    node_lt_local = (
        df_lt.groupby(["Product", "To_Location"])[["Lead_Time_Days", "Lead_Time_Std_Dev"]]
        .mean()
        .reset_index()
    )
    node_lt_local.columns = ["Product", "Location", "LT_Mean", "LT_Std"]

    res = pd.merge(
        network_stats,
        df_d[["Product", "Location", "Period", "Forecast"]],
        on=["Product", "Location", "Period"],
        how="left",
    )
    res = pd.merge(res, node_lt_local, on=["Product", "Location"], how="left")

    res = res.fillna(
        {
            "Forecast": 0,
            "Agg_Std_Hist": np.nan,
            "LT_Mean": 7,
            "LT_Std": 2,
            "Agg_Future_Demand": 0,
            "Agg_Future_Internal": 0,
            "Agg_Future_External": 0,
        }
    )

    product_median_localstd = stats.groupby("Product")["Local_Std"].median().to_dict()
    global_median_std = stats["Local_Std"].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0

    def fill_agg_std(row):
        if not pd.isna(row["Agg_Std_Hist"]):
            return row["Agg_Std_Hist"]
        return product_median_localstd.get(row["Product"], global_median_std)

    res["Agg_Std_Hist"] = res.apply(fill_agg_std, axis=1)

    for c in [
        "Agg_Std_Hist",
        "LT_Mean",
        "LT_Std",
        "Agg_Future_Demand",
        "Agg_Future_Internal",
        "Agg_Future_External",
        "Forecast",
    ]:
        res[c] = res[c].astype(float)

    res["Sigma_D_Day"] = res["Agg_Std_Hist"] / np.sqrt(float(days_per_month))
    res["D_day"] = res["Agg_Future_Demand"] / float(days_per_month)
    res["Var_D_Day"] = res["Sigma_D_Day"] ** 2

    low_demand_monthly_threshold = 20.0
    low_mask = res["Agg_Future_Demand"] < low_demand_monthly_threshold
    res.loc[low_mask, "Var_D_Day"] = res.loc[low_mask, "Var_D_Day"].where(
        res.loc[low_mask, "Var_D_Day"] >= res.loc[low_mask, "D_day"],
        res.loc[low_mask, "D_day"],
    )

    demand_component = res["Var_D_Day"] * res["LT_Mean"]

    lt_component_list = []
    for _, row in res.iterrows():
        d_day = float(row["D_day"])
        D_for_LT = d_day
        if lt_mode_param == "Ignore LT variance":
            lt_comp = 0.0
        elif lt_mode_param == "Apply LT variance":
            lt_comp = float(row["LT_Std"]) ** 2 * (D_for_LT ** 2)
        elif lt_mode_param == "Average LT Std across downstream":
            reachable = reachable_map.get((row["Product"], row["Location"], row["Period"]), set())
            if not reachable:
                lt_used = float(row["LT_Std"])
            else:
                vals = []
                for rn in reachable:
                    match = node_lt_local[
                        (node_lt_local["Product"] == row["Product"])
                        & (node_lt_local["Location"] == rn)
                    ]
                    if not match.empty:
                        vals.append(float(match["LT_Std"].iloc[0]))
                lt_used = float(np.mean(vals)) if vals else float(row["LT_Std"])
            lt_comp = lt_used ** 2 * (D_for_LT ** 2)
        else:
            lt_comp = float(row["LT_Std"]) ** 2 * (D_for_LT ** 2)
        lt_component_list.append(lt_comp)

    res["lt_component"] = np.array(lt_component_list)
    combined_variance = (demand_component + res["lt_component"]).clip(lower=0)

    def compute_hop_distances_for_product(p_lt_df, prod_nodes):
        """Compute hop distance (node → nearest leaf downstream) used for tiered SL."""
        children = {}
        for _, r in p_lt_df.iterrows():
            f = r.get("From_Location", None)
            t = r.get("To_Location", None)
            if pd.isna(f) or pd.isna(t):
                continue
            children.setdefault(f, set()).add(t)

        all_nodes = set(prod_nodes)
        if not p_lt_df.empty:
            all_nodes = all_nodes.union(
                set(p_lt_df["From_Location"].dropna().unique())
            ).union(set(p_lt_df["To_Location"].dropna().unique()))

        leaf_nodes = {n for n in all_nodes if n not in children or not children.get(n)}

        distances = {}
        if not leaf_nodes:
            for n in all_nodes:
                distances[n] = 0
            return distances

        for n in all_nodes:
            if n in leaf_nodes:
                distances[n] = 0
                continue
            q = collections.deque()
            q.append((n, 0))
            visited = {n}
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

    products = res["Product"].unique()
    prod_to_nodes = {p: set(res[res["Product"] == p]["Location"].unique().tolist()) for p in products}
    prod_to_routes = {}
    if "Product" in df_lt.columns:
        for p in df_lt["Product"].unique():
            prod_to_routes[p] = df_lt[df_lt["Product"] == p].copy()
    else:
        prod_to_routes[None] = df_lt.copy()

    prod_distances = {}
    for p in products:
        p_routes = prod_to_routes.get(p, df_lt.copy())
        nodes = prod_to_nodes.get(p, set())
        prod_distances[p] = compute_hop_distances_for_product(p_routes, nodes)

    # Hard-coded overrides for special hubs
    special_hops = {"B616": 4, "BEEX": 3, "LUEX": 2}
    for p, distances in prod_distances.items():
        for node, fixed_hop in special_hops.items():
            distances[node] = int(fixed_hop)

    product_tiering_params = {}
    for p, distances in prod_distances.items():
        max_hops = int(max(distances.values())) if distances else 0
        tier_map_pct = {f"SL_hop_{h}_pct": float(sl_for_hop(h) * 100.0) for h in range(0, 4)}
        tier_map_pct["max_tier_hops"] = max_hops
        product_tiering_params[p] = tier_map_pct

    sl_list = []
    hop_distance_list = []
    for _, row in res.iterrows():
        prod = row["Product"]
        loc = row["Location"]
        distances = prod_distances.get(prod, {})
        dist = int(distances.get(loc, 0))
        sl_node = sl_for_hop(dist)
        sl_list.append(sl_node)
        hop_distance_list.append(dist)
    res["Tier_Hops"] = np.array(hop_distance_list)
    res["Service_Level_Node"] = np.array(sl_list)
    res["Z_node"] = res["Service_Level_Node"].apply(lambda x: float(norm.ppf(x)))

    res["SS_stat"] = res.apply(
        lambda r: r["Z_node"] * math.sqrt(max(0.0, (demand_component.loc[r.name] + r["lt_component"]))),
        axis=1,
    )

    min_floor_fraction_of_LT_demand = 0.01
    res["Mean_Demand_LT"] = res["D_day"] * res["LT_Mean"]
    res["SS_floor"] = res["Mean_Demand_LT"] * min_floor_fraction_of_LT_demand
    res["Pre_Rule_SS"] = res[["SS_stat", "SS_floor"]].max(axis=1)
    res["Adjustment_Status"] = "Optimal (Statistical)"
    res["Safety_Stock"] = res["Pre_Rule_SS"]

    if zero_if_no_net_fcst:
        zero_mask = res["Agg_Future_Demand"] <= 0
        res.loc[zero_mask, "Adjustment_Status"] = "Forced to Zero"
        res.loc[zero_mask, "Safety_Stock"] = 0.0

    res["Pre_Cap_SS"] = res["Safety_Stock"]
    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100.0, cap_range[1] / 100.0
        l_lim = res["Agg_Future_Demand"] * l_cap
        u_lim = res["Agg_Future_Demand"] * u_cap
        high_mask = (res["Safety_Stock"] > u_lim) & (res["Adjustment_Status"] == "Optimal (Statistical)")
        low_mask = (
            (res["Safety_Stock"] < l_lim)
            & (res["Adjustment_Status"] == "Optimal (Statistical)")
            & (res["Agg_Future_Demand"] > 0)
        )
        res.loc[high_mask, "Adjustment_Status"] = "Capped (High)"
        res.loc[low_mask, "Adjustment_Status"] = "Capped (Low)"
        res["Safety_Stock"] = res["Safety_Stock"].clip(lower=l_lim, upper=u_lim)

    res["Safety_Stock"] = res["Safety_Stock"].round(0)
    res.loc[res["Location"] == "B616", "Safety_Stock"] = 0
    res["Max_Corridor"] = res["Safety_Stock"] + res["Forecast"]

    def compute_days_covered(r):
        try:
            d = float(r.get("D_day", 0.0))
            if d <= 0:
                return np.nan
            return float(r.get("Safety_Stock", 0.0)) / d
        except Exception:
            return np.nan

    res["Days_Covered_by_SS"] = res.apply(compute_days_covered, axis=1)

    tier_summary = []
    for p, params in product_tiering_params.items():
        tier_summary.append(
            {
                "Product": p,
                "SL_hop_0_pct": params.get("SL_hop_0_pct", 99.0),
                "SL_hop_1_pct": params.get("SL_hop_1_pct", 95.0),
                "SL_hop_2_pct": params.get("SL_hop_2_pct", 90.0),
                "SL_hop_3_pct": params.get("SL_hop_3_pct", 85.0),
                "max_tier_hops": params.get("max_tier_hops", 0),
            }
        )
    tier_df = pd.DataFrame(tier_summary)
    res.attrs["tiering_params"] = tier_df

    return res, reachable_map


# -------- Load data & run model --------
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

    needed_sales_cols = {"Product", "Location", "Period", "Consumption", "Forecast"}
    needed_demand_cols = {"Product", "Location", "Period", "Forecast"}
    needed_lt_cols = {"Product", "From_Location", "To_Location", "Lead_Time_Days", "Lead_Time_Std_Dev"}

    if not needed_sales_cols.issubset(df_s.columns):
        st.error(f"sales.csv missing columns: {needed_sales_cols - set(df_s.columns)}")
        st.stop()
    if not needed_demand_cols.issubset(df_d.columns):
        st.error(f"demand.csv missing columns: {needed_demand_cols - set(df_d.columns)}")
        st.stop()
    if not needed_lt_cols.issubset(df_lt.columns):
        st.error(f"leadtime.csv missing columns: {needed_lt_cols - set(df_lt.columns)}")
        st.stop()

    df_s["Period"] = pd.to_datetime(df_s["Period"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df_d["Period"] = pd.to_datetime(df_d["Period"], errors="coerce").dt.to_period("M").to_timestamp()

    df_s["Consumption"] = clean_numeric(df_s["Consumption"])
    df_s["Forecast"] = clean_numeric(df_s["Forecast"])
    df_d["Forecast"] = clean_numeric(df_d["Forecast"])
    df_lt["Lead_Time_Days"] = clean_numeric(df_lt["Lead_Time_Days"])
    df_lt["Lead_Time_Std_Dev"] = clean_numeric(df_lt["Lead_Time_Std_Dev"])

    stats = (
        df_s.groupby(["Product", "Location"])["Consumption"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["Product", "Location", "Local_Mean", "Local_Std"]
    global_median_std = stats["Local_Std"].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0
    prod_medians = stats.groupby("Product")["Local_Std"].median().to_dict()

    def fill_local_std(row):
        if not pd.isna(row["Local_Std"]) and row["Local_Std"] > 0:
            return row["Local_Std"]
        pm = prod_medians.get(row["Product"], np.nan)
        return pm if not pd.isna(pm) else global_median_std

    stats["Local_Std"] = stats.apply(fill_local_std, axis=1)

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
        cap_range=cap_range,
    )

    # Historical accuracy metrics
    hist = df_s[["Product", "Location", "Period", "Consumption", "Forecast"]].copy()
    hist.rename(columns={"Forecast": "Forecast_Hist"}, inplace=True)
    hist["Deviation"] = hist["Consumption"] - hist["Forecast_Hist"]
    hist["Abs_Error"] = hist["Deviation"].abs()
    hist["APE_%"] = hist["Abs_Error"] / hist["Consumption"].replace(0, np.nan) * 100
    hist["APE_%"] = hist["APE_%"].fillna(0)
    hist["Accuracy_%"] = (1 - hist["APE_%"] / 100) * 100
    hist_net = df_s.groupby(["Product", "Period"], as_index=False).agg(
        Network_Consumption=("Consumption", "sum"),
        Network_Forecast_Hist=("Forecast", "sum"),
    )

    meaningful_mask = (
        results[["Agg_Future_Demand", "Forecast", "Safety_Stock", "Pre_Rule_SS"]]
        .fillna(0)
        .abs()
        .sum(axis=1)
        > 0
    )
    meaningful_results = results[meaningful_mask].copy()

    all_products = sorted(meaningful_results["Product"].unique().tolist())
    if not all_products:
        all_products = sorted(results["Product"].unique().tolist())
    default_product = (
        DEFAULT_PRODUCT_CHOICE
        if DEFAULT_PRODUCT_CHOICE in all_products
        else (all_products[0] if all_products else "")
    )

    def default_location_for(prod):
        locs = sorted(
            meaningful_results[meaningful_results["Product"] == prod]["Location"]
            .unique()
            .tolist()
        )
        if not locs:
            locs = sorted(
                results[results["Product"] == prod]["Location"].unique().tolist()
            )
        return (
            DEFAULT_LOCATION_CHOICE
            if DEFAULT_LOCATION_CHOICE in locs
            else (locs[0] if locs else "")
        )

    all_periods = sorted(results["Period"].unique().tolist())
    default_period = (
        CURRENT_MONTH_TS
        if CURRENT_MONTH_TS in all_periods
        else (all_periods[-1] if all_periods else None)
    )

    period_label_map = {period_label(p): p for p in all_periods}
    period_labels = list(period_label_map.keys())

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📈 Inventory Corridor",
            "🕸️ Network Topology",
            "📋 Full Plan",
            "⚖️ Efficiency Analysis",
            "📉 Forecast Accuracy",
            "🧮 Calculation Trace & Sim",
            "📦 By Material",
        ]
    )

    # -------- TAB 1: Inventory Corridor --------
    with tab1:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            sku_default = default_product
            sku_index = all_products.index(sku_default) if sku_default in all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="tab1_sku")

            loc_opts = sorted(
                meaningful_results[
                    (meaningful_results["Product"] == sku)
                    & (meaningful_results["Period"] == CURRENT_MONTH_TS)
                ]["Location"]
                .unique()
                .tolist()
            )
            if not loc_opts:
                loc_opts = sorted(
                    results[
                        (results["Product"] == sku)
                        & (results["Period"] == CURRENT_MONTH_TS)
                    ]["Location"]
                    .unique()
                    .tolist()
                )
            if not loc_opts:
                loc_opts = sorted(
                    meaningful_results[meaningful_results["Product"] == sku]["Location"]
                    .unique()
                    .tolist()
                )
            if not loc_opts:
                loc_opts = sorted(
                    results[results["Product"] == sku]["Location"].unique().tolist()
                )
            if not loc_opts:
                loc_opts = ["(no location)"]

            loc_default = (
                DEFAULT_LOCATION_CHOICE
                if DEFAULT_LOCATION_CHOICE in loc_opts
                else (loc_opts[0] if loc_opts else "(no location)")
            )
            loc_index = loc_opts.index(loc_default) if loc_default in loc_opts else 0
            loc = st.selectbox("LOCATION", loc_opts, index=loc_index, key="tab1_loc")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            try:
                summary_row = results[
                    (results["Product"] == sku)
                    & (results["Location"] == loc)
                    & (results["Period"] == CURRENT_MONTH_TS)
                ]
                if not summary_row.empty:
                    srow = summary_row.iloc[0]
                    avg_daily = srow.get("D_day", np.nan)
                    days_cov = srow.get("Days_Covered_by_SS", np.nan)
                    avg_daily_txt = f"{avg_daily:.2f} units/day" if pd.notna(avg_daily) else "N/A"
                    days_cov_txt = f"{days_cov:.1f} days" if pd.notna(days_cov) else "N/A"
                    st.markdown(f"Avg Daily Demand: **{avg_daily_txt}**")
                    st.markdown(f"Safety Stock coverage: **{days_cov_txt}**")
            except Exception:
                pass

        with col_main:
            render_selection_line("Selected:", product=sku, location=loc)
            st.subheader("📈 Inventory Corridor")

            plot_df = results[(results["Product"] == sku) & (results["Location"] == loc)].sort_values("Period")
            df_all_periods = pd.DataFrame({"Period": all_periods})
            plot_full = pd.merge(
                df_all_periods,
                plot_df[
                    [
                        "Period",
                        "Max_Corridor",
                        "Safety_Stock",
                        "Forecast",
                        "Agg_Future_Internal",
                        "Agg_Future_External",
                    ]
                ],
                on="Period",
                how="left",
            )
            num_cols = [
                "Max_Corridor",
                "Safety_Stock",
                "Forecast",
                "Agg_Future_Internal",
                "Agg_Future_External",
            ]
            plot_full[num_cols] = plot_full[num_cols].fillna(0)

            traces = [
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Max_Corridor"],
                    name="Max Corridor (SS + Forecast)",
                    line=dict(width=1, color="rgba(0,0,0,0.1)"),
                ),
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Safety_Stock"],
                    name="Safety Stock",
                    fill="tonexty",
                    fillcolor="rgba(0,176,246,0.2)",
                ),
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Forecast"],
                    name="Local Direct Demand (Internal)",
                    line=dict(color="black", dash="dot"),
                ),
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Agg_Future_External"],
                    name="External Network Demand (Downstream)",
                    line=dict(color="blue", dash="dash"),
                ),
            ]
            fig = go.Figure(traces)
            fig.update_layout(
                legend=dict(orientation="h"),
                xaxis_title="Period",
                yaxis_title="Units",
                xaxis=dict(tickformat="%b\n%Y"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Experimental risk-focused view
            st.markdown(
                "<div style='margin-top:10px; font-size:0.9rem; color:#666;'>"
                "⬇︎ Experimental risk-focused view (for comparison)</div>",
                unsafe_allow_html=True,
            )

            color_ss_line = "#d32f2f"
            color_ss_fill = "rgba(211,47,47,0.18)"
            color_fcst = "#212121"
            color_ext = "#1976d2"
            color_corridor = "rgba(0,0,0,0.12)"
            color_corridor_points = "rgba(0,0,0,0.35)"

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Forecast"],
                    mode="lines",
                    name="Local Forecast",
                    line=dict(color=color_fcst, width=2, dash="dot"),
                    hovertemplate="<b>%{x|%b %Y}</b><br>"
                    "Local Forecast: %{y:,.0f}<extra></extra>",
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Forecast"] + plot_full["Safety_Stock"],
                    mode="lines",
                    name="Safety Stock (Risk Buffer)",
                    line=dict(color=color_ss_line, width=2.5),
                    fill="tonexty",
                    fillcolor=color_ss_fill,
                    hovertemplate="<b>%{x|%b %Y}</b><br>"
                    "Corridor top (Fcst + SS): %{y:,.0f}<extra></extra>",
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Agg_Future_External"],
                    mode="lines+markers",
                    name="External Network Demand (Downstream)",
                    line=dict(color=color_ext, width=2, dash="dash"),
                    marker=dict(size=5, symbol="circle"),
                    hovertemplate="<b>%{x|%b %Y}</b><br>"
                    "External Network Demand: %{y:,.0f}<extra></extra>",
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Max_Corridor"],
                    mode="lines+markers",
                    name="Max Corridor (SS + Forecast)",
                    line=dict(color=color_corridor, width=1),
                    marker=dict(size=4, color=color_corridor_points),
                    hovertemplate="<b>%{x|%b %Y}</b><br>"
                    "Max Corridor: %{y:,.0f}<extra></extra>",
                )
            )

            # Highlight current period (robust across Plotly versions)
            current_idx = None
            period_values = list(plot_full["Period"].values)
            if CURRENT_MONTH_TS in period_values:
                current_idx = period_values.index(CURRENT_MONTH_TS)

            if current_idx is not None:
                current_x = plot_full["Period"].iloc[current_idx]

                # Keep x as scalar; Plotly will handle pandas.Timestamp or datetime
                current_x_val = current_x

                current_fcst = float(plot_full["Forecast"].iloc[current_idx])
                current_ss = float(plot_full["Safety_Stock"].iloc[current_idx])

                try:
                    # Use the newer add_vline API if available
                    fig2.add_vline(
                        x=current_x_val,
                        line_width=1.5,
                        line_dash="dot",
                        line_color="#9e9e9e",
                        annotation_text="Current period",
                        annotation_position="top left",
                        annotation_font=dict(size=11, color="#616161"),
                    )
                except Exception:
                    # Fallback for older Plotly versions: add shape + annotation manually
                    fig2.add_shape(
                        type="line",
                        x0=current_x_val,
                        x1=current_x_val,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(width=1.5, dash="dot", color="#9e9e9e"),
                    )
                    fig2.add_annotation(
                        x=current_x_val,
                        y=1,
                        xref="x",
                        yref="paper",
                        text="Current period",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=11, color="#616161"),
                    )

                row_current = results[
                    (results["Product"] == sku)
                    & (results["Location"] == loc)
                    & (results["Period"] == CURRENT_MONTH_TS)
                ]
                if not row_current.empty:
                    dcov = row_current["Days_Covered_by_SS"].iloc[0]
                    if pd.notna(dcov):
                        coverage_label = f"{dcov:.1f} days coverage"
                        fig2.add_annotation(
                            x=current_x_val,
                            y=current_fcst + current_ss,
                            text=coverage_label,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            ax=0,
                            ay=-40,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor=color_ss_line,
                            borderwidth=1,
                            font=dict(color=color_ss_line, size=11),
                        )

            fig2.update_layout(
                title=dict(
                    text="Risk-focused Inventory Corridor (Safety Stock highlighted in red)",
                    x=0,
                    xanchor="left",
                    font=dict(size=14),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=11),
                ),
                xaxis_title="Period",
                yaxis_title="Units",
                xaxis=dict(tickformat="%b\n%Y", showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False),
                margin=dict(l=40, r=10, t=60, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # -------- TAB 2: Network Topology --------
    with tab2:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            sku_default = default_product
            sku_index = all_products.index(sku_default) if all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="network_sku")

            if period_labels:
                try:
                    default_label = period_label(default_period) if default_period is not None else period_labels[-1]
                    period_index = period_labels.index(default_label) if default_label in period_labels else len(period_labels) - 1
                except Exception:
                    period_index = len(period_labels) - 1
                chosen_label = st.selectbox(
                    "PERIOD",
                    period_labels,
                    index=period_index,
                    key="network_period",
                )
                chosen_period = period_label_map.get(chosen_label, default_period)
            else:
                chosen_period = CURRENT_MONTH_TS

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            render_selection_line("Selected:", product=sku, period_text=period_label(chosen_period))
            st.subheader("🕸️ Network Topology")

            st.markdown(
                """
                <style>
                    iframe {
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        border: none;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            label_data = (
                results[results["Period"] == chosen_period]
                .set_index(["Product", "Location"])
                .to_dict("index")
            )
            sku_lt = df_lt[df_lt["Product"] == sku] if "Product" in df_lt.columns else df_lt.copy()

            net = Network(
                height="700px",
                width="100%",
                directed=True,
                bgcolor="#ffffff",
                font_color="#222222",
            )

            hubs = {"B616", "BEEX", "LUEX"}
            if not sku_lt.empty:
                froms = set(sku_lt["From_Location"].dropna().unique().tolist())
                tos = set(sku_lt["To_Location"].dropna().unique().tolist())
                all_nodes = froms.union(tos).union(hubs)
            else:
                all_nodes = set(hubs)

            demand_lookup = {}
            for n in all_nodes:
                demand_lookup[n] = label_data.get(
                    (sku, n),
                    {
                        "Forecast": 0,
                        "Agg_Future_Internal": 0,
                        "Agg_Future_External": 0,
                        "Safety_Stock": 0,
                        "Tier_Hops": np.nan,
                        "Service_Level_Node": np.nan,
                        "D_day": 0,
                        "Days_Covered_by_SS": np.nan,
                    },
                )

            for n in sorted(all_nodes):
                m = demand_lookup.get(
                    n,
                    {
                        "Forecast": 0,
                        "Agg_Future_Internal": 0,
                        "Agg_Future_External": 0,
                        "Safety_Stock": 0,
                        "Tier_Hops": np.nan,
                        "Service_Level_Node": np.nan,
                        "D_day": 0,
                        "Days_Covered_by_SS": np.nan,
                    },
                )
                used = float(m.get("Agg_Future_External", 0)) > 0 or float(m.get("Forecast", 0)) > 0

                if n == "B616":
                    bg, border, font_color, size = "#dcedc8", "#8bc34a", "#0b3d91", 14
                elif n in {"BEEX", "LUEX"}:
                    bg, border, font_color, size = "#bbdefb", "#64b5f6", "#0b3d91", 14
                else:
                    if used:
                        bg, border, font_color, size = "#fff9c4", "#fbc02d", "#222222", 12
                    else:
                        bg, border, font_color, size = "#f7f7f7", "#e0e0e0", "#b0b0b0", 10

                sl_node = m.get("Service_Level_Node", None)
                if pd.notna(sl_node):
                    try:
                        sl_label = f"{float(sl_node) * 100:.2f}%"
                    except Exception:
                        sl_label = str(sl_node)
                else:
                    sl_label = "-"

                lbl = (
                    f"{n}\n"
                    f"LDD: {euro_format(m.get('Forecast', 0), show_zero=True)}\n"
                    f"EXT: {euro_format(m.get('Agg_Future_External', 0), show_zero=True)}\n"
                    f"SS: {euro_format(m.get('Safety_Stock', 0), show_zero=True)}\n"
                    f"SL: {sl_label}"
                )
                net.add_node(
                    n,
                    label=lbl,
                    title=lbl,
                    color={"background": bg, "border": border},
                    shape="box",
                    font={"color": font_color, "size": size},
                )

            if not sku_lt.empty:
                for _, r in sku_lt.iterrows():
                    from_n, to_n = r["From_Location"], r["To_Location"]
                    if pd.isna(from_n) or pd.isna(to_n):
                        continue
                    from_used = float(demand_lookup.get(from_n, {}).get("Agg_Future_External", 0)) > 0 or float(
                        demand_lookup.get(from_n, {}).get("Forecast", 0)
                    ) > 0
                    to_used = float(demand_lookup.get(to_n, {}).get("Agg_Future_External", 0)) > 0 or float(
                        demand_lookup.get(to_n, {}).get("Forecast", 0)
                    ) > 0
                    edge_color = "#dddddd" if not from_used and not to_used else "#888888"
                    lt_val = r.get("Lead_Time_Days", 0)
                    label = f"{int(lt_val)}d" if not pd.isna(lt_val) else ""
                    net.add_edge(from_n, to_n, label=label, color=edge_color)

            net.set_options(
                """
                {
                  "physics": {
                    "stabilization": { "iterations": 200, "fit": true }
                  },
                  "nodes": { "borderWidthSelected": 2 },
                  "interaction": { "hover": true, "zoomView": true },
                  "layout": { "improvedLayout": true }
                }
                """
            )
            tmpfile = "net.html"
            net.save_graph(tmpfile)
            html_text = open(tmpfile, "r", encoding="utf-8").read()
            injection_css = """
            <style>
              html, body { height: 100%; margin: 0; padding: 0; }
              #mynetwork { display:flex !important; align-items:center; justify-content:center; height:700px !important; width:100% !important; }
              .vis-network { display:block !important; margin: 0 auto !important; }
            </style>
            """
            injection_js = """
            <script>
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
            if "</head>" in html_text:
                html_text = html_text.replace("</head>", injection_css + "</head>", 1)
            if "</body>" in html_text:
                html_text = html_text.replace("</body>", injection_js + "</body>", 1)
            else:
                html_text += injection_js
            components.html(html_text, height=750)

            st.markdown(
                """
                <div style="text-align:center; font-size:12px; padding:8px 0;">
                  <div style="display:inline-block; background:#f7f9fc; padding:8px 12px; border-radius:8px;">
                    <strong>Legend:</strong><br/>
                    LDD = Local Direct Demand (local forecast) &nbsp;&nbsp;|&nbsp;&nbsp;
                    EXT = External Demand (downstream forecasts rolled-up) &nbsp;&nbsp;|&nbsp;&nbsp;
                    SS  = Safety Stock (final policy value)
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- TAB 3 ----------------
    with tab3:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            st.markdown(
                "<div style='padding:6px 0;'></div>",
                unsafe_allow_html=True,
            )
            prod_choices = (
                sorted(meaningful_results["Product"].unique())
                if not meaningful_results.empty
                else sorted(results["Product"].unique())
            )
            loc_choices = (
                sorted(meaningful_results["Location"].unique())
                if not meaningful_results.empty
                else sorted(results["Location"].unique())
            )

            period_choices_labels = period_labels

            default_prod_list = (
                [default_product] if default_product in prod_choices else []
            )
            default_period_list = []
            cur_label = period_label(CURRENT_MONTH_TS)
            if cur_label in period_choices_labels:
                default_period_list = [cur_label]
            else:
                if default_period is not None:
                    dp_label = period_label(default_period)
                    if dp_label in period_choices_labels:
                        default_period_list = [dp_label]

            f_prod = st.multiselect(
                "MATERIAL",
                prod_choices,
                default=default_prod_list,
                key="full_f_prod",
            )
            f_loc = st.multiselect(
                "LOCATION", loc_choices, default=[], key="full_f_loc"
            )
            f_period_labels = st.multiselect(
                "PERIOD",
                period_choices_labels,
                default=default_period_list,
                key="full_f_period",
            )

            f_period = (
                [period_label_map[lbl] for lbl in f_period_labels]
                if f_period_labels
                else []
            )

            with st.container():
                st.markdown(
                    '<div class="export-csv-btn">', unsafe_allow_html=True
                )
                st.download_button(
                    "💾 Export CSV",
                    data=results.to_csv(index=False),
                    file_name="filtered_plan.csv",
                    mime="text/csv",
                    key="full_plan_export",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div style='height:6px'></div>", unsafe_allow_html=True
            )

        with col_main:
            badge_product = (
                f_prod[0]
                if f_prod
                else (
                    default_product
                    if default_product in all_products
                    else (all_products[0] if all_products else "")
                )
            )
            badge_loc = ", ".join(f_loc) if f_loc else ""
            badge_period = ", ".join(f_period_labels) if f_period_labels else ""
            selected_text_parts = []
            if badge_product:
                selected_text_parts.append(badge_product)
            if badge_loc:
                selected_text_parts.append(badge_loc)
            if badge_period:
                selected_text_parts.append(badge_period)
            render_selection_line(
                "Selected:",
                product=" — ".join(selected_text_parts) if selected_text_parts else None,
            )

            st.subheader("📋 Global Inventory Plan")
            filtered = results.copy()
            if f_prod:
                filtered = filtered[filtered["Product"].isin(f_prod)]
            if f_loc:
                filtered = filtered[filtered["Location"].isin(f_loc)]
            if f_period:
                filtered = filtered[filtered["Period"].isin(f_period)]
            filtered = filtered.sort_values("Safety_Stock", ascending=False)

            filtered_display = hide_zero_rows(filtered)

            display_cols = [
                c
                for c in [
                    "Product",
                    "Location",
                    "Period",
                    "Forecast",
                    "D_day",
                    "Days_Covered_by_SS",
                    "Safety_Stock",
                    "Adjustment_Status",
                ]
                if c in filtered_display.columns
            ]
            fmt_cols = [
                c
                for c in [
                    "Forecast",
                    "D_day",
                    "Days_Covered_by_SS",
                    "Safety_Stock",
                ]
                if c in filtered_display.columns
            ]

            disp_df = filtered_display.copy()
            if "Period" in disp_df.columns:
                try:
                    disp_df["Period"] = disp_df["Period"].apply(period_label)
                except Exception:
                    pass

            disp = df_format_for_display(
                disp_df[display_cols].copy(),
                cols=fmt_cols,
                two_decimals_cols=["D_day", "Days_Covered_by_SS"],
            )
            st.dataframe(disp, use_container_width=True, height=700)

    # ---------------- TAB 4 ----------------
    with tab4:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            sku_default = default_product
            sku_index = (
                all_products.index(sku_default)
                if all_products
                else 0
            )
            sku = st.selectbox(
                "MATERIAL", all_products, index=sku_index, key="eff_sku"
            )

            if period_labels:
                try:
                    default_label = (
                        period_label(default_period)
                        if default_period is not None
                        else period_labels[-1]
                    )
                    period_index = (
                        period_labels.index(default_label)
                        if default_label in period_labels
                        else len(period_labels) - 1
                    )
                except Exception:
                    period_index = len(period_labels) - 1
                chosen_label = st.selectbox(
                    "PERIOD",
                    period_labels,
                    index=period_index,
                    key="eff_period",
                )
                eff_period = period_label_map.get(chosen_label, default_period)
            else:
                eff_period = CURRENT_MONTH_TS

            snapshot_period = (
                eff_period
                if eff_period in all_periods
                else (all_periods[-1] if all_periods else None)
            )
            if snapshot_period is None:
                eff_export = results[results["Product"] == sku].copy()
            else:
                eff_export = results[
                    (results["Product"] == sku)
                    & (results["Period"] == snapshot_period)
                ].copy()

            with st.container():
                st.markdown(
                    '<div class="export-csv-btn">', unsafe_allow_html=True
                )
            st.download_button(
                "💾 Export CSV",
                data=eff_export.to_csv(index=False),
                file_name=f"efficiency_{sku}_{period_label(snapshot_period) if snapshot_period is not None else 'all'}.csv",
                mime="text/csv",
                key="eff_export_btn",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div style='height:6px'></div>", unsafe_allow_html=True
            )

        with col_main:
            render_selection_line(
                "Selected:",
                product=sku,
                period_text=period_label(eff_period),
            )
            st.subheader("⚖️ Efficiency & Policy Analysis")

            snapshot_period = (
                eff_period
                if eff_period in all_periods
                else (all_periods[-1] if all_periods else None)
            )
            if snapshot_period is None:
                st.warning("No period data available for Efficiency Analysis.")
                eff = results[(results["Product"] == sku)].copy()
            else:
                eff = results[
                    (results["Product"] == sku)
                    & (results["Period"] == snapshot_period)
                ].copy()
            eff["SS_to_FCST_Ratio"] = (
                eff["Safety_Stock"]
                / eff["Agg_Future_Demand"].replace(0, np.nan)
            ).fillna(0)
            eff_display = hide_zero_rows(eff)
            total_ss_sku = eff["Safety_Stock"].sum()
            total_net_demand_sku = eff["Agg_Future_Demand"].sum()
            sku_ratio = (
                total_ss_sku / total_net_demand_sku
                if total_net_demand_sku > 0
                else 0
            )
            all_res = (
                results[results["Period"] == snapshot_period]
                if snapshot_period is not None
                else results
            )
            global_ratio = (
                all_res["Safety_Stock"].sum()
                / all_res["Agg_Future_Demand"].replace(0, np.nan).sum()
                if not all_res.empty
                else 0
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Network Ratio (Material)", f"{sku_ratio:.2f}")
            m2.metric(
                "Global Network Ratio (All Items)", f"{global_ratio:.2f}"
            )
            m3.metric("Total SS for Material", euro_format(int(total_ss_sku), True))
            st.markdown("---")
            c1, c2 = st.columns([3, 2])
            with c1:
                fig_eff = px.scatter(
                    eff_display,
                    x="Agg_Future_Demand",
                    y="Safety_Stock",
                    color="Adjustment_Status",
                    size="SS_to_FCST_Ratio",
                    hover_name="Location",
                    color_discrete_map={
                        "Optimal (Statistical)": "#00CC96",
                        "Capped (High)": "#EF553B",
                        "Capped (Low)": "#636EFA",
                        "Forced to Zero": "#AB63FA",
                    },
                    title="Policy Impact & Efficiency Ratio (Bubble Size = SS_to_FCST_Ratio)",
                )
                st.plotly_chart(fig_eff, use_container_width=True)
            with c2:
                st.markdown("**Status Breakdown**")
                st.table(eff_display["Adjustment_Status"].value_counts())
                st.markdown(
                    "**Top Nodes by Safety Stock (snapshot)**"
                )
                eff_top = eff_display.sort_values(
                    "Safety_Stock", ascending=False
                )
                eff_top_display = (
                    eff_top[
                        [
                            "Location",
                            "Adjustment_Status",
                            "Safety_Stock",
                            "SS_to_FCST_Ratio",
                        ]
                    ]
                    .head(10)
                    .reset_index(drop=True)
                )
                eff_top_display["Safety_Stock"] = eff_top_display[
                    "Safety_Stock"
                ].round(0)
                eff_top_fmt = df_format_for_display(
                    eff_top_display,
                    cols=["Safety_Stock", "SS_to_FCST_Ratio"],
                    two_decimals_cols=["SS_to_FCST_Ratio"],
                )
                # styled table inside a wrapper to allow header wrapping and avoid truncation
                eff_top_styled = eff_top_fmt.style.set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("white-space", "normal"),
                                ("word-break", "break-word"),
                                ("max-width", "120px"),
                            ],
                        },
                        {
                            "selector": "td",
                            "props": [("font-size", "11px")],
                        },
                    ]
                )
                st.markdown('<div class="ss-top-table">', unsafe_allow_html=True)
                st.dataframe(
                    eff_top_styled,
                    use_container_width=True,
                    height=300,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 5 ----------------
    with tab5:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            h_sku_default = default_product
            h_sku_index = (
                all_products.index(h_sku_default)
                if all_products
                else 0
            )
            h_sku = st.selectbox(
                "MATERIAL", all_products, index=h_sku_index, key="h1"
            )

            h_loc_opts = sorted(
                results[results["Product"] == h_sku]["Location"]
                .unique()
                .tolist()
            )
            if not h_loc_opts:
                h_loc_opts = sorted(
                    hist[hist["Product"] == h_sku]["Location"]
                    .unique()
                    .tolist()
                )
            if not h_loc_opts:
                h_loc_opts = ["(no location)"]
            h_loc_default = (
                DEFAULT_LOCATION_CHOICE
                if DEFAULT_LOCATION_CHOICE in h_loc_opts
                else (h_loc_opts[0] if h_loc_opts else "(no location)")
            )
            h_loc_index = (
                h_loc_opts.index(h_loc_default)
                if h_loc_default in h_loc_opts
                else 0
            )
            h_loc = st.selectbox(
                "LOCATION", h_loc_opts, index=h_loc_index, key="h2"
            )

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            render_selection_line(
                "Selected:",
                product=h_sku,
                location=(h_loc if h_loc != "(no location)" else None),
            )
            st.subheader("📉 Historical Forecast vs Actuals")
            hdf = hist.copy()
            if h_loc != "(no location)":
                hdf = hdf[
                    (hdf["Product"] == h_sku) & (hdf["Location"] == h_loc)
                ].sort_values("Period")
            else:
                hdf = hdf[hdf["Product"] == h_sku].sort_values("Period")

            if not hdf.empty:
                k1, k2, k3 = st.columns(3)
                denom_consumption = (
                    hdf["Consumption"].replace(0, np.nan).sum()
                )
                if denom_consumption > 0:
                    wape_val = (
                        hdf["Abs_Error"].sum()
                        / denom_consumption
                        * 100
                    )
                    bias_val = (
                        hdf["Deviation"].sum()
                        / denom_consumption
                        * 100
                    )
                    k1.metric("WAPE (%)", f"{wape_val:.1f}")
                    k2.metric("Bias (%)", f"{bias_val:.1f}")
                else:
                    k1.metric("WAPE (%)", "N/A")
                    k2.metric("Bias (%)", "N/A")
                avg_acc = (
                    hdf["Accuracy_%"].mean()
                    if not hdf["Accuracy_%"].isna().all()
                    else np.nan
                )
                k3.metric(
                    "Avg Accuracy (%)",
                    f"{avg_acc:.1f}" if not np.isnan(avg_acc) else "N/A",
                )

                fig_hist = go.Figure(
                    [
                        go.Scatter(
                            x=hdf["Period"],
                            y=hdf["Consumption"],
                            name="Actuals",
                            line=dict(color="black"),
                        ),
                        go.Scatter(
                            x=hdf["Period"],
                            y=hdf["Forecast_Hist"],
                            name="Forecast",
                            line=dict(color="blue", dash="dot"),
                        ),
                    ]
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("---")

                st.subheader(
                    "Aggregated Network History (Selected Product)"
                )
                net_table = (
                    hist_net[hist_net["Product"] == h_sku]
                    .merge(
                        hdf[["Period"]].drop_duplicates(),
                        on="Period",
                        how="inner",
                    )
                    .sort_values("Period")
                    .drop(columns=["Product"])
                )
                if not net_table.empty:
                    net_table["Net_Abs_Error"] = (
                        net_table["Network_Consumption"]
                        - net_table["Network_Forecast_Hist"]
                    ).abs()
                    denom_net = (
                        net_table["Network_Consumption"]
                        .replace(0, np.nan)
                        .sum()
                    )
                    net_wape = (
                        net_table["Net_Abs_Error"].sum()
                        / denom_net
                        * 100
                        if denom_net > 0
                        else np.nan
                    )
                else:
                    net_wape = np.nan
                c_net1, c_net2 = st.columns([3, 1])
                with c_net1:
                    if not net_table.empty:
                        st.dataframe(
                            df_format_for_display(
                                net_table[
                                    [
                                        "Period",
                                        "Network_Consumption",
                                        "Network_Forecast_Hist",
                                    ]
                                ].copy(),
                                cols=[
                                    "Network_Consumption",
                                    "Network_Forecast_Hist",
                                ],
                                two_decimals_cols=[
                                    "Network_Consumption",
                                    "Network_Forecast_Hist",
                                ],
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.write(
                            "No aggregated network history available for the chosen selection."
                        )
                with c_net2:
                    c_val = (
                        f"{net_wape:.1f}"
                        if not np.isnan(net_wape)
                        else "N/A"
                    )
                    st.metric("Network WAPE (%)", c_val)

    # ---------------- TAB 6 ----------------
    with tab6:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            calc_sku_default = default_product
            calc_sku_index = (
                all_products.index(calc_sku_default)
                if all_products
                else 0
            )
            calc_sku = st.selectbox(
                "MATERIAL", all_products, index=calc_sku_index, key="c_sku"
            )

            avail_locs = sorted(
                meaningful_results[meaningful_results["Product"] == calc_sku][
                    "Location"
                ]
                .unique()
                .tolist()
            )
            if not avail_locs:
                avail_locs = sorted(
                    results[results["Product"] == calc_sku]["Location"]
                    .unique()
                    .tolist()
                )
            if not avail_locs:
                avail_locs = ["(no location)"]
            calc_loc_default = (
                DEFAULT_LOCATION_CHOICE
                if DEFAULT_LOCATION_CHOICE in avail_locs
                else (avail_locs[0] if avail_locs else "(no location)")
            )
            calc_loc_index = (
                avail_locs.index(calc_loc_default)
                if calc_loc_default in avail_locs
                else 0
            )
            calc_loc = st.selectbox(
                "LOCATION", avail_locs, index=calc_loc_index, key="c_loc"
            )

            if period_labels:
                try:
                    default_label = (
                        period_label(default_period)
                        if default_period is not None
                        else period_labels[-1]
                    )
                    calc_period_index = (
                        period_labels.index(default_label)
                        if default_label in period_labels
                        else len(period_labels) - 1
                    )
                except Exception:
                    calc_period_index = len(period_labels) - 1
                chosen_label = st.selectbox(
                    "PERIOD",
                    period_labels,
                    index=calc_period_index,
                    key="c_period",
                )
            else:
                chosen_label = period_label(CURRENT_MONTH_TS)
            calc_period = period_label_map.get(
                chosen_label, default_period
            )

            row_export = results[
                (results["Product"] == calc_sku)
                & (results["Location"] == calc_loc)
                & (results["Period"] == calc_period)
            ]
            export_data = row_export if not row_export.empty else pd.DataFrame()

            with st.container():
                st.markdown(
                    '<div class="export-csv-btn">', unsafe_allow_html=True
                )
                st.download_button(
                    "💾 Export CSV",
                    data=export_data.to_csv(index=False),
                    file_name=f"calc_trace_{calc_sku}_{calc_loc}_{period_label(calc_period)}.csv",
                    mime="text/csv",
                    key="calc_export_btn",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div style='height:6px'></div>", unsafe_allow_html=True
            )

        with col_main:
            st.markdown(
                """
                  <style>
                    .calc-mapping-container {
                      max-width: 560px;
                    }
                  </style>
                  """,
                unsafe_allow_html=True,
            )

            render_selection_line(
                "Selected:",
                product=calc_sku,
                location=calc_loc,
                period_text=period_label(calc_period),
            )
            st.subheader("🧮 Transparent Calculation Engine & Scenario Simulation")
            st.write(
                "See how changing service level or lead-time assumptions affects safety stock."
            )

            z_current = norm.ppf(service_level)

            row_df = results[
                (results["Product"] == calc_sku)
                & (results["Location"] == calc_loc)
                & (results["Period"] == calc_period)
            ]
            if row_df.empty:
                st.warning("Selection not found in results.")
            else:
                row = row_df.iloc[0]

                node_sl = float(row.get("Service_Level_Node", service_level))
                node_z = float(row.get("Z_node", norm.ppf(node_sl)))
                hops = int(row.get("Tier_Hops", 0))

                mapping_rows = [
                    (0, "99%", "End-node"),
                    (1, "95%", "Internal + external demand"),
                    (2, "90%", "Level-1 hub"),
                    (3, "85%", "Level-2 hub"),
                ]
                table_html = """
                <div class="calc-mapping-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; font-size:13px;">
                  <table style="width:40%; border-collapse:collapse;">
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
                    row_style = (
                        "background:#FFF59D; font-weight:700;"
                        if h == hops
                        else ""
                    )
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
                st.markdown(
                    "**Applied Hop → Service Level mapping (highlight shows which row was used for this node):**"
                )
                try:
                    components.html(table_html, height=200)
                except Exception:
                    st.markdown(table_html, unsafe_allow_html=True)

                avg_daily = row.get("D_day", np.nan)
                days_cov = row.get("Days_Covered_by_SS", np.nan)
                avg_daily_txt = (
                    f"{avg_daily:.2f}" if pd.notna(avg_daily) else "N/A"
                )
                days_cov_txt = (
                    f"{days_cov:.1f}" if pd.notna(days_cov) else "N/A"
                )
                st.markdown("---")
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
                st.markdown(
                    "**Values used for the calculation (highlighted above):**"
                )
                st.markdown(summary_html, unsafe_allow_html=True)

                # ---- Scenario planning banner ABOVE expander ----
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
                      SCENARIO PLANNING TOOL — simulate alternative SL / LT assumptions (analysis‑only)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(
                    "Show detailed scenario controls", expanded=False
                ):
                    st.markdown(
                        """
                        <div style="border:1px solid #0b3d91;border-radius:10px;background:#fff9e0;padding:12px;color:#0b3d91;font-size:0.95rem;">
                          Use scenarios to test sensitivity to Service Level or Lead Time. Scenarios do not change implemented policy — they are analysis-only.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if "n_scen" not in st.session_state:
                        st.session_state["n_scen"] = 1
                    options = [1, 2, 3]
                    default_index = (
                        options.index(st.session_state.get("n_scen", 1))
                        if st.session_state.get("n_scen", 1) in options
                        else 0
                    )
                    n_scen = st.selectbox(
                        "Number of Scenarios to compare",
                        options,
                        index=default_index,
                        key="n_scen",
                    )
                    scenarios = []
                    for s in range(n_scen):
                        with st.expander(
                            f"Scenario {s+1} inputs", expanded=False
                        ):
                            sc_sl_default = (
                                float(service_level * 100)
                                if s == 0
                                else min(
                                    99.9, float(service_level * 100) + 0.5 * s
                                )
                            )
                            sc_sl = st.slider(
                                f"Scenario {s+1} Service Level (%)",
                                50.0,
                                99.9,
                                sc_sl_default,
                                key=f"sc_sl_{s}",
                            )
                            sc_lt_default = (
                                float(row["LT_Mean"])
                                if s == 0
                                else float(row["LT_Mean"])
                            )
                            sc_lt = st.slider(
                                f"Scenario {s+1} Avg Lead Time (Days)",
                                0.0,
                                max(
                                    30.0,
                                    float(row["LT_Mean"]) * 2,
                                ),
                                value=sc_lt_default,
                                key=f"sc_lt_{s}",
                            )
                            sc_lt_std_default = (
                                float(row["LT_Std"])
                                if s == 0
                                else float(row["LT_Std"])
                            )
                            sc_lt_std = st.slider(
                                f"Scenario {s+1} LT Std Dev (Days)",
                                0.0,
                                max(
                                    10.0,
                                    float(row["LT_Std"]) * 2,
                                ),
                                value=sc_lt_std_default,
                                key=f"sc_lt_std_{s}",
                            )
                            scenarios.append(
                                {
                                    "SL_pct": sc_sl,
                                    "LT_mean": sc_lt,
                                    "LT_std": sc_lt_std,
                                }
                            )

                    scen_rows = []
                    for idx, sc in enumerate(scenarios):
                        sc_z = norm.ppf(sc["SL_pct"] / 100.0)
                        d_day = float(row["Agg_Future_Demand"]) / float(
                            days_per_month
                        )
                        sigma_d_day = float(row["Agg_Std_Hist"]) / math.sqrt(
                            float(days_per_month)
                        )
                        var_d = sigma_d_day ** 2
                        if row["Agg_Future_Demand"] < 20.0:
                            var_d = max(var_d, d_day)
                        sc_ss = sc_z * math.sqrt(
                            var_d * sc["LT_mean"]
                            + (sc["LT_std"] ** 2) * (d_day ** 2)
                        )
                        sc_floor = d_day * sc["LT_mean"] * 0.01
                        sc_ss = max(sc_ss, sc_floor)
                        scen_rows.append(
                            {
                                "Scenario": f"S{idx+1}",
                                "Service_Level_%": sc["SL_pct"],
                                "LT_mean_days": sc["LT_mean"],
                                "LT_std_days": sc["LT_std"],
                                "Simulated_SS": sc_ss,
                            }
                        )
                    scen_df = pd.DataFrame(scen_rows)

                    base_row = {
                        "Scenario": "Base (Stat)",
                        "Service_Level_%": service_level * 100,
                        "LT_mean_days": row["LT_Mean"],
                        "LT_std_days": row["LT_Std"],
                        "Simulated_SS": row["Pre_Rule_SS"],
                    }
                    impl_row = {
                        "Scenario": "Implemented",
                        "Service_Level_%": np.nan,
                        "LT_mean_days": np.nan,
                        "LT_std_days": np.nan,
                        "Simulated_SS": row["Safety_Stock"],
                    }
                    compare_df = pd.concat(
                        [pd.DataFrame([base_row, impl_row]), scen_df],
                        ignore_index=True,
                        sort=False,
                    )
                    display_comp = compare_df.copy()
                    display_comp["Simulated_SS"] = display_comp[
                        "Simulated_SS"
                    ].astype(float)

                    st.markdown(
                        "Scenario comparison (Simulated SS). 'Implemented' shows the final Safety_Stock after rules."
                    )
                    st.markdown(
                        '<div class="scenario-table-container">',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        df_format_for_display(
                            display_comp[
                                [
                                    "Scenario",
                                    "Service_Level_%",
                                    "LT_mean_days",
                                    "LT_std_days",
                                    "Simulated_SS",
                                ]
                            ].copy(),
                            cols=[
                                "Service_Level_%",
                                "LT_mean_days",
                                "LT_std_days",
                                "Simulated_SS",
                            ],
                            two_decimals_cols=[
                                "Service_Level_%",
                                "Simulated_SS",
                            ],
                        ),
                        use_container_width=False,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    fig_bar = go.Figure()
                    colors = px.colors.qualitative.Pastel
                    fig_bar.add_trace(
                        go.Bar(
                            x=display_comp["Scenario"],
                            y=display_comp["Simulated_SS"],
                            marker_color=colors[: len(display_comp)],
                        )
                    )
                    fig_bar.update_layout(
                        title="Scenario SS Comparison",
                        yaxis_title="SS (units)",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- TAB 7 ----------------
    with tab7:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            sel_prod_default = default_product
            sel_prod_index = (
                all_products.index(sel_prod_default)
                if all_products
                else 0
            )
            selected_product = st.selectbox(
                "MATERIAL",
                all_products,
                index=sel_prod_index,
                key="mat_sel",
            )

            if period_labels:
                try:
                    sel_label = (
                        period_label(default_period)
                        if default_period is not None
                        else period_labels[-1]
                    )
                    sel_period_index = (
                        period_labels.index(sel_label)
                        if sel_label in period_labels
                        else len(period_labels) - 1
                    )
                except Exception:
                    sel_period_index = len(period_labels) - 1
                chosen_label = st.selectbox(
                    "PERIOD",
                    period_labels,
                    index=sel_period_index,
                    key="mat_period",
                )
                selected_period = period_label_map.get(
                    chosen_label, default_period
                )
            else:
                selected_period = CURRENT_MONTH_TS

            mat_period_export = results[
                (results["Product"] == selected_product)
                & (results["Period"] == selected_period)
            ].copy()

            with st.container():
                st.markdown(
                    '<div class="export-csv-btn">', unsafe_allow_html=True
                )
                st.download_button(
                    "💾 Export CSV",
                    data=mat_period_export.to_csv(index=False),
                    file_name=f"material_view_{selected_product}_{period_label(selected_period)}.csv",
                    mime="text/csv",
                    key="mat_export_btn",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div style='height:6px'></div>", unsafe_allow_html=True
            )

        with col_main:
            st.markdown(
                """
                <style>
                  div[data-testid="stVerticalBlock"] div:nth-child(7) .stMarkdown p {
                    font-size: 0.82rem !important;
                  }
                </style>
                """,
                unsafe_allow_html=True,
            )

            render_selection_line(
                "Selected:",
                product=selected_product,
                period_text=period_label(selected_period),
            )
            st.subheader("📦 View by Material (+ 8 Reasons for Inventory)")

            mat_period_df = results[
                (results["Product"] == selected_product)
                & (results["Period"] == selected_period)
            ].copy()
            mat_period_df_display = hide_zero_rows(mat_period_df)
            total_forecast = mat_period_df["Forecast"].sum()
            network_total_forecast = df_d[
                (df_d["Product"] == selected_product)
                & (df_d["Period"] == selected_period)
            ]["Forecast"].sum()
            total_net = network_total_forecast
            total_ss = mat_period_df["Safety_Stock"].sum()
            nodes_count = mat_period_df["Location"].nunique()
            avg_ss_per_node = (
                mat_period_df["Safety_Stock"].mean()
                if nodes_count > 0
                else 0
            )

            try:
                avg_days_covered = mat_period_df[
                    "Days_Covered_by_SS"
                ].replace([np.inf, -np.inf], np.nan).mean()
            except Exception:
                avg_days_covered = np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Total Local Forecast", euro_format(total_forecast, True)
            )
            k2.metric(
                "Total Safety Stock (sum nodes)",
                euro_format(total_ss, True),
            )
            k3.metric("Nodes", f"{nodes_count}")
            k4.metric(
                "Avg Days Covered (nodes)",
                f"{avg_days_covered:.1f}"
                if not pd.isna(avg_days_covered)
                else "N/A",
            )

            st.markdown("---")
            st.markdown(
                "### Why do we carry this SS? — 8 Reasons breakdown (aggregated for selected material)"
            )
            if mat_period_df_display.empty:
                st.warning(
                    "No data for this material/period (non-zero rows filtered)."
                )
            else:
                mat = mat_period_df.copy()
                for c in [
                    "LT_Mean",
                    "LT_Std",
                    "Agg_Std_Hist",
                    "Pre_Rule_SS",
                    "Safety_Stock",
                    "Forecast",
                    "Agg_Future_Demand",
                    "Agg_Future_Internal",
                    "Agg_Future_External",
                    "D_day",
                    "Days_Covered_by_SS",
                ]:
                    mat[c] = mat[c].fillna(0)

                mat["term1"] = (
                    mat["Agg_Std_Hist"] ** 2 / float(days_per_month)
                ) * mat["LT_Mean"]
                mat["term2"] = (mat["LT_Std"] ** 2) * (
                    mat["Agg_Future_Demand"] / float(days_per_month)
                ) ** 2
                z_current = norm.ppf(service_level)
                mat["demand_uncertainty_raw"] = z_current * np.sqrt(
                    mat["term1"].clip(lower=0)
                )
                mat["lt_uncertainty_raw"] = z_current * np.sqrt(
                    mat["term2"].clip(lower=0)
                )
                mat["direct_forecast_raw"] = mat["Forecast"].clip(lower=0)
                mat["indirect_network_raw"] = mat[
                    "Agg_Future_External"
                ].clip(lower=0)
                mat["cap_reduction_raw"] = (
                    (mat["Pre_Rule_SS"] - mat["Safety_Stock"]).clip(lower=0)
                ).fillna(0)
                mat["cap_increase_raw"] = (
                    (mat["Safety_Stock"] - mat["Pre_Rule_SS"]).clip(lower=0)
                ).fillna(0)
                mat["forced_zero_raw"] = mat.apply(
                    lambda r: r["Pre_Rule_SS"]
                    if r["Adjustment_Status"] == "Forced to Zero"
                    else 0,
                    axis=1,
                )
                mat["b616_override_raw"] = mat.apply(
                    lambda r: r["Pre_Rule_SS"]
                    if (r["Location"] == "B616" and r["Safety_Stock"] == 0)
                    else 0,
                    axis=1,
                )

                raw_drivers = {
                    "Demand Uncertainty (z*sqrt(term1))": mat[
                        "demand_uncertainty_raw"
                    ].sum(),
                    "Lead-time Uncertainty (z*sqrt(term2))": mat[
                        "lt_uncertainty_raw"
                    ].sum(),
                    "Direct Local Forecast (sum Fcst)": mat[
                        "direct_forecast_raw"
                    ].sum(),
                    "Indirect Network Demand (sum extra downstream)": mat[
                        "indirect_network_raw"
                    ].sum(),
                    "Caps — Reductions (policy lowering SS)": mat[
                        "cap_reduction_raw"
                    ].sum(),
                    "Caps — Increases (policy increasing SS)": mat[
                        "cap_increase_raw"
                    ].sum(),
                    "Forced Zero Overrides (policy)": mat[
                        "forced_zero_raw"
                    ].sum(),
                    "B616 Policy Override": mat["b616_override_raw"].sum(),
                }

                drv_df = pd.DataFrame(
                    {
                        "driver": list(raw_drivers.keys()),
                        "amount": [float(v) for v in raw_drivers.values()],
                    }
                )
                drv_df_display = drv_df[drv_df["amount"] != 0].copy()
                drv_denom = drv_df["amount"].sum()
                drv_df_display["pct_of_total_ss"] = (
                    drv_df_display["amount"]
                    / (drv_denom if drv_denom > 0 else 1.0)
                    * 100
                )

                st.markdown(
                    "#### A. Original — Raw driver values (interpretation view)"
                )
                pastel_colors = px.colors.qualitative.Pastel
                fig_drv_raw = go.Figure()
                color_slice = (
                    pastel_colors[: len(drv_df_display)]
                    if len(drv_df_display) > 0
                    else pastel_colors
                )
                fig_drv_raw.add_trace(
                    go.Bar(
                        x=drv_df_display["driver"],
                        y=drv_df_display["amount"],
                        marker_color=color_slice,
                    )
                )
                annotations_raw = []
                for _, rowd in drv_df_display.iterrows():
                    annotations_raw.append(
                        dict(
                            x=rowd["driver"],
                            y=rowd["amount"],
                            text=f"{rowd['pct_of_total_ss']:.1f}%",
                            showarrow=False,
                            yshift=8,
                        )
                    )
                fig_drv_raw.update_layout(
                    title=f"{selected_product} — Raw Drivers (not SS-attribution)",
                    xaxis_title="Driver",
                    yaxis_title="Units",
                    annotations=annotations_raw,
                    height=420,
                )
                st.plotly_chart(fig_drv_raw, use_container_width=True)
                st.dataframe(
                    df_format_for_display(
                        drv_df_display.rename(
                            columns={
                                "driver": "Driver",
                                "amount": "Units",
                                "pct_of_total_ss": "Pct_of_raw_sum",
                            }
                        ).round(2),
                        cols=["Units", "Pct_of_raw_sum"],
                        two_decimals_cols=["Pct_of_raw_sum"],
                    ),
                    use_container_width=True,
                )

                # B. Attribution
                st.markdown("---")
                st.markdown(
                    "#### B. SS Attribution — Mutually exclusive components that SUM EXACTLY to Total Safety Stock"
                )
                per_node = mat.copy()
                per_node["is_forced_zero"] = (
                    per_node["Adjustment_Status"] == "Forced to Zero"
                )
                per_node["is_b616_override"] = (
                    per_node["Location"] == "B616"
                ) & (per_node["Safety_Stock"] == 0)
                per_node["pre_ss"] = per_node["Pre_Rule_SS"].clip(lower=0)
                per_node["share_denom"] = (
                    per_node["demand_uncertainty_raw"]
                    + per_node["lt_uncertainty_raw"]
                )

                def demand_share_calc(r):
                    if r["share_denom"] > 0:
                        return r["pre_ss"] * (
                            r["demand_uncertainty_raw"] / r["share_denom"]
                        )
                    else:
                        return (r["pre_ss"] / 2) if r["pre_ss"] > 0 else 0.0

                def lt_share_calc(r):
                    if r["share_denom"] > 0:
                        return r["pre_ss"] * (
                            r["lt_uncertainty_raw"] / r["share_denom"]
                        )
                    else:
                        return (r["pre_ss"] / 2) if r["pre_ss"] > 0 else 0.0

                per_node["demand_share"] = per_node.apply(
                    demand_share_calc, axis=1
                )
                per_node["lt_share"] = per_node.apply(
                    lt_share_calc, axis=1
                )
                per_node["forced_zero_amount"] = per_node.apply(
                    lambda r: r["pre_ss"]
                    if r["is_forced_zero"]
                    else 0.0,
                    axis=1,
                )
                per_node["b616_override_amount"] = per_node.apply(
                    lambda r: r["pre_ss"]
                    if r["is_b616_override"]
                    else 0.0,
                    axis=1,
                )

                def retained_ratio_calc(r):
                    if r["pre_ss"] <= 0:
                        return 0.0
                    if r["is_forced_zero"] or r["is_b616_override"]:
                        return 0.0
                    return (
                        float(r["Safety_Stock"]) / float(r["pre_ss"])
                        if r["pre_ss"] > 0
                        else 0.0
                    )

                per_node["retained_ratio"] = per_node.apply(
                    retained_ratio_calc, axis=1
                )
                per_node["retained_demand"] = (
                    per_node["demand_share"] * per_node["retained_ratio"]
                )
                per_node["retained_lt"] = (
                    per_node["lt_share"] * per_node["retained_ratio"]
                )
                per_node["retained_stat_total"] = (
                    per_node["retained_demand"]
                    + per_node["retained_lt"]
                )

                def direct_frac_calc(r):
                    if r["Agg_Future_Demand"] > 0:
                        return float(r["Forecast"]) / float(
                            r["Agg_Future_Demand"]
                        )
                    return 0.0

                per_node["direct_frac"] = per_node.apply(
                    direct_frac_calc, axis=1
                ).clip(lower=0, upper=1)
                per_node["direct_retained_ss"] = (
                    per_node["retained_stat_total"]
                    * per_node["direct_frac"]
                )
                per_node["indirect_retained_ss"] = (
                    per_node["retained_stat_total"]
                    * (1 - per_node["direct_frac"])
                )
                per_node["cap_reduction"] = per_node.apply(
                    lambda r: max(
                        r["pre_ss"] - r["Safety_Stock"], 0.0
                    )
                    if not (r["is_forced_zero"] or r["is_b616_override"])
                    else 0.0,
                    axis=1,
                )
                per_node["cap_increase"] = per_node.apply(
                    lambda r: max(
                        r["Safety_Stock"] - r["pre_ss"], 0.0
                    )
                    if not (r["is_forced_zero"] or r["is_b616_override"])
                    else 0.0,
                    axis=1,
                )

                ss_attrib = {
                    "Demand Uncertainty (SS portion)": per_node[
                        "retained_demand"
                    ].sum(),
                    "Lead-time Uncertainty (SS portion)": per_node[
                        "retained_lt"
                    ].sum(),
                    "Direct Local Forecast (SS portion)": per_node[
                        "direct_retained_ss"
                    ].sum(),
                    "Indirect Network Demand (SS portion)": per_node[
                        "indirect_retained_ss"
                    ].sum(),
                    "Caps — Reductions (policy lowering SS)": per_node[
                        "cap_reduction"
                    ].sum(),
                    "Caps — Increases (policy increasing SS)": per_node[
                        "cap_increase"
                    ].sum(),
                    "Forced Zero Overrides (policy)": per_node[
                        "forced_zero_amount"
                    ].sum(),
                    "B616 Policy Override": per_node[
                        "b616_override_amount"
                    ].sum(),
                }
                for k in ss_attrib:
                    ss_attrib[k] = float(ss_attrib[k])
                ss_sum = sum(ss_attrib.values())
                residual = float(total_ss) - ss_sum
                if abs(residual) > 1e-6:
                    ss_attrib["Caps — Reductions (policy lowering SS)"] += (
                        residual
                    )
                    ss_sum = sum(ss_attrib.values())

                ss_drv_df = pd.DataFrame(
                    {
                        "driver": list(ss_attrib.keys()),
                        "amount": [float(v) for v in ss_attrib.values()],
                    }
                )
                ss_drv_df_display = ss_drv_df[ss_drv_df["amount"] != 0].copy()
                denom = total_ss if total_ss > 0 else ss_drv_df[
                    "amount"
                ].sum()
                denom = denom if denom > 0 else 1.0
                ss_drv_df_display["pct_of_total_ss"] = (
                    ss_drv_df_display["amount"] / denom * 100
                )

                labels = ss_drv_df_display["driver"].tolist() + ["Total SS"]
                values = ss_drv_df_display["amount"].tolist() + [total_ss]
                measures = ["relative"] * len(ss_drv_df_display) + ["total"]
                pastel_colors = px.colors.qualitative.Pastel
                pastel_inc = (
                    pastel_colors[0] if len(pastel_colors) > 0 else "#A3C1DA"
                )
                pastel_dec = (
                    pastel_colors[1] if len(pastel_colors) > 1 else "#F6C3A0"
                )
                pastel_tot = (
                    pastel_colors[2] if len(pastel_colors) > 2 else "#CFCFCF"
                )
                fig_drv = go.Figure(
                    go.Waterfall(
                        name="SS Attribution",
                        orientation="v",
                        measure=measures,
                        x=labels,
                        y=values,
                        text=[
                            f"{v:,.0f}"
                            for v in ss_drv_df_display["amount"].tolist()
                        ]
                        + [f"{total_ss:,.0f}"],
                        connector={"line": {"color": "rgba(63,63,63,0.25)"}},
                        decreasing=dict(
                            marker=dict(color=pastel_dec)
                        ),
                        increasing=dict(
                            marker=dict(color=pastel_inc)
                        ),
                        totals=dict(marker=dict(color=pastel_tot)),
                    )
                )
                fig_drv.update_layout(
                    title=f"{selected_product} — SS Attribution Waterfall (adds to {euro_format(total_ss, True)})",
                    xaxis_title="Driver",
                    yaxis_title="Units",
                    height=420,
                )
                st.plotly_chart(fig_drv, use_container_width=True)

                ss_attrib_df_formatted = df_format_for_display(
                    ss_drv_df_display.rename(
                        columns={
                            "driver": "Driver",
                            "amount": "Units",
                            "pct_of_total_ss": "Pct_of_total_SS",
                        }
                    ).round(2),
                    cols=["Units", "Pct_of_total_SS"],
                    two_decimals_cols=["Pct_of_total_SS"],
                )
                st.dataframe(
                    ss_attrib_df_formatted, use_container_width=True
                )


                st.markdown(summary_html, unsafe_allow_html=True)

else:
    st.info(
        "Please upload sales.csv, demand.csv and leadtime.csv in the sidebar to run the optimizer."
    )
