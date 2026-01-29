# Multi-Echelon Inventory Optimizer — Raw Materials
# Developed by mat635418 — JAN 2026
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

st.set_page_config(page_title="MEIO for RM", layout="wide", page_icon="📦")

LOGO_FILENAME = "GY_logo.jpg"
LOGO_BASE_WIDTH = 160
days_per_month = 30

# --- New: show logo.svg above the main title ---
st.image("logo.jpg", width=300)

st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        gap:10px;
        margin:4px 0 0 0;
        font-size:0.9rem;
    ">
      <span style="
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding:4px 10px;
          border-radius:999px;
          background:#e3f2fd;
          color:#0b3d91;
          font-weight:600;
      ">
        <span style="
            display:inline-flex;
            align-items:center;
            justify-content:center;
            width:18px;
            height:18px;
            border-radius:50%;
            background:#0b3d91;
            color:#ffffff;
            font-size:0.75rem;
        ">
          V
        </span>
        v1.15
      </span>
      <span style="
          display:inline-flex;
          align-items:center;
          gap:6px;
          padding:4px 10px;
          border-radius:999px;
          background:#f3e5f5;
          color:#6a1b9a;
          font-weight:600;
      ">
        <span style="
            display:inline-flex;
            align-items:center;
            justify-content:center;
            width:18px;
            height:18px;
            border-radius:50%;
            background:#6a1b9a;
            color:#ffffff;
            font-size:0.75rem;
        ">
          ⏱
        </span>
        Released Feb&nbsp;2026
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

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
      details summary {
        cursor: pointer;
      }
      details summary::-webkit-details-marker {
        display: none;
      }
      details summary:before {
        content: "▶ ";
        font-size: 0.8rem;
      }
      details[open] summary:before {
        content: "▼ ";
      }
      .all-mat-details {
        margin: 0;
        padding: 6px 10px;
        background: #f8faff;
        border-radius: 6px;
        border: 1px solid #e0e7ff;
        font-size: 0.85rem;
      }
      .all-mat-details summary {
        font-weight: 600;
        color: #0b3d91;
      }
      .kpi-2x2-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.8rem;
      }
      .kpi-2x2-table td {
        padding: 3px 6px;
        border: 1px solid #e0e0e0;
      }
      .kpi-label {
        color: #555555;
        font-weight: 500;
        white-space: nowrap;
        text-align: left;
      }
      .kpi-value {
        color: #111111;
        font-weight: 700;
        text-align: left;
        white-space: nowrap;
      }
      .violin-box {
        border: 1px solid #dddddd;
        border-radius: 8px;
        padding: 8px 10px 4px 10px;
        margin-bottom: 10px;
        background: #f7f7f7;
      }
      .violin-box-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #333333;
      }
      /* Compact run configuration + snapshot header */
      .run-snapshot-container {
        display: grid;
        grid-template-columns: minmax(220px, 1.2fr) repeat(3, minmax(180px, 1fr));
        gap: 10px;
        margin: 10px 0 6px 0;
      }
      .run-card,
      .snap-card {
        border-radius: 8px;
        padding: 8px 10px;
        background: #ffffff;
        border: 1px solid #e0e0e0;
      }
      .run-card {
        background: linear-gradient(135deg,#e3f2fd,#e8f5e9);
      }
      .run-card-title,
      .snap-card-label {
        font-size: 0.78rem;
        color: #0b3d91;
        font-weight: 700;
        margin-bottom: 2px;
      }
      .run-card-body {
        font-size: 0.78rem;
        color: #154360;
        line-height: 1.25;
      }
      .snap-card-value {
        font-size: 1.0rem;
        font-weight: 800;
        color: #111111;
      }
      .snap-card-value-green {
        color: #00695c;
      }
      .snap-card-value-orange {
        color: #ef6c00;
      }
      .snap-card-sub {
        font-size: 0.75rem;
        color: #666666;
        margin-top: 2px;
      }
      .run-snapshot-header {
        padding: 6px 10px 4px 10px;
        margin: 4px 0 2px 0;
        border-radius: 8px;
        background: linear-gradient(90deg,#e3f2fd,#e8f5e9);
        font-size: 0.80rem;
        font-weight: 700;
        color: #0b3d91;
      }
      .exec-takeaway-selection {
        color: #b71c1c;
        font-weight: 700;
      }
      .exec-takeaway-percentage {
        color: #b71c1c;
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Small reusable helpers for UX / explanations / logging
# ------------------------------------------------------------------


def render_data_dictionary():
    """Show a small data dictionary so users know the expected CSV schema."""
    with st.sidebar.expander("📚 Data Dictionary / Template Schema", expanded=False):
        st.markdown(
            """
            **1. `sales.csv` — Historical Sales / Consumption**

            Required columns:
            - `Product`: material code (string)
            - `Location`: node / plant / warehouse (string)
            - `Period`: month in date format (e.g. `2024-01-01`, `2024-01-31`)
            - `Consumption`: actual consumption in the month (numeric)
            - `Forecast`: historical forecast for that month (numeric)

            **2. `demand.csv` — Future Local Demand (Planned)**

            Required columns:
            - `Product`: material code
            - `Location`: node / plant / warehouse
            - `Period`: month in date format
            - `Forecast`: local future demand (numeric)

            **3. `leadtime.csv` — Network Routes & Lead Times**

            Required columns:
            - `Product`: material code (or generic if same for all)
            - `From_Location`: upstream node
            - `To_Location`: downstream node
            - `Lead_Time_Days`: average lead time (days)
            - `Lead_Time_Std_Dev`: standard deviation of lead time (days)

            _Tip: use the sample templates from the project as a starting point._
            """,
            unsafe_allow_html=True,
        )


def render_run_and_snapshot_header(
    run_id: str,
    now_str: str,
    service_level: float,
    zero_if_no_net_fcst: bool,
    apply_cap: bool,
    cap_range,
    snapshot_label: str,
    tot_local_demand: float,
    tot_ss: float,
    ss_ratio_pct: float,
    coverage_months: float,
    n_active_materials: int,
    n_active_nodes: int,
):
    """Single banner that matches the pic2 layout: snapshot row on top, run config at bottom-left."""
    st.markdown(
        f"""
        <div class="run-snapshot-header">
          Network snapshot – {snapshot_label}
        </div>
        <div class="run-snapshot-container">
          <div class="snap-card">
            <div class="snap-card-label">Total Local Demand (month)</div>
            <div class="snap-card-value">{euro_format(tot_local_demand, True)}</div>
          </div>
          <div class="snap-card">
            <div class="snap-card-label">Safety Stock (sum)</div>
            <div class="snap-card-value snap-card-value-green">{euro_format(tot_ss, True)}</div>
          </div>
          <div class="snap-card">
            <div class="snap-card-label">SS / Demand Coverage</div>
            <div class="snap-card-value snap-card-value-orange">
              {ss_ratio_pct:.1f}%&nbsp;({coverage_months:.2f} months)
            </div>
          </div>
          <div class="snap-card">
            <div class="snap-card-label">Active Materials (with corridor)</div>
            <div class="snap-card-value">{n_active_materials}</div>
          </div>
        </div>
        <div class="run-snapshot-container" style="grid-template-columns: minmax(220px, 1.2fr) repeat(2, minmax(180px, 1fr)); margin-top:4px;">
          <div class="run-card">
            <div class="run-card-title">Run configuration</div>
            <div class="run-card-body">
              <div><strong>ID:</strong> {run_id}</div>
              <div><strong>Time:</strong> {now_str}</div>
              <div><strong>End-node SL:</strong> {service_level*100:.2f}%</div>
              <div><strong>Zero SS if no demand:</strong> {str(zero_if_no_net_fcst)}</div>
              <div><strong>SS capping:</strong> {str(apply_cap)} ({cap_range[0]}–{cap_range[1]} % of network demand)</div>
            </div>
          </div>
          <div class="snap-card">
            <div class="snap-card-label">Active Nodes (with corridor)</div>
            <div class="snap-card-value">{n_active_nodes}</div>
          </div>
          <div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab1_explainer():
    """Short guide for reading the Inventory Corridor tab."""
    with st.expander("ℹ️ How to read this view", expanded=False):
        st.markdown(
            """
            - **Max Corridor (SS + Forecast)**: the upper bound of the inventory corridor for the node.
            - **Safety Stock**: the buffer inventory needed to protect against uncertainty.
            - **Local Direct Demand (Internal)**: the node's own demand (forecast).
            - **External Network Demand (Downstream)**: rolled‑up demand of downstream nodes that this node serves.

            Reading tip: for a healthy node, **Max Corridor** should move broadly in line with
            total demand; if SS grows much faster than demand, the node becomes more inventory‑intensive.
            """,
            unsafe_allow_html=True,
        )


def render_tab7_explainer():
    """Short guide for the 'By Material' / 8 reasons tab."""
    with st.expander("ℹ️ How to interpret the 8 Inventory Reasons", expanded=False):
        st.markdown(
            """
            **Core drivers (statistical):**
            - **Demand Uncertainty**: SS needed because forecast varies month to month.
            - **Lead‑time Uncertainty**: SS needed because supply lead times are volatile.
            - **Direct Local Forecast**: share of SS linked to protecting a node's own demand.
            - **Indirect Network Demand**: share of SS linked to protecting downstream locations.

            **Policy / rule‑based effects:**
            - **Caps — Reductions**: rules that reduce SS vs the raw statistical suggestion.
            - **Caps — Increases**: rules that increase SS (e.g. minimum coverage).
            - **Forced Zero Overrides**: decisions to set SS to zero even if statistics would suggest > 0.
            - **B616 Policy Override**: specific override for location B616 in this model.

            Together, these eight components **sum exactly** to the implemented Safety Stock
            for the material and period, so you can explain “why we carry this SS” in business terms.
            """,
            unsafe_allow_html=True,
        )


def render_ss_formula_explainer():
    """Explain the core SS formula in the scenario tab."""
    with st.expander("📐 Safety Stock formula used in scenarios", expanded=False):
        st.markdown(
            r"""
            The scenario engine uses the classical formula:

            $$
            SS = z \cdot \sqrt{
                \underbrace{\mathrm{Var}(D)\cdot LT_{\text{mean}}}_{\text{demand uncertainty}}
                \;+\;
                \underbrace{(LT_{\text{std}})^2\cdot D^2}_{\text{lead-time uncertainty}}
            }
            $$

            where:

            - $z$ is the z-score for the chosen Service Level (e.g. $99\% \Rightarrow 2.33$),
            - $D$ is the average daily demand,
            - $\mathrm{Var}(D)$ is the variance of daily demand,
            - $LT_{\text{mean}}$ is the average lead time (days),
            - $LT_{\text{std}}$ is the lead-time standard deviation (days).

            Additional rules applied:

            - For very low monthly demand, we enforce $\mathrm{Var}(D) \ge D$ (Poisson-like floor).
            - A small floor of $1\%$ of the mean demand over lead time is also enforced,  
              so tiny variances do not result in zero $SS$.
            - Scenario $SS$ is for **analysis only** and does not overwrite the implemented policy.
            """,
            unsafe_allow_html=True,
        )


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


def df_format_for_display(
    df: pd.DataFrame,
    cols=None,
    two_decimals_cols=None,
) -> pd.DataFrame:
    """
    Format numeric columns in a DataFrame for display:
    - Integers with euro_format (thousands separated by dots).
    - Selected columns in `two_decimals_cols` forced to 2 decimals.
    """
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


# ------------------------------------------------------------------
# ACTIVE DEFINITIONS (single source of truth)
# ------------------------------------------------------------------
def get_active_mask(results: pd.DataFrame) -> pd.Series:
    """
    Row is ACTIVE if there is any meaningful demand or final corridor
    implemented on that row.

    We consider as signals:
      - Agg_Future_Demand  (network demand seen by node)
      - Forecast           (local demand)
      - Safety_Stock       (final implemented SS)
      - Max_Corridor       (SS + Forecast, final corridor)

    Any of these being non-zero marks the row as active.
    """
    if results is None or results.empty:
        return pd.Series(False, index=results.index if results is not None else [])

    cols = ["Agg_Future_Demand", "Forecast", "Safety_Stock", "Max_Corridor"]
    existing = [c for c in cols if c in results.columns]
    if not existing:
        return pd.Series(False, index=results.index)

    return (
        results[existing]
        .fillna(0)
        .abs()
        .sum(axis=1)
        > 0
    )


def get_active_snapshot(results: pd.DataFrame, period) -> pd.DataFrame:
    """Return only ACTIVE rows for a given period."""
    if results is None or results.empty:
        return results.iloc[0:0]
    snap = results[results["Period"] == period].copy()
    if snap.empty:
        return snap
    mask = get_active_mask(snap)
    return snap[mask].copy()


def active_materials(results: pd.DataFrame, period=None):
    """
    Active materials = products with at least one ACTIVE row.

    If period is provided, we restrict to that period. Otherwise, any period
    counts.
    """
    df = results
    if period is not None:
        df = df[df["Period"] == period]
    if df is None or df.empty:
        return []
    mask = get_active_mask(df)
    return sorted(df[mask]["Product"].dropna().unique().tolist())


def active_nodes(results: pd.DataFrame, period=None, product=None):
    """
    Active nodes = locations with at least one ACTIVE row.

    Optional filters:
      - period: only rows from that Period
      - product: only rows for that Product
    """
    df = results
    if period is not None:
        df = df[df["Period"] == period]
    if product is not None:
        df = df[df["Product"] == product]
    if df is None or df.empty:
        return []
    mask = get_active_mask(df)
    return sorted(df[mask]["Location"].dropna().unique().tolist())


def hide_zero_rows(df: pd.DataFrame, check_cols=None) -> pd.DataFrame:
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


def render_logo_above_parameters(scale: float = 1.5) -> None:
    if LOGO_FILENAME and os.path.exists(LOGO_FILENAME):
        try:
            width = int(LOGO_BASE_WIDTH * float(scale))
            st.image(LOGO_FILENAME, width=width)
        except Exception:
            pass


def render_selection_line(label, product=None, location=None, period_text=None) -> None:
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


def aggregate_network_stats(df_forecast, df_stats, df_lt, transitive: bool = True, rho: float = 1.0):
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


with st.sidebar.expander("⚙️ Service Level Configuration", expanded=True):
    service_level = st.slider(
        "Service Level (%) for the end-nodes",
        50.0,
        99.9,
        99.0,
        help="Target probability of not stocking out for end-nodes (hop 0). Upstream nodes get fixed SLs by hop-distance.",
    ) / 100
    z = norm.ppf(service_level)

with st.sidebar.expander("⚙️ Safety Stock Rules", expanded=True):
    zero_if_no_net_fcst = st.checkbox(
        "Force zero SS if no Demand",
        value=True,
        help="When enabled, nodes with zero aggregated network demand will have Safety Stock forced to zero.",
    )
    apply_cap = st.checkbox(
        "Enable SS Capping",
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
render_data_dictionary()
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

DEFAULT_PRODUCT_CHOICE = "NOKANDO2"
DEFAULT_LOCATION_CHOICE = "DEW1"
CURRENT_MONTH_TS = pd.Timestamp.now().to_period("M").to_timestamp()

if s_file and d_file and lt_file:
    try:
        df_s = pd.read_csv(s_file)
        df_d = pd.read_csv(d_file)
        df_lt = pd.read_csv(lt_file)
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
        st.stop()

    # (Data preview & sanity checks block removed to declutter the UI)

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

    # Period parsing with explicit diagnostics
    df_s["Period"] = pd.to_datetime(df_s["Period"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df_d["Period"] = pd.to_datetime(df_d["Period"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    bad_s = df_s["Period"].isna().sum()
    bad_d = df_d["Period"].isna().sum()

    if bad_s > 0:
        st.warning(
            f"sales.csv: {bad_s} row(s) have invalid Period values and will be dropped. "
            "Please check the date format in those rows."
        )
        df_s = df_s.dropna(subset=["Period"])
    if bad_d > 0:
        st.warning(
            f"demand.csv: {bad_d} row(s) have invalid Period values and will be dropped. "
            "Please check the date format in those rows."
        )
        df_d = df_d.dropna(subset=["Period"])

    if df_s.empty:
        st.error("After parsing Period, sales.csv has no valid rows. Please fix the file and reload.")
        st.stop()
    if df_d.empty:
        st.error("After parsing Period, demand.csv has no valid rows. Please fix the file and reload.")
        st.stop()

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

    meaningful_mask = get_active_mask(results)
    meaningful_results = results[meaningful_mask].copy()

    all_products = active_materials(results) or sorted(results["Product"].unique().tolist())
    default_product = (
        DEFAULT_PRODUCT_CHOICE
        if DEFAULT_PRODUCT_CHOICE in all_products
        else (all_products[0] if all_products else "")
    )

    def default_location_for(prod):
        locs = active_nodes(results, product=prod) or sorted(
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

    # --- New combined banner matching pic2 layout (ACTIVE only) ---
    run_id = datetime.now().strftime("RUN-%Y%m%d-%H%M%S")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if default_period is not None:
        global_period = default_period
        active_snapshot = get_active_snapshot(results, global_period)

        tot_local_demand = active_snapshot["Forecast"].sum() if "Forecast" in active_snapshot.columns else 0.0
        tot_ss = active_snapshot["Safety_Stock"].sum() if "Safety_Stock" in active_snapshot.columns else 0.0
        coverage_months = (tot_ss / tot_local_demand) if tot_local_demand > 0 else 0.0
        ss_ratio_pct = coverage_months * 100.0
        n_active_materials = active_snapshot["Product"].nunique() if "Product" in active_snapshot.columns else 0
        n_active_nodes = active_snapshot["Location"].nunique() if "Location" in active_snapshot.columns else 0

        render_run_and_snapshot_header(
            run_id=run_id,
            now_str=now_str,
            service_level=service_level,
            zero_if_no_net_fcst=zero_if_no_net_fcst,
            apply_cap=apply_cap,
            cap_range=cap_range,
            snapshot_label=period_label(global_period),
            tot_local_demand=tot_local_demand,
            tot_ss=tot_ss,
            ss_ratio_pct=ss_ratio_pct,
            coverage_months=coverage_months,
            n_active_materials=n_active_materials,
            n_active_nodes=n_active_nodes,
        )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📈 Inventory Corridor",
            "🕸️ Network Topology",
            "📋 Full Plan",
            "⚖️ Efficiency Analysis",
            "📉 Forecast Accuracy",
            "🧮 Calculation Trace & Sim",
            "📦 By Material",
            "📊 All Materials View",
        ]
    )

    # TAB 1 -----------------------------------------------------------------
    with tab1:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

            sku_default = default_product
            sku_index = all_products.index(sku_default) if sku_default in all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="tab1_sku")

            loc_opts = active_nodes(results, period=CURRENT_MONTH_TS, product=sku)
            if not loc_opts:
                loc_opts = active_nodes(results, product=sku)
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
                    avg_daily_val = f"{avg_daily:.2f}" if pd.notna(avg_daily) else ""
                    days_cov_val = f"{days_cov:.1f}" if pd.notna(days_cov) else ""

                    kpi_html = f"""
                    <table class="kpi-2x2-table">
                      <tr>
                        <td class="kpi-label">Avg Daily Demand<br/>[units/day]</td>
                        <td class="kpi-label">Safety Stock Coverage<br/>[days]</td>
                      </tr>
                      <tr>
                        <td class="kpi-value">{avg_daily_val}</td>
                        <td class="kpi-value">{days_cov_val}</td>
                      </tr>
                    </table>
                    """
                    st.markdown(kpi_html, unsafe_allow_html=True)
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

            def int_dot(v):
                try:
                    return "{:,.0f}".format(float(v)).replace(",", ".")
                except Exception:
                    return str(v)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Max_Corridor"],
                    name="Max Corridor (SS + Forecast)",
                    mode="lines",
                    line=dict(width=1.5, color="#9e9e9e", dash="dot"),
                    hovertemplate=(
                        "Period: %{x|%b %Y}<br>"
                        "Max Corridor: %{customdata} units<extra></extra>"
                    ),
                    customdata=plot_full["Max_Corridor"].apply(int_dot),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Safety_Stock"],
                    name="Safety Stock",
                    mode="lines",
                    line=dict(width=0.5, color="#42a5f5"),
                    fill="tozeroy",
                    fillcolor="rgba(66,165,245,0.25)",
                    hovertemplate=(
                        "Period: %{x|%b %Y}<br>"
                        "Safety Stock: %{customdata} units<extra></extra>"
                    ),
                    customdata=plot_full["Safety_Stock"].apply(int_dot),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Forecast"],
                    name="Local Direct Demand (Internal)",
                    mode="lines+markers",
                    line=dict(color="#212121", width=2),
                    marker=dict(size=5),
                    hovertemplate=(
                        "Period: %{x|%b %Y}<br>"
                        "Local Forecast: %{customdata} units<extra></extra>"
                    ),
                    customdata=plot_full["Forecast"].apply(int_dot),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_full["Period"],
                    y=plot_full["Agg_Future_External"],
                    name="External Network Demand (Downstream)",
                    mode="lines+markers",
                    line=dict(color="#00897b", width=2, dash="dash"),
                    marker=dict(size=5),
                    hovertemplate=(
                        "Period: %{x|%b %Y}<br>"
                        "External Demand: %{customdata} units<extra></extra>"
                    ),
                    customdata=plot_full["Agg_Future_External"].apply(int_dot),
                )
            )

            if CURRENT_MONTH_TS in plot_full["Period"].values:
                cm = CURRENT_MONTH_TS
                try:
                    fig.add_vrect(
                        x0=cm - pd.Timedelta(days=15),
                        x1=cm + pd.Timedelta(days=15),
                        fillcolor="rgba(255,235,59,0.18)",
                        line_width=0,
                        layer="below",
                    )
                except Exception:
                    pass

            fig.update_layout(
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e0e0e0",
                    borderwidth=1,
                    font=dict(size=10),
                ),
                margin=dict(l=20, r=10, t=10, b=10),
                xaxis_title="Period",
                yaxis_title="Units",
                xaxis=dict(
                    tickformat="%b\n%Y",
                    dtick="M1",
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.05)",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.05)",
                    zeroline=False,
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # separator + explainer directly after the graph
            st.markdown(
                "<hr style='margin:6px 0 10px 0; border: none; border-top: 1px solid #cccccc;'/>",
                unsafe_allow_html=True,
            )
            render_tab1_explainer()


    # TAB 2 -----------------------------------------------------------------
    with tab2:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

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
                <div style="font-size:0.85rem; margin-bottom:4px; color:#555;">
                  Only nodes with non‑zero demand or corridor in the selected period are drawn.
                  Routes that connect to fully inactive nodes are hidden to keep the view focused.
                </div>
                """,
                unsafe_allow_html=True,
            )

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
                height="1200px",
                width="100%",
                directed=True,
                bgcolor="#ffffff",
                font_color="#222222",
            )

            hubs = {"B616", "BEEX", "LUEX"}

            active_nodes_for_sku = set(
                active_nodes(results, period=chosen_period, product=sku)
            )

            if not sku_lt.empty:
                froms = set(sku_lt["From_Location"].dropna().unique().tolist())
                tos = set(sku_lt["To_Location"].dropna().unique().tolist())
                route_nodes = (froms.union(tos)).intersection(active_nodes_for_sku)
                all_nodes = route_nodes.union(hubs.intersection(active_nodes_for_sku))
            else:
                all_nodes = hubs.intersection(active_nodes_for_sku)

            if not all_nodes:
                all_nodes = hubs

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
                        "Max_Corridor": 0,
                        "Agg_Future_Demand": 0,
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
                        "Max_Corridor": 0,
                        "Agg_Future_Demand": 0,
                    },
                )

                node_active = (
                    abs(float(m.get("Agg_Future_Demand", 0)))
                    + abs(float(m.get("Forecast", 0)))
                    + abs(float(m.get("Safety_Stock", 0)))
                    + abs(float(m.get("Max_Corridor", float(m.get("Safety_Stock", 0)) + float(m.get("Forecast", 0)))))
                    > 0
                )

                if not node_active:
                    continue

                tier_hops = m.get("Tier_Hops", np.nan)
                try:
                    hops_val = int(tier_hops) if not pd.isna(tier_hops) else 0
                except Exception:
                    hops_val = 0

                if n == "B616":
                    bg, border, font_color, size = "#dcedc8", "#8bc34a", "#0b3d91", 14
                elif n in {"BEEX", "LUEX"}:
                    bg, border, font_color, size = "#bbdefb", "#64b5f6", "#0b3d91", 14
                else:
                    if hops_val <= 0:
                        bg = "#fffde7"
                        border = "#fbc02d"
                    elif hops_val == 1:
                        bg = "#fff9c4"
                        border = "#f9a825"
                    elif hops_val == 2:
                        bg = "#fff59d"
                        border = "#f57f17"
                    else:
                        bg = "#fff176"
                        border = "#f57f17"
                    font_color, size = "#222222", 12

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
                    f"FC: {euro_format(m.get('Forecast', 0), show_zero=True)}\n"
                    f"DFC: {euro_format(m.get('Agg_Future_External', 0), show_zero=True)}\n"
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

            visible_nodes = {n["id"] for n in net.nodes}
            if not sku_lt.empty:
                for _, r in sku_lt.iterrows():
                    from_n, to_n = r["From_Location"], r["To_Location"]
                    if pd.isna(from_n) or pd.isna(to_n):
                        continue
                    if (from_n not in visible_nodes) or (to_n not in visible_nodes):
                        continue
                    edge_color = "#888888"
                    lt_val = r.get("Lead_Time_Days", 0)
                    label = f"{int(lt_val)}d" if not pd.isna(lt_val) else ""
                    net.add_edge(
                        from_n,
                        to_n,
                        label=label,
                        color=edge_color,
                        smooth={"enabled": True, "type": "dynamic", "roundness": 0.4},
                    )

            net.set_options(
                """
                {
                  "physics": {
                    "enabled": true,
                    "stabilization": {
                      "enabled": true,
                      "iterations": 300,
                      "fit": true
                    },
                    "barnesHut": {
                      "gravitationalConstant": -3000,
                      "centralGravity": 0.1,
                      "springLength": 50,
                      "springConstant": 0.02,
                      "damping": 0.09,
                      "avoidOverlap": 1.0
                    }
                  },
                  "nodes": {
                    "borderWidthSelected": 2
                  },
                  "edges": {
                    "smooth": {
                      "enabled": true,
                      "type": "dynamic",
                      "roundness": 0.4
                    }
                  },
                  "interaction": {
                    "hover": true,
                    "zoomView": true,
                    "dragView": true,
                    "dragNodes": true
                  },
                  "layout": {
                    "improvedLayout": true
                  }
                }
                """
            )
            tmpfile = "net.html"
            net.save_graph(tmpfile)
            html_text = open(tmpfile, "r", encoding="utf-8").read()
            injection_css = """
            <style>
              html, body { height: 100%; margin: 0; padding: 0; }
              #mynetwork {
                display:flex !important;
                align-items:center;
                justify-content:center;
                height:1200px !important;
                width:100% !important;
              }
              .vis-network {
                display:block !important;
                margin: 0 auto !important;
              }
            </style>
            """
            injection_js = """
            <script>
              function fitAndCenterNetwork() {
                try {
                  if (typeof network !== 'undefined') {
                    network.fit({ animation: false });
                  }
                } catch (e) {
                  console.warn("Network fit failed:", e);
                }
              }
              setTimeout(fitAndCenterNetwork, 700);
            </script>
            """
            if "</head>" in html_text:
                html_text = html_text.replace("</head>", injection_css + "</head>", 1)
            if "</body>" in html_text:
                html_text = html_text.replace("</body>", injection_js + "</body>", 1)
            else:
                html_text += injection_js
            components.html(html_text, height=1250)

            st.markdown(
                """
                <div style="text-align:center; font-size:12px; padding:8px 0;">
                  <div style="display:inline-block; background:#f7f9fc; padding:8px 12px; border-radius:8px;">
                    <strong>Legend:</strong><br/>
                    FC = Local Forecast &nbsp;&nbsp;|&nbsp;&nbsp;
                    DFC = Downstream FC (rolled-up) &nbsp;&nbsp;|&nbsp;&nbsp;
                    SS  = Safety Stock (final policy value)<br/>
                    Node border intensity loosely reflects hop distance from end-nodes.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # TAB 3 -----------------------------------------------------------------
    with tab3:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)
            st.markdown("<div style='padding:6px 0;'></div>", unsafe_allow_html=True)

            prod_choices = active_materials(results) or sorted(results["Product"].unique())
            loc_choices = active_nodes(results) or sorted(results["Location"].unique())
            period_choices_labels = period_labels

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

            f_prod = st.multiselect(
                "MATERIAL",
                prod_choices,
                default=default_prod_list,
                key="full_f_prod",
            )
            f_loc = st.multiselect("LOCATION", loc_choices, default=[], key="full_f_loc")
            f_period_labels = st.multiselect(
                "PERIOD",
                period_choices_labels,
                default=default_period_list,
                key="full_f_period",
            )
            f_period = [period_label_map[lbl] for lbl in f_period_labels] if f_period_labels else []

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=results.to_csv(index=False),
                    file_name="filtered_plan.csv",
                    mime="text/csv",
                    key="full_plan_export",
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

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
            filtered = filtered[get_active_mask(filtered)]
            filtered = filtered.sort_values("Safety_Stock", ascending=False)

            filtered_display = hide_zero_rows(
                filtered,
                check_cols=[
                    "Safety_Stock",
                    "Forecast",
                    "Agg_Future_Demand",
                    "Agg_Future_Internal",
                    "Agg_Future_External",
                ],
            )

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

    # TAB 4 -----------------------------------------------------------------
    with tab4:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

            sku_default = default_product
            sku_index = all_products.index(sku_default) if all_products else 0
            sku = st.selectbox("MATERIAL", all_products, index=sku_index, key="eff_sku")

            if period_labels:
                try:
                    default_label = period_label(default_period) if default_period is not None else period_labels[-1]
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

            snapshot_period = eff_period if eff_period in all_periods else (all_periods[-1] if all_periods else None)
            if snapshot_period is None:
                eff_export = results[results["Product"] == sku].copy()
            else:
                eff_export = get_active_snapshot(results, snapshot_period)
                eff_export = eff_export[eff_export["Product"] == sku]

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
            st.download_button(
                "💾 Export CSV",
                data=eff_export.to_csv(index=False),
                file_name=f"efficiency_{sku}_{period_label(snapshot_period) if snapshot_period is not None else 'all'}.csv",
                mime="text/csv",
                key="eff_export_btn",
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            render_selection_line("Selected:", product=sku, period_text=period_label(eff_period))
            st.subheader("⚖️ Efficiency & Policy Analysis — Summary Metrics")

            snapshot_period = eff_period if eff_period in all_periods else (all_periods[-1] if all_periods else None)
            if snapshot_period is None:
                st.warning("No period data available for Efficiency Analysis.")
                eff = results[(results["Product"] == sku)].copy()
            else:
                eff = get_active_snapshot(results, snapshot_period)
                eff = eff[eff["Product"] == sku].copy()

            eff["SS_to_Demand_Ratio"] = (
                eff["Safety_Stock"] / eff["Forecast"].replace(0, np.nan)
            ).fillna(0)

            eff_display = hide_zero_rows(eff)

            total_ss_sku = eff["Safety_Stock"].sum()
            total_forecast_sku = eff["Forecast"].sum()
            sku_ratio = total_ss_sku / total_forecast_sku if total_forecast_sku > 0 else 0

            all_res = get_active_snapshot(results, snapshot_period) if snapshot_period is not None else results
            global_total_ss = all_res["Safety_Stock"].sum()
            global_total_fc = all_res["Forecast"].sum()
            global_ratio = global_total_ss / global_total_fc if global_total_fc > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Months of FC held by SS (selection)", f"{sku_ratio:.2f}")
            m2.metric("Months of FC held by SS (all materials)", f"{global_ratio:.2f}")
            m3.metric("Total SS for Material", euro_format(int(total_ss_sku), True))
            m4.metric("Total Forecast", euro_format(int(total_forecast_sku), True))
            st.markdown("---")

            c1, c2 = st.columns([7, 3])
            with c1:
                st.markdown("**Top Nodes by Safety Stock (snapshot)**")
                if not eff_display.empty:
                    eff_top = eff_display.sort_values("Safety_Stock", ascending=False)
                    eff_top_display = (
                        eff_top[
                            [
                                "Location",
                                "Adjustment_Status",
                                "Safety_Stock",
                                "SS_to_Demand_Ratio",
                            ]
                        ]
                        .head(10)
                        .reset_index(drop=True)
                    )
                    eff_top_display["Safety_Stock"] = eff_top_display["Safety_Stock"].round(0)
                    eff_top_fmt = df_format_for_display(
                        eff_top_display,
                        cols=["Safety_Stock", "SS_to_Demand_Ratio"],
                        two_decimals_cols=["SS_to_Demand_Ratio"],
                    )
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
                        height=420,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.write("No non-zero nodes for this selection.")

            with c2:
                st.markdown("**Status Breakdown**")
                if not eff_display.empty:
                    st.table(eff_display["Adjustment_Status"].value_counts())
                else:
                    st.write("No non-zero nodes for this selection.")

    # TAB 5 -----------------------------------------------------------------
    with tab5:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

            h_sku_default = default_product
            h_sku_index = all_products.index(h_sku_default) if all_products else 0
            h_sku = st.selectbox("MATERIAL", all_products, index=h_sku_index, key="h1")

            h_loc_opts = active_nodes(results, product=h_sku)
            if not h_loc_opts:
                h_loc_opts = sorted(
                    results[results["Product"] == h_sku]["Location"].unique().tolist()
                )
            if not h_loc_opts:
                h_loc_opts = ["(no location)"]
            h_loc_default = (
                DEFAULT_LOCATION_CHOICE
                if DEFAULT_LOCATION_CHOICE in h_loc_opts
                else (h_loc_opts[0] if h_loc_opts else "(no location)")
            )
            h_loc_index = h_loc_opts.index(h_loc_default) if h_loc_default in h_loc_opts else 0
            h_loc = st.selectbox("LOCATION", h_loc_opts, index=h_loc_index, key="h2")

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
                hdf = hdf[(hdf["Product"] == h_sku) & (hdf["Location"] == h_loc)].sort_values("Period")
            else:
                hdf = hdf[hdf["Product"] == h_sku].sort_values("Period")

            if not hdf.empty:
                k1, k2, k3 = st.columns(3)
                denom_consumption = hdf["Consumption"].replace(0, np.nan).sum()
                if denom_consumption > 0:
                    wape_val = hdf["Abs_Error"].sum() / denom_consumption * 100
                    bias_val = hdf["Deviation"].sum() / denom_consumption * 100
                    k1.metric("WAPE (%)", f"{wape_val:.1f}")
                    k2.metric("Bias (%)", f"{bias_val:.1f}")
                else:
                    k1.metric("WAPE (%)", "N/A")
                    k2.metric("Bias (%)", "N/A")
                avg_acc = hdf["Accuracy_%"].mean() if not hdf["Accuracy_%"].isna().all() else np.nan
                k3.metric("Avg Accuracy (%)", f"{avg_acc:.1f}" if not np.isnan(avg_acc) else "N/A")

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
                fig_hist.update_layout(xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("---")

                st.subheader("Aggregated Network History (Selected Product) — formatted by month")
                net_table = (
                    hist_net[hist_net["Product"] == h_sku]
                    .merge(hdf[["Period"]].drop_duplicates(), on="Period", how="inner")
                    .sort_values("Period")
                    .drop(columns=["Product"])
                )
                if not net_table.empty:
                    net_table["Net_Abs_Error"] = (
                        net_table["Network_Consumption"] - net_table["Network_Forecast_Hist"]
                    ).abs()
                    denom_net = net_table["Network_Consumption"].replace(0, np.nan).sum()
                    net_wape = (
                        net_table["Net_Abs_Error"].sum() / denom_net * 100 if denom_net > 0 else np.nan
                    )
                else:
                    net_wape = np.nan

                c_net1, c_net2 = st.columns([3, 1])
                with c_net1:
                    if not net_table.empty:
                        net_table_fmt = net_table.copy()
                        net_table_fmt["Period"] = net_table_fmt["Period"].apply(period_label)
                        for col in ["Network_Consumption", "Network_Forecast_Hist"]:
                            net_table_fmt[col] = net_table_fmt[col].apply(
                                lambda v: euro_format(v, always_two_decimals=False, show_zero=True)
                            )
                        st.dataframe(
                            net_table_fmt[
                                ["Period", "Network_Consumption", "Network_Forecast_Hist"]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.write("No aggregated network history available for the chosen selection.")
                with c_net2:
                    c_val = f"{net_wape:.1f}" if not np.isnan(net_wape) else "N/A"
                    st.metric("Network WAPE (%)", c_val)

    # TAB 6 -----------------------------------------------------------------
    with tab6:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

            calc_sku_default = default_product
            calc_sku_index = all_products.index(calc_sku_default) if all_products else 0
            calc_sku = st.selectbox("MATERIAL", all_products, index=calc_sku_index, key="c_sku")

            avail_locs = active_nodes(results, product=calc_sku)
            if not avail_locs:
                avail_locs = sorted(
                    results[results["Product"] == calc_sku]["Location"].unique().tolist()
                )
            if not avail_locs:
                avail_locs = ["(no location)"]
            calc_loc_default = (
                DEFAULT_LOCATION_CHOICE
                if DEFAULT_LOCATION_CHOICE in avail_locs
                else (avail_locs[0] if avail_locs else "(no location)")
            )
            calc_loc_index = avail_locs.index(calc_loc_default) if calc_loc_default in avail_locs else 0
            calc_loc = st.selectbox("LOCATION", avail_locs, index=calc_loc_index, key="c_loc")

            if period_labels:
                try:
                    default_label = period_label(default_period) if default_period is not None else period_labels[-1]
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
            calc_period = period_label_map.get(chosen_label, default_period)

            row_export = get_active_snapshot(results, calc_period if calc_period is not None else default_period)
            row_export = row_export[
                (row_export["Product"] == calc_sku)
                & (row_export["Location"] == calc_loc)
            ]
            export_data = row_export if not row_export.empty else pd.DataFrame()

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=export_data.to_csv(index=False),
                    file_name=f"calc_trace_{calc_sku}_{calc_loc}_{period_label(calc_period)}.csv",
                    mime="text/csv",
                    key="calc_export_btn",
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

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
                "See how changing the **end-node** service level (SL) or lead-time assumptions affects safety stock. "
                "Hop 1–3 service levels are automatically recomputed to keep the same relative gaps as in the base policy."
            )
            render_ss_formula_explainer()

            z_current = norm.ppf(service_level)

            row_df = get_active_snapshot(results, calc_period if calc_period is not None else default_period)
            row_df = row_df[
                (row_df["Product"] == calc_sku)
                & (row_df["Location"] == calc_loc)
            ]
            if row_df.empty:
                st.warning("Selection not found in results.")
            else:
                row = row_df.iloc[0]

                base_hop_sl = {0: 99.0, 1: 95.0, 2: 90.0, 3: 85.0}
                base_end_sl_pct = base_hop_sl[0]
                hop_ratios = {h: base_hop_sl[h] / base_end_sl_pct for h in base_hop_sl}

                node_sl = float(row.get("Service_Level_Node", service_level))
                node_z = float(row.get("Z_node", norm.ppf(node_sl)))
                hops = int(row.get("Tier_Hops", 0))

                st.markdown(
                    "**Applied Hop → Service Level mapping (highlight shows which row was used for this node):**"
                )
                hop_image_path = "HOP_SLjpg.jpg"
                if os.path.exists(hop_image_path):
                    st.image(hop_image_path, width=500)
                else:
                    st.info(
                        "Network hop illustration not found on the server "
                        f"(expected at '{hop_image_path}'). "
                        "Please add this image file next to MEIO_V2.py."
                    )

                avg_daily = row.get("D_day", np.nan)
                days_cov = row.get("Days_Covered_by_SS", np.nan)
                avg_daily_txt = f"{avg_daily:.2f}" if pd.notna(avg_daily) else "N/A"
                days_cov_txt = f"{days_cov:.1f}" if pd.notna(days_cov) else "N/A"
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
                st.markdown("**Values used for the calculation (highlighted above):**")
                st.markdown(summary_html, unsafe_allow_html=True)

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
                      SCENARIO PLANNING TOOL — simulate alternative end-node SL / LT assumptions (analysis‑only).
                      Hop 1–3 SLs are automatically recalculated to keep the same relative gaps as in the policy.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Show detailed scenario controls", expanded=False):
                    st.markdown(
                        """
                        Use the sliders below to set an **end-node** Service Level (SL) for each scenario.
                        Hop 1–3 SLs are recomputed automatically based on the base-grid ratios.
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
                        with st.expander(f"Scenario {s+1} inputs", expanded=False):
                            sc_sl_default = (
                                float(service_level * 100)
                                if s == 0
                                else min(99.9, float(service_level * 100) + 0.5 * s)
                            )
                            sc_sl = st.slider(
                                f"Scenario {s+1} end-node SL (%)",
                                50.0,
                                99.9,
                                sc_sl_default,
                                help="End-node Service Level used for this scenario. Hop 1–3 SLs are recomputed automatically.",
                                key=f"sc_sl_{s}",
                            )
                            hop0 = sc_sl
                            hop1 = max(0.0, min(99.9, hop0 * hop_ratios[1]))
                            hop2 = max(0.0, min(99.9, hop0 * hop_ratios[2]))
                            hop3 = max(0.0, min(99.9, hop0 * hop_ratios[3]))

                            st.markdown(
                                f"""
                                <div style="font-size:0.85rem; margin-top:4px;">
                                  <strong>Derived hop SLs used in this scenario:</strong><br/>
                                  Hop 0 (end-node): <strong>{hop0:.2f}%</strong><br/>
                                  Hop 1: <strong>{hop1:.2f}%</strong> &nbsp;&nbsp; Hop 2: <strong>{hop2:.2f}%</strong> &nbsp;&nbsp; Hop 3: <strong>{hop3:.2f}%</strong>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            sc_lt_default = float(row["LT_Mean"])
                            sc_lt = st.slider(
                                f"Scenario {s+1} Avg Lead Time (Days)",
                                0.0,
                                max(30.0, float(row["LT_Mean"]) * 2),
                                value=sc_lt_default,
                                key=f"sc_lt_{s}",
                            )
                            sc_lt_std_default = float(row["LT_Std"])
                            sc_lt_std = st.slider(
                                f"Scenario {s+1} LT Std Dev (Days)",
                                0.0,
                                max(10.0, float(row["LT_Std"]) * 2),
                                value=sc_lt_std_default,
                                key=f"sc_lt_std_{s}",
                            )
                            scenarios.append(
                                {
                                    "SL_pct": sc_sl,
                                    "LT_mean": sc_lt,
                                    "LT_std": sc_lt_std,
                                    "Hop0": hop0,
                                    "Hop1": hop1,
                                    "Hop2": hop2,
                                    "Hop3": hop3,
                                }
                            )

                    scen_rows = []
                    for idx, sc in enumerate(scenarios):
                        sc_z = norm.ppf(sc["SL_pct"] / 100.0)
                        d_day = float(row["Agg_Future_Demand"]) / float(days_per_month)
                        sigma_d_day = float(row["Agg_Std_Hist"]) / math.sqrt(float(days_per_month))
                        var_d = sigma_d_day**2
                        if row["Agg_Future_Demand"] < 20.0:
                            var_d = max(var_d, d_day)
                        sc_ss = sc_z * math.sqrt(
                            var_d * sc["LT_mean"] + (sc["LT_std"] ** 2) * (d_day**2)
                        )
                        sc_floor = d_day * sc["LT_mean"] * 0.01
                        sc_ss = max(sc_ss, sc_floor)
                        scen_rows.append(
                            {
                                "Scenario": f"S{idx+1}",
                                "EndNode_SL_%": sc["SL_pct"],
                                "Hop1_SL_%": sc["Hop1"],
                                "Hop2_SL_%": sc["Hop2"],
                                "Hop3_SL_%": sc["Hop3"],
                                "LT_mean_days": sc["LT_mean"],
                                "LT_std_days": sc["LT_std"],
                                "Simulated_SS": sc_ss,
                            }
                        )
                    scen_df = pd.DataFrame(scen_rows)

                    base_row = {
                        "Scenario": "Base (Stat)",
                        "EndNode_SL_%": service_level * 100,
                        "Hop1_SL_%": base_hop_sl[1],
                        "Hop2_SL_%": base_hop_sl[2],
                        "Hop3_SL_%": base_hop_sl[3],
                        "LT_mean_days": row["LT_Mean"],
                        "LT_std_days": row["LT_Std"],
                        "Simulated_SS": row["Pre_Rule_SS"],
                    }
                    impl_row = {
                        "Scenario": "Implemented",
                        "EndNode_SL_%": np.nan,
                        "Hop1_SL_%": np.nan,
                        "Hop2_SL_%": np.nan,
                        "Hop3_SL_%": np.nan,
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
                    display_comp["Simulated_SS"] = display_comp["Simulated_SS"].astype(float)

                    implemented_ss = float(row["Safety_Stock"])

                    def pct_vs_impl(v):
                        try:
                            if implemented_ss <= 0 or pd.isna(v):
                                return np.nan
                            return (float(v) / implemented_ss - 1.0) * 100.0
                        except Exception:
                            return np.nan

                    display_comp["Pct_vs_Implemented_%"] = display_comp["Simulated_SS"].apply(pct_vs_impl)

                    st.markdown(
                        "Scenario comparison (Simulated SS). 'Implemented' shows the final Safety_Stock after rules. "
                        "Service Levels shown are for **end-nodes** and the derived hop tiers."
                    )
                    st.markdown('<div class="scenario-table-container">', unsafe_allow_html=True)
                    st.dataframe(
                        df_format_for_display(
                            display_comp[
                                [
                                    "Scenario",
                                    "EndNode_SL_%",
                                    "Hop1_SL_%",
                                    "Hop2_SL_%",
                                    "Hop3_SL_%",
                                    "LT_mean_days",
                                    "LT_std_days",
                                    "Simulated_SS",
                                    "Pct_vs_Implemented_%",
                                ]
                            ].copy(),
                            cols=[
                                "EndNode_SL_%",
                                "Hop1_SL_%",
                                "Hop2_SL_%",
                                "Hop3_SL_%",
                                "LT_mean_days",
                                "LT_std_days",
                                "Simulated_SS",
                                "Pct_vs_Implemented_%",
                            ],
                            two_decimals_cols=[
                                "EndNode_SL_%",
                                "Hop1_SL_%",
                                "Hop2_SL_%",
                                "Hop3_SL_%",
                                "Simulated_SS",
                                "Pct_vs_Implemented_%",
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

    # TAB 7 -----------------------------------------------------------------
    with tab7:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

            sel_prod_default = default_product
            sel_prod_index = all_products.index(sel_prod_default) if all_products else 0
            selected_product = st.selectbox(
                "MATERIAL",
                all_products,
                index=sel_prod_index,
                key="mat_sel",
            )

            if period_labels:
                try:
                    sel_label = period_label(default_period) if default_period is not None else period_labels[-1]
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
                selected_period = period_label_map.get(chosen_label, default_period)
            else:
                selected_period = CURRENT_MONTH_TS

            mat_period_export = get_active_snapshot(results, selected_period if selected_period is not None else default_period)
            mat_period_export = mat_period_export[
                (mat_period_export["Product"] == selected_product)
            ].copy()

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=mat_period_export.to_csv(index=False),
                    file_name=f"material_view_{selected_product}_{period_label(selected_period)}.csv",
                    mime="text/csv",
                    key="mat_export_btn",
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

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
            render_tab7_explainer()

            mat_period_df = get_active_snapshot(results, selected_period if selected_period is not None else default_period)
            mat_period_df = mat_period_df[
                (mat_period_df["Product"] == selected_product)
            ].copy()
            mat_period_df_display = hide_zero_rows(mat_period_df)
            total_forecast = mat_period_df["Forecast"].sum()
            total_ss = mat_period_df["Safety_Stock"].sum()
            nodes_count = mat_period_df["Location"].nunique()

            try:
                avg_days_covered = (
                    mat_period_df["Days_Covered_by_SS"]
                    .replace([np.inf, -np.inf], np.nan)
                    .mean()
                )
            except Exception:
                avg_days_covered = np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Local Forecast", euro_format(total_forecast, True))
            k2.metric("Total Safety Stock (sum nodes)", euro_format(total_ss, True))
            k3.metric("Nodes", f"{nodes_count}")
            k4.metric(
                "Avg Days Covered (nodes)",
                f"{avg_days_covered:.1f}" if not pd.isna(avg_days_covered) else "N/A",
            )

            st.markdown("---")
            st.markdown("### Why do we carry this SS? — 8 Reasons breakdown")
            if mat_period_df_display.empty:
                st.warning("No data for this material/period (non-zero rows filtered).")
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

                mat["term1"] = (mat["Agg_Std_Hist"] ** 2 / float(days_per_month)) * mat["LT_Mean"]
                mat["term2"] = (mat["LT_Std"] ** 2) * (
                    mat["Agg_Future_Demand"] / float(days_per_month)
                ) ** 2
                z_current = norm.ppf(service_level)
                mat["demand_uncertainty_raw"] = z_current * np.sqrt(mat["term1"].clip(lower=0))
                mat["lt_uncertainty_raw"] = z_current * np.sqrt(mat["term2"].clip(lower=0))
                mat["direct_forecast_raw"] = mat["Forecast"].clip(lower=0)
                mat["indirect_network_raw"] = mat["Agg_Future_External"].clip(lower=0)
                mat["cap_reduction_raw"] = (
                    (mat["Pre_Rule_SS"] - mat["Safety_Stock"]).clip(lower=0)
                ).fillna(0)
                mat["cap_increase_raw"] = (
                    (mat["Safety_Stock"] - mat["Pre_Rule_SS"]).clip(lower=0)
                ).fillna(0)
                mat["forced_zero_raw"] = mat.apply(
                    lambda r: r["Pre_Rule_SS"] if r["Adjustment_Status"] == "Forced to Zero" else 0,
                    axis=1,
                )
                mat["b616_override_raw"] = mat.apply(
                    lambda r: r["Pre_Rule_SS"]
                    if (r["Location"] == "B616" and r["Safety_Stock"] == 0)
                    else 0,
                    axis=1,
                )

                raw_drivers = {
                    "Demand Uncertainty (z*sqrt(term1))": mat["demand_uncertainty_raw"].sum(),
                    "Lead-time Uncertainty (z*sqrt(term2))": mat["lt_uncertainty_raw"].sum(),
                    "Direct Local Forecast (sum Fcst)": mat["direct_forecast_raw"].sum(),
                    "Indirect Network Demand (sum extra downstream)": mat["indirect_network_raw"].sum(),
                    "Caps — Reductions (policy lowering SS)": mat["cap_reduction_raw"].sum(),
                    "Caps — Increases (policy increasing SS)": mat["cap_increase_raw"].sum(),
                    "Forced Zero Overrides (policy)": mat["forced_zero_raw"].sum(),
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

                st.markdown("#### A. Original — Raw driver values (interpretation view)")
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

                st.markdown("---")
                st.markdown(
                    "#### B. SS Attribution — Mutually exclusive components that SUM EXACTLY to Total Safety Stock"
                )
                per_node = mat.copy()
                per_node["is_forced_zero"] = per_node["Adjustment_Status"] == "Forced to Zero"
                per_node["is_b616_override"] = (per_node["Location"] == "B616") & (
                    per_node["Safety_Stock"] == 0
                )
                per_node["pre_ss"] = per_node["Pre_Rule_SS"].clip(lower=0)
                per_node["share_denom"] = (
                    per_node["demand_uncertainty_raw"] + per_node["lt_uncertainty_raw"]
                )

                def demand_share_calc(r):
                    if r["share_denom"] > 0:
                        return r["pre_ss"] * (r["demand_uncertainty_raw"] / r["share_denom"])
                    return (r["pre_ss"] / 2) if r["pre_ss"] > 0 else 0.0

                def lt_share_calc(r):
                    if r["share_denom"] > 0:
                        return r["pre_ss"] * (r["lt_uncertainty_raw"] / r["share_denom"])
                    return (r["pre_ss"] / 2) if r["pre_ss"] > 0 else 0.0

                per_node["demand_share"] = per_node.apply(demand_share_calc, axis=1)
                per_node["lt_share"] = per_node.apply(lt_share_calc, axis=1)
                per_node["forced_zero_amount"] = per_node.apply(
                    lambda r: r["pre_ss"] if r["is_forced_zero"] else 0.0,
                    axis=1,
                )
                per_node["b616_override_amount"] = per_node.apply(
                    lambda r: r["pre_ss"] if r["is_b616_override"] else 0.0,
                    axis=1,
                )

                def retained_ratio_calc(r):
                    if r["pre_ss"] <= 0:
                        return 0.0
                    if r["is_forced_zero"] or r["is_b616_override"]:
                        return 0.0
                    return float(r["Safety_Stock"]) / float(r["pre_ss"]) if r["pre_ss"] > 0 else 0.0

                per_node["retained_ratio"] = per_node.apply(retained_ratio_calc, axis=1)
                per_node["retained_demand"] = per_node["demand_share"] * per_node["retained_ratio"]
                per_node["retained_lt"] = per_node.apply(
                    lambda r: r["lt_share"] * r["retained_ratio"], axis=1
                )
                per_node["retained_stat_total"] = per_node["retained_demand"] + per_node["retained_lt"]

                def direct_frac_calc(r):
                    if r["Agg_Future_Demand"] > 0:
                        return float(r["Forecast"]) / float(r["Agg_Future_Demand"])
                    return 0.0

                per_node["direct_frac"] = per_node.apply(direct_frac_calc, axis=1).clip(lower=0, upper=1)
                per_node["direct_retained_ss"] = per_node["retained_stat_total"] * per_node["direct_frac"]
                per_node["indirect_retained_ss"] = per_node["retained_stat_total"] * (
                    1 - per_node["direct_frac"]
                )
                per_node["cap_reduction"] = per_node.apply(
                    lambda r: max(r["pre_ss"] - r["Safety_Stock"], 0.0)
                    if not (r["is_forced_zero"] or r["is_b616_override"])
                    else 0.0,
                    axis=1,
                )
                per_node["cap_increase"] = per_node.apply(
                    lambda r: max(r["Safety_Stock"] - r["pre_ss"], 0.0)
                    if not (r["is_forced_zero"] or r["is_b616_override"])
                    else 0.0,
                    axis=1,
                )

                ss_attrib = {
                    "Demand Uncertainty (SS portion)": per_node["retained_demand"].sum(),
                    "Lead-time Uncertainty (SS portion)": per_node["retained_lt"].sum(),
                    "Direct Local Forecast (SS portion)": per_node["direct_retained_ss"].sum(),
                    "Indirect Network Demand (SS portion)": per_node["indirect_retained_ss"].sum(),
                    "Caps — Reductions (policy lowering SS)": per_node["cap_reduction"].sum(),
                    "Caps — Increases (policy increasing SS)": per_node["cap_increase"].sum(),
                    "Forced Zero Overrides (policy)": per_node["forced_zero_amount"].sum(),
                    "B616 Policy Override": per_node["b616_override_amount"].sum(),
                }
                for k in ss_attrib:
                    ss_attrib[k] = float(ss_attrib[k])
                ss_sum = sum(ss_attrib.values())
                residual = float(total_ss) - ss_sum
                if abs(residual) > 1e-6:
                    ss_attrib["Caps — Reductions (policy lowering SS)"] += residual
                    ss_sum = sum(ss_attrib.values())

                ss_drv_df = pd.DataFrame(
                    {
                        "driver": list(ss_attrib.keys()),
                        "amount": [float(v) for v in ss_attrib.values()],
                    }
                )
                ss_drv_df_display = ss_drv_df[ss_drv_df["amount"] != 0].copy()
                denom = total_ss if total_ss > 0 else ss_drv_df["amount"].sum()
                denom = denom if denom > 0 else 1.0
                ss_drv_df_display["pct_of_total_ss"] = (
                    ss_drv_df_display["amount"] / denom * 100
                )

                labels = ss_drv_df_display["driver"].tolist() + ["Total SS"]
                values = ss_drv_df_display["amount"].tolist() + [total_ss]
                measures = ["relative"] * len(ss_drv_df_display) + ["total"]

                decreasing_color = "rgba(255, 138, 128, 0.8)"
                increasing_color = "rgba(129, 199, 132, 0.8)"
                total_color = "rgba(144, 202, 249, 0.8)"

                fig_drv = go.Figure(
                    go.Waterfall(
                        name="SS Attribution",
                        orientation="v",
                        measure=measures,
                        x=labels,
                        y=values,
                        text=[f"{v:,.0f}" for v in ss_drv_df_display["amount"].tolist()]
                        + [f"{total_ss:,.0f}"],
                        connector={"line": {"color": "rgba(63,63,63,0.25)"}},
                        decreasing=dict(marker=dict(color=decreasing_color)),
                        increasing=dict(marker=dict(color=increasing_color)),
                        totals=dict(marker=dict(color=total_color)),
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
                st.dataframe(ss_attrib_df_formatted, use_container_width=True)

                # Executive takeaway with bold percentages and larger font
                try:
                    top3 = ss_drv_df_display.sort_values("pct_of_total_ss", ascending=False).head(3)
                    pieces = []
                    for _, r in top3.iterrows():
                        pieces.append(
                            f"{r['driver']} (<strong>{r['pct_of_total_ss']:.1f}%</strong>)"
                        )
                    if pieces:
                        takeaway = (
                            f"For <strong>{selected_product}</strong> in <strong>{period_label(selected_period)}</strong>, "
                            f"safety stock is mainly explained by: " + "; ".join(pieces) + "."
                        )
                        st.markdown(
                            f"""
                            <div style="margin-top:8px;padding:8px 10px;border-radius:8px;
                                background:#f5f9ff;border:1px solid #c5cae9;font-size:1.1rem;">
                              <strong>Executive takeaway:</strong><br/>
                              {takeaway}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass

    # TAB 8 -----------------------------------------------------------------
    with tab8:
        col_main, col_badge = st.columns([17, 3])
        with col_badge:
            render_logo_above_parameters(scale=1.5)

            if period_labels:
                try:
                    sel_label = period_label(default_period) if default_period is not None else period_labels[-1]
                    sel_period_index = (
                        period_labels.index(sel_label)
                        if sel_label in period_labels
                        else len(period_labels) - 1
                    )
                except Exception:
                    sel_period_index = len(period_labels) - 1
                chosen_label_all = st.selectbox(
                    "PERIOD",
                    period_labels,
                    index=sel_period_index,
                    key="allmat_period",
                )
                selected_period_all = period_label_map.get(chosen_label_all, default_period)
            else:
                selected_period_all = CURRENT_MONTH_TS

            snapshot_all = get_active_snapshot(results, selected_period_all if selected_period_all is not None else default_period)

            agg_all = snapshot_all.groupby("Product", as_index=False).agg(
                Network_Demand_Month=("Agg_Future_Demand", "sum"),
                Local_Forecast_Month=("Forecast", "sum"),
                Safety_Stock=("Safety_Stock", "sum"),
                Max_Corridor=("Max_Corridor", "sum"),
                Avg_Day_Demand=("D_day", "mean"),
                Avg_SS_Days_Coverage=("Days_Covered_by_SS", "mean"),
                Nodes=("Location", "nunique"),
            )

            if "Tier_Hops" in snapshot_all.columns:
                end_nodes = (
                    snapshot_all[snapshot_all["Tier_Hops"] == 0]
                    .groupby("Product")["Location"]
                    .nunique()
                    .reset_index(name="End_Nodes")
                )
                agg_all = agg_all.merge(end_nodes, on="Product", how="left")
            else:
                agg_all["End_Nodes"] = np.nan

            agg_all = agg_all[agg_all["Safety_Stock"] > 0].copy()

            agg_all["Reorder_Point"] = agg_all["Safety_Stock"] + agg_all["Local_Forecast_Month"]

            agg_all["SS_to_Demand_Ratio_%"] = np.where(
                agg_all["Network_Demand_Month"] > 0,
                (agg_all["Safety_Stock"] / agg_all["Network_Demand_Month"]) * 100.0,
                0.0,
            )

            for c in [
                "Network_Demand_Month",
                "Local_Forecast_Month",
                "Safety_Stock",
                "Max_Corridor",
                "Reorder_Point",
            ]:
                if c in agg_all.columns:
                    agg_all[c] = agg_all[c].fillna(0.0)

            if "Avg_Day_Demand" in agg_all.columns:
                agg_all["Avg_Day_Demand"] = agg_all["Avg_Day_Demand"].fillna(0.0)
            if "Avg_SS_Days_Coverage" in agg_all.columns:
                agg_all["Avg_SS_Days_Coverage"] = agg_all["Avg_SS_Days_Coverage"].fillna(0.0)
            if "SS_to_Demand_Ratio_%" in agg_all.columns:
                agg_all["SS_to_Demand_Ratio_%" ] = agg_all["SS_to_Demand_Ratio_%"].fillna(0.0)

            with st.container():
                st.markdown('<div class="export-csv-btn">', unsafe_allow_html=True)
                st.download_button(
                    "💾 Export CSV",
                    data=agg_all.to_csv(index=False),
                    file_name=f"all_materials_{period_label(selected_period_all)}.csv",
                    mime="text/csv",
                    key="allmat_export_btn",
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        with col_main:
            render_selection_line("Selected:", period_text=period_label(selected_period_all))
            st.subheader("📊 All Materials View")

            st.markdown(
                "High-level snapshot by material (one row per material for the selected period). "
                "Values are aggregated across all ACTIVE locations; all numeric values are rounded to integers."
            )

            if agg_all.empty:
                st.warning("No data available for the selected period.")
            else:
                display_cols_all = [
                    "Product",
                    "Avg_Day_Demand",
                    "Safety_Stock",
                    "Avg_SS_Days_Coverage",
                    "Local_Forecast_Month",
                    "SS_to_Demand_Ratio_%",
                ]
                display_cols_all = [c for c in display_cols_all if c in agg_all.columns]

                agg_view = agg_all.sort_values("Avg_Day_Demand", ascending=False)[display_cols_all].reset_index(
                    drop=True
                )

                rename_map = {
                    "Avg_Day_Demand": "Avg Daily Demand",
                    "Safety_Stock": "Calculated Safety Stock",
                    "Avg_SS_Days_Coverage": "SS Coverage (days)",
                    "Local_Forecast_Month": "Local Forecast (month)",
                    "SS_to_Demand_Ratio_%" : "SS / Demand (%)",
                }
                agg_view = agg_view.rename(columns=rename_map)

                formatted = agg_view.copy()
                if "Avg Daily Demand" in formatted.columns:
                    formatted["Avg Daily Demand"] = formatted["Avg Daily Demand"].apply(
                        lambda v: "{:.3f}".format(v) if pd.notna(v) else ""
                    )
                if "Calculated Safety Stock" in formatted.columns:
                    formatted["Calculated Safety Stock"] = formatted["Calculated Safety Stock"].apply(
                        lambda v: euro_format(v, always_two_decimals=False, show_zero=True)
                    )
                if "Local Forecast (month)" in formatted.columns:
                    formatted["Local Forecast (month)"] = formatted["Local Forecast (month)"].apply(
                        lambda v: euro_format(v, always_two_decimals=False, show_zero=True)
                    )
                if "SS Coverage (days)" in formatted.columns:
                    formatted["SS Coverage (days)"] = formatted["SS Coverage (days)"].apply(
                        lambda v: "{:.0f}".format(v) if pd.notna(v) else ""
                    )
                if "SS / Demand (%)" in formatted.columns:
                    formatted["SS / Demand (%)"] = formatted["SS / Demand (%)"].apply(
                        lambda v: "{:.0f}".format(v) if pd.notna(v) else ""
                    )

                st.dataframe(formatted, use_container_width=True, height=430)

else:
    st.info(
        "Please upload sales.csv, demand.csv and leadtime.csv in the sidebar to run the optimizer."
    )
