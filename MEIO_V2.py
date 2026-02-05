"""
Multi-Echelon Inventory Optimizer — Raw Materials
Developed by mat635418 — JAN 2026
"""
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
        v2.30
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
        Released Jan&nbsp;2026
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
        max-width: 100%;
        margin-left: 0;
        margin-right: 0;
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
      /* Compact run configuration + snapshot header (single-row snapshot) */
      .run-snapshot-container {
        display: grid;
        grid-template-columns: minmax(220px, 1.2fr) repeat(3, minmax(180px, 1fr));
        gap: 10px;
        margin: 8px 0 4px 0;
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
        margin: 0;
        border-radius: 8px 8px 0 0;
        background: linear-gradient(90deg,#e3f2fd,#e8f5e9);
        font-size: 0.80rem;
        font-weight: 700;
        color: #0b3d91;
      }
      .run-snapshot-wrapper {
        border-radius: 8px;
        background: linear-gradient(90deg,#e3f2fd,#e8f5e9);
        padding: 0;
        margin-top: 6px;
        margin-bottom: 4px;
      }
      .run-snapshot-inner {
        padding: 6px 8px 8px 8px;
      }
      .exec-takeaway-selection {
        color: #b71c1c;
        font-weight: 700;
      }
      .exec-takeaway-percentage {
        color: #b71c1c;
        font-weight: 700;
      }

      /* --- Generic card-based "table row" layout used across tabs --- */
      .strip-container {
        width: 100%;
        overflow-x: auto;
        padding-bottom: 6px;
      }
      .cards-row {
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-width: 650px;
      }
      .card-row {
        display: grid;
        grid-template-columns: repeat(6, minmax(90px, 1fr));
        align-items: stretch;
        border-radius: 10px;
        padding: 6px 10px;
        background: linear-gradient(90deg,#f8fbff,#f5fff9);
        box-shadow: 0 1px 2px rgba(15,23,42,0.12);
      }
      .card-row-col-header {
        font-size: 0.70rem;
        font-weight: 600;
        color: #616161;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 1px;
      }
      .card-row-col-value {
        font-size: 0.92rem;
        font-weight: 650;
        color: #111827;
        white-space: nowrap;
      }
      .card-row-col-sub {
        font-size: 0.72rem;
        color: #9e9e9e;
        margin-left: 4px;
      }
      .card-row-badge-main {
        display: inline-flex;
        align-items: center;
        justify-content: flex-start;
        padding: 4px 8px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.88rem;
        color: #424242;
        white-space: nowrap;
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
      }
      .card-row-badge-chevron {
        margin-right: 6px;
        font-size: 0.80rem;
        color: #757575;
      }
      .card-row-pill-grey {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 42px;
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #424242;
        background: #eeeeee;
        border: 1px solid #e0e0e0;
      }
      .card-row-pill-blue {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 42px;
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(90deg,#2196f3,#64b5f6);
      }
      .card-row-pill-green {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 42px;
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(90deg,#43a047,#66bb6a);
      }
      .card-row-pill-orange {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 42px;
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(90deg,#fb8c00,#ffb74d);
      }
      .card-row-pill-red {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 42px;
        padding: 2px 6px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(90deg,#e53935,#ef5350);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Helper Functions: UI Renderers & Data Dictionary
# =============================================================================


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
    """Collapsible banner: single-row snapshot + run configuration, initially collapsed."""
    with st.expander("📊 Network snapshot & run configuration", expanded=False):
        st.markdown(
            f"""
            <div class="run-snapshot-wrapper">
              <div class="run-snapshot-header">
                Network snapshot – {snapshot_label}
              </div>
              <div class="run-snapshot-inner">
                <div style="display: flex; gap: 10px;">
                  <div style="flex: 0 0 75%; display: flex; flex-direction: column; gap: 10px;">
                    <div class="run-snapshot-container" style="grid-template-columns: repeat(3, 1fr); margin: 0;">
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
                        <div class="snap-card-value">
                          {coverage_months:.2f} months
                        </div>
                      </div>
                    </div>
                    <div class="run-snapshot-container" style="grid-template-columns: repeat(2, 1fr); margin: 0;">
                      <div class="snap-card">
                        <div class="snap-card-label">Active Materials (with corridor)</div>
                        <div class="snap-card-value">{n_active_materials}</div>
                      </div>
                      <div class="snap-card">
                        <div class="snap-card-label">Active Nodes (with corridor)</div>
                        <div class="snap-card-value">{n_active_nodes}</div>
                      </div>
                    </div>
                  </div>
                  <div style="flex: 0 0 25%;">
                    <div class="run-card" style="height: 100%;">
                      <div class="run-card-title">Run configuration</div>
                      <div class="run-card-body">
                        <div><strong>ID:</strong> {run_id}</div>
                        <div><strong>Time:</strong> {now_str}</div>
                        <div><strong>End-node SL:</strong> {service_level*100:.2f}%</div>
                        <div><strong>Zero SS if no demand:</strong> {str(zero_if_no_net_fcst)}</div>
                        <div><strong>SS capping:</strong> {str(apply_cap)} ({cap_range[0]}–{cap_range[1]} % of network demand)</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
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
    """Short guide for the 'By Material' / SS attribution tab."""
    with st.expander("ℹ️ How to interpret the SS attribution view", expanded=False):
        st.markdown(
            """
            This tab explains **why we hold Safety Stock (SS)** for a given material and period
            by decomposing the implemented SS into **mutually exclusive components** that
            **sum exactly** to the total SS.

            The components are grouped as:

            **Statistical drivers (what the math suggests):**
            - **Demand Uncertainty (SS portion)**: part of SS driven by forecast variability.
            - **Lead‑time Uncertainty (SS portion)**: part of SS driven by supply lead‑time volatility.
            - **Direct Local Forecast (SS portion)**: SS share linked to protecting a node's own demand.
            - **Indirect Network Demand (SS portion)**: SS share linked to protecting downstream locations.

            **Policy / rule‑based effects (how policy reshapes SS):**
            - **Caps — Reductions**: policy rules that reduce SS vs the raw statistical result.
            - **Caps — Increases**: policy rules that increase SS (e.g. minimum coverage).
            - **Forced Zero Overrides**: decisions that set SS to zero even if statistics suggest > 0.
            - **B616 Policy Override**: specific override for location B616 in this model.

            Use the waterfall chart and the table to **trace the contribution of each driver**
            to the final implemented SS for the selected material and period.
            """,
            unsafe_allow_html=True,
        )


def render_ss_formula_explainer(hops=None, service_level=None):
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
            - Scenario $SS$ is **policy-consistent** with the main engine:
              we apply zero-if-no-demand, capping and B616 overrides.
            """
        )
        
        # If hop information is provided, display the Applied Hop Logic table
        if hops is not None and service_level is not None:
            st.markdown("---")
            st.markdown("**Applied Hop Logic → Service Level mapping:**")
            
            # Calculate SL values for each hop based on current service_level from sidebar
            hop_sl_values = {
                0: service_level * 100,  # Convert to percentage
                1: service_level * 0.9596 * 100,
                2: service_level * 0.9091 * 100,
                3: service_level * 0.8586 * 100
            }
            
            # Node type descriptions
            node_type_descriptions = {
                0: "End-node",
                1: "Internal+External Demand",
                2: "Level-1 Hub (e.g. LUEX)",
                3: "Level-2 Hub (e.g. BEEX)"
            }
            
            # Create dataframe for the SL tiering table
            sl_tier_data = []
            for hop in range(4):
                sl_tier_data.append({
                    'Hop': hop,
                    'Service Level (%)': hop_sl_values[hop],
                    'Node Type': node_type_descriptions[hop]
                })
            sl_tier_df = pd.DataFrame(sl_tier_data)
            
            # Apply highlighting to the row matching current node's hop level
            def highlight_current_hop(row):
                if row['Hop'] == hops:
                    return ['background-color: #fffacd'] * len(row)
                return [''] * len(row)
            
            sl_tier_styled = sl_tier_df.style.apply(highlight_current_hop, axis=1).format({
                'Service Level (%)': '{:.2f}'
            }).set_properties(**{
                'text-align': 'left'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]}
            ])
            
            # Display table with 1/3 width and more space for Node Type column
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1:
                st.dataframe(sl_tier_styled, hide_index=True)


def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Robustly convert a pandas-like column to numeric.
    Handles various formats: parentheses for negatives, commas, spaces, and NA-like strings.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    s = series.astype("string")
    s = s.str.strip()

    s = s.replace({
        "": pd.NA, "-": pd.NA, "—": pd.NA, "na": pd.NA, "NA": pd.NA,
        "n/a": pd.NA, "N/A": pd.NA, "None": pd.NA,
    })

    paren_mask = s.str.startswith("(") & s.str.endswith(")")
    if paren_mask.any():
        s.loc[paren_mask] = "-" + s.loc[paren_mask].str[1:-1]

    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
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


def render_card_table(
    df: pd.DataFrame,
    *,
    id_col: str,
    col_defs: list,
    container_height: int = 500,
) -> None:
    """
    Render a DataFrame as a list of horizontally-aligned 'card rows'.

    Parameters
    ----------
    df : DataFrame
        Data to render (already filtered & ordered).
    id_col : str
        Column used as the left 'identifier badge'.
    col_defs : list of dict
        Column definition dicts with keys: "header", "col", "fmt", "pill", "sub_col", "sub_fmt"
    container_height : int
        Height hint for the HTML container.
    """
    if df is None or df.empty:
        st.info("No rows to display for this selection.")
        return

    parts = ['<div class="strip-container" style="max-height:{}px;">'.format(container_height)]
    parts.append('<div class="cards-row">')

    for _, r in df.iterrows():
        parts.append('<div class="card-row">')

        id_val = str(r.get(id_col, ""))
        parts.append(
            '<div style="display:flex;align-items:center;">'
            '<div class="card-row-badge-main">'
            '<span class="card-row-badge-chevron">▶</span>'
            f"{id_val}"
            "</div>"
            "</div>"
        )

        for d in col_defs:
            header = d.get("header", "")
            col_name = d.get("col", "")
            fmt = d.get("fmt", None)
            pill = d.get("pill", None)
            sub_col = d.get("sub_col", None)
            sub_fmt = d.get("sub_fmt", None)

            raw_val = r.get(col_name, "")
            if fmt is not None:
                try:
                    val_txt = fmt(raw_val)
                except Exception:
                    val_txt = str(raw_val)
            else:
                val_txt = str(raw_val)

            sub_val_txt = ""
            if sub_col is not None:
                raw_sub = r.get(sub_col, "")
                if sub_fmt is not None:
                    try:
                        sub_val_txt = sub_fmt(raw_sub)
                    except Exception:
                        sub_val_txt = str(raw_sub)
                else:
                    sub_val_txt = str(raw_sub)

            pill_cls = {
                "grey": "card-row-pill-grey",
                "blue": "card-row-pill-blue",
                "green": "card-row-pill-green",
                "orange": "card-row-pill-orange",
                "red": "card-row-pill-red"
            }.get(pill, None)

            parts.append("<div>")
            parts.append(f'<div class="card-row-col-header">{header}</div>')
            if pill_cls:
                parts.append(
                    f'<div class="card-row-col-value"><span class="{pill_cls}">{val_txt}</span>'
                )
            else:
                parts.append(f'<div class="card-row-col-value">{val_txt}')

            if sub_val_txt not in ("", " ", None):
                parts.append(f'<span class="card-row-col-sub">{sub_val_txt}</span>')
            parts.append("</div></div>")

        parts.append("</div>")

    parts.append("</div></div>")
    html = "".join(parts)
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# Active Row Detection & Filtering
# =============================================================================


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
            # Handle potential duplicate Location entries by aggregating
            p_stats_filtered = df_stats[df_stats["Product"] == prod]
            if p_stats_filtered["Location"].duplicated().any():
                # Group by Location and keep first occurrence for stats
                # This handles rare cases where stats have duplicate locations
                p_stats_filtered = p_stats_filtered.groupby("Location").first().reset_index()
            p_stats = p_stats_filtered.set_index("Location").to_dict("index")
            
            # Handle potential duplicate Location entries by aggregating Forecast values
            p_fore_filtered = df_month[df_month["Product"] == prod]
            if p_fore_filtered["Location"].duplicated().any():
                # Sum Forecast values for duplicate locations (e.g., from date ambiguity)
                # Build aggregation dict: sum for Forecast and numeric columns, first for others
                agg_dict = {}
                for col in p_fore_filtered.columns:
                    if col in ['Product', 'Period', 'PurchasingGroupName']:
                        agg_dict[col] = 'first'  # Keep first value for categorical/identifier columns
                    elif col == 'Forecast' or pd.api.types.is_numeric_dtype(p_fore_filtered[col]):
                        agg_dict[col] = 'sum'  # Sum numeric values
                    else:
                        agg_dict[col] = 'first'  # Default: keep first for unknown columns
                p_fore_filtered = p_fore_filtered.groupby("Location", as_index=False).agg(agg_dict)
            p_fore = p_fore_filtered.set_index("Location").to_dict("index")
            
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


def apply_policy_to_scalar_ss(
    ss_value: float,
    agg_future_demand: float,
    location: str,
    zero_if_no_net_fcst: bool,
    apply_cap: bool,
    cap_range,
) -> float:
    """
    Apply policy rules to a single SS value:
      1) Zero SS if no aggregated network demand (if enabled)
      2) Cap SS within [l_cap, u_cap] * Agg_Future_Demand (if enabled)
      3) B616 override → SS = 0
    """
    ss = float(ss_value)

    if zero_if_no_net_fcst and agg_future_demand <= 0:
        ss = 0.0

    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100.0, cap_range[1] / 100.0
        l_lim = agg_future_demand * l_cap
        u_lim = agg_future_demand * u_cap
        ss = max(l_lim, min(ss, u_lim))

    if location == "B616":
        ss = 0.0

    return ss


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
    hop1_sl: float = None,
    hop2_sl: float = None,
    hop3_sl: float = None,
):
    # Calculate hop-based SL values
    # If hop SLs are provided, use them; otherwise use the original ratios
    if hop1_sl is None:
        hop1_sl = service_level * 0.9596
    else:
        hop1_sl = hop1_sl / 100.0  # Convert from percentage
    
    if hop2_sl is None:
        hop2_sl = service_level * 0.9091
    else:
        hop2_sl = hop2_sl / 100.0  # Convert from percentage
    
    if hop3_sl is None:
        hop3_sl = service_level * 0.8586
    else:
        hop3_sl = hop3_sl / 100.0  # Convert from percentage
    
    hop_to_sl = {
        0: service_level,
        1: hop1_sl,
        2: hop2_sl,
        3: hop3_sl
    }

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

    # Zero if no demand
    if zero_if_no_net_fcst:
        zero_mask = res["Agg_Future_Demand"] <= 0
        res.loc[zero_mask, "Adjustment_Status"] = "Forced to Zero"
        res.loc[zero_mask, "Safety_Stock"] = 0.0

    # Capping
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
    # B616 override
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


with st.sidebar.expander("⚙️ Service Level Configuration", expanded=False):
    service_level = st.slider(
        "Service Level (%) for the end-nodes",
        50.0,
        99.9,
        99.0,
        help="Target probability of not stocking out for end-nodes (hop 0). Upstream nodes get fixed SLs by hop-distance.",
    ) / 100
    z = norm.ppf(service_level)
    
    # Define default hop service levels for use across tabs
    # These can be overridden in Tab6 for scenario planning
    hop1_sl_global = 95.0  # Default 95%
    hop2_sl_global = 90.0  # Default 90%
    hop3_sl_global = 85.0  # Default 85%

with st.sidebar.expander("⚙️ Safety Stock Rules", expanded=False):
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
        hop1_sl=hop1_sl_global,
        hop2_sl=hop2_sl_global,
        hop3_sl=hop3_sl_global,
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

    # =============================================================================
    # TAB 1: Inventory Corridor
    # =============================================================================
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

        # Add cost per kilo slider below PERIOD filter (PR #3)
        cost_per_kilo = st.slider(
            "Cost per kilo (USD)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.10,
            help="USD cost per kilo for SS simulation",
            key="tab2_costperkilo"
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    with col_main:
        render_selection_line("Selected:", product=sku, period_text=period_label(chosen_period))
        st.subheader("🕸️ Network Topology")
        
        # ---- SCENARIO PLANNING BOX ----
        st.markdown("---")
        with st.expander("🚚 Distribution Strategy Scenario Planning ('What-If' Routing)", expanded=False):
            st.markdown(
                """
                **'What if I move a location from being served by node A to being served by node B?'**
                - Select a node to reroute, and the new supplier for that node.
                - Heating time is automatically determined by plant and season.
                """
            )

            scenario_products = active_materials(results, period=chosen_period)
            scenario_product_default = sku if sku in scenario_products else (scenario_products[0] if scenario_products else "")
            scenario_product = st.selectbox("Which material?", scenario_products, index=(scenario_products.index(scenario_product_default) if scenario_product_default in scenario_products else 0), key="scen_sku")

            all_service_nodes = active_nodes(results, period=chosen_period, product=scenario_product)
            location_default = all_service_nodes[0] if all_service_nodes else ""
            location_to_move = st.selectbox("Which location do you want to reroute?", all_service_nodes, index=(all_service_nodes.index(location_default) if location_default in all_service_nodes else 0), key="scen_locmove")

            scenario_lt = df_lt[df_lt["Product"] == scenario_product] if "Product" in df_lt.columns else df_lt.copy()
            mask_to = scenario_lt["To_Location"] == location_to_move
            if not scenario_lt[mask_to].empty:
                current_suppliers = scenario_lt[mask_to]["From_Location"].dropna().unique().tolist()
            else:
                current_suppliers = []

            if not current_suppliers:
                st.info("This location has no supplier in the current network topology for this period (cannot reroute).")
                reroute_enabled = False
                current_supplier = None
            else:
                current_supplier = current_suppliers[0]
                st.markdown(f"**Current supplier (from node):** `{current_supplier}`")
                possible_suppliers = [n for n in all_service_nodes if n != location_to_move and n != current_supplier]
                new_supplier_default = possible_suppliers[0] if possible_suppliers else ""
                new_supplier = st.selectbox("New supplier (route to node):", possible_suppliers, index=(possible_suppliers.index(new_supplier_default) if new_supplier_default in possible_suppliers else 0), key="scen_newsup")
                reroute_enabled = True

            heating_winter = {
                "BEEX": 0, "DEF1": 0, "DEF2": 21, "DEH1": 0, "DEW1": 14,
                "FRA1": 2, "FRM1": 0, "LUEX": 14, "LUL1": 14, "PLPL": 1,
                "RSKR": 1, "SIS1": 4, "TRTU": 1, "TRTZ": 1, "ZASA": 0
            }
            heating_summer = {
                "BEEX": 0, "DEF1": 0, "DEF2": 14, "DEH1": 0, "DEW1": 14,
                "FRA1": 1.5, "FRM1": 0, "LUEX": 14, "LUL1": 14, "PLPL": 1,
                "RSKR": 1, "SIS1": 4, "TRTU": 1, "TRTZ": 1, "ZASA": 0
            }
            def infer_season(ts):
                if isinstance(ts, pd.Timestamp):
                    month = ts.month
                else:
                    try:
                        month = pd.to_datetime(ts).month
                    except Exception:
                        return "WINTER"
                if month in [10, 11, 12, 1, 2, 3]:
                    return "WINTER"
                elif month in [7, 8, 9]:
                    return "SUMMER"
                else:
                    return "OTHER"

            reroute_button_style = """
                <style>
                .reroute-btn button {
                    background: #4fc3e4 !important;
                    color: #fff !important;
                    border-radius: 20px !important;
                    padding: 0.45rem 1.1rem !important;
                    border: none;
                    font-weight: 700;
                    font-size: 1.03rem;
                    margin-bottom: 7px;
                }
                </style>
            """
            st.markdown(reroute_button_style, unsafe_allow_html=True)

            if reroute_enabled and st.button("Simulate reroute scenario", key="dist_scen_run", help="Run the scenario with the current choices.", type="primary"):
                rerouted_lt = scenario_lt.copy()
                removal_mask = (
                    (rerouted_lt["From_Location"] == current_supplier) &
                    (rerouted_lt["To_Location"] == location_to_move)
                )
                rerouted_lt = rerouted_lt[~removal_mask]
                season = infer_season(chosen_period)
                if season == "WINTER": heating_table = heating_winter
                elif season == "SUMMER": heating_table = heating_summer
                else: heating_table = heating_winter
                heating_days = heating_table.get(location_to_move, 0)
                direct_route = scenario_lt[
                    (scenario_lt["From_Location"] == new_supplier) & (scenario_lt["To_Location"] == location_to_move)
                ]
                extra_note = ""
                if not direct_route.empty:
                    new_lt_days  = float(direct_route.iloc[0]["Lead_Time_Days"]) + heating_days
                    new_lt_stddev = float(direct_route.iloc[0]["Lead_Time_Std_Dev"])
                    lead_time_note = f"Used <strong>explicit lead time from {new_supplier} to {location_to_move}</strong> plus <strong>{heating_days}d heating</strong>: <strong>{new_lt_days:.1f}d</strong>, stddev <strong>{new_lt_stddev:.1f}d</strong>."
                else:
                    poss_lts = scenario_lt[scenario_lt["From_Location"] == new_supplier]
                    if not poss_lts.empty:
                        min_idx = poss_lts["Lead_Time_Days"].idxmin()
                        new_lt_days = float(poss_lts.loc[min_idx, "Lead_Time_Days"]) + heating_days
                        new_lt_stddev = float(poss_lts.loc[min_idx, "Lead_Time_Std_Dev"])
                        extra_note = f" (minimal outbound LT from {new_supplier})"
                        lead_time_note = f"No direct LT; used <strong>minimal outbound LT from {new_supplier}{extra_note}</strong> plus <strong>{heating_days}d heating</strong>: <strong>{new_lt_days:.1f}d</strong>, stddev <strong>{new_lt_stddev:.1f}d</strong>."
                    else:
                        new_lt_days = (float(rerouted_lt["Lead_Time_Days"].median()) if not rerouted_lt.empty else 7.0) + heating_days
                        new_lt_stddev = float(rerouted_lt["Lead_Time_Std_Dev"].median()) if not rerouted_lt.empty else 2.0
                        lead_time_note = f"No data: used <strong>network median LT for this SKU</strong> plus <strong>{heating_days}d heating</strong>: <strong>{new_lt_days:.1f}d</strong>, stddev <strong>{new_lt_stddev:.1f}d</strong>."
                new_row = {
                    "Product": scenario_product,
                    "From_Location": new_supplier,
                    "To_Location": location_to_move,
                    "Lead_Time_Days": new_lt_days,
                    "Lead_Time_Std_Dev": new_lt_stddev
                }
                rerouted_lt = pd.concat([rerouted_lt, pd.DataFrame([new_row])], ignore_index=True)

                # --------- Run pipeline ---------
                reroute_df_d = df_d[(df_d["Product"] == scenario_product) & (df_d["Period"] == chosen_period)].copy()
                reroute_stats = stats[(stats["Product"] == scenario_product)].copy()
                rerouted_results, _ = run_pipeline(
                    df_d=reroute_df_d,
                    stats=reroute_stats,
                    df_lt=rerouted_lt,
                    service_level=service_level,
                    transitive=use_transitive,
                    rho=var_rho,
                    lt_mode_param=lt_mode,
                    zero_if_no_net_fcst=zero_if_no_net_fcst,
                    apply_cap=apply_cap,
                    cap_range=cap_range,
                    hop1_sl=hop1_sl_global,
                    hop2_sl=hop2_sl_global,
                    hop3_sl=hop3_sl_global,
                )
                old_results = results[
                    (results["Product"] == scenario_product) &
                    (results["Period"] == chosen_period)
                ]
                left = old_results.copy()
                right = rerouted_results.copy()
                for col in ['Safety_Stock', 'LT_Mean', 'Tier_Hops', 'Service_Level_Node']:
                    left.rename(columns={col: col+'_old'}, inplace=True)
                    right.rename(columns={col: col+'_new'}, inplace=True)
                comparison = pd.merge(
                    left[['Location', 'Safety_Stock_old', 'LT_Mean_old', 'Tier_Hops_old', 'Service_Level_Node_old']],
                    right[['Location', 'Safety_Stock_new', 'LT_Mean_new', 'Tier_Hops_new', 'Service_Level_Node_new']],
                    on='Location', how='outer'
                )

                comparison['SS_old_usd'] = comparison['Safety_Stock_old'].fillna(0) * cost_per_kilo
                comparison['SS_new_usd'] = comparison['Safety_Stock_new'].fillna(0) * cost_per_kilo
                comparison['ΔSS'] = comparison['Safety_Stock_new'].fillna(0) - comparison['Safety_Stock_old'].fillna(0)
                comparison['%ΔSS'] = 100 * comparison['ΔSS'] / comparison['Safety_Stock_old'].replace(0, np.nan)
                comparison['ΔSS_usd'] = comparison['SS_new_usd'] - comparison['SS_old_usd']
                comparison['Safety_Stock_old'] = comparison['Safety_Stock_old'].fillna(0).astype(int)
                comparison['Safety_Stock_new'] = comparison['Safety_Stock_new'].fillna(0).astype(int)
                comparison['LT_Mean_old'] = comparison['LT_Mean_old'].fillna(0).round().astype(int)
                comparison['LT_Mean_new'] = comparison['LT_Mean_new'].fillna(0).round().astype(int)
                comparison['Tier_Hops_old'] = comparison['Tier_Hops_old'].fillna(0).astype(int)
                comparison['Tier_Hops_new'] = comparison['Tier_Hops_new'].fillna(0).astype(int)

                always_include = set([new_supplier, current_supplier, location_to_move])
                key_nodes_mask = comparison['Location'].isin(always_include)
                mask = (
                    (comparison['Safety_Stock_old'] != comparison['Safety_Stock_new']) |
                    (comparison['LT_Mean_old'] != comparison['LT_Mean_new']) |
                    (comparison['Tier_Hops_old'] != comparison['Tier_Hops_new']) |
                    (comparison['Service_Level_Node_old'] != comparison['Service_Level_Node_new']) |
                    key_nodes_mask
                )
                changed_rows = comparison[mask].copy()

                gt_before = left['Safety_Stock_old'].sum()
                gt_after = right['Safety_Stock_new'].sum()
                gt_before_usd = gt_before * cost_per_kilo
                gt_after_usd = gt_after * cost_per_kilo
                gt_delta = gt_after - gt_before
                gt_delta_usd = gt_after_usd - gt_before_usd
                gt_pct   = 100 * gt_delta / gt_before if gt_before else float('nan')

                if changed_rows.empty:
                    st.info("No nodes changed their Safety Stock, tier, or lead time due to this reroute.")
                else:
                    col_map = {
                        'Location': 'Node',
                        'Safety_Stock_old': 'SS Before',
                        'Safety_Stock_new': 'SS After',
                        'ΔSS': 'ΔSS units',
                        '%ΔSS': 'ΔSS %',
                        'LT_Mean_old': 'LT Before',
                        'LT_Mean_new': 'LT After',
                        'Tier_Hops_old': 'Hops Before',
                        'Tier_Hops_new': 'Hops After',
                        'Service_Level_Node_old': 'SL Before',
                        'Service_Level_Node_new': 'SL After',
                        'SS_old_usd': 'SS Before (USD)',
                        'SS_new_usd': 'SS After (USD)',
                        'ΔSS_usd': 'ΔSS USD'
                    }
                    disp_df = changed_rows.rename(columns=col_map)
                    def format_int_dot(v):
                        if pd.isna(v): return ""
                        try: return "{:,.0f}".format(int(round(v))).replace(",", ".")
                        except: return str(int(v))
                    def format_usd(v):
                        try: return f"${v:,.0f}".replace(",", ".")
                        except: return ""
                    def ss_delta_color(val):
                        if pd.isna(val): return "black"
                        return "green" if val < 0 else "red" if val > 0 else "black"
                    table_md = "<table style='width:100%;text-align:center;background:#f6faf7;font-size:0.75em;'><tr>"+ "".join(
                        f"<th>{col}</th>" for col in list(col_map.values())
                    ) + "</tr>"
                    for _, row in disp_df.iterrows():
                        table_md += "<tr>"
                        for col in list(col_map.values()):
                            v = row[col]
                            style = ""
                            if col == "ΔSS %":
                                color = ss_delta_color(v)
                                v_display = f"<strong>{v:+.1f}%</strong>" if pd.notnull(v) else ""
                                style = f" style='color:{color}'"
                            elif col == "ΔSS units":
                                color = ss_delta_color(row["ΔSS %"])
                                v_display = f"{format_int_dot(v)}" if pd.notnull(v) else ""
                                style = f" style='color:{color}'"
                            elif col == "ΔSS USD":
                                color = ss_delta_color(row["ΔSS %"])
                                v_display = format_usd(v)
                                style = f" style='color:{color}'"
                            elif col in ("SS Before (USD)", "SS After (USD)"):
                                v_display = format_usd(v)
                            elif col in ("SL Before", "SL After"):
                                v_display = f"{v:.2%}" if pd.notnull(v) else ""
                            elif col in ("LT Before", "LT After"):
                                v_display = format_int_dot(v)
                            elif col in ("SS Before", "SS After"):
                                v_display = format_int_dot(v)
                            elif col in ("Hops Before", "Hops After"):
                                v_display = f"{v:d}" if not pd.isna(v) else "0"
                            else:
                                v_display = str(v)
                            table_md += f"<td{style}>{v_display}</td>"
                        table_md += "</tr>"

                    gt_delta_color = ss_delta_color(gt_pct)
                    table_md += ("<tr style='font-weight:bold;background:#e6f5e4;'>"
                                 "<td>Grand Total</td>"  # Column 1: Node
                                 "<td colspan='2'></td>"  # Columns 2-3: Skip SS Before, SS After
                                 f"<td style='color:{gt_delta_color}'>{format_int_dot(gt_delta)}</td>"  # Column 4: ΔSS units total
                                 f"<td style='color:{gt_delta_color}'><strong>{gt_pct:+.1f}%</strong></td>"  # Column 5: ΔSS % total
                                 "<td colspan='6'></td>"  # Columns 6-11: Skip LT, Hops, SL columns
                                 f"<td>{format_usd(gt_before_usd)}</td>"  # Column 12: SS Before (USD)
                                 f"<td>{format_usd(gt_after_usd)}</td>"   # Column 13: SS After (USD)
                                 f"<td style='color:{gt_delta_color}'>{format_usd(gt_delta_usd)}</td>"   # Column 14: ΔSS USD
                                 "</tr>")
                    table_md += "</table>"

                    # Diagnostic if BEEX's SS doesn't change but demand does
                    beex_old = left[left['Location'] == new_supplier]['Safety_Stock_old'].sum()
                    beex_new = right[right['Location'] == new_supplier]['Safety_Stock_new'].sum()
                    beex_demand_old = left[left['Location'] == new_supplier].get('Agg_Future_Demand', pd.Series([0])).sum()
                    beex_demand_new = right[right['Location'] == new_supplier].get('Agg_Future_Demand', pd.Series([0])).sum()
                    if beex_new == beex_old and beex_demand_new != beex_demand_old:
                        st.warning(f"Note: BEEX's SS did not change, even though demand increased. This is due to capping (cap_range={cap_range}) for SS as a % of demand in your scenario.")

                    st.markdown(
                        f"""
                        <div style="background:#f6faf7; border:1px solid #7fd47c; border-radius:10px; padding:12px 16px; margin:0 0 8px 0;">
                        <b>Result – Each changed node:</b>
                        {table_md}
                        <div style='font-size:0.9em;color:#444;margin-top:6px;'>SS change color: <span style='color:green'>green</span> if SS decreases, <span style='color:red'>red</span> if SS increases.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"<div style='margin-top:7px;font-size:1em;'>{lead_time_note}</div>", unsafe_allow_html=True)
                st.write("You can repeat with different routing choices. To get back to the base plan, reload data or change period.")

        st.markdown(
            """
            <div style="font-size:0.85rem; margin-bottom:4px; color:#555;">
              Only nodes with non‑zero demand or corridor in the selected period are drawn.
              Routes that connect to fully inactive nodes are hidden to keep the view focused.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------- MAP CODE UNCHANGED, always after scenario box --------
        # Prepare label data for network nodes
        # Drop duplicates to ensure unique index for to_dict("index")
        label_data = (
            results[results["Period"] == chosen_period]
            .drop_duplicates(subset=["Product", "Location"], keep="first")
            .set_index(["Product", "Location"])
            .to_dict("index")
        )
        sku_lt = df_lt[df_lt["Product"] == sku] if "Product" in df_lt.columns else df_lt.copy()
        net = Network(
            height="900px",
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
            height:900px !important;
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
        components.html(html_text, height=940)
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

        display_df = hide_zero_rows(
            filtered,
            check_cols=[
                "Safety_Stock",
                "Forecast",
                "Agg_Future_Demand",
                "Agg_Future_Internal",
                "Agg_Future_External",
            ],
        )
        if "Period" in display_df.columns:
            display_df = display_df.copy()
            display_df["Period_Label"] = display_df["Period"].apply(period_label)
        else:
            display_df["Period_Label"] = ""

        col_map = {
            "Product": "Material",
            "Location": "Node",
            "Period_Label": "Month",
            "Forecast": "Fcst [unit]",
            "D_day": "Avg/day [unit]",
            "Safety_Stock": "SS [unit]",
            "Days_Covered_by_SS": "SS Cov [days]",
            "Adjustment_Status": "Status",
            "Agg_Future_Demand": "Net Dem [unit]",
            "Agg_Future_Internal": "Local Dem [unit]",
            "Agg_Future_External": "NW Dem [unit]",
        }
        display_cols = [c for c in [
            "Product", "Location", "Period_Label", "Forecast", "D_day", "Safety_Stock",
            "Days_Covered_by_SS", "Adjustment_Status", "Agg_Future_Demand", "Agg_Future_Internal", "Agg_Future_External"
        ] if c in display_df.columns]
        nice = display_df[display_cols].copy()
        nice = nice.rename(columns=col_map)

        # --- Formatting ---
        def _fmt_int(val):
            try:
                return f"{int(round(val)):,}".replace(",", ".")
            except:
                return ""
        def _fmt_2dec(val):
            try:
                # Use comma as decimal separator for avg/day column
                formatted = f"{val:.2f}"
                # Replace dot with comma for decimal separator
                formatted = formatted.replace(".", ",")
                return formatted
            except:
                return ""
        # header styles for wrapping
        header_props = {'white-space': 'normal', 'word-break': 'break-word', 'font-size': '0.85em'}

        # Add grand total row at the bottom
        if not nice.empty and 'SS [unit]' in nice.columns:
            total_ss = display_df['Safety_Stock'].sum() if pd.api.types.is_numeric_dtype(display_df['Safety_Stock']) else 0
            # Create a total row with empty values except for SS column
            total_row = pd.Series({col: '' for col in nice.columns})
            total_row['SS [unit]'] = total_ss
            nice = pd.concat([nice, pd.DataFrame([total_row])], ignore_index=True)
        
        pandas_fmt = {
            "Fcst [unit]": _fmt_int,
            "Avg/day [unit]": _fmt_2dec,
            "SS [unit]": lambda val: _fmt_int(val) if val != '' and val is not None else '',
            "SS Cov [days]": _fmt_int,
            "Net Dem [unit]": _fmt_int,
            "Local Dem [unit]": _fmt_int,
            "NW Dem [unit]": _fmt_int,
        }
        # Add light red highlighting to SS column (PR #4)
        ss_highlight = {'background-color': '#ffcccc'}
        
        # Define status values and their corresponding background colors
        STATUS_COLORS = {
            'Capped (High)': '#fffacd',      # light yellow
            'Capped (Low)': '#fffacd',       # light yellow
            'Optimal (Statistical)': '#90ee90'  # light green
        }
        
        # Function to get background color style for Status column values
        def get_status_background_color(val):
            val_str = str(val)
            return f'background-color: {STATUS_COLORS[val_str]}' if val_str in STATUS_COLORS else ''
        
        # Function to generate styles for Status column
        def get_status_column_styles(col):
            if col.name == 'Status':
                return col.map(get_status_background_color)
            return [''] * len(col)
        
        # Function to make the Total row bold
        def make_total_bold(row):
            # The total row has empty Material column (after renaming Product to Material)
            if 'Material' in row.index and row['Material'] == '':
                return ['font-weight: bold'] * len(row)
            return [''] * len(row)
        
        styled = (
            nice.style
            .format(pandas_fmt, escape="html")
            .set_properties(**header_props, axis=1)
            .set_properties(**ss_highlight, subset=['SS [unit]'])
            .apply(get_status_column_styles, axis=0)
            .apply(make_total_bold, axis=1)
        )
        st.dataframe(styled, use_container_width=True)

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

        # --- AGGREGATE: SS Coverage = sum of SS over sum of daily avg demand ---
        total_ss_sku = eff["Safety_Stock"].sum()
        total_avg_daily_sku = eff["Forecast"].sum() / days_per_month if days_per_month > 0 else 0
        sku_coverage = total_ss_sku / total_avg_daily_sku if total_avg_daily_sku > 0 else 0

        all_res = get_active_snapshot(results, snapshot_period) if snapshot_period is not None else results
        global_total_ss = all_res["Safety_Stock"].sum()
        global_avg_day = all_res["Forecast"].sum() / days_per_month if days_per_month > 0 else 0
        global_coverage = global_total_ss / global_avg_day if global_avg_day > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Days of Coverage (selection)", f"{int(round(sku_coverage))}")
        m2.metric("Days Coverage (all materials)", f"{int(round(global_coverage))}")
        m3.metric("Total SS", euro_format(int(total_ss_sku), True))
        m4.metric("Total Forecast", euro_format(int(eff['Forecast'].sum()), True))
        st.markdown("---")

        c1, c2 = st.columns([7, 3])
        with c1:
            st.markdown("**Top Nodes by Safety Stock (snapshot)**")
            if not eff.empty:
                eff_top = eff.sort_values("Safety_Stock", ascending=False).head(10)
                if "Period" in eff_top.columns:
                    eff_top = eff_top.copy()
                    eff_top["Period_Label"] = eff_top["Period"].apply(period_label)
                else:
                    eff_top["Period_Label"] = ""
                cols = [
                    "Location", "Period_Label", "Safety_Stock", "Days_Covered_by_SS", "Adjustment_Status", "Forecast", "Agg_Future_Demand"
                ]
                display_cols = [c for c in cols if c in eff_top.columns]
                compact_labels = {
                    "Location":"Node", "Period_Label":"Month", "Safety_Stock":"SS [unit]",
                    "Days_Covered_by_SS":"SS Cov [days]", "Adjustment_Status":"Status",
                    "Forecast":"Fcst [unit]", "Agg_Future_Demand":"Net Dem [unit]"
                }
                eff_top_std = eff_top[display_cols].rename(columns=compact_labels)
                
                # Add grand total row at the bottom
                if not eff_top_std.empty and 'SS [unit]' in eff_top_std.columns:
                    total_ss_top = eff_top['Safety_Stock'].sum() if pd.api.types.is_numeric_dtype(eff_top['Safety_Stock']) else 0
                    # Create a total row with empty values except for SS column
                    total_row = pd.Series({col: '' for col in eff_top_std.columns})
                    total_row['SS [unit]'] = total_ss_top
                    eff_top_std = pd.concat([eff_top_std, pd.DataFrame([total_row])], ignore_index=True)
                
                # Formatting function
                def _fmt_int(val):
                    try:
                        return f"{int(round(val)):,}".replace(",", ".")
                    except:
                        return ""
                
                tbl_fmt = {
                    "SS [unit]": lambda val: _fmt_int(val) if val != '' and val is not None else '',
                    "SS Cov [days]": _fmt_int,
                    "Fcst [unit]": _fmt_int,
                    "Net Dem [unit]": _fmt_int,
                }
                header_style = {'white-space': 'normal', 'word-break': 'break-word', 'font-size': '0.85em'}
                # Add light red highlighting to SS column (PR #5)
                ss_highlight = {'background-color': '#ffcccc'}
                
                # Function to make the Total row bold
                def make_total_bold(row):
                    # The total row has empty Node column (first column)
                    if 'Node' in row.index and row['Node'] == '':
                        return ['font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled = (eff_top_std.style
                         .format(tbl_fmt, escape="html")
                         .set_properties(**header_style, axis=1)
                         .set_properties(**ss_highlight, subset=['SS [unit]'])
                         .apply(make_total_bold, axis=1))
                st.dataframe(styled, use_container_width=True)
            else:
                st.write("No non-zero nodes for this selection.")

        with c2:
            st.markdown("**Status Breakdown**")
            if not eff.empty:
                st.table(eff["Adjustment_Status"].value_counts())
            else:
                st.write("No non-zero nodes for this selection.")
        
        # Policy Rules Recap Box
        st.markdown("---")
        
        # Get current policy settings for display
        zero_rule_status = "✅ Enabled" if zero_if_no_net_fcst else "❌ Disabled"
        cap_rule_status = "✅ Enabled" if apply_cap else "❌ Disabled"
        cap_lower, cap_upper = cap_range if apply_cap else (0, 0)
        
        # Wrap in collapsible expander with reduced font size
        with st.expander("#### 📋 Policy Rules Explanation", expanded=False):
            policy_recap_html = f"""<div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); border: 2px solid #1976d2; border-radius: 12px; padding: 20px; margin-top: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-size: 0.9rem;">
    <div style="font-size: 1.0rem; font-weight: 700; color: #0b3d91; margin-bottom: 15px;">📖 Safety Stock Policy Rules Summary</div>
    <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 12px;">
        <div style="font-size: 0.85rem; font-weight: 600; color: #1565c0; margin-bottom: 8px;">1️⃣ Zero-If-No-Demand Rule: {zero_rule_status}</div>
        <div style="font-size: 0.75rem; color: #424242; line-height: 1.5;">When enabled, nodes with zero or negative aggregated network demand (Agg_Future_Demand ≤ 0) will have their Safety Stock forced to zero. This prevents holding inventory for items with no expected demand.</div>
    </div>
    <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 12px;">
        <div style="font-size: 0.85rem; font-weight: 600; color: #1565c0; margin-bottom: 8px;">2️⃣ Safety Stock Capping Rule: {cap_rule_status}</div>
        <div style="font-size: 0.75rem; color: #424242; line-height: 1.5;">When enabled, Safety Stock is constrained within <strong>{cap_lower}% - {cap_upper}%</strong> of the aggregated network demand. This prevents extreme SS values that are disproportionate to demand levels.</div>
        <ul style="margin: 8px 0 0 20px; padding: 0; font-size: 0.75rem;">
            <li><strong>Capped (Low):</strong> SS below {cap_lower}% of demand is raised to the minimum</li>
            <li><strong>Capped (High):</strong> SS above {cap_upper}% of demand is reduced to the maximum</li>
        </ul>
    </div>
    <div style="background: white; border-radius: 8px; padding: 15px;">
        <div style="font-size: 0.85rem; font-weight: 600; color: #1565c0; margin-bottom: 8px;">3️⃣ Minimum Floor Rule: ✅ Always Applied</div>
        <div style="font-size: 0.75rem; color: #424242; line-height: 1.5;">A minimum safety stock floor is always applied to prevent unrealistically low values:<br/>SS_floor = Daily_Demand × Lead_Time_Mean × 0.01<br/>The final statistical SS is always at least this floor value, ensuring basic coverage.</div>
    </div>
    <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.7); border-radius: 6px; border-left: 4px solid #ff9800;">
        <div style="font-size: 0.7rem; color: #424242; line-height: 1.4;"><strong>Note:</strong> These rules are applied in sequence after the statistical SS calculation. The <em>Adjustment_Status</em> column in the Status Breakdown above shows which rule was applied to each node.</div>
    </div>
</div>"""
            
            st.markdown(policy_recap_html, unsafe_allow_html=True)

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

            st.subheader("Aggregated Network History (Selected Product across all locations)")
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
                    net_table_fmt["Period_Label"] = net_table_fmt["Period"].apply(period_label)
                    cols = [
                        "Period_Label", "Network_Consumption", "Network_Forecast_Hist", "Net_Abs_Error"
                    ]
                    display_cols = [c for c in cols if c in net_table_fmt.columns]
                    table_std = net_table_fmt[display_cols].copy()
                    # Format with dots as thousands separator (European style)
                    def fmt_with_dots(val):
                        try:
                            return f"{int(val):,}".replace(",", ".")
                        except:
                            return str(val)
                    fmt_dict = {
                        "Network_Consumption": fmt_with_dots,
                        "Network_Forecast_Hist": fmt_with_dots,
                        "Net_Abs_Error": fmt_with_dots
                    }
                    def highlight_err(val):
                        try:
                            return "background-color: #ffe7e7" if float(val) > 0 else ""
                        except: return ""
                    styled = table_std.style.format(fmt_dict).applymap(highlight_err, subset=["Net_Abs_Error"])
                    st.dataframe(styled, use_container_width=True)
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
            avail_locs = sorted(results[results["Product"] == calc_sku]["Location"].unique().tolist())
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
        
        # Cost per kilo slider
        cost_per_kilo_tab6 = st.slider(
            "Cost per kilo (USD)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.10,
            help="USD cost per kilo for scenario simulation",
            key="sim_costperkilo"
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
            node_sl = float(row.get("Service_Level_Node", service_level))
            node_z = float(row.get("Z_node", norm.ppf(node_sl)))
            hops = int(row.get("Tier_Hops", 0))
            
            # Render the formula explainer with hop logic table embedded
            render_ss_formula_explainer(hops=hops, service_level=service_level)
            
            # Display calculation values in a single row
            st.markdown("---")
            st.markdown("**Values used for the calculation:**")
            avg_daily = row.get("D_day", np.nan)
            days_cov = row.get("Days_Covered_by_SS", np.nan)
            avg_daily_txt = f"{avg_daily:.2f}" if pd.notna(avg_daily) else "N/A"
            days_cov_txt = f"{days_cov:.1f}" if pd.notna(days_cov) else "N/A"
            
            # Display all values in a single row using smaller flex items
            summary_html = f"""
            <div style="display:flex;flex-wrap:nowrap;gap:8px;margin-top:10px;overflow-x:auto;" tabindex="0">
              <div style="flex:1;min-width:120px;background:#e8f0ff;border-radius:8px;padding:10px;">
                <div style="font-size:11px;color:#0b3d91;font-weight:600;">Node Service Level</div>
                <div style="font-size:18px;font-weight:800;color:#0b3d91;">{node_sl*100:.2f}%</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(hops={hops})</div>
              </div>
              <div style="flex:1;min-width:110px;background:#fff3e0;border-radius:8px;padding:10px;min-height:70px;">
                <div style="font-size:11px;color:#a64d00;font-weight:600;">Z-Score</div>
                <div style="font-size:18px;font-weight:800;color:#a64d00;">{node_z:.4f}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(std devs)</div>
              </div>
              <div style="flex:1;min-width:120px;background:#e8f8f0;border-radius:8px;padding:10px;">
                <div style="font-size:11px;color:#00695c;font-weight:600;">Net Demand</div>
                <div style="font-size:18px;font-weight:800;color:#00695c;">{euro_format(row['Agg_Future_Demand'], True)}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(units/month)</div>
              </div>
              <div style="flex:1;min-width:120px;background:#fbeff2;border-radius:8px;padding:10px;">
                <div style="font-size:11px;color:#880e4f;font-weight:600;">Net StdDev</div>
                <div style="font-size:18px;font-weight:800;color:#880e4f;">{euro_format(row['Agg_Std_Hist'], True)}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(units/month)</div>
              </div>
              <div style="flex:1;min-width:110px;background:#f0f4c3;border-radius:8px;padding:10px;">
                <div style="font-size:11px;color:#827717;font-weight:600;">Avg Lead Time</div>
                <div style="font-size:18px;font-weight:800;color:#827717;">{row['LT_Mean']}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(days)</div>
              </div>
              <div style="flex:1;min-width:110px;background:#e1f5fe;border-radius:8px;padding:10px;">
                <div style="font-size:11px;color:#01579b;font-weight:600;">LT StdDev</div>
                <div style="font-size:18px;font-weight:800;color:#01579b;">{row['LT_Std']}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(days)</div>
              </div>
              <div style="flex:1;min-width:110px;background:#ffffff;border-radius:8px;padding:10px;border:1px solid #ddd;">
                <div style="font-size:11px;color:#333;font-weight:600;">Daily Demand</div>
                <div style="font-size:18px;font-weight:800;color:#333;">{avg_daily_txt}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(units/day)</div>
              </div>
              <div style="flex:1;min-width:110px;background:#ffffff;border-radius:8px;padding:10px;border:1px solid #ddd;">
                <div style="font-size:11px;color:#333;font-weight:600;">SS Coverage</div>
                <div style="font-size:18px;font-weight:800;color:#333;">{days_cov_txt}</div>
                <div style="font-size:10px;color:#444;margin-top:3px;">(days)</div>
              </div>
            </div>
            """
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
                    color:#0b3d91;
                    font-weight:500;">
                  <div style="font-size:1.00rem;"><b>SCENARIO PLANNING TOOL</b></div>
                  <div style="height:8px;"></div>
                  <div style="font-size:0.85rem;">Simulate alternative end-node SL / LT assumptions (<b>analysis‑only</b>), but using the same policy rules as the implemented plan (<b>zero-if-no demand</b>, capping, overrides).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Show detailed scenario controls", expanded=False):
                
                # Hop 0 uses the service_level from the sidebar (top left corner slider)
                hop0_sl_tab6 = service_level * 100  # Convert to percentage
                
                # Use hop service levels from the Scenario Input section above
                endnode_sl_tab6 = hop0_sl_tab6  # Hop 0 uses service_level from sidebar

                base_sl_node = float(row["Service_Level_Node"])
                base_z_node = float(row["Z_node"])
                base_LT_mean = float(row["LT_Mean"])
                base_LT_std = float(row["LT_Std"])
                base_agg_demand = float(row["Agg_Future_Demand"])
                base_agg_std = float(row["Agg_Std_Hist"])

                base_sigma_d_day = base_agg_std / math.sqrt(float(days_per_month))
                base_d_day = base_agg_demand / float(days_per_month)
                base_var_d_day = base_sigma_d_day**2
                if base_agg_demand < 20.0:
                    base_var_d_day = max(base_var_d_day, base_d_day)

                base_demand_component = base_var_d_day * base_LT_mean
                base_lt_component = base_LT_std**2 * (base_d_day**2)
                base_combined_var = max(base_demand_component + base_lt_component, 0.0)
                base_ss_stat = base_z_node * math.sqrt(base_combined_var)

                base_floor = base_d_day * base_LT_mean * 0.01
                base_pre_rule_ss = max(base_ss_stat, base_floor)

                base_ss_policy = apply_policy_to_scalar_ss(
                    ss_value=base_pre_rule_ss,
                    agg_future_demand=base_agg_demand,
                    location=str(row["Location"]),
                    zero_if_no_net_fcst=zero_if_no_net_fcst,
                    apply_cap=apply_cap,
                    cap_range=cap_range,
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
                    "Number of additional Scenarios to compare (besides 'Base-calibrated')",
                    options,
                    index=default_index,
                    key="n_scen",
                )
                
                scenarios = []
                
                # Render scenario expanders
                for s in range(n_scen):
                    with st.expander(f"Scenario {s+1} inputs", expanded=False):
                        # Add hop service level configuration to the first scenario
                        if s == 0:
                            st.markdown("### 📥 Scenario Input")
                            st.markdown("Configure the hop service levels for scenario planning:")
                            
                            # Show frozen message for upstream hops (hops < current level)
                            if hops > 0:
                                frozen_hops = list(range(0, hops))
                                frozen_msg = ", ".join([f"Hop {h}" for h in frozen_hops])
                                st.markdown(f"_**Frozen (upstream):** {frozen_msg}_")
                            
                            # Show slider for Hop 0 only if the node is Hop 0
                            if hops == 0:
                                st.slider(
                                    "Hop 0 (End-nodes) SL (%)",
                                    min_value=50.0,
                                    max_value=99.9,
                                    value=hop0_sl_tab6,
                                    step=0.1,
                                    help="Service Level for Hop 0 (End-nodes) in scenario simulations",
                                    key="hop0_sl_tab6"
                                )
                            
                            # Show Hop 1 slider only if current node is Hop 0 or Hop 1
                            if hops <= 1:
                                hop1_sl_tab6 = st.slider(
                                    "Hop 1 SL (%)",
                                    min_value=50.0,
                                    max_value=99.9,
                                    value=95.0,
                                    step=0.1,
                                    help="Service Level for Hop 1 nodes in scenario simulations",
                                    key="hop1_sl_tab6"
                                )
                            
                            # Show Hop 2 slider only if current node is Hop 0, 1, or 2
                            if hops <= 2:
                                hop2_sl_tab6 = st.slider(
                                    "Hop 2 SL (%)",
                                    min_value=50.0,
                                    max_value=99.9,
                                    value=90.0,
                                    step=0.1,
                                    help="Service Level for Hop 2 nodes in scenario simulations",
                                    key="hop2_sl_tab6"
                                )
                            
                            # Show Hop 3 slider only if current node is Hop 0, 1, 2, or 3
                            if hops <= 3:
                                hop3_sl_tab6 = st.slider(
                                    "Hop 3 SL (%)",
                                    min_value=50.0,
                                    max_value=99.9,
                                    value=85.0,
                                    step=0.1,
                                    help="Service Level for Hop 3 nodes in scenario simulations",
                                    key="hop3_sl_tab6"
                                )
                            
                            st.markdown("---")
                        
                        # Get hop SL values from session state (set by sliders in scenario 1)
                        # Use default values as fallback
                        # Note: hop0_sl_tab6_value from session state is only used when hops==0 in hop_sl_map
                        hop0_sl_tab6_value = st.session_state.get("hop0_sl_tab6", hop0_sl_tab6)
                        hop1_sl_tab6 = st.session_state.get("hop1_sl_tab6", 95.0)
                        hop2_sl_tab6 = st.session_state.get("hop2_sl_tab6", 90.0)
                        hop3_sl_tab6 = st.session_state.get("hop3_sl_tab6", 85.0)
                        
                        # Use tab6 hop SL sliders to determine SL based on node's hop tier
                        # Map hop tier to the appropriate SL from Tab6
                        hop_sl_map = {
                            0: hop0_sl_tab6_value,  # End-nodes (Hop 0) - from slider if available (when viewing Hop 0 node), otherwise from sidebar
                            1: hop1_sl_tab6,     # Hop 1 - from Tab6 calculations
                            2: hop2_sl_tab6,     # Hop 2 - from Tab6 calculations
                            3: hop3_sl_tab6,     # Hop 3 - from Tab6 calculations
                        }
                        
                        # Determine the Service Level based on the node's hop tier
                        sc_sl = hop_sl_map.get(hops, base_sl_node * 100.0)
                        
                        st.markdown(
                            f"""
                            <div style="font-size:0.85rem; margin-bottom:8px;">
                              <strong>Service Level for this node:</strong> {sc_sl:.2f}% (Hop {hops})<br/>
                              <em>Service Levels are automatically calculated from the primary SL slider in the sidebar based on tiering ratios.</em>
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

                sc_ss_raw = sc_z * math.sqrt(
                    var_d * sc["LT_mean"] + (sc["LT_std"] ** 2) * (d_day**2)
                )
                sc_floor = d_day * sc["LT_mean"] * 0.01
                sc_ss_raw = max(sc_ss_raw, sc_floor)

                sc_ss = apply_policy_to_scalar_ss(
                    ss_value=sc_ss_raw,
                    agg_future_demand=float(row["Agg_Future_Demand"]),
                    location=str(row["Location"]),
                    zero_if_no_net_fcst=zero_if_no_net_fcst,
                    apply_cap=apply_cap,
                    cap_range=cap_range,
                )
                sc_ss_usd = sc_ss * cost_per_kilo_tab6

                scen_rows.append(
                    {
                        "Scenario": f"S{idx+1}",
                        "Service_Level_%": sc["SL_pct"],
                        "LT_mean_days": sc["LT_mean"],
                        "LT_std_days": sc["LT_std"],
                        "Simulated_SS": sc_ss,
                        "Simulated_SS_USD": sc_ss_usd,
                    }
                )
            base_ss_policy_usd = base_ss_policy * cost_per_kilo_tab6
            base_calibrated_row = {
                "Scenario": "Base-calibrated",
                "Service_Level_%": base_sl_node * 100.0,
                "LT_mean_days": base_LT_mean,
                "LT_std_days": base_LT_std,
                "Simulated_SS": base_ss_policy,
                "Simulated_SS_USD": base_ss_policy_usd,
            }
            impl_row = {
                "Scenario": "Implemented",
                "Service_Level_%": np.nan,
                "LT_mean_days": np.nan,
                "LT_std_days": np.nan,
                "Simulated_SS": float(row["Safety_Stock"]),
                "Simulated_SS_USD": float(row["Safety_Stock"]) * cost_per_kilo_tab6,
            }
            compare_df = pd.concat(
                [pd.DataFrame([base_calibrated_row, impl_row]), pd.DataFrame(scen_rows)],
                ignore_index=True,
                sort=False,
            )
            display_comp = compare_df.copy()
            display_comp["Simulated_SS"] = display_comp["Simulated_SS"].astype(float)
            display_comp["Simulated_SS_USD"] = display_comp["Simulated_SS_USD"].astype(float)

            implemented_ss = float(row["Safety_Stock"])
            implemented_ss_usd = implemented_ss * cost_per_kilo_tab6
            
            def pct_vs_impl(v):
                try:
                    if implemented_ss <= 0 or pd.isna(v):
                        return np.nan
                    return (float(v) / implemented_ss - 1.0) * 100.0
                except Exception:
                    return np.nan
            
            def delta_usd_vs_impl(v):
                try:
                    if pd.isna(v):
                        return np.nan
                    return float(v) - implemented_ss_usd
                except Exception:
                    return np.nan
            
            display_comp["% vs Implemented"] = display_comp["Simulated_SS"].apply(pct_vs_impl)
            display_comp["Delta USD"] = display_comp["Simulated_SS_USD"].apply(delta_usd_vs_impl)

            # Rename columns to shorter versions to reduce table width
            display_comp_renamed = display_comp.rename(columns={
                "Service_Level_%": "SL%",
                "LT_mean_days": "LT μ",
                "LT_std_days": "LT σ",
                "Simulated_SS": "SS (units)",
                "Simulated_SS_USD": "SS (USD)",
                "% vs Implemented": "% vs Impl",
                "Delta USD": "Δ USD"
            })

            # ----------- Pretty pandas styling for DataFrame -----------
            def highlight_pct(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val < 0 else 'red' if val > 0 else 'black'
                return f'color: {color}; font-weight: bold;'
            
            def highlight_delta_usd(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val < 0 else 'red' if val > 0 else 'black'
                return f'color: {color}; font-weight: bold;'
            
            styled = (
                display_comp_renamed
                .style
                .format({
                    "SL%": "{:.2f}",
                    "LT μ": "{:.2f}",
                    "LT σ": "{:.2f}",
                    "SS (units)": "{:,.0f}",
                    "SS (USD)": "${:,.0f}",
                    "% vs Impl": "{:+.2f}%",
                    "Δ USD": "${:+,.0f}"
                })
                .applymap(highlight_pct, subset=['% vs Impl'])
                .applymap(highlight_delta_usd, subset=['Δ USD'])
                .set_properties(**{'background-color': '#f7fafd', 'font-size': '0.85rem'}, subset=pd.IndexSlice[:, :])
            )
            
            # Put scenario comparison table and graph into a collapsible expander
            with st.expander("📊 View Scenario Comparison Table & Visualization", expanded=False):
                st.subheader("Scenario Comparison Table")
                st.dataframe(styled, use_container_width=True)

                # Add column chart to visualize scenario impacts (PR #6)
                st.markdown("---")
                st.subheader("Scenario Impact Visualization")
                
                # Define unique light pastel colors for each scenario
                scenario_colors = {
                    "Base-calibrated": "#B3E5FC",  # Light pastel blue
                    "Implemented": "#FFF9C4",      # Light pastel yellow
                    "S1": "#C8E6C9",               # Light pastel green
                    "S2": "#FFCCBC",               # Light pastel orange
                    "S3": "#E1BEE7"                # Light pastel purple
                }
                
                # Assign colors based on scenario names
                bar_colors = [scenario_colors.get(scenario, "#E0E0E0") for scenario in display_comp["Scenario"]]
                
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Bar(
                    x=display_comp["Scenario"],
                    y=display_comp["Simulated_SS_USD"],
                    text=[f"${v:,.0f}" for v in display_comp["Simulated_SS_USD"]],
                    textposition='auto',
                    marker=dict(
                        color=bar_colors
                    )
                ))
                fig_sim.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Simulated SS (USD)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_sim, use_container_width=True)

                st.markdown(
                    """
                    <small>
                    <ul>
                      <li><b>Implemented</b> = pipeline SS after all rules.</li>
                      <li><b>Base-calibrated</b> = scenario rebuilt from node's own SL / LT / demand, then run through the same policy engine. Should match Implemented.</li>
                      <li><b>S1, S2, S3</b> = user scenarios</li>
                      <li><b>% vs Implemented</b> highlights decreases in green, increases in red.</li>
                    </ul>
                    </small>
                    """,
                    unsafe_allow_html=True,
                )
    
# ---- TAB 7: Remove first table. Correct SS coverage KPI calculation. ----
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

        mat_period_export = get_active_snapshot(
            results,
            selected_period if selected_period is not None else default_period,
        )
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
        render_selection_line(
            "Selected:",
            product=selected_product,
            period_text=period_label(selected_period),
        )
        st.subheader("📦 View by Material (+ SS attribution)")
        render_tab7_explainer()

        mat_period_df = get_active_snapshot(
            results,
            selected_period if selected_period is not None else default_period,
        )
        mat_period_df = mat_period_df[
            (mat_period_df["Product"] == selected_product)
        ].copy()

        if mat_period_df.empty:
            st.warning("No data for this material/period.")
        else:
            mat_period_df_display = hide_zero_rows(mat_period_df)

            st.markdown("---")
            st.markdown("### Why do we carry this SS? — SS attribution")

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

            # >>>>>>>>>>> ADD THIS DEFINITION <<<<<<<<<<<<<<<<<<<
            total_ss = per_node["Safety_Stock"].sum()
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

            # Simple table for attribution breakdown (normal table)
            attrib_display = ss_drv_df_display.rename(
                columns={
                    "driver": "Driver",
                    "amount": "Units",
                    "pct_of_total_ss": "Pct_of_total_SS",
                }
            )

            attrib_display_fmt = attrib_display.copy()
            attrib_display_fmt["Units"] = attrib_display_fmt["Units"].apply(
                lambda v: euro_format(v, False, True)
            )
            attrib_display_fmt["Pct_of_total_SS"] = attrib_display_fmt[
                "Pct_of_total_SS"
            ].apply(lambda v: f"{v:.1f}%")

            st.markdown("#### SS attribution breakdown (table)")
            st.dataframe(attrib_display_fmt, use_container_width=True)
            
            # Horizontal line marker
            st.markdown("---")
            
            st.plotly_chart(fig_drv, use_container_width=True)

            # Exec takeaway (kept)
            try:
                top3 = (
                    ss_drv_df_display.sort_values(
                        "pct_of_total_ss", ascending=False
                    ).head(3).copy()
                )
                total_top3 = top3["pct_of_total_ss"].sum()

                pieces = []
                for _, r in top3.iterrows():
                    if total_top3 > 0:
                        norm_pct = r["pct_of_total_ss"] / total_top3 * 100.0
                    else:
                        norm_pct = 0.0
                    pieces.append(
                        f"{r['driver']} (<strong>{norm_pct:.1f}%</strong>)"
                    )

                if pieces:
                    takeaway = (
                        f"For <strong>{selected_product}</strong> in "
                        f"<strong>{period_label(selected_period)}</strong>, "
                        f"safety stock is mainly explained by: " + "; ".join(pieces) + "."
                    )
                    st.markdown(
                        f"""
                        <div style="margin-top:8px;padding:8px 10px;border-radius:8px;
                            background:#f5f9ff;border:1px solid #c5cae9;font-size:0.95rem;">
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

        # --- Period selector as before ---
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
            chosen_label_all = st.selectbox(
                "PERIOD",
                period_labels,
                index=sel_period_index,
                key="allmat_period",
            )
            selected_period_all = period_label_map.get(
                chosen_label_all, default_period
            )
        else:
            selected_period_all = CURRENT_MONTH_TS

        # --- Choose alarm threshold for SS coverage in days ---
        alarm_threshold = st.slider(
            "Coverage Days Alarm Threshold",
            min_value=1, max_value=120, value=30,
            help="Cards with coverage below this will be shown in red, otherwise in green."
        )

        # --- Aggregate data exactly as before ---
        snapshot_all = get_active_snapshot(
            results,
            selected_period_all if selected_period_all is not None else default_period,
        )

        agg_all = snapshot_all.groupby("Product", as_index=False).agg(
            Network_Demand_Month=("Agg_Future_Demand", "sum"),
            Local_Forecast_Month=("Forecast", "sum"),
            Safety_Stock=("Safety_Stock", "sum"),
            Max_Corridor=("Max_Corridor", "sum"),
            Avg_Day_Demand=("Forecast", lambda x: x.sum() / days_per_month if days_per_month > 0 else 0),
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

        # Only rows with SS>0
        agg_all = agg_all[agg_all["Safety_Stock"] > 0].copy()
        agg_all["Reorder_Point"] = (
            agg_all["Safety_Stock"] + agg_all["Local_Forecast_Month"]
        )

        # Proper total SS coverage in days per material
        agg_all["SS_Coverage_Days"] = agg_all.apply(
            lambda r: r["Safety_Stock"] / r["Avg_Day_Demand"] if r["Avg_Day_Demand"] > 0 else 0,
            axis=1
        )

        agg_all["SS_to_Demand_Ratio_%"] = (
            agg_all["Safety_Stock"] / agg_all["Network_Demand_Month"] * 100.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Formatting functions
        def fmt_int(v):
            try: return f"{int(round(v)):,}".replace(",", ".")
            except: return ""
        def fmt_2d(v):
            try: return f"{v:,.2f}".replace(",", ".")
            except: return ""
        def color_cov(days):
            try:
                val = float(days)
                return "red" if val < alarm_threshold else "green"
            except: return "grey"

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
        render_selection_line(
            "Selected:", period_text=period_label(selected_period_all)
        )
        st.subheader("📊 All Materials View")

        st.markdown(
            """
            <div style="font-size:0.86rem; color:#555; margin-bottom:6px;">
              One card per material for the selected month. Values are aggregated across all
              <strong>active</strong> locations.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display the table/cards FIRST
        if agg_all.empty:
            st.warning("No data available for the selected period.")
        else:
            # Sort by Avg Daily Demand descending (as requested)
            agg_view = agg_all.sort_values(
                "Avg_Day_Demand", ascending=False
            ).reset_index(drop=True)
            st.markdown(
                """
                <style>
                  .mat-strip-container { width: 100%; overflow-x: auto; padding-bottom: 6px;}
                  .mat-cards-row { display: flex; flex-direction: column; gap: 8px; min-width: 650px;}
                  .mat-card {
                    display: grid;
                    grid-template-columns: 150px repeat(4, minmax(110px, 1fr));
                    align-items: stretch;
                    border-radius: 14px;
                    padding: 6px 10px;
                    background: linear-gradient(90deg,#f8fbff,#f5fff9);
                    box-shadow: 0 2px 4px rgba(15,23,42,0.10);
                  }
                  .mat-card-col-header {
                    font-size: 0.70rem; font-weight: 600; color: #616161;
                    text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 2px;
                  }
                  .mat-card-col-value {
                    font-size: 0.95rem; font-weight: 700; color: #111827;
                  }
                  .mat-product-badge {
                    display: flex; align-items: center; justify-content: flex-start;
                    padding: 6px 10px; border-radius: 999px; font-weight: 800; font-size: 0.92rem;
                    color: #424242; white-space: nowrap; background: #f5f5f5; border: 1px solid #e0e0e0;
                  }
                  .mat-product-chevron { margin-right: 6px; font-size: 0.80rem; color: #757575;}
                  .mat-pill-red {
                    display: inline-flex; align-items: center; justify-content: center;
                    min-width: 54px; padding: 2px 8px; border-radius: 999px;
                    font-size: 0.85rem; font-weight: 700; color: #fff;
                    background: linear-gradient(90deg,#e53935,#ef5350);
                  }
                  .mat-pill-green {
                    display: inline-flex; align-items: center; justify-content: center;
                    min-width: 54px; padding: 2px 8px; border-radius: 999px;
                    font-size: 0.85rem; font-weight: 700; color: #fff;
                    background: linear-gradient(90deg,#43a047,#66bb6a);}
                  .mat-pill-grey {
                    display: inline-flex; align-items: center; justify-content: center;
                    min-width: 54px; padding: 2px 8px; border-radius: 999px;
                    font-size: 0.85rem; font-weight: 700; color: #424242;
                    background: #eeeeee; border: 1px solid #e0e0e0;
                  }
                </style>
                """,
                unsafe_allow_html=True,
            )
            # Build HTML for all cards
            cards_html_parts = [
                '<div class="mat-strip-container"><div class="mat-cards-row">'
            ]
            for _, r in agg_view.iterrows():
                prod = str(r["Product"])
                avg_daily = r["Avg_Day_Demand"]
                ss_units = r["Safety_Stock"]
                ss_cov = r["SS_Coverage_Days"]
                local_fc = r["Local_Forecast_Month"]
                avg_daily_txt = fmt_2d(avg_daily)
                ss_txt = fmt_int(ss_units)
                cov_txt = fmt_int(ss_cov)
                fc_txt = fmt_int(local_fc)
                pill_cov_cls = 'mat-pill-red' if ss_cov < alarm_threshold else 'mat-pill-green'
                card_html = (
                    '<div class="mat-card">'
                    '<div style="display:flex;align-items:center;">'
                    '<div class="mat-product-badge">'
                    '<span class="mat-product-chevron">▶</span>'
                    f"{prod}"
                    "</div>"
                    "</div>"
                    '<div>'
                    '<div class="mat-card-col-header">Avg Daily Demand</div>'
                    f'<div class="mat-card-col-value">{avg_daily_txt}</div>'
                    "</div>"
                    '<div>'
                    '<div class="mat-card-col-header">Safety Stock</div>'
                    f'<div class="mat-card-col-value">{ss_txt}</div>'
                    "</div>"
                    '<div>'
                    '<div class="mat-card-col-header">SS Coverage (days)</div>'
                    '<div class="mat-card-col-value">'
                    f'<span class="{pill_cov_cls}">{cov_txt}</span>'
                    "</div>"
                    "</div>"
                    '<div>'
                    '<div class="mat-card-col-header">Local Forecast (month)</div>'
                    '<div class="mat-card-col-value">'
                    f'<span class="mat-pill-grey">{fc_txt}</span>'
                    "</div>"
                    "</div>"
                    "</div>"
                )
                cards_html_parts.append(card_html)
            cards_html_parts.append("</div></div>")
            final_html = "".join(cards_html_parts)
            st.markdown(final_html, unsafe_allow_html=True)
            
        # Now add graph showing coverage vs threshold (AFTER the table)
        if not agg_all.empty:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            # Sort by SS_Coverage_Days for better visualization
            agg_graph = agg_all.sort_values("SS_Coverage_Days", ascending=True).copy()
            
            # Create the bar chart
            fig_coverage = go.Figure()
            
            # Add bars for coverage with light pastel blue color
            fig_coverage.add_trace(go.Bar(
                x=agg_graph["Product"],
                y=agg_graph["SS_Coverage_Days"],
                name="Safety Stock Coverage",
                marker_color="rgba(173, 216, 230, 0.7)",  # Light blue pastel color
                hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f} days<extra></extra>"
            ))
            
            # Add threshold line with light pastel red color
            fig_coverage.add_trace(go.Scatter(
                x=agg_graph["Product"],
                y=[alarm_threshold] * len(agg_graph),
                mode="lines",
                name="Coverage Threshold",
                line=dict(color="rgba(255, 182, 193, 0.9)", width=3, dash="dash"),  # Light red pastel color
                hovertemplate=f"<b>Threshold</b><br>{alarm_threshold} days<extra></extra>"
            ))
            
            # Update layout with clean design
            fig_coverage.update_layout(
                title=dict(
                    text="Coverage Position vs Threshold",
                    font=dict(size=16, color="#424242"),
                    x=0.5,
                    xanchor="center"
                ),
                xaxis_title="Material",
                yaxis_title="Coverage (days)",
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor="rgba(248, 251, 255, 0.5)",
                paper_bgcolor="white",
                xaxis=dict(
                    showgrid=False,
                    tickangle=-45
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(200, 200, 200, 0.2)"
                ),
                hovermode="closest"
            )
            
            st.plotly_chart(fig_coverage, use_container_width=True)
        # END of else for agg_all.empty in tab8

# END OF MAIN CODE
# Now, handle no-data fallback if main data was not uploaded
if not (s_file and d_file and lt_file):
    st.info(
        "Please upload sales.csv, demand.csv and leadtime.csv in the sidebar to run the optimizer."
    )
