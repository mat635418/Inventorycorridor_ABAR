# ============================================================
# MEIO for Raw Materials — Excel-Compatible Version
# Option 1: Non-recursive demand, no LT variance
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import math
import os
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="MEIO for Raw Materials", layout="wide")

# ============================================================
# HEADER WITH OPEN-SOURCE LOGO (MIT LICENSED SVG)
# ============================================================
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
  <svg width="64" height="64" viewBox="0 0 64 64" fill="none"
       xmlns="http://www.w3.org/2000/svg">
    <circle cx="32" cy="32" r="28" stroke="#0b3d91" stroke-width="6"/>
    <circle cx="32" cy="32" r="14" stroke="#0b3d91" stroke-width="4"/>
    <circle cx="32" cy="32" r="3" fill="#0b3d91"/>
  </svg>
  <div>
    <div style="font-size:28px;font-weight:700;color:#0b3d91;">
      MEIO for Raw Materials
    </div>
    <div style="font-size:14px;color:#555;">
      Excel-Compatible Mode — Network-Aware but Policy-Aligned
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "🧮 **Excel-Compatibility Mode active**  \n"
    "- Network demand = Local + Direct Children only  \n"
    "- Lead-Time variance disabled  \n"
    "- Designed to align with Final Corridor Calculation logic"
)

# ============================================================
# HELPERS
# ============================================================
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '')
        .str.replace(' ', '')
        .replace(['', '-', 'na', 'n/a'], np.nan),
        errors='coerce'
    )

def euro(x):
    if pd.isna(x):
        return ""
    return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Parameters")

service_level = st.sidebar.slider("Service Level (%)", 50.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

days_per_month = st.sidebar.number_input(
    "Days per month",
    value=30,
    min_value=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Input Data")

s_file = st.sidebar.file_uploader("Sales (historical)", type="csv")
d_file = st.sidebar.file_uploader("Demand (future)", type="csv")
lt_file = st.sidebar.file_uploader("Lead Times", type="csv")

# ============================================================
# EXCEL-COMPATIBLE NETWORK AGGREGATION
# ============================================================
def aggregate_network_stats_excel(df_forecast, df_stats, df_lt):
    """
    Excel-compatible logic:
    - Local forecast
    - + Direct children only
    - No recursion
    - Variance aggregated once
    """
    out = []

    for _, r in df_forecast.iterrows():
        prod = r['Product']
        loc = r['Location']
        per = r['Period']

        local_fcst = r['Forecast']

        children = df_lt[
            (df_lt['Product'] == prod) &
            (df_lt['From_Location'] == loc)
        ]['To_Location'].unique()

        child_fcst = df_forecast[
            (df_forecast['Product'] == prod) &
            (df_forecast['Period'] == per) &
            (df_forecast['Location'].isin(children))
        ]['Forecast'].sum()

        agg_demand = local_fcst + child_fcst

        local_std = df_stats.loc[
            (df_stats['Product'] == prod) &
            (df_stats['Location'] == loc),
            'Local_Std'
        ]

        local_var = local_std.iloc[0]**2 if not local_std.empty else 0

        child_vars = []
        for c in children:
            c_std = df_stats.loc[
                (df_stats['Product'] == prod) &
                (df_stats['Location'] == c),
                'Local_Std'
            ]
            if not c_std.empty:
                child_vars.append(c_std.iloc[0]**2)

        agg_std = np.sqrt(local_var + sum(child_vars))

        out.append({
            'Product': prod,
            'Location': loc,
            'Period': per,
            'Agg_Future_Demand': agg_demand,
            'Agg_Std_Hist': agg_std
        })

    return pd.DataFrame(out)

# ============================================================
# MAIN
# ============================================================
if s_file and d_file and lt_file:

    df_s = pd.read_csv(s_file)
    df_d = pd.read_csv(d_file)
    df_lt = pd.read_csv(lt_file)

    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]

    df_s['Period'] = pd.to_datetime(df_s['Period']).dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = pd.to_datetime(df_d['Period']).dt.to_period('M').dt.to_timestamp()

    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])

    stats = (
        df_s
        .groupby(['Product', 'Location'])['Consumption']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'std': 'Local_Std'})
    )

    stats['Local_Std'] = stats['Local_Std'].fillna(
        stats.groupby('Product')['Local_Std'].transform('median')
    )

    network = aggregate_network_stats_excel(df_d, stats, df_lt)

    lt = (
        df_lt
        .groupby(['Product', 'To_Location'])['Lead_Time_Days']
        .mean()
        .reset_index()
        .rename(columns={'To_Location': 'Location', 'Lead_Time_Days': 'LT_Mean'})
    )

    res = (
        network
        .merge(df_d, on=['Product', 'Location', 'Period'], how='left')
        .merge(lt, on=['Product', 'Location'], how='left')
        .fillna({'Forecast': 0, 'LT_Mean': 7})
    )

    # ========================================================
    # SAFETY STOCK — EXCEL COMPATIBLE (NO LT VARIANCE)
    # ========================================================
    res['Safety_Stock'] = z * np.sqrt(
        (res['Agg_Std_Hist']**2 / days_per_month) * res['LT_Mean']
    )

    res['Safety_Stock'] = res['Safety_Stock'].round(0)
    res['Max_Corridor'] = res['Forecast'] + res['Safety_Stock']

    # ========================================================
    # UI
    # ========================================================
    st.subheader("📈 Inventory Corridor")

    sku = st.selectbox("Material", sorted(res['Product'].unique()))
    loc = st.selectbox(
        "Location",
        sorted(res[res['Product'] == sku]['Location'].unique())
    )

    plot_df = res[
        (res['Product'] == sku) &
        (res['Location'] == loc)
    ].sort_values('Period')

    fig = go.Figure()
    fig.add_scatter(
        x=plot_df['Period'],
        y=plot_df['Max_Corridor'],
        name="Max Corridor",
        line=dict(color='rgba(0,0,0,0.15)')
    )
    fig.add_scatter(
        x=plot_df['Period'],
        y=plot_df['Safety_Stock'],
        name="Safety Stock",
        fill='tonexty',
        fillcolor='rgba(0,176,246,0.25)'
    )
    fig.add_scatter(
        x=plot_df['Period'],
        y=plot_df['Forecast'],
        name="Local Forecast",
        line=dict(color='black', dash='dot')
    )
    fig.add_scatter(
        x=plot_df['Period'],
        y=plot_df['Agg_Future_Demand'],
        name="Network Demand",
        line=dict(color='blue', dash='dash')
    )

    fig.update_layout(
        legend=dict(orientation="h"),
        xaxis_title="Period",
        yaxis_title="Units"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Key Figures")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Forecast", euro(plot_df['Forecast'].sum()))
    c2.metric("Total Network Demand", euro(plot_df['Agg_Future_Demand'].sum()))
    c3.metric("Total Safety Stock", euro(plot_df['Safety_Stock'].sum()))

else:
    st.warning("Please upload Sales, Demand and Lead Time CSV files.")
