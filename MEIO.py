import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer", layout="wide")
st.title("📊 Multi-Echelon Network Inventory Optimizer")

# -------------------------------
# HELPERS
# -------------------------------
def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', '').str.replace('-', '0').str.strip(), errors='coerce').fillna(0)

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    results = []
    months = df_forecast['Period'].unique()
    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in df_forecast['Product'].unique():
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]

            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(set(p_lt['From_Location'])).union(set(p_lt['To_Location']))
            if not nodes:
                continue

            agg_demand = {n: p_fore.get(n, {'Forecast': 0})['Forecast'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': 0})['Local_Std'])**2 for n in nodes}

            children = {}
            for _, row in p_lt.iterrows():
                children.setdefault(row['From_Location'], []).append(row['To_Location'])

            for _ in range(15):
                changed = False
                for parent in nodes:
                    if parent in children:
                        new_d = p_fore.get(parent, {'Forecast': 0})['Forecast'] + sum(agg_demand.get(c, 0) for c in children[parent])
                        new_v = (p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(agg_var.get(c, 0) for c in children[parent])
                        if abs(new_d - agg_demand[parent]) > 0.01:
                            agg_demand[parent] = new_d
                            agg_var[parent] = new_v
                            changed = True
                if not changed:
                    break

            for n in nodes:
                results.append({'Product': prod,'Location': n,'Period': month,'Agg_Future_Demand': agg_demand[n],'Agg_Std_Hist': np.sqrt(agg_var[n])})
    return pd.DataFrame(results)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 90.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Stock Rules")
ss_formula = st.sidebar.selectbox("Safety Stock Formula", ["Statistical (Demand + Lead Time Variability)", "Demand-only", "Fixed Coverage (Days)", "Percent of Network Demand"])
coverage_days = st.sidebar.slider("Coverage Days", 0, 120, 30)
percent_demand = st.sidebar.slider("Percent of Network Demand (%)", 0, 200, 50)
dv_cap_days = st.sidebar.slider("Cap Demand Variability Impact (Days)", 0, 120, 45)
zero_if_no_net_fcst = st.sidebar.checkbox("Force Zero SS if No Network Demand", value=True)
apply_cap = st.sidebar.checkbox("Enable SS Capping (% of Network Demand)", value=True)
cap_range = st.sidebar.slider("Cap Range (%)", 0, 500, (0, 200))

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Sources")
DEFAULT_FILES = {"sales": "sales.csv", "demand": "demand.csv", "lt": "leadtime.csv"}
s_upload = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

s_file = s_upload if s_upload else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)

if s_file: st.sidebar.success(f"✅ Sales Loaded: {s_file}")
if d_file: st.sidebar.success(f"✅ Demand Loaded: {d_file}")
if lt_file: st.sidebar.success(f"✅ Lead Time Loaded: {lt_file}")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if s_file and d_file and lt_file:
    df_s = pd.read_csv(s_file)
    df_d = pd.read_csv(d_file)
    df_lt = pd.read_csv(lt_file)

    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]

    df_s['Period'] = pd.to_datetime(df_s['Period']).dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = pd.to_datetime(df_d['Period']).dt.to_period('M').dt.to_timestamp()

    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)

    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # Compute SS based on selected formula
    if ss_formula == "Statistical (Demand + Lead Time Variability)":
        demand_component = (np.minimum(results['LT_Mean'], dv_cap_days) / 30) * (results['Agg_Std_Hist']**2)
        supply_component = (results['LT_Std']**2) * (results['Agg_Future_Demand'] / 30)**2
        results['SS_Raw'] = z * np.sqrt(demand_component + supply_component)
    elif ss_formula == "Demand-only":
        results['SS_Raw'] = z * results['Agg_Std_Hist'] * np.sqrt(np.minimum(results['LT_Mean'], dv_cap_days) / 30)
    elif ss_formula == "Fixed Coverage (Days)":
        results['SS_Raw'] = (coverage_days / 30) * results['Agg_Future_Demand']
    elif ss_formula == "Percent of Network Demand":
        results['SS_Raw'] = (percent_demand / 100) * results['Agg_Future_Demand']

    results['Adjustment_Status'] = 'Optimal'
    results['Safety_Stock'] = results['SS_Raw']

    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0

    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100, cap_range[1] / 100
        l_lim, u_lim = results['Agg_Future_Demand'] * l_cap, results['Agg_Future_Demand'] * u_cap
        results['Safety_Stock'] = results['Safety_Stock'].clip(lower=l_lim, upper=u_lim)

    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis", "📉 Forecast Accuracy", "🧮 Calculation Trace & Sim"])

    with tab6:
        st.header("🧮 Transparent Calculation Engine")
        st.write(f"Selected Formula: **{ss_formula}**")
        calc_sku = st.selectbox("Select Product", sorted(results['Product'].unique()), key="c_sku")
        calc_loc = st.selectbox("Select Location", sorted(results[results['Product'] == calc_sku]['Location'].unique()), key="c_loc")
        calc_period = st.selectbox("Select Period", sorted(results['Period'].unique()), key="c_period")

        row = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)].iloc[0]

        st.subheader("Calculation Steps")
        if ss_formula == "Statistical (Demand + Lead Time Variability)":
            st.write(f"SS = Z * sqrt(((min(L,{dv_cap_days})/30)*σ_D²) + ((σ_L²)*(D/30)²))")
        elif ss_formula == "Demand-only":
            st.write(f"SS = Z * σ_D * sqrt(min(L,{dv_cap_days})/30)")
        elif ss_formula == "Fixed Coverage (Days)":
            st.write(f"SS = CoverageDays/30 * NetworkDemand")
        elif ss_formula == "Percent of Network Demand":
            st.write(f"SS = Percent * NetworkDemand")

        st.metric("Safety Stock", f"{int(row['Safety_Stock'])}")
