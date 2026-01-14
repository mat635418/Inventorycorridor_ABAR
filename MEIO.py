import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer", layout="wide")
st.title("📊 Multi-Echelon Network Inventory Optimizer")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def clean_numeric(series):
    """Cleans strings/objects into numeric values, handling formatting issues."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '')
        .str.replace('(', '-')
        .str.replace(')', '')
        .str.replace('-', '0')
        .str.strip(),
        errors='coerce'
    ).fillna(0)

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    """Propagates demand and variance up the supply chain network."""
    results = []
    months = df_forecast['Period'].unique()
    
    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in df_forecast['Product'].unique():
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]

            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(
                set(p_lt['To_Location'])
            )

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
                        new_d = p_fore.get(parent, {'Forecast': 0})['Forecast'] + \
                                sum(agg_demand.get(c, 0) for c in children[parent])
                        new_v = (p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + \
                                sum(agg_var.get(c, 0) for c in children[parent])

                        if abs(new_d - agg_demand[parent]) > 0.01:
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
                    'Agg_Demand': agg_demand[n],
                    'Agg_Std_Hist': np.sqrt(agg_var[n])
                })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# SIDEBAR & FILE LOADING LOGIC
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parameters")

service_level = st.sidebar.slider("Service Level (%)", 90.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Stock Rules")

zero_if_no_net_fcst = st.sidebar.checkbox("Force Zero SS if No Network Demand", value=True)
apply_cap = st.sidebar.checkbox("Enable SS Capping (% of Network Demand)", value=True)
cap_range = st.sidebar.slider("Cap Range (%)", 0, 500, (0, 200), help="Ensures SS stays between these % of total network demand for that node.")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Sources")

DEFAULT_FILES = {
    "sales": "sales.csv",
    "demand": "demand.csv",
    "lt": "leadtime.csv"
}

s_upload = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

s_file = s_upload if s_upload is not None else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload is not None else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload is not None else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)

if s_file: st.sidebar.success(f"✅ Sales Loaded")
if d_file: st.sidebar.success(f"✅ Demand Loaded")
if lt_file: st.sidebar.success(f"✅ Lead Time Loaded")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if s_file and d_file and lt_file:

    # LOAD & CLEAN DATA
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

    # HISTORICAL VARIABILITY
    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    # NETWORK AGGREGATION (Future)
    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)
    network_stats.rename(columns={'Agg_Demand': 'Agg_Future_Demand'}, inplace=True)

    # LEAD TIME RECEIVING NODES
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # MERGE FOR SS CALCULATION
    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # RULE-BASED SAFETY STOCK LOGIC
    results['SS_Raw'] = (
        z * np.sqrt(
            (results['LT_Mean'] / 30) * (results['Agg_Std_Hist']**2) +
            (results['LT_Std']**2) * (results['Agg_Future_Demand'] / 30)**2
        )
    )
    
    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['SS_Raw']

    if zero_if_no_net_fcst:
        results.loc[results['Agg_Future_Demand'] <= 0, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[results['Agg_Future_Demand'] <= 0, 'Safety_Stock'] = 0

    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100, cap_range[1] / 100
        l_lim, u_lim = results['Agg_Future_Demand'] * l_cap, results['Agg_Future_Demand'] * u_cap
        results['Safety_Stock'] = results['Safety_Stock'].clip(lower=l_lim, upper=u_lim)

    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # --- ACCURACY DATA WITH AGGREGATED DEMAND ---
    # 1. Aggregate Historical Network Demand
    hist_agg = aggregate_network_stats(df_forecast=df_s, df_stats=stats, df_lt=df_lt)
    hist_agg.rename(columns={'Agg_Demand': 'Agg_Hist_Demand'}, inplace=True)

    # 2. Prepare History Table
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    
    # 3. Merge Aggregated Historical Demand
    hist = pd.merge(hist, hist_agg[['Product', 'Location', 'Period', 'Agg_Hist_Demand']], on=['Product', 'Location', 'Period'], how='left')
    
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)).fillna(0) * 100
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    # ---------------------------------------------------------
    # TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Inventory Corridor", 
        "🕸️ Network Topology", 
        "📋 Full Plan", 
        "⚖️ Efficiency Analysis", 
        "📉 Forecast Accuracy",
        "🧮 Calculation Trace"
    ])

    # (Tabs 1-4 remain same as previous version)
    with tab1:
        sku = st.selectbox("Product", sorted(results['Product'].unique()))
        loc = st.selectbox("Location", sorted(results[results['Product'] == sku]['Location'].unique()))
        plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')
        fig = go.Figure([
            go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor', line=dict(width=0)),
            go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Network Demand', line=dict(color='blue', dash='dash'))
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        next_month = sorted(results['Period'].unique())[0]
        label_data = results[results['Period'] == next_month].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku]
        net = Network(height="700px", width="100%", directed=True, bgcolor="#eeeeee")
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        for n in all_nodes:
            m = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label = f"{n}\nFcst: {int(m['Forecast'])}\nNet: {int(m['Agg_Future_Demand'])}\nSS: {int(m['Safety_Stock'])}"
            net.add_node(n, label=label, shape='box', color='#31333F', font={'color': 'white'})
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
        net.save_graph("net.html")
        components.html(open("net.html").read(), height=750)

    with tab3:
        st.dataframe(results[['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status']], use_container_width=True)

    with tab4:
        eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()
        fig_eff = px.scatter(eff, x="Agg_Future_Demand", y="Safety_Stock", color="Adjustment_Status", size="Agg_Future_Demand", hover_name="Location")
        st.plotly_chart(fig_eff, use_container_width=True)

    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")
        
        h_sku = st.selectbox("Select Product", sorted(hist['Product'].unique()), key="h1")
        h_loc = st.selectbox("Select Location", sorted(hist[hist['Product'] == h_sku]['Location'].unique()), key="h2")
        hdf = hist[(hist['Product'] == h_sku) & (hist['Location'] == h_loc)].sort_values('Period')
        
        if not hdf.empty:
            k1, k2, k3 = st.columns(3)
            k1.metric("WAPE (%)", f"{(hdf['Abs_Error'].sum() / hdf['Consumption'].replace(0, np.nan).sum() * 100):.1f}")
            k2.metric("Bias (%)", f"{(hdf['Deviation'].sum() / hdf['Consumption'].replace(0, np.nan).sum() * 100):.1f}")
            k3.metric("Avg Accuracy (%)", f"{hdf['Accuracy_%'].mean():.1f}")

            fig_hist = go.Figure([
                go.Scatter(x=hdf['Period'], y=hdf['Consumption'], name='Actuals', line=dict(color='black')),
                go.Scatter(x=hdf['Period'], y=hdf['Forecast_Hist'], name='Forecast', line=dict(color='blue', dash='dot')),
            ])
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("📊 Detailed Accuracy by Month")
            # Added Agg_Hist_Demand to the column selection here:
            st.dataframe(
                hdf[['Period','Consumption','Forecast_Hist','Agg_Hist_Demand','Deviation','Abs_Error','APE_%','Accuracy_%']], 
                use_container_width=True, 
                height=500
            )

    with tab6:
        # Same logic as provided in the previous turn for Calculation Trace
        st.header("🧮 Transparent Calculation Engine")
        calc_sku = st.selectbox("Select Product", sorted(results['Product'].unique()), key="c_sku")
        calc_loc = st.selectbox("Select Location", sorted(results[results['Product'] == calc_sku]['Location'].unique()), key="c_loc")
        calc_period = st.selectbox("Select Period", sorted(results['Period'].unique()), key="c_period")
        row = results[(results['Product'] == calc_sku) & (results['Location'] == calc_loc) & (results['Period'] == calc_period)].iloc[0]

        st.latex(r"SS_{raw} = Z \times \sqrt{ \left( \frac{L}{30} \times \sigma_D^2 \right) + \left( \sigma_L^2 \times \left( \frac{D}{30} \right)^2 \right) }")
        st.metric("Aggregated Network Demand (Used in Calc)", f"{row['Agg_Future_Demand']}")
        st.info(f"The Final Safety Stock is: {row['Safety_Stock']} (Status: {row['Adjustment_Status']})")

else:
    st.info("Please upload data files to begin.")
