import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm

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
                    'Agg_Future_Demand': agg_demand[n],
                    'Agg_Std_Hist': np.sqrt(agg_var[n])
                })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 90.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Stock Rules")

# Rule 1: Zero if no aggregated network demand
zero_if_no_net_fcst = st.sidebar.checkbox("Force Zero SS if No Network Demand", value=True)

# Rule 2: Capping logic relative to Network Demand
apply_cap = st.sidebar.checkbox("Enable SS Capping (% of Network Demand)", value=True)
cap_range = st.sidebar.slider("Cap Range (%)", 0, 500, (0, 200), help="Ensures SS stays between these % of total network demand for that node.")

st.sidebar.markdown("---")
s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

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

    # NETWORK AGGREGATION
    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)

    # LEAD TIME RECEIVING NODES
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # MERGE
    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']], on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # -----------------------------
    # RULE-BASED SAFETY STOCK LOGIC
    # -----------------------------
    results['SS_Raw'] = (
        z * np.sqrt(
            (results['LT_Mean'] / 30) * (results['Agg_Std_Hist']**2) +
            (results['LT_Std']**2) * (results['Agg_Future_Demand'] / 30)**2
        )
    )
    
    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['SS_Raw']

    # Rule: Zero if no NETWORK demand
    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0

    # Rule: Capping based on NETWORK demand
    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100, cap_range[1] / 100
        l_lim, u_lim = results['Agg_Future_Demand'] * l_cap, results['Agg_Future_Demand'] * u_cap

        high_mask = (results['Safety_Stock'] > u_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)')
        results.loc[high_mask, 'Adjustment_Status'] = 'Capped (High)'
        
        low_mask = (results['Safety_Stock'] < l_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)') & (results['Agg_Future_Demand'] > 0)
        results.loc[low_mask, 'Adjustment_Status'] = 'Capped (Low)'

        results['Safety_Stock'] = results['Safety_Stock'].clip(lower=l_lim, upper=u_lim)

    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0 
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # ACCURACY DATA
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)).fillna(0) * 100
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    # ---------------------------------------------------------
    # TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis", "📉 Forecast Accuracy"
    ])

    with tab1:
        sku = st.selectbox("Product", sorted(results['Product'].unique()))
        loc = st.selectbox("Location", sorted(results[results['Product'] == sku]['Location'].unique()))
        plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')

        fig = go.Figure([
            go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=0)),
            go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Direct Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
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
            color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
            net.add_node(n, label=label, title=label, color=color, shape='box', font={'color': 'white'})
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
        net.save_graph("net.html")
        components.html(open("net.html").read(), height=750)

    with tab3:
        st.subheader("📋 Global Inventory Plan")
        # --- RE-INTRODUCED FILTERS ---
        col1, col2, col3 = st.columns(3)
        f_prod = col1.multiselect("Filter Product", sorted(results['Product'].unique()))
        f_loc = col2.multiselect("Filter Location", sorted(results['Location'].unique()))
        f_period = col3.multiselect("Filter Period", sorted(results['Period'].unique()))

        filtered = results.copy()
        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]

        st.dataframe(filtered[['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']], use_container_width=True, height=700)

    with tab4:
        st.subheader(f"⚖️ Efficiency & Policy Analysis: {next_month}")
        eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_eff = px.scatter(
                eff, x="Agg_Future_Demand", y="Safety_Stock", color="Adjustment_Status",
                size="Agg_Future_Demand", hover_name="Location",
                color_discrete_map={'Optimal (Statistical)': '#00CC96', 'Capped (High)': '#EF553B', 'Capped (Low)': '#636EFA', 'Forced to Zero': '#AB63FA'},
                title="Policy Impact: Network Demand vs Safety Stock"
            )
            st.plotly_chart(fig_eff, use_container_width=True)
        with c2:
            st.markdown("**Status Breakdown**")
            st.table(eff['Adjustment_Status'].value_counts())
            st.markdown("**Top Adjusted Nodes**")
            eff['Gap'] = (eff['Safety_Stock'] - eff['SS_Raw']).abs()
            st.dataframe(eff.sort_values('Gap', ascending=False)[['Location','Adjustment_Status','SS_Raw','Safety_Stock']].head(10))

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
                go.Bar(x=hdf['Period'], y=hdf['Deviation'], name='Error', marker_color='red', opacity=0.3)
            ])
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- RE-INTRODUCED DATA TABLE ---
            st.subheader("📊 Detailed Accuracy by Month")
            st.dataframe(hdf[['Period','Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']], use_container_width=True)

else:
    st.info("Please upload all three CSV files in the sidebar to begin.")
