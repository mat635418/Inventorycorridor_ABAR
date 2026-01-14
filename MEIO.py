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
                    'Agg_Future_Demand': agg_demand[n],
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

# Define default local filenames
DEFAULT_FILES = {
    "sales": "sales.csv",
    "demand": "demand.csv",
    "lt": "leadtime.csv"
}

# File Uploaders
s_upload = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

# Logic: Priority to Uploaded File -> then Local File -> then None
s_file = s_upload if s_upload is not None else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload is not None else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload is not None else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)

# Visual indicators for the user
if s_file:
    st.sidebar.success(f"✅ Sales Loaded: {s_file.name if hasattr(s_file, 'name') else s_file}")
if d_file:
    st.sidebar.success(f"✅ Demand Loaded: {d_file.name if hasattr(d_file, 'name') else d_file}")
if lt_file:
    st.sidebar.success(f"✅ Lead Time Loaded: {lt_file.name if hasattr(lt_file, 'name') else lt_file}")

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Inventory Corridor", 
        "🕸️ Network Topology", 
        "📋 Full Plan", 
        "⚖️ Efficiency Analysis", 
        "📉 Forecast Accuracy",
        "🧮 Calculation Trace & Sim"
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
        net = Network(height="1200px", width="100%", directed=True, bgcolor="#eeeeee")
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        
        for n in all_nodes:
            m = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            
            if m['Agg_Future_Demand'] == 0:
                node_color = '#DDDDDD' 
                font_color = '#888888'
            else:
                node_color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
                font_color = 'white'
            
            label = f"{n}\nFcst: {int(m['Forecast'])}\nNet: {int(m['Agg_Future_Demand'])}\nSS: {int(m['Safety_Stock'])}"
            
            net.add_node(
                n, 
                label=label, 
                title=label, 
                color=node_color, 
                shape='box', 
                font={'color': font_color}
            )

        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
        
        net.save_graph("net.html")
        components.html(open("net.html").read(), height=1250)

    with tab3:
        st.subheader("📋 Global Inventory Plan")
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
        
        eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Forecast'].replace(0, np.nan)).fillna(0)
        total_ss_sku = eff['Safety_Stock'].sum()
        total_fcst_sku = eff['Forecast'].sum()
        sku_ratio = total_ss_sku / total_fcst_sku if total_fcst_sku > 0 else 0
        
        all_res = results[results['Period'] == next_month]
        global_ratio = all_res['Safety_Stock'].sum() / all_res['Forecast'].replace(0, np.nan).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Network Ratio ({sku})", f"{sku_ratio:.2f}")
        m2.metric("Global Network Ratio (All Items)", f"{global_ratio:.2f}")
        m3.metric("Total SS for Material", f"{int(total_ss_sku)}")

        st.markdown("---")

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_eff = px.scatter(
                eff, x="Agg_Future_Demand", y="Safety_Stock", color="Adjustment_Status",
                size="SS_to_FCST_Ratio", hover_name="Location",
                color_discrete_map={'Optimal (Statistical)': '#00CC96', 'Capped (High)': '#EF553B', 'Capped (Low)': '#636EFA', 'Forced to Zero': '#AB63FA'},
                title=f"Policy Impact & Efficiency Ratio (Bubble Size = SS/FCST Ratio)"
            )
            st.plotly_chart(fig_eff, use_container_width=True)
            
        with c2:
            st.markdown("**Status Breakdown**")
            st.table(eff['Adjustment_Status'].value_counts())
            
            st.markdown("**Top Nodes by Efficiency Gap**")
            eff['Gap'] = (eff['Safety_Stock'] - eff['SS_Raw']).abs()
            st.dataframe(
                eff.sort_values('Gap', ascending=False)[
                    ['Location','Adjustment_Status','Safety_Stock','SS_to_FCST_Ratio']
                ].head(10), 
                use_container_width=True
            )

    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")
        # OPTION 3: Selectboxes pull from 'results' to show ALL materials/locations
        h_sku = st.selectbox("Select Product", sorted(results['Product'].unique()), key="h1")
        h_loc = st.selectbox("Select Location", sorted(results[results['Product'] == h_sku]['Location'].unique()), key="h2")
        
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
            st.dataframe(
                hdf[['Period','Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']], 
                use_container_width=True, 
                height=500
            )
        else:
            st.warning("⚠️ No historical sales data (sales.csv) found for this selection. Accuracy metrics cannot be calculated.")
    # ---------------------------------------------------------
    # TAB 6: CALCULATION TRACE & SIMULATION
    # ---------------------------------------------------------
    with tab6:
        st.header("🧮 Transparent Calculation Engine")
        st.write("Select a specific node and period to see exactly how the Safety Stock number was derived.")

        # 1. Selection Controls
        c1, c2, c3 = st.columns(3)
        calc_sku = c1.selectbox("Select Product", sorted(results['Product'].unique()), key="c_sku")
        
        # Filter locations based on SKU
        avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique())
        calc_loc = c2.selectbox("Select Location", avail_locs, key="c_loc")
        
        # Filter periods
        avail_periods = sorted(results['Period'].unique())
        calc_period = c3.selectbox("Select Period", avail_periods, key="c_period")

        # Get specific row data
        row = results[
            (results['Product'] == calc_sku) & 
            (results['Location'] == calc_loc) & 
            (results['Period'] == calc_period)
        ].iloc[0]

        st.markdown("---")

        # 2. Input Variables Display
        st.subheader("1. Actual Inputs (Frozen)")
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Service Level", f"{service_level*100}%", help=f"Z-Score: {z:.2f}")
        i2.metric("Network Demand (D)", f"{row['Agg_Future_Demand']:,.1f}", help="Aggregated Future Demand")
        i3.metric("Network Std Dev (σ_D)", f"{row['Agg_Std_Hist']:,.1f}", help="Aggregated Historical Variability")
        i4.metric("Avg Lead Time (L)", f"{row['LT_Mean']} days")
        i5.metric("LT Std Dev (σ_L)", f"{row['LT_Std']} days")

        # 3. Mathematical Trace
        st.subheader("2. Statistical Calculation (Actual)")
        
        # Terms calculation for display
        term1_demand_var = (row['LT_Mean'] / 30) * (row['Agg_Std_Hist']**2)
        term2_supply_var = (row['LT_Std']**2) * ((row['Agg_Future_Demand'] / 30)**2)
        combined_sd = np.sqrt(term1_demand_var + term2_supply_var)
        raw_ss_calc = z * combined_sd

        st.markdown("The standard formula for Safety Stock with variable Demand and variable Lead Time is:")
        
        st.latex(r"SS_{raw} = Z \times \sqrt{ \underbrace{\left( \frac{L}{30} \times \sigma_D^2 \right)}_{\text{Demand Var}} + \underbrace{\left( \sigma_L^2 \times \left( \frac{D}{30} \right)^2 \right)}_{\text{Supply Var}} }")

        st.markdown("**Step-by-Step Substitution:**")
        
        st.code(f"""
        1. Demand Component = ({row['LT_Mean']} / 30) * ({row['Agg_Std_Hist']:.2f})² 
                            = {term1_demand_var:,.2f}

        2. Supply Component = ({row['LT_Std']}²) * ({row['Agg_Future_Demand']:.2f} / 30)² 
                            = {term2_supply_var:,.2f}

        3. Combined Variance = {term1_demand_var:,.2f} + {term2_supply_var:,.2f} 
                             = {(term1_demand_var + term2_supply_var):,.2f}

        4. Combined Std Dev  = sqrt(Combined Variance) 
                             = {combined_sd:,.2f}

        5. Raw SS            = {z:.2f} (Z-Score) * {combined_sd:,.2f}
        """)
        
        st.info(f"🧮 **Resulting Statistical SS:** {raw_ss_calc:,.2f} units")

        # 4. Rules Application
        st.subheader("3. Business Rules Application")
        
        col_rule_1, col_rule_2 = st.columns(2)
        
        with col_rule_1:
            st.markdown("**Check 1: Zero Demand Rule**")
            if zero_if_no_net_fcst and row['Agg_Future_Demand'] <= 0:
                st.error("❌ Network Demand is 0. SS Forced to 0.")
            else:
                st.success("✅ Network Demand exists. Keep Statistical SS.")

        with col_rule_2:
            st.markdown("**Check 2: Capping (Min/Max)**")
            if apply_cap:
                lower_limit = row['Agg_Future_Demand'] * (cap_range[0]/100)
                upper_limit = row['Agg_Future_Demand'] * (cap_range[1]/100)
                
                st.write(f"Constraint: {int(cap_range[0])}% to {int(cap_range[1])}% of Demand")
                st.write(f"Range: [{lower_limit:,.1f}, {upper_limit:,.1f}]")
                
                if raw_ss_calc > upper_limit:
                    st.warning(f"⚠️ Raw SS ({raw_ss_calc:,.1f}) > Max Cap ({upper_limit:,.1f}). Capping downwards.")
                elif raw_ss_calc < lower_limit and row['Agg_Future_Demand'] > 0:
                    st.warning(f"⚠️ Raw SS ({raw_ss_calc:,.1f}) < Min Cap ({lower_limit:,.1f}). Buffering upwards.")
                else:
                    st.success("✅ Raw SS is within efficient boundaries.")
            else:
                st.write("Capping logic disabled.")

        st.markdown("---")

        # 5. SIMULATION MODE
        st.subheader("4. What-If Simulation")
        st.write("Tweak parameters below to see how Safety Stock reacts *without* changing the global settings.")
        
        

        # Simulation Sliders
        # We use dynamic keys to ensure sliders reset when you change SKU/Loc
        sim_cols = st.columns(3)
        
        sim_sl = sim_cols[0].slider(
            "Simulated Service Level (%)", 
            min_value=50.0, max_value=99.9, 
            value=service_level*100, 
            key=f"sim_sl_{calc_sku}_{calc_loc}"
        )
        
        sim_lt = sim_cols[1].slider(
            "Simulated Avg Lead Time (Days)", 
            min_value=0.0, max_value=max(30.0, row['LT_Mean']*2), 
            value=float(row['LT_Mean']),
            key=f"sim_lt_{calc_sku}_{calc_loc}"
        )

        sim_lt_std = sim_cols[2].slider(
            "Simulated LT Variability (Days)", 
            min_value=0.0, max_value=max(10.0, row['LT_Std']*2), 
            value=float(row['LT_Std']),
            key=f"sim_lt_std_{calc_sku}_{calc_loc}"
        )

        # Dynamic Recalculation
        sim_z = norm.ppf(sim_sl / 100)
        sim_ss = sim_z * np.sqrt(
            (sim_lt / 30) * (row['Agg_Std_Hist']**2) +
            (sim_lt_std**2) * (row['Agg_Future_Demand'] / 30)**2
        )
        
        # Display Results
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Original SS (Actual)", f"{int(raw_ss_calc)}")
        res_col2.metric(
            "Simulated SS (New)", 
            f"{int(sim_ss)}", 
            delta=f"{int(sim_ss - raw_ss_calc)} Units",
            delta_color="inverse"
        )
        
        if sim_ss < raw_ss_calc:
            st.success(f"📉 Reducing uncertainty could lower inventory by **{int(raw_ss_calc - sim_ss)}** units.")
        elif sim_ss > raw_ss_calc:
            st.warning(f"📈 Increasing service or lead time requires **{int(sim_ss - raw_ss_calc)}** more units.")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
