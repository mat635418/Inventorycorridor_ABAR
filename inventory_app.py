import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer", layout="wide")
st.title("📊 Multi-Echelon Network Inventory Optimizer")

# --- Helper Functions ---
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', '').str.replace('-', '0').str.strip(), 
        errors='coerce'
    ).fillna(0)

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    results = []
    months = df_forecast['Future_Forecast_Month'].unique()
    
    for month in months:
        df_month = df_forecast[df_forecast['Future_Forecast_Month'] == month]
        for prod in df_forecast['Product'].unique():
            # FIX: Added numeric_only=True to prevent the mean() error on string columns
            p_stats = df_stats[df_stats['Product'] == prod].groupby('Location').mean(numeric_only=True).to_dict('index')
            
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]
            
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(set(p_lt['To_Location']))
            
            if not nodes: continue

            agg_demand = {n: p_fore.get(n, {'Forecast_Quantity': 0})['Forecast_Quantity'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': 0})['Local_Std'])**2 for n in nodes}
            
            children = {}
            for _, row in p_lt.iterrows():
                if row['From_Location'] not in children: children[row['From_Location']] = []
                children[row['From_Location']].append(row['To_Location'])
                
            for _ in range(15):
                changed = False
                for parent in nodes:
                    if parent in children:
                        new_d = p_fore.get(parent, {'Forecast_Quantity': 0})['Forecast_Quantity'] + sum(agg_demand.get(c, 0) for c in children[parent])
                        new_v = (p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(agg_var.get(c, 0) for c in children[parent])
                        
                        if abs(new_d - agg_demand[parent]) > 0.01:
                            agg_demand[parent], agg_var[parent] = new_d, new_v
                            changed = True
                if not changed: break
                
            for n in nodes:
                results.append({
                    'Product': prod, 'Location': n, 'Future_Forecast_Month': month,
                    'Agg_Future_Demand': agg_demand[n], 'Agg_Std_Hist': np.sqrt(agg_var[n])
                })
    return pd.DataFrame(results)

# --- Sidebar ---
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 90.0, 99.9, 99.0)/100
z = norm.ppf(service_level)

s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    for df in [df_s, df_d, df_lt]: df.columns = [c.strip() for c in df.columns]
    
    df_s['Quantity'] = clean_numeric(df_s['Quantity'])
    
    # --- Month/Year Integration ---
    if 'Month/Year' in df_s.columns:
        df_s['Date_Sort'] = pd.to_datetime(df_s['Month/Year'], errors='coerce')
    
    df_d['Forecast_Quantity'] = clean_numeric(df_d['Forecast_Quantity'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # Calculate historical stats
    stats = df_s.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)
    stats['CV'] = (stats['Local_Std'] / stats['Local_Mean'].replace(0, np.nan)).fillna(0)

    # Core Calculations
    network_stats = aggregate_network_stats(df_d, stats, df_lt)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean(numeric_only=True).reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d, on=['Product', 'Location', 'Future_Forecast_Month'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    
    # Fill missing values for hubs
    results = results.fillna({
        'Forecast_Quantity': 0, 'Agg_Std_Hist': 0, 
        'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0
    })
    
    results['Safety_Stock'] = (z * np.sqrt(
        (results['LT_Mean']/30) * (results['Agg_Std_Hist']**2) + 
        (results['LT_Std']**2) * (results['Agg_Future_Demand']/30)**2
    )).round(0)
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast_Quantity']

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis", "🕰️ Historical Analysis"])
    
    with tab1:
        sku = st.selectbox("Product", sorted(results['Product'].unique()))
        loc = st.selectbox("Location", sorted(results[results['Product']==sku]['Location'].unique()))
        plot_df = results[(results['Product']==sku) & (results['Location']==loc)].sort_values('Future_Forecast_Month')
        
        fig = go.Figure([
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Local Forecast)', line=dict(width=0)),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Forecast_Quantity'], name='Local Direct Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand (Aggregated)', line=dict(color='blue', dash='dash'))
        ])
        fig.update_layout(title=f"Inventory Plan for {sku} at {loc}", xaxis_title="Month", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Topology code remains unchanged
        next_month = sorted(results['Future_Forecast_Month'].unique())[0]
        label_data = results[results['Future_Forecast_Month'] == next_month].set_index(['Product', 'Location']).to_dict('index')
        net = Network(height="600px", width="100%", directed=True, bgcolor="#eeeeee")
        sku_lt = df_lt[df_lt['Product'] == sku]
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        for n in all_nodes:
            m = label_data.get((sku, n), {'Forecast_Quantity': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label_text = f"{n}\nNet: {int(m['Agg_Future_Demand'])}\nSS: {int(m['Safety_Stock'])}"
            color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
            net.add_node(n, label=label_text, color=color, shape='box', font={'color': 'white'})
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
        net.save_graph("net.html")
        components.html(open("net.html", 'r').read(), height=650)

    with tab3:
        st.subheader("Global Inventory Plan")
        st.dataframe(results[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity', 'Agg_Future_Demand', 'Safety_Stock', 'Max_Corridor']], use_container_width=True)

    with tab4:
        st.subheader(f"⚖️ Efficiency Snapshot: {next_month}")
        # Analysis code remains unchanged
        eff_df = results[(results['Product'] == sku) & (results['Future_Forecast_Month'] == next_month)].copy()
        st.metric("Total Safety Stock", f"{int(eff_df['Safety_Stock'].sum()):,}")
        fig_eff = px.scatter(eff_df, x="Forecast_Quantity", y="Safety_Stock", size="Agg_Future_Demand", color="Location", title="Inventory Positioning")
        st.plotly_chart(fig_eff, use_container_width=True)

    with tab5:
        st.subheader("🕰️ Past Data & Demand Variability")
        # Filters historical sales for selected SKU/Location
        hist_df = df_s[(df_s['Product'] == sku) & (df_s['Location'] == loc)].sort_values('Date_Sort')
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_hist = px.line(hist_df, x='Month/Year', y='Quantity', markers=True, 
                               title=f"Historical Sales Trend: {sku} @ {loc}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_b:
            loc_stats = stats[(stats['Product'] == sku) & (stats['Location'] == loc)].iloc[0]
            st.metric("Avg Monthly Demand", f"{loc_stats['Local_Mean']:.1f}")
            st.metric("Coefficient of Variation (CV)", f"{loc_stats['CV']:.2f}")
            
            # Contextual insight
            if loc_stats['CV'] > 0.5:
                st.warning("High Variability detected. Consider increasing Safety Stock.")
            else:
                st.success("Stable demand pattern.")

        st.divider()
        st.markdown("**Network-wide Demand Variability (CV)**")
        cv_fig = px.bar(stats[stats['Product']==sku], x='Location', y='CV', color='CV', 
                        title="Variability Comparison: Which nodes are most erratic?")
        st.plotly_chart(cv_fig, use_container_width=True)

else:
    st.info("Please upload all three CSV files in the sidebar to begin.")
