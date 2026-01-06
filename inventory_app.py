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
            # Get stats for the product
            p_stats_df = df_stats[df_stats['Product'] == prod]
            p_stats = p_stats_df.groupby('Location').mean(numeric_only=True).to_dict('index')
            
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]
            
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(set(p_lt['To_Location']))
            
            if not nodes: continue

            # If a node has no history, we default its Std Dev to 20% of its forecast
            agg_demand = {n: p_fore.get(n, {'Forecast_Quantity': 0})['Forecast_Quantity'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': agg_demand[n] * 0.2})['Local_Std'])**2 for n in nodes}
            
            children = {}
            for _, row in p_lt.iterrows():
                if row['From_Location'] not in children: children[row['From_Location']] = []
                children[row['From_Location']].append(row['To_Location'])
                
            for _ in range(15):
                changed = False
                for parent in nodes:
                    if parent in children:
                        # Safety check for missing stats in the parent node
                        parent_local_std = p_stats.get(parent, {'Local_Std': agg_demand[parent] * 0.2})['Local_Std']
                        new_d = p_fore.get(parent, {'Forecast_Quantity': 0})['Forecast_Quantity'] + sum(agg_demand.get(c, 0) for c in children[parent])
                        new_v = (parent_local_std)**2 + sum(agg_var.get(c, 0) for c in children[parent])
                        
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
s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 90.0, 99.9, 99.0)/100
z = norm.ppf(service_level)

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    for df in [df_s, df_d, df_lt]: df.columns = [c.strip() for c in df.columns]
    
    # Cleaning
    df_s['Quantity'] = clean_numeric(df_s['Quantity'])
    if 'Historical_Forecast' in df_s.columns:
        df_s['Historical_Forecast'] = clean_numeric(df_s['Historical_Forecast'])
    
    if 'Month/Year' in df_s.columns:
        df_s['Date_Sort'] = pd.to_datetime(df_s['Month/Year'], errors='coerce')
    
    df_d['Forecast_Quantity'] = clean_numeric(df_d['Forecast_Quantity'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # Historical Stats
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Inventory Corridor", "🕸️ Network Topology", "📋 Full Plan", "⚖️ Efficiency Analysis", "🕰️ Forecast Accuracy & Variability"])
    
    with tab1:
        sku = st.selectbox("Product", sorted(results['Product'].unique()))
        loc = st.selectbox("Location", sorted(results[results['Product']==sku]['Location'].unique()))
        plot_df = results[(results['Product']==sku) & (results['Location']==loc)].sort_values('Future_Forecast_Month')
        
        fig = go.Figure([
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Max_Corridor'], name='Max Corridor', line=dict(width=0)),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Forecast_Quantity'], name='Local Forecast', line=dict(color='black', dash='dot')),
            go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Agg_Future_Demand'], name='Aggregated Demand', line=dict(color='blue', dash='dash'))
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Topology visualization based on lead time routes.")
        # Network code...

    with tab3:
        st.subheader("Global Inventory Plan")
        st.dataframe(results[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity', 'Agg_Future_Demand', 'Safety_Stock', 'Max_Corridor']], use_container_width=True)

    with tab4:
        st.subheader("Efficiency Analysis")
        # Efficiency code...

    with tab5:
        st.subheader("🕰️ Historical Forecast vs Actuals")
        hist_df = df_s[(df_s['Product'] == sku) & (df_s['Location'] == loc)].sort_values('Date_Sort')
        
        if not hist_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=hist_df['Month/Year'], y=hist_df['Quantity'], name='Actual Sales', line=dict(color='green')))
                if 'Historical_Forecast' in hist_df.columns:
                    fig_hist.add_trace(go.Scatter(x=hist_df['Month/Year'], y=hist_df['Historical_Forecast'], name='Past Forecast', line=dict(color='orange', dash='dot')))
                fig_hist.update_layout(title=f"Trend for {sku} @ {loc}")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with c2:
                # SAFE ACCESS FIX: Check if stats exist for this combination
                loc_stats_filter = stats[(stats['Product'] == sku) & (stats['Location'] == loc)]
                
                if not loc_stats_filter.empty:
                    loc_stats = loc_stats_filter.iloc[0]
                    st.metric("Avg Monthly Sales", f"{loc_stats['Local_Mean']:.1f}")
                    st.metric("Demand CV", f"{loc_stats['CV']:.2f}")
                    
                    if 'Historical_Forecast' in hist_df.columns:
                        # Calculate MAPE: Mean Absolute Percentage Error
                        abs_err = np.abs(hist_df['Quantity'] - hist_df['Historical_Forecast'])
                        mape = (abs_err / hist_df['Quantity'].replace(0, np.nan)).mean() * 100
                        bias = (hist_df['Historical_Forecast'].sum() / hist_df['Quantity'].sum() - 1) * 100 if hist_df['Quantity'].sum() != 0 else 0
                        
                        st.metric("Forecast Accuracy (MAPE)", f"{max(0, 100-mape):.1f}%")
                        st.metric("Forecast Bias", f"{bias:+.1f}%")
                        st.caption("Bias > 0 means over-forecasting; Bias < 0 means under-forecasting.")
                else:
                    st.warning("No summary statistics available for this selection.")
        else:
            st.warning("No historical data found in sales.csv for this Product/Location combination.")

        st.divider()
        st.markdown("**Variability (CV) Heatmap**")
        st.plotly_chart(px.bar(stats[stats['Product']==sku], x='Location', y='CV', color='CV', title="Demand Volatility by Site"), use_container_width=True)

else:
    st.info("Please upload all three CSV files in the sidebar to begin.")
