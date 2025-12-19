import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer", layout="wide")
st.title("📈 Multi-Echelon Network Inventory Optimizer")

# --- Helper Functions ---
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', '').str.replace('-', '0').str.strip(), 
        errors='coerce'
    ).fillna(0)

# --- Parameters ---
st.sidebar.header("⚙️ Optimization Parameters")
service_level = st.sidebar.slider("Target Service Level (%)", 80.0, 99.9, 99.0, step=0.1)
z_score = norm.ppf(service_level / 100)

def process_network_optimization(s_raw, d_raw, lt_raw):
    # 1. Clean Data
    for df in [s_raw, d_raw, lt_raw]:
        df.columns = [c.strip() for c in df.columns]
    
    s_raw['Quantity'] = clean_numeric(s_raw['Quantity'])
    d_raw['Forecast_Quantity'] = clean_numeric(d_raw['Forecast_Quantity'])
    lt_raw['Lead_Time_Days'] = clean_numeric(lt_raw['Lead_Time_Days'])
    lt_raw['Lead_Time_Std_Dev'] = clean_numeric(lt_raw['Lead_Time_Std_Dev'])

    # 2. Upstream Demand Aggregation (The "True Network" Step)
    # Map which 'To_Location' demand is served by which 'From_Location'
    network_map = lt_raw[['From_Location', 'To_Location', 'Product']].drop_duplicates()
    
    # Group sales by the UPSTREAM parent location
    child_demand = pd.merge(s_raw, network_map, left_on=['Location', 'Product'], right_on=['To_Location', 'Product'])
    parent_stats = child_demand.groupby(['Product', 'From_Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    parent_stats.columns = ['Product', 'Location', 'Aggregated_Avg_Demand', 'Aggregated_Std_Demand']

    # 3. Lead Time Stats
    lt_stats = lt_raw.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    lt_stats.rename(columns={'To_Location': 'Location', 'Lead_Time_Days': 'LT_Avg', 'Lead_Time_Std_Dev': 'LT_SD'}, inplace=True)

    # 4. Final Merge with Future Forecast
    data = pd.merge(d_raw, parent_stats, on=['Product', 'Location'], how='left')
    data = pd.merge(data, lt_stats, on=['Product', 'Location'], how='left')

    # Defaults for endpoints
    data['Aggregated_Std_Demand'] = data['Aggregated_Std_Demand'].fillna(data['Forecast_Quantity'] * 0.2)
    data['LT_Avg'] = data['LT_Avg'].fillna(7)
    data['LT_SD'] = data['LT_SD'].fillna(2)

    # 5. Calculation
    data['Safety_Stock'] = z_score * np.sqrt(
        ((data['LT_Avg'] / 30) * (data['Aggregated_Std_Demand']**2)) + 
        ((data['Forecast_Quantity']**2) * ((data['LT_SD'] / 30)**2))
    ).round(0)
    
    data['Min_Corridor'] = data['Safety_Stock']
    data['Max_Corridor'] = data['Safety_Stock'] + data['Forecast_Quantity']
    
    return data, lt_raw

# --- File Loading ---
s_file = st.sidebar.file_uploader("1. Sales Data", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data", type="csv")

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    results, network_ref = process_network_optimization(df_s, df_d, df_lt)

    tab1, tab2, tab3 = st.tabs(["📊 Inventory Corridor", "🌐 Network View", "📋 Full Plan & Filters"])

    with tab1:
        sku = st.selectbox("Select Product", results['Product'].unique(), key="sb_sku")
        loc = st.selectbox("Select Location", results[results['Product']==sku]['Location'].unique(), key="sb_loc")
        plot_df = results[(results['Product']==sku) & (results['Location']==loc)].sort_values('Future_Forecast_Month')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Max_Corridor'], name='Max Corridor', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Min_Corridor'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'))
        fig.add_trace(go.Scatter(x=plot_df['Future_Forecast_Month'], y=plot_df['Forecast_Quantity'], name='Forecast', line=dict(color='black', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Supply Chain Network for {sku}")
        net = Network(height="850px", width="100%", directed=True, bgcolor="#eeeeee")
        
        # Filter data for selected SKU
        sku_lt = network_ref[network_ref['Product'] == sku]
        sku_demand = raw_demand[raw_demand['Product'] == sku]
        
        # Add Nodes with Raw Demand info
        nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        for n in nodes:
            # Find the raw demand for this specific node
            d_val = sku_demand[sku_demand['Location'] == n]['Raw_Demand_Avg'].values
            d_str = f"{int(d_val[0]):,}" if len(d_val) > 0 else "0"
            
            node_color = "#31333F" if n in sku_lt['From_Location'].values else "#ff4b4b"
            net.add_node(n, label=f"{n}\n(Dem: {d_str})", title=f"Location: {n}\nAvg Monthly Demand: {d_str}", color=node_color, size=25)
        
        # Add Edges with Lead Time Labels
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']} days", title=f"LT: {r['Lead_Time_Days']}d | SD: {r['Lead_Time_Std_Dev']}d")
        
        net.toggle_physics(True)
        net.save_graph("network.html")
        components.html(open("network.html", 'r').read(), height=850)
        st.caption("⚫ Dark Node: Source/Parent | 🔴 Red Node: Destination | (Dem: X) = Raw Historical Demand")

    
    with tab3:
        st.subheader("Global Inventory Plan - Dynamic Filters")
        
        # Filter Columns
        col1, col2, col3 = st.columns(3)
        with col1:
            f_prod = st.multiselect("Filter Product", results['Product'].unique())
        with col2:
            f_loc = st.multiselect("Filter Location", results['Location'].unique())
        with col3:
            f_month = st.multiselect("Filter Month", results['Future_Forecast_Month'].unique())

        # Apply Filters
        filtered_df = results.copy()
        if f_prod:
            filtered_df = filtered_df[filtered_df['Product'].isin(f_prod)]
        if f_loc:
            filtered_df = filtered_df[filtered_df['Location'].isin(f_loc)]
        if f_month:
            filtered_df = filtered_df[filtered_df['Future_Forecast_Month'].isin(f_month)]

        st.dataframe(filtered_df[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity', 'Safety_Stock', 'Min_Corridor', 'Max_Corridor']], use_container_width=True, height=1500)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Plan", csv, "filtered_inventory_plan.csv", "text/csv")
