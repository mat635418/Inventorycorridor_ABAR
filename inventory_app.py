import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Time-Phased Inventory Corridor", layout="wide")

st.title("📈 Time-Phased Inventory Corridor")
st.markdown("Dynamic Safety Stock calculated per month based on Lead Time variability and Forecasted Demand.")

# --- Helper Functions ---
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '').str.replace('-', '0').str.strip(), 
        errors='coerce'
    ).fillna(0)

# --- Sidebar ---
st.sidebar.header("⚙️ Optimization Parameters")
# Defaulting to 99.5% service level
service_level = st.sidebar.slider("Service Level (%)", 80.0, 99.9, 99.5, step=0.1)
z_score = norm.ppf(service_level / 100)

st.sidebar.header("📂 Data Upload")
sales_file = st.sidebar.file_uploader("1. Sales Data (sales.csv)", type="csv")
demand_file = st.sidebar.file_uploader("2. Demand Data (Demand.csv)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Lead time.csv)", type="csv")

def process_data(s_raw, d_raw, lt_raw):
    # Standardize column names (strip spaces)
    s_raw.columns = [c.strip() for c in s_raw.columns]
    d_raw.columns = [c.strip() for c in d_raw.columns]
    lt_raw.columns = [c.strip() for c in lt_raw.columns]

    # Clean Numeric Values
    s_raw['Quantity'] = clean_numeric(s_raw['Quantity'])
    d_raw['Forecast_Quantity'] = clean_numeric(d_raw['Forecast_Quantity'])
    lt_raw['Lead_Time_Days'] = clean_numeric(lt_raw['Lead_Time_Days'])
    lt_raw['Lead_Time_Std_Dev'] = clean_numeric(lt_raw['Lead_Time_Std_Dev'])

    # 1. Calculate Historical Sales Volatility (per Product/Location)
    s_stats = s_raw.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    s_stats.columns = ['Product', 'Location', 'Avg_Hist_Sales', 'Std_Hist_Sales']
    
    # 2. Prepare Lead Time Data (per Product/To_Location)
    lt_stats = lt_raw.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    lt_stats.rename(columns={'To_Location': 'Location', 'Lead_Time_Days': 'LT_Avg', 'Lead_Time_Std_Dev': 'LT_SD'}, inplace=True)

    # 3. Merge Demand with Stats to calculate Time-Phased Safety Stock
    # This keeps multiple rows per month for each SKU/Loc
    merged = pd.merge(d_raw, s_stats, on=['Product', 'Location'], how='left')
    merged = pd.merge(merged, lt_stats, on=['Product', 'Location'], how='left')
    
    # Handle missing values
    merged['Std_Hist_Sales'] = merged['Std_Hist_Sales'].fillna(0)
    merged['LT_Avg'] = merged['LT_Avg'].fillna(30)
    merged['LT_SD'] = merged['LT_SD'].fillna(5)

    # 4. TIME-PHASED CALCULATION
    # SS_t = Z * sqrt( (LT/30)*Std_Sales^2 + Forecast_t^2*(LT_SD/30)^2 )
    lt_m = merged['LT_Avg'] / 30
    ltsd_m = merged['LT_SD'] / 30
    
    merged['Safety_Stock'] = z_score * np.sqrt(
        (lt_m * (merged['Std_Hist_Sales']**2)) + 
        ((merged['Forecast_Quantity']**2) * (ltsd_m**2))
    )
    
    merged['Min_Corridor'] = merged['Safety_Stock']
    merged['Max_Corridor'] = merged['Safety_Stock'] + merged['Forecast_Quantity']
    
    return merged, s_raw, d_raw, lt_raw

# --- Main App ---
if sales_file and demand_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(sales_file), pd.read_csv(demand_file), pd.read_csv(lt_file)
    data, s_full, d_full, lt_full = process_data(df_s, df_d, df_lt)

    tab1, tab2, tab3 = st.tabs(["📉 Corridor Analysis", "🌐 Network Topology", "📋 Global Summary"])

    with tab1:
        c1, c2 = st.columns(2)
        sku = c1.selectbox("Select SKU", options=sorted(data['Product'].unique()))
        loc = c2.selectbox("Select Location", options=sorted(data[data['Product']==sku]['Location'].unique()))
        
        # Filter data for chart
        sku_loc_data = data[(data['Product'] == sku) & (data['Location'] == loc)].sort_values('Future_Forecast_Month')
        
        # Plotly Corridor
        fig = go.Figure()

        # Shaded Corridor
        fig.add_trace(go.Scatter(
            x=sku_loc_data['Future_Forecast_Month'], y=sku_loc_data['Max_Corridor'],
            mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sku_loc_data['Future_Forecast_Month'], y=sku_loc_data['Min_Corridor'],
            fill='tonexty', fillcolor='rgba(0, 123, 255, 0.15)', name='Target Inventory Corridor',
            line_color='rgba(0,0,0,0)'
        ))

        # Lines
        fig.add_trace(go.Scatter(x=sku_loc_data['Future_Forecast_Month'], y=sku_loc_data['Forecast_Quantity'], 
                                 name='Forecast', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=sku_loc_data['Future_Forecast_Month'], y=sku_loc_data['Safety_Stock'], 
                                 name='Safety Stock (Time-Phased)', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title=f"Time-Phased Corridor: {sku} @ {loc}", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 The Safety Stock (red line) now adjusts every month based on the forecasted volume and lead time uncertainty.")

    with tab2:
        st.subheader(f"Supply Chain Topology: {sku}")
        sku_lt = lt_full[lt_full['Product'] == sku]
        net = Network(height="1500px", width="100%", directed=True, bgcolor="#ffffff")
        
        nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        for n in nodes:
            # Color destinations differently
            node_color = "#ff4b4b" if n in sku_lt['To_Location'].values else "#31333F"
            net.add_node(n, label=n, color=node_color, size=20)
            
        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
            
        net.save_graph("net.html")
        components.html(open("net.html", 'r').read(), height=550)

    with tab3:
        st.subheader("Global Time-Phased Inventory Plan")
        # Show the full table of all combinations and months
        st.dataframe(
            data[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity', 'Safety_Stock', 'Min_Corridor', 'Max_Corridor']],
            use_container_width=True,
            hide_index=True,
            height=1000
        )

else:
    st.info("👋 Please upload your CSV files to see the time-phased corridor.")
