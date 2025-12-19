import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Inventory Corridor Explorer", layout="wide")

st.title("📦 Inventory Corridor Explorer")
st.markdown("Calculate **Safety Stock** and **Inventory Corridors** with automatic data cleaning.")

# --- Sidebar: Parameters ---
st.sidebar.header("⚙️ Calculation Parameters")
service_level = st.sidebar.slider("Desired Service Level (%)", 80, 99, 95)
z_map = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 98: 2.05, 99: 2.33}
z_score = z_map.get(service_level, 1.65)

# --- Data Loading ---
st.sidebar.header("📂 Upload Datasets")
sales_file = st.sidebar.file_uploader("1. Upload Sales (consumption) CSV", type="csv")
demand_file = st.sidebar.file_uploader("2. Upload Demand (forecast) CSV", type="csv")
lt_file = st.sidebar.file_uploader("3. Upload Lead Time CSV", type="csv")

def process_data(s_df, d_df, l_df):
    try:
        # 1. Robust Column Assignment (use position if names fail)
        # Rename first two columns to Product/Location regardless of their name
        s_df.columns.values[0] = 'Product'
        s_df.columns.values[1] = 'Location'
        d_df.columns.values[0] = 'Product'
        d_df.columns.values[1] = 'Location'
        
        # Clean Lead Time column names (remove spaces)
        l_df.columns = [c.strip() for c in l_df.columns]
        l_df = l_df.rename(columns={
            'to :': 'Location', 
            'Average measured leadtime': 'LT_Avg', 
            'std dev of leadtime used for SS calculation': 'LT_SD'
        })

        # 2. Identify Data Columns
        s_cols = [c for c in s_df.columns if any(x in c.lower() for x in ['consumption', 'm-'])]
        f_cols = [c for c in d_df.columns if any(x in c.lower() for x in ['forecast', 'm+'])]

        if not s_cols:
            st.error("Could not find 'Consumption' columns in Sales file.")
            return None, None, None

        # 3. FORCE NUMERIC (This fixes your current error)
        for col in s_cols:
            s_df[col] = pd.to_numeric(s_df[col], errors='coerce')
        for col in f_cols:
            d_df[col] = pd.to_numeric(d_df[col], errors='coerce')

        # 4. Calculate Stats
        s_df['Avg_Sales'] = s_df[s_cols].mean(axis=1, skipna=True)
        s_df['Std_Sales'] = s_df[s_cols].std(axis=1, skipna=True).fillna(0)
        
        # 5. Merge
        merged = pd.merge(s_df, d_df[['Product', 'Location'] + f_cols], on=['Product', 'Location'], how='inner')
        merged = pd.merge(merged, l_df, on='Location', how='left')

        # Fill missing Lead Time data with defaults to prevent crashes
        merged['LT_Avg'] = merged['LT_Avg'].fillna(30) # Default 30 days
        merged['LT_SD'] = merged['LT_SD'].fillna(5)    # Default 5 days std dev

        # 6. Safety Stock Calculation
        merged['LT_M'] = merged['LT_Avg'] / 30
        merged['LT_SD_M'] = merged['LT_SD'] / 30
        
        merged['Safety_Stock'] = z_score * np.sqrt(
            (merged['LT_M'] * (merged['Std_Sales']**2)) + 
            ((merged['Avg_Sales']**2) * (merged['LT_SD_M']**2))
        )
        
        merged['Min_Corridor'] = merged['Safety_Stock']
        merged['Max_Corridor'] = merged['Safety_Stock'] + merged['Avg_Sales']
        
        return merged, s_cols, f_cols
    
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return None, None, None

if sales_file and demand_file and lt_file:
    df_sales = pd.read_csv(sales_file)
    df_demand = pd.read_csv(demand_file)
    df_lt = pd.read_csv(lt_file)
    
    data, s_cols, f_cols = process_data(df_sales, df_demand, df_lt)
    
    if data is not None:
        # Selection UI
        col1, col2 = st.columns(2)
        u_products = sorted(data['Product'].dropna().unique())
        product = col1.selectbox("Select Product", options=u_products)
        
        u_locations = sorted(data[data['Product']==product]['Location'].dropna().unique())
        location = col2.selectbox("Select Location", options=u_locations)
        
        row = data[(data['Product'] == product) & (data['Location'] == location)].iloc[0]
        
        # Dashboard
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Monthly Sales", f"{row['Avg_Sales']:.0f}")
        m2.metric("Safety Stock", f"{row['Safety_Stock']:.0f}")
        m3.metric("Lead Time (Days)", f"{row['LT_Avg']:.1f}")
        m4.metric("Volatility (SD)", f"{row['Std_Sales']:.1f}")

        # Chart
        st.subheader(f"Inventory Corridor Analysis")
        
        hist_y = row[s_cols].values
        fore_y = row[f_cols].values
        labels = [c.replace('consumption ', '').replace('forecast ', '') for c in s_cols + f_cols]
        
        fig = go.Figure()
        # Area for Corridor
        fig.add_trace(go.Scatter(x=labels, y=[row['Max_Corridor']]*len(labels), mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=labels, y=[row['Min_Corridor']]*len(labels), fill='tonexty', fillcolor='rgba(0, 123, 255, 0.15)', name='Ideal Corridor', line_color='rgba(0,0,0,0)'))
        # Lines
        fig.add_trace(go.Scatter(x=labels[:len(s_cols)], y=hist_y, name='Hist. Sales', line=dict(color='black', width=3)))
        fig.add_trace(go.Scatter(x=labels[len(s_cols)-1:], y=np.concatenate([[hist_y[-1]], fore_y]), name='Forecast', line=dict(color='blue', dash='dash')))

        fig.update_layout(height=450, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Upload all three CSV files to begin.")
