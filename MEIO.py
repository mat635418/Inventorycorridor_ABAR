**Implementing and adjusting metrics**

I need to calculate APE (%) and add it to the `hist` dataframe, along with other key metrics. For `mape`, `wape`, `bias`, and `TS`, I'll compute them using existing columns, and make sure that the calculations hold by ensuring no zeroes in the quantities. I'll need to integrate the new Month/Year field properly while ensuring old code doesn’t break. This also means updating references to this new column across the board. Adjustments are required for tabs too, but no new imports or libraries, just the ones already in place. Let’s finalize this script, ensuring everything works smoothly.
```python
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
        series.astype(str)
        .str.replace(',', '')
        .str.replace('(', '-')
        .str.replace(')', '')
        .str.replace('-', '0')
        .str.strip(),
        errors='coerce'
    ).fillna(0)

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    results = []
    months = df_forecast['Future_Forecast_Month'].unique()
    
    for month in months:
        df_month = df_forecast[df_forecast['Future_Forecast_Month'] == month]
        for prod in df_forecast['Product'].unique():
            # Filter data for this specific product
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]
            
            # Critical: Include ALL nodes (those with forecast AND those only in lead time routes)
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(
                set(p_lt['To_Location'])
            )
            
            if not nodes:
                continue

            # Initialize demand and variance
            agg_demand = {n: p_fore.get(n, {'Forecast_Quantity': 0})['Forecast_Quantity'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': 0})['Local_Std'])**2 for n in nodes}
            
            # Map parent-child relationships
            children = {}
            for _, row in p_lt.iterrows():
                if row['From_Location'] not in children:
                    children[row['From_Location']] = []
                children[row['From_Location']].append(row['To_Location'])
                
            # Propagate demand up the network (Iterative approach)
            for _ in range(15):  # Increased iterations for deeper networks
                changed = False
                for parent in nodes:
                    if parent in children:
                        # Parent demand = Local Forecast + Sum of children's aggregated demand
                        new_d = p_fore.get(parent, {'Forecast_Quantity': 0})['Forecast_Quantity'] + sum(
                            agg_demand.get(c, 0) for c in children[parent]
                        )
                        new_v = (p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(
                            agg_var.get(c, 0) for c in children[parent]
                        )
                        
                        if abs(new_d - agg_demand[parent]) > 0.01:
                            agg_demand[parent], agg_var[parent] = new_d, new_v
                            changed = True
                if not changed:
                    break
                
            for n in nodes:
                results.append({
                    'Product': prod, 
                    'Location': n, 
                    'Future_Forecast_Month': month,
                    'Agg_Future_Demand': agg_demand[n], 
                    'Agg_Std_Hist': np.sqrt(agg_var[n])
                })
    return pd.DataFrame(results)

# --- Sidebar Parameter & File Upload ---
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 90.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

if s_file and d_file and lt_file:
    df_s, df_d, df_lt = pd.read_csv(s_file), pd.read_csv(d_file), pd.read_csv(lt_file)
    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]
    
    # --- Use Month/Year from sales and align with forecast ---
    # Assuming the new column in sales.csv is named exactly "Month/Year"
    df_s['Month_Year'] = pd.to_datetime(df_s['Month/Year'])
    df_d['Future_Forecast_Month'] = pd.to_datetime(df_d['Future_Forecast_Month'])

    df_s['Quantity'] = clean_numeric(df_s['Quantity'])
    df_d['Forecast_Quantity'] = clean_numeric(df_d['Forecast_Quantity'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # Calculate historical stats for variability
    stats = df_s.groupby(['Product', 'Location'])['Quantity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    # 1. Calculate Aggregated Network Demand
    network_stats = aggregate_network_stats(df_d, stats, df_lt)

    # 2. Process Lead Times per node
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # 3. CORRECTED MERGE LOGIC: 
    # Use network_stats as the left table to keep nodes with 0 direct forecast.
    results = pd.merge(network_stats, df_d, on=['Product', 'Location', 'Future_Forecast_Month'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    
    # Fill values for hubs and external locations
    results = results.fillna({
        'Forecast_Quantity': 0, 
        'Agg_Std_Hist': 0, 
        'LT_Mean': 7, 
        'LT_Std': 2, 
        'Agg_Future_Demand': 0
    })
    
    # 4. Final Calculations
    # Safety Stock formula accounting for Lead Time variability and Demand variability
    results['Safety_Stock'] = (z * np.sqrt(
        (results['LT_Mean'] / 30) * (results['Agg_Std_Hist']**2) + 
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / 30)**2
    )).round(0)

    # --- NEW: Force B616 to zero ---
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    # ------------------------------

    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast_Quantity']

    # --- Historical Forecast Accuracy (Actuals vs Forecast) ---
    # Merge actuals from sales with forecasts on month/product/location
    hist = pd.merge(
        df_s[['Product', 'Location', 'Month_Year', 'Quantity']],
        df_d[['Product', 'Location', 'Future_Forecast_Month', 'Forecast_Quantity']],
        left_on=['Product', 'Location', 'Month_Year'],
        right_on=['Product', 'Location', 'Future_Forecast_Month'],
        how='left'
    )

    hist['Forecast_Quantity'] = hist['Forecast_Quantity'].fillna(0)

    # Basic error metrics per period
    hist['Deviation'] = hist['Quantity'] - hist['Forecast_Quantity']  # Actual - Forecast
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['Accuracy_%'] = (1 - (hist['Abs_Error'] / hist['Quantity'].replace(0, np.nan))).fillna(0) * 100
    hist['APE_%'] = (hist['Abs_Error'] / hist['Quantity'].replace(0, np.nan)).fillna(0) * 100  # per-period APE (for MAPE)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Inventory Corridor",
        "🕸️ Network Topology",
        "📋 Full Plan",
        "⚖️ Efficiency Analysis",
        "📉 Forecast Accuracy"
    ])
    
    with tab1:
        sku = st.selectbox("Product", sorted(results['Product'].unique()))
        loc = st.selectbox("Location", sorted(results[results['Product'] == sku]['Location'].unique()))
        
        plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Future_Forecast_Month')
        
        fig = go.Figure([
            go.Scatter(
                x=plot_df['Future_Forecast_Month'],
                y=plot_df['Max_Corridor'],
                name='Max Corridor (SS + Local Forecast)',
                line=dict(width=0)
            ),
            go.Scatter(
                x=plot_df['Future_Forecast_Month'],
                y=plot_df['Safety_Stock'],
                name='Safety Stock',
                fill='tonexty',
                fillcolor='rgba(0,176,246,0.2)'
            ),
            go.Scatter(
                x=plot_df['Future_Forecast_Month'],
                y=plot_df['Forecast_Quantity'],
                name='Local Direct Forecast',
                line=dict(color='black', dash='dot')
            ),
            go.Scatter(
                x=plot_df['Future_Forecast_Month'],
                y=plot_df['Agg_Future_Demand'],
                name='Total Network Demand (Aggregated)',
                line=dict(color='blue', dash='dash')
            )
        ])
        fig.update_layout(title=f"Inventory Plan for {sku} at {loc}", xaxis_title="Month", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Nodes with 0 direct forecast (Hubs) are now included and show aggregated network demand.")
        next_month = sorted(results['Future_Forecast_Month'].unique())[0]
        label_data = results[results['Future_Forecast_Month'] == next_month].set_index(['Product', 'Location']).to_dict('index')

        net = Network(height="900px", width="100%", directed=True, bgcolor="#eeeeee")
        sku_lt = df_lt[df_lt['Product'] == sku]
        
        # Include all nodes mentioned in routes for this SKU
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        
        for n in all_nodes:
            m = label_data.get((sku, n), {'Forecast_Quantity': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label_text = (
                f"{n}\n"
                f"Local: {int(m['Forecast_Quantity'])}\n"
                f"Net: {int(m['Agg_Future_Demand'])}\n"
                f"SS: {int(m['Safety_Stock'])}"
            )
            
            # Color coding: Gray for suppliers/hubs, Red for customer-facing nodes
            color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
            net.add_node(n, label=label_text, title=label_text, color=color, shape='box', font={'color': 'white'})

        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")
            
        net.save_graph("net.html")
        components.html(open("net.html", 'r').read(), height=950)

    with tab3:
        st.subheader("Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        with col1:
            f_prod = st.multiselect("Filter Product", sorted(results['Product'].unique()), key="f1")
        with col2:
            f_loc = st.multiselect("Filter Location", sorted(results['Location'].unique()), key="f2")
        with col3:
            f_month = st.multiselect("Filter Month", sorted(results['Future_Forecast_Month'].unique()), key="f3")

        filtered_df = results.copy()
        if f_prod:
            filtered_df = filtered_df[filtered_df['Product'].isin(f_prod)]
        if f_loc:
            filtered_df = filtered_df[filtered_df['Location'].isin(f_loc)]
        if f_month:
            filtered_df = filtered_df[filtered_df['Future_Forecast_Month'].isin(f_month)]

        st.dataframe(
            filtered_df[
                [
                    'Product',
                    'Location',
                    'Future_Forecast_Month',
                    'Forecast_Quantity',
                    'Agg_Future_Demand',
                    'Safety_Stock',
                    'Max_Corridor'
                ]
            ],
            use_container_width=True,
            height=1000
        )

    with tab4:
        st.subheader(f"⚖️ Efficiency Snapshot: {next_month}")
        eff_df = results[(results['Product'] == sku) & (results['Future_Forecast_Month'] == next_month)].copy()
        
        eff_df['SS_to_Fcst_Ratio'] = (eff_df['Safety_Stock'] / eff_df['Forecast_Quantity'].replace(0, np.nan)).fillna(0)
        total_ss = eff_df['Safety_Stock'].sum()
        avg_ratio = eff_df[eff_df['Forecast_Quantity'] > 0]['SS_to_Fcst_Ratio'].mean()
        high_risk_nodes = eff_df[eff_df['SS_to_Fcst_Ratio'] > 1.5].shape[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Safety Stock (Units)", f"{int(total_ss):,}")
        m2.metric("Avg SS-to-Forecast Ratio", f"{avg_ratio:.2f}")
        m3.metric("High Buffering Locations", high_risk_nodes)

        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_eff = px.scatter(
                eff_df,
                x="Forecast_Quantity",
                y="Safety_Stock",
                color="Location",
                size="Agg_Future_Demand",
                hover_name="Location",
                labels={"Forecast_Quantity": "Local Direct Demand", "Safety_Stock": "Proposed Safety Stock"},
                title="Inventory Positioning: Local Demand vs Safety Stock"
            )
            st.plotly_chart(fig_eff, use_container_width=True)
        with c2:
            st.markdown("**Top Stock-Heavy Locations**")
            heavy_ranking = eff_df.sort_values('Safety_Stock', ascending=False)[['Location', 'Safety_Stock', 'Forecast_Quantity']]
            st.dataframe(heavy_ranking.head(10), use_container_width=True)

    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")

        # Selectors for product/location
        sku_hist = st.selectbox("Product", sorted(hist['Product'].unique()), key="h1")
        loc_hist = st.selectbox(
            "Location",
            sorted(hist[hist['Product'] == sku_hist]['Location'].unique()),
            key="h2"
        )

        hdf = hist[(hist['Product'] == sku_hist) & (hist['Location'] == loc_hist)].sort_values('Month_Year')

        # --- Aggregate accuracy metrics for this plant/material ---
        total_actual = hdf['Quantity'].replace(0, np.nan).sum()
        total_abs_error = hdf['Abs_Error'].sum()
        total_error = hdf['Deviation'].sum()

        # MAPE: mean of APE_%
        mape = hdf['APE_%'].mean() if not hdf.empty else 0.0

        # WAPE: sum(|error|) / sum(actual)
        wape = (total_abs_error / total_actual * 100) if total_actual and not np.isnan(total_actual) else 0.0

        # Bias: signed error as % of total actuals
        bias = (total_error / total_actual * 100) if total_actual and not np.isnan(total_actual) else 0.0

        # Tracking Signal: cumulative error / MAD
        mad = hdf['Abs_Error'].mean() if not hdf.empty else 0.0
        cumulative_error = hdf['Deviation'].cumsum().iloc[-1] if not hdf.empty else 0.0
        tracking_signal = cumulative_error / mad if mad != 0 else 0.0

        # --- Top level KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAPE (%)", f"{mape:.1f}")
        k2.metric("WAPE (%)", f"{wape:.1f}")
        k3.metric("Bias (% of Actuals)", f"{bias:.1f}")
        k4.metric("Tracking Signal", f"{tracking_signal:.2f}")

        st.divider()

        # --- Time series plot: Actual vs Forecast + Deviation ---
        fig_hist = go.Figure([
            go.Scatter(
                x=hdf['Month_Year'],
                y=hdf['Quantity'],
                name='Actual Sales',
                line=dict(color='black')
            ),
            go.Scatter(
                x=hdf['Month_Year'],
                y=hdf['Forecast_Quantity'],
                name='Forecast',
                line=dict(color='blue', dash='dot')
            ),
            go.Bar(
                x=hdf['Month_Year'],
                y=hdf['Deviation'],
                name='Deviation (Actual - Forecast)',
                marker_color='red',
                opacity=0.4
            )
        ])

        fig_hist.update_layout(
            title=f"Historical Forecast Accuracy for {sku_hist} at {loc_hist}",
            xaxis_title="Month",
            yaxis_title="Units",
            barmode='overlay'
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📊 Detailed Accuracy by Month")
        st.dataframe(
            hdf[
                [
                    'Month_Year',
                    'Quantity',
                    'Forecast_Quantity',
                    'Deviation',
                    'Abs_Error',
                    'Accuracy_%',
                    'APE_%'
                ]
            ],
            use_container_width=True
        )

else:
    st.info("Please upload all three CSV files in the sidebar to begin.")
```
