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
    """
    df_forecast: expects columns Product, Location, Period, Forecast
    df_stats:    expects columns Product, Location, Local_Mean, Local_Std
    df_lt:       expects columns Product, From_Location, To_Location, Lead_Time_Days, Lead_Time_Std_Dev
    """
    results = []
    months = df_forecast['Period'].unique()
    
    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in df_forecast['Product'].unique():

            # Per-product slices
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod]

            # All nodes (forecast + routes)
            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt['From_Location'])
            ).union(
                set(p_lt['To_Location'])
            )

            if not nodes:
                continue

            # Initialize demand and variance at nodes
            agg_demand = {n: p_fore.get(n, {'Forecast': 0})['Forecast'] for n in nodes}
            agg_var = {n: (p_stats.get(n, {'Local_Std': 0})['Local_Std'])**2 for n in nodes}

            # Parent-child mapping
            children = {}
            for _, row in p_lt.iterrows():
                children.setdefault(row['From_Location'], []).append(row['To_Location'])

            # Propagate demand up the network
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

s_file = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_file = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_file = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")

# ---------------------------------------------------------
# MAIN LOGIC (ONLY RUNS IF ALL FILES ARE PRESENT)
# ---------------------------------------------------------
if s_file and d_file and lt_file:

    # -----------------------------
    # LOAD & CLEAN DATA
    # -----------------------------
    df_s = pd.read_csv(s_file)
    df_d = pd.read_csv(d_file)
    df_lt = pd.read_csv(lt_file)

    # Strip column names
    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]

    # Expecting:
    # sales.csv:  Product, Location, Consumption, Forecast, Period, PurchasingGroupName
    # demand.csv: Product, Location, Forecast, Period, PurchasingGroupName

    # Normalize periods to month start
    df_s['Period'] = pd.to_datetime(df_s['Period']).dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = pd.to_datetime(df_d['Period']).dt.to_period('M').dt.to_timestamp()

    # Clean numeric
    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_s['Forecast'] = clean_numeric(df_s['Forecast'])           # historical forecast
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])           # future forecast

    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    # -----------------------------
    # HISTORICAL VARIABILITY
    # -----------------------------
    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    # -----------------------------
    # NETWORK AGGREGATION
    # -----------------------------
    # df_d now has Period and Forecast
    network_stats = aggregate_network_stats(
        df_forecast=df_d.rename(columns={'Forecast': 'Forecast'}),
        df_stats=stats,
        df_lt=df_lt
    )

    # Lead time per receiving node
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    # Merge: network stats + future forecast + LT
    results = pd.merge(
        network_stats,
        df_d[['Product', 'Location', 'Period', 'Forecast']],
        on=['Product', 'Location', 'Period'],
        how='left'
    )
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')

    # Fill defaults for missing values
    results = results.fillna({
        'Forecast': 0,
        'Agg_Std_Hist': 0,
        'LT_Mean': 7,
        'LT_Std': 2,
        'Agg_Future_Demand': 0
    })

    # -----------------------------
    # SAFETY STOCK
    # -----------------------------
    results['Safety_Stock'] = (
        z * np.sqrt(
            (results['LT_Mean'] / 30) * (results['Agg_Std_Hist']**2) +
            (results['LT_Std']**2) * (results['Agg_Future_Demand'] / 30)**2
        )
    ).round(0)

    # Force B616 SS to zero
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0

    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # -----------------------------
    # FORECAST ACCURACY (HISTORICAL)
    # -----------------------------
    # Using ONLY sales.csv: Consumption vs (historical) Forecast
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
        "📈 Inventory Corridor",
        "🕸️ Network Topology",
        "📋 Full Plan",
        "⚖️ Efficiency Analysis",
        "📉 Forecast Accuracy"
    ])

    # ---------------------------------------------------------
    # TAB 1 — INVENTORY CORRIDOR
    # ---------------------------------------------------------
    with tab1:
        sku = st.selectbox("Product", sorted(results['Product'].unique()))
        loc = st.selectbox("Location", sorted(results[results['Product'] == sku]['Location'].unique()))

        plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')

        fig = go.Figure([
            go.Scatter(
                x=plot_df['Period'],
                y=plot_df['Max_Corridor'],
                name='Max Corridor (SS + Local Forecast)',
                line=dict(width=0)
            ),
            go.Scatter(
                x=plot_df['Period'],
                y=plot_df['Safety_Stock'],
                name='Safety Stock',
                fill='tonexty',
                fillcolor='rgba(0,176,246,0.2)'
            ),
            go.Scatter(
                x=plot_df['Period'],
                y=plot_df['Forecast'],
                name='Local Direct Forecast',
                line=dict(color='black', dash='dot')
            ),
            go.Scatter(
                x=plot_df['Period'],
                y=plot_df['Agg_Future_Demand'],
                name='Total Network Demand (Aggregated)',
                line=dict(color='blue', dash='dash')
            )
        ])

        fig.update_layout(
            title=f"Inventory Plan for {sku} at {loc}",
            xaxis_title="Month",
            yaxis_title="Units"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2 — NETWORK TOPOLOGY
    # ---------------------------------------------------------
    with tab2:
        st.info("Nodes with 0 direct forecast (Hubs) are included and show aggregated network demand.")
        next_month = sorted(results['Period'].unique())[0]

        label_data = results[results['Period'] == next_month].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku]

        net = Network(height="900px", width="100%", directed=True, bgcolor="#eeeeee")

        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))

        for n in all_nodes:
            m = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            label = (
                f"{n}\n"
                f"Local: {int(m['Forecast'])}\n"
                f"Net: {int(m['Agg_Future_Demand'])}\n"
                f"SS: {int(m['Safety_Stock'])}"
            )
            color = '#31333F' if n in sku_lt['From_Location'].values else '#ff4b4b'
            net.add_node(n, label=label, title=label, color=color, shape='box', font={'color': 'white'})

        for _, r in sku_lt.iterrows():
            net.add_edge(r['From_Location'], r['To_Location'], label=f"{r['Lead_Time_Days']}d")

        net.save_graph("net.html")
        components.html(open("net.html").read(), height=950)

    # ---------------------------------------------------------
    # TAB 3 — FULL PLAN
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Global Inventory Plan")

        col1, col2, col3 = st.columns(3)
        f_prod = col1.multiselect("Filter Product", sorted(results['Product'].unique()))
        f_loc = col2.multiselect("Filter Location", sorted(results['Location'].unique()))
        f_period = col3.multiselect("Filter Period", sorted(results['Period'].unique()))

        filtered = results.copy()
        if f_prod:
            filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc:
            filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period:
            filtered = filtered[filtered['Period'].isin(f_period)]

        st.dataframe(
            filtered[
                [
                    'Product',
                    'Location',
                    'Period',
                    'Forecast',
                    'Agg_Future_Demand',
                    'Safety_Stock',
                    'Max_Corridor'
                ]
            ],
            use_container_width=True,
            height=900
        )

    # ---------------------------------------------------------
    # TAB 4 — EFFICIENCY ANALYSIS
    # ---------------------------------------------------------
    with tab4:
        st.subheader(f"⚖️ Efficiency Snapshot: {next_month}")

        eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()
        eff['SS_to_Fcst_Ratio'] = (eff['Safety_Stock'] / eff['Forecast'].replace(0, np.nan)).fillna(0)

        total_ss = eff['Safety_Stock'].sum()
        avg_ratio = eff[eff['Forecast'] > 0]['SS_to_Fcst_Ratio'].mean()
        high_risk_nodes = eff[eff['SS_to_Fcst_Ratio'] > 1.5].shape[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Safety Stock (Units)", f"{int(total_ss):,}")
        m2.metric("Avg SS-to-Forecast Ratio", f"{avg_ratio:.2f}")
        m3.metric("High Buffering Locations", high_risk_nodes)

        st.divider()

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_eff = px.scatter(
                eff,
                x="Forecast",
                y="Safety_Stock",
                color="Location",
                size="Agg_Future_Demand",
                hover_name="Location",
                labels={"Forecast": "Local Direct Demand", "Safety_Stock": "Proposed Safety Stock"},
                title="Inventory Positioning: Local Demand vs Safety Stock"
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with c2:
            st.markdown("**Top Stock-Heavy Locations**")
            heavy_ranking = eff.sort_values('Safety_Stock', ascending=False)[['Location', 'Safety_Stock', 'Forecast']]
            st.dataframe(heavy_ranking.head(10), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 5 — FORECAST ACCURACY (HISTORICAL)
    # ---------------------------------------------------------
    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")

        sku_hist = st.selectbox("Product", sorted(hist['Product'].unique()), key="h1")
        loc_hist = st.selectbox(
            "Location",
            sorted(hist[hist['Product'] == sku_hist]['Location'].unique()),
            key="h2"
        )

        hdf = hist[(hist['Product'] == sku_hist) & (hist['Location'] == loc_hist)].sort_values('Period')

        total_actual = hdf['Consumption'].replace(0, np.nan).sum()
        total_abs_error = hdf['Abs_Error'].sum()
        total_error = hdf['Deviation'].sum()

        mape = hdf['APE_%'].mean() if not hdf.empty else 0.0
        wape = (total_abs_error / total_actual * 100) if total_actual and not np.isnan(total_actual) else 0.0
        bias = (total_error / total_actual * 100) if total_actual and not np.isnan(total_actual) else 0.0
        mad = hdf['Abs_Error'].mean() if not hdf.empty else 0.0
        tracking_signal = (hdf['Deviation'].cumsum().iloc[-1] / mad) if mad else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAPE (%)", f"{mape:.1f}")
        k2.metric("WAPE (%)", f"{wape:.1f}")
        k3.metric("Bias (% of Actuals)", f"{bias:.1f}")
        k4.metric("Tracking Signal", f"{tracking_signal:.2f}")

        st.divider()

        fig_hist = go.Figure([
            go.Scatter(
                x=hdf['Period'],
                y=hdf['Consumption'],
                name='Actual Consumption',
                line=dict(color='black')
            ),
            go.Scatter(
                x=hdf['Period'],
                y=hdf['Forecast_Hist'],
                name='Historical Forecast',
                line=dict(color='blue', dash='dot')
            ),
            go.Bar(
                x=hdf['Period'],
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
                    'Period',
                    'Consumption',
                    'Forecast_Hist',
                    'Deviation',
                    'Abs_Error',
                    'APE_%',
                    'Accuracy_%'
                ]
            ],
            use_container_width=True
        )

else:
    st.info("Please upload all three CSV files in the sidebar to begin.")
