# Multi-Echelon Inventory Optimizer — Enhanced Version
# Enhanced by Copilot for mat635418 — 2026-01-15
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from scipy.stats import norm
import os
import math
from io import StringIO

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer (Method 5 SS)", layout="wide")
st.title("📊 Multi-Echelon Network Inventory Optimizer — SS Method 5 (σD & σLT) — Enhanced")

# -------------------------------
# HELPERS / FORMATTING
# -------------------------------
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

def euro_format(x, always_two_decimals=True):
    """
    Formats numbers with '.' as thousand separator and ',' as decimal separator.
    Examples: 1234.5 -> '1.234,50' (if always_two_decimals True)
    """
    try:
        if pd.isna(x):
            return ""
        neg = x < 0
        x_abs = abs(float(x))
        if always_two_decimals:
            s = f"{x_abs:,.2f}"  # 1,234,567.89
        else:
            # choose integer if it's whole
            if math.isclose(x_abs, round(x_abs)):
                s = f"{x_abs:,.0f}"
            else:
                s = f"{x_abs:,.2f}"
        # swap separators: comma->tmp, dot->comma, tmp->dot
        s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"-{s}" if neg else s
    except Exception:
        return str(x)

def df_format_for_display(df, cols=None, two_decimals_cols=None):
    """
    Returns a copy of df with selected numeric columns formatted to euro_format strings.
    If cols is None, attempt to format common columns.
    two_decimals_cols: list of columns to force two decimals; others will be formatted with integer where possible.
    """
    d = df.copy()
    if cols is None:
        cols = [c for c in d.columns if d[c].dtype.kind in 'biufc']
    for c in cols:
        if c in d.columns:
            if two_decimals_cols and c in two_decimals_cols:
                d[c] = d[c].apply(lambda v: euro_format(v, always_two_decimals=True))
            else:
                d[c] = d[c].apply(lambda v: euro_format(v, always_two_decimals=False))
    return d

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

# -------------------------------
# SIDEBAR & FILE LOADING LOGIC
# -------------------------------
st.sidebar.header("⚙️ Parameters")
service_level = st.sidebar.slider("Service Level (%)", 50.0, 99.9, 99.0) / 100
z = norm.ppf(service_level)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Calculation Settings")
days_per_month = st.sidebar.number_input("Days per month (used to convert monthly->daily)", value=30, min_value=1)
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Safety Stock Rules")
zero_if_no_net_fcst = st.sidebar.checkbox("Force Zero SS if No Network Demand", value=True)
apply_cap = st.sidebar.checkbox("Enable SS Capping (% of Network Demand)", value=True)
cap_range = st.sidebar.slider("Cap Range (%)", 0, 500, (0, 200),
                              help="Ensures SS stays between these % of total network demand for that node.")

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

# -------------------------------
# MAIN LOGIC
# -------------------------------
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
    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']],
                       on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # --------------------------------
    # SAFETY STOCK — SS METHOD 5 (vectorized)
    # --------------------------------
    # Convert monthly -> daily using days_per_month
    results['SS_Raw'] = z * np.sqrt(
        (results['Agg_Std_Hist']**2 / float(days_per_month)) * results['LT_Mean'] +
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / float(days_per_month))**2
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

    # Round Safety Stock for final presentation, but keep raw in SS_Raw for trace
    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    # Domain-specific overrides (kept from original)
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # ACCURACY DATA (LOCAL)
    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)).fillna(0) * 100
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    # --- NEW: Aggregated historical network view (per Product, Period) ---
    hist_net = (
        df_s.groupby(['Product', 'Period'], as_index=False)
            .agg(Network_Consumption=('Consumption', 'sum'),
                 Network_Forecast_Hist=('Forecast', 'sum'))
    )

    # -------------------------------
    # TABS (added "By Material")
    # -------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Inventory Corridor",
        "🕸️ Network Topology",
        "📋 Full Plan",
        "⚖️ Efficiency Analysis",
        "📉 Forecast Accuracy",
        "🧮 Calculation Trace & Sim",
        "📦 By Material"
    ])

    # -------------------------------
    # TAB 1: Inventory Corridor (with highlighted selection)
    # -------------------------------
    with tab1:
        # Left: selectors + plot ; Right: highlighted badge + quick KPIs
        left, right = st.columns([3, 1])
        with left:
            sku = st.selectbox("Product", sorted(results['Product'].unique()))
            loc = st.selectbox("Location", sorted(results[results['Product'] == sku]['Location'].unique()))
            plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')

            # show corridor chart
            fig = go.Figure([
                go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=0)),
                go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
                go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Direct Forecast', line=dict(color='black', dash='dot')),
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
            ])
            fig.update_layout(legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

        # Right: big highlighted badge with selection and key numbers (top-right)
        with right:
            # badge with large font and background
            badge_html = f"""
            <div style="background:#0b3d91;padding:18px;border-radius:8px;color:white;text-align:right;">
                <div style="font-size:14px;opacity:0.8">Selected</div>
                <div style="font-size:22px;font-weight:700">{sku} — {loc}</div>
                <div style="margin-top:8px;font-size:13px;opacity:0.95">
                    Fcst (Local): <strong>{euro_format(float(plot_df['Forecast'].sum()))}</strong><br>
                    Net Demand: <strong>{euro_format(float(plot_df['Agg_Future_Demand'].sum()))}</strong><br>
                    SS (Current): <strong>{euro_format(float(plot_df['Safety_Stock'].sum()), True)}</strong>
                </div>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

            # quick KPIs
            s1, s2 = st.columns(2)
            s1.metric("Total SS (sku/loc)", euro_format(float(plot_df['Safety_Stock'].sum()), True))
            s2.metric("Total Net Demand", euro_format(float(plot_df['Agg_Future_Demand'].sum()), True))

    # -------------------------------
    # TAB 2: Network Topology (enhanced greying)
    # -------------------------------
    with tab2:
        sku = st.selectbox("Product for Network View", sorted(results['Product'].unique()), key="network_sku")
        next_month = sorted(results['Period'].unique())[0]
        label_data = results[results['Period'] == next_month].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku]

        net = Network(height="1200px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))

        # Build a quick lookup to know which nodes have demand for this sku in the period
        demand_lookup = {loc: label_data.get((sku, loc), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0}) for loc in all_nodes}

        for n in sorted(all_nodes):
            m = demand_lookup.get(n, {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            used = (m['Agg_Future_Demand'] > 0) or (m['Forecast'] > 0)
            # format label numbers using euro_format
            lbl = f"{n}\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            if not used:
                # much greyer, smaller font, muted background
                net.add_node(
                    n,
                    label=lbl,
                    title=lbl,
                    color={'background': '#f0f0f0', 'border': '#cccccc'},
                    shape='box',
                    font={'color': '#888888', 'size': 10}
                )
            else:
                # stronger node color
                is_source = n in set(sku_lt['From_Location'])
                bg = '#2E7D32' if is_source else '#FF5252'
                net.add_node(
                    n,
                    label=lbl,
                    title=lbl,
                    color={'background': bg, 'border': '#222222'},
                    shape='box',
                    font={'color': 'white', 'size': 12}
                )

        # Add edges and grey them if both ends unused
        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
            to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
            edge_color = '#bbbbbb' if (not from_used and not to_used) else '#666666'
            net.add_edge(from_n, to_n, label=f"{int(r['Lead_Time_Days'])}d", color=edge_color)

        net.set_options("""
        var options = {
          "physics": {"stabilization": {"iterations": 250}},
          "nodes": {"borderWidthSelected":2}
        }
        """)
        net.save_graph("net.html")
        components.html(open("net.html").read(), height=1250)

    # -------------------------------
    # TAB 3: Full Plan (with exports)
    # -------------------------------
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

        # Format for display
        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
        disp = df_format_for_display(filtered[display_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'], two_decimals_cols=['Forecast'])
        st.dataframe(disp, use_container_width=True, height=700)

        # Export CSV of the filtered view
        csv_buf = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # -------------------------------
    # TAB 4: Efficiency Analysis
    # -------------------------------
    with tab4:
        st.subheader("⚖️ Efficiency & Policy Analysis")
        sku = st.selectbox("Material", sorted(results['Product'].unique()), key="eff_sku")
        next_month = sorted(results['Period'].unique())[0]
        eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()

        # Ratio based on network demand
        eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Agg_Future_Demand'].replace(0, np.nan)).fillna(0)

        total_ss_sku = eff['Safety_Stock'].sum()
        total_net_demand_sku = eff['Agg_Future_Demand'].sum()
        sku_ratio = total_ss_sku / total_net_demand_sku if total_net_demand_sku > 0 else 0

        all_res = results[results['Period'] == next_month]
        global_ratio = all_res['Safety_Stock'].sum() / all_res['Agg_Future_Demand'].replace(0, np.nan).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Network Ratio ({sku})", f"{sku_ratio:.2f}")
        m2.metric("Global Network Ratio (All Items)", f"{global_ratio:.2f}")
        m3.metric("Total SS for Material", euro_format(int(total_ss_sku), True))

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_eff = px.scatter(
                eff, x="Agg_Future_Demand", y="Safety_Stock", color="Adjustment_Status",
                size="SS_to_FCST_Ratio", hover_name="Location",
                color_discrete_map={'Optimal (Statistical)': '#00CC96', 'Capped (High)': '#EF553B','Capped (Low)': '#636EFA', 'Forced to Zero': '#AB63FA'},
                title="Policy Impact & Efficiency Ratio (Bubble Size = SS_to_FCST_Ratio)"
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with c2:
            st.markdown("**Status Breakdown**")
            st.table(eff['Adjustment_Status'].value_counts())

            st.markdown("**Top Nodes by Efficiency Gap**")
            eff['Gap'] = (eff['Safety_Stock'] - eff['SS_Raw']).abs()
            st.dataframe(
                df_format_for_display(
                    eff.sort_values('Gap', ascending=False)[
                        ['Location', 'Adjustment_Status', 'Safety_Stock', 'SS_to_FCST_Ratio']
                    ],
                    cols=['Safety_Stock'],
                    two_decimals_cols=['Safety_Stock']
                ).head(10),
                use_container_width=True
            )

    # -------------------------------
    # TAB 5: Forecast Accuracy
    # -------------------------------
    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")

        # Selectboxes pull from 'results' to show ALL materials/locations
        h_sku = st.selectbox("Select Product", sorted(results['Product'].unique()), key="h1")
        h_loc = st.selectbox("Select Location", sorted(results[results['Product'] == h_sku]['Location'].unique()), key="h2")

        hdf = hist[(hist['Product'] == h_sku) & (hist['Location'] == h_loc)].sort_values('Period')

        if not hdf.empty:
            k1, k2, k3 = st.columns(3)
            wape_val = (hdf['Abs_Error'].sum() / hdf['Consumption'].replace(0, np.nan).sum() * 100)
            bias_val = (hdf['Deviation'].sum() / hdf['Consumption'].replace(0, np.nan).sum() * 100)
            avg_acc = hdf['Accuracy_%'].mean()
            k1.metric("WAPE (%)", f"{wape_val:.1f}")
            k2.metric("Bias (%)", f"{bias_val:.1f}")
            k3.metric("Avg Accuracy (%)", f"{avg_acc:.1f}")

            fig_hist = go.Figure([
                go.Scatter(x=hdf['Period'], y=hdf['Consumption'], name='Actuals', line=dict(color='black')),
                go.Scatter(x=hdf['Period'], y=hdf['Forecast_Hist'], name='Forecast', line=dict(color='blue', dash='dot')),
            ])
            st.plotly_chart(fig_hist, use_container_width=True)

            # Aggregated network totals table for the selected product
            st.subheader("🌐 Aggregated Network History (Selected Product)")
            net_table = (
                hist_net[hist_net['Product'] == h_sku]
                        .merge(hdf[['Period']].drop_duplicates(), on='Period', how='inner')
                        .sort_values('Period')
                        .drop(columns=['Product'])
            )

            # Network-level WAPE
            if not net_table.empty:
                net_table['Net_Abs_Error'] = (net_table['Network_Consumption'] - net_table['Network_Forecast_Hist']).abs()
                net_wape = (net_table['Net_Abs_Error'].sum() /
                            net_table['Network_Consumption'].replace(0, np.nan).sum() * 100)
            else:
                net_wape = 0.0

            c_net1, c_net2 = st.columns([3, 1])
            with c_net1:
                st.dataframe(df_format_for_display(net_table[['Period', 'Network_Consumption', 'Network_Forecast_Hist']].copy(),
                                                   cols=['Network_Consumption','Network_Forecast_Hist'], two_decimals_cols=['Network_Consumption']), use_container_width=True, height=500)
            with c_net2:
                st.metric("Network WAPE (%)", f"{net_wape:.1f}")

            st.subheader("📊 Detailed Accuracy by Month")
            st.dataframe(df_format_for_display(hdf[['Period','Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']].copy(),
                                              cols=['Consumption','Forecast_Hist','Deviation','Abs_Error','APE_%','Accuracy_%']), use_container_width=True, height=500)
        else:
            st.warning("⚠️ No historical sales data (sales.csv) found for this selection. Accuracy metrics cannot be calculated.")

    # --------------------------------
    # TAB 6: Calculation Trace & Simulation
    # --------------------------------
    with tab6:
        st.header("🧮 Transparent Calculation Engine")
        st.write("Select a specific node and period to see exactly how the Safety Stock number was derived.")

        # 1. Selection Controls
        c1, c2, c3 = st.columns(3)
        calc_sku = c1.selectbox("Select Product", sorted(results['Product'].unique()), key="c_sku")

        avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique())
        calc_loc = c2.selectbox("Select Location", avail_locs, key="c_loc")

        avail_periods = sorted(results['Period'].unique())
        calc_period = c3.selectbox("Select Period", avail_periods, key="c_period")

        row = results[
            (results['Product'] == calc_sku) &
            (results['Location'] == calc_loc) &
            (results['Period'] == calc_period)
        ].iloc[0]

        st.markdown("---")

        # 2. Input Variables Display (formatted)
        st.subheader("1. Actual Inputs (Frozen)")
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Service Level", f"{service_level*100:.2f}%", help=f"Z-Score: {z:.4f}")
        i2.metric("Network Demand (D, monthly)", euro_format(row['Agg_Future_Demand'], True), help="Aggregated Future Demand (monthly)")
        i3.metric("Network Std Dev (σ_D, monthly)", euro_format(row['Agg_Std_Hist'], True), help="Aggregated Historical Std Dev (monthly totals)")
        i4.metric("Avg Lead Time (L)", f"{row['LT_Mean']} days")
        i5.metric("LT Std Dev (σ_L)", f"{row['LT_Std']} days")

        # 3. Mathematical Trace
        st.subheader("2. Statistical Calculation (Actual)")
        term1_demand_var = (row['Agg_Std_Hist']**2 / float(days_per_month)) * row['LT_Mean']
        term2_supply_var = (row['LT_Std']**2) * ((row['Agg_Future_Demand'] / float(days_per_month))**2)
        combined_sd = np.sqrt(term1_demand_var + term2_supply_var)
        raw_ss_calc = z * combined_sd

        st.markdown("Using Safety Stock Method 5 (daily form):")
        st.latex(r"SS_{\text{raw}} = Z \times \sqrt{\,\sigma_D^2 \times L \;+\; \sigma_L^2 \times D^2\,}")
        st.markdown("Where σ_D and D are daily values (converted from monthly inputs in the dataset).")

        st.markdown("**Step-by-Step Substitution (values used):**")
        st.code(f"""
1. σ_D_daily^2 (from monthly agg std) = ({euro_format(row['Agg_Std_Hist'], True)})^2 / {days_per_month}
   Demand Component = σ_D_daily^2 * L = {euro_format(term1_demand_var, True)}
2. Supply Component = σ_L^2 * D_daily^2 = ({euro_format(row['LT_Std'], True)})^2 * ({euro_format(row['Agg_Future_Demand'], True)} / {days_per_month})^2
   = {euro_format(term2_supply_var, True)}
3. Combined Variance = {euro_format(term1_demand_var, True)} + {euro_format(term2_supply_var, True)}
   = {euro_format(term1_demand_var + term2_supply_var, True)}
4. Combined Std Dev = sqrt(Combined Variance)
   = {euro_format(combined_sd, True)}
5. Raw SS = {z:.4f} (Z-Score) * {euro_format(combined_sd, True)}
   = {euro_format(raw_ss_calc, True)} units
""")
        st.info(f"🧮 **Resulting Statistical SS (Method 5):** {euro_format(raw_ss_calc, True)} units")

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
                st.write(f"Range: [{euro_format(lower_limit, True)}, {euro_format(upper_limit, True)}]")
                if raw_ss_calc > upper_limit:
                    st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) > Max Cap ({euro_format(upper_limit, True)}). Capping downwards.")
                elif raw_ss_calc < lower_limit and row['Agg_Future_Demand'] > 0:
                    st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) < Min Cap ({euro_format(lower_limit, True)}). Buffering upwards.")
                else:
                    st.success("✅ Raw SS is within efficient boundaries.")
            else:
                st.write("Capping logic disabled.")

        st.markdown("---")

        # 5. SIMULATION MODE
        st.subheader("4. What-If Simulation")
        st.write("Tweak parameters below to see how Safety Stock reacts *without* changing the global settings.")

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

        sim_z = norm.ppf(sim_sl / 100.0)
        sim_ss = sim_z * np.sqrt(
            (row['Agg_Std_Hist']**2 / float(days_per_month)) * sim_lt +
            (sim_lt_std**2) * (row['Agg_Future_Demand'] / float(days_per_month))**2
        )

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Original SS (Actual)", euro_format(raw_ss_calc, True))
        res_col2.metric(
            "Simulated SS (New)",
            euro_format(sim_ss, True),
            delta=euro_format(sim_ss - raw_ss_calc, True),
            delta_color="inverse"
        )
        if sim_ss < raw_ss_calc:
            st.success(f"📉 Reducing uncertainty could lower inventory by **{euro_format(raw_ss_calc - sim_ss, True)}** units.")
        elif sim_ss > raw_ss_calc:
            st.warning(f"📈 Increasing service or lead time requires **{euro_format(sim_ss - raw_ss_calc, True)}** more units.")

    # -------------------------------
    # TAB 7: By Material (new tab)
    # -------------------------------
    with tab7:
        st.header("📦 View by Material (No Locations)")
        t_period = st.selectbox("Select Period", sorted(results['Period'].unique()), key="mat_period")
        mat_agg = (
            results[results['Period'] == t_period]
            .groupby('Product', as_index=False)
            .agg(
                Forecast=('Forecast', 'sum'),
                Net_Demand=('Agg_Future_Demand', 'sum'),
                Safety_Stock=('Safety_Stock', 'sum'),
                Nodes=('Location', lambda s: s.nunique())
            )
            .sort_values('Net_Demand', ascending=False)
        )

        st.plotly_chart(
            px.bar(mat_agg.melt(id_vars=['Product'], value_vars=['Net_Demand','Safety_Stock']),
                   x='Product', y='value', color='variable', barmode='group',
                   labels={'value': 'Units'}, title=f"Material Totals — {t_period.strftime('%Y-%m')}")
            , use_container_width=True)

        st.subheader("Table — Aggregated by Material")
        st.dataframe(df_format_for_display(mat_agg.copy(), cols=['Forecast','Net_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=500)

        # Export
        st.download_button("📥 Download Material Aggregation (CSV)", data=mat_agg.to_csv(index=False), file_name="material_aggregation.csv", mime="text/csv")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
