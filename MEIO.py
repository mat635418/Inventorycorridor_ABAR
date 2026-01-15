# MEIO for Raw Materials — SS Method 5 (σD & σLT)
# Updated for mat635418 — 2026-01-15
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

# -------------------------------
# PAGE CONFIG & TITLE (changed)
# -------------------------------
st.set_page_config(page_title="MEIO for Raw Materials — SS Method 5", layout="wide")
st.title("MEIO for Raw Materials — SS Method 5 (σD & σLT)")

# -------------------------------
# HELPERS / FORMATTING
# -------------------------------
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

def euro_format(x, always_two_decimals=True):
    try:
        if pd.isna(x):
            return ""
        neg = x < 0
        x_abs = abs(float(x))
        if always_two_decimals:
            s = f"{x_abs:,.2f}"
        else:
            if math.isclose(x_abs, round(x_abs)):
                s = f"{x_abs:,.0f}"
            else:
                s = f"{x_abs:,.2f}"
        s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"-{s}" if neg else s
    except Exception:
        return str(x)

def df_format_for_display(df, cols=None, two_decimals_cols=None):
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

def font_size_for_value(val, base=22):
    """Reduce font size when numbers are large (e.g., >10M)."""
    try:
        v = abs(float(val))
    except Exception:
        return base
    if v > 50_000_000:
        return max(12, base - 10)
    if v > 10_000_000:
        return max(14, base - 6)
    if v > 1_000_000:
        return max(16, base - 4)
    return base

def aggregate_network_stats(df_forecast, df_stats, df_lt):
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
# SIDEBAR & FILE LOADING
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
st.sidebar.subheader("🔒 Initial selection behaviour")
lock_initial = st.sidebar.checkbox("Lock first selected material/location across tabs (still editable)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Sources")
DEFAULT_FILES = {"sales": "sales.csv", "demand": "demand.csv", "lt": "leadtime.csv"}
s_upload = st.sidebar.file_uploader("1. Sales Data (Historical)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes)", type="csv")
s_file = s_upload if s_upload is not None else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload is not None else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload is not None else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)
if s_file: st.sidebar.success(f"✅ Sales Loaded: {s_file.name if hasattr(s_file,'name') else s_file}")
if d_file: st.sidebar.success(f"✅ Demand Loaded: {d_file.name if hasattr(d_file,'name') else d_file}")
if lt_file: st.sidebar.success(f"✅ Lead Time Loaded: {lt_file.name if hasattr(lt_file,'name') else lt_file}")

# Initialize session state for initial selection
if 'initial_sku' not in st.session_state:
    st.session_state['initial_sku'] = None
if 'initial_loc' not in st.session_state:
    st.session_state['initial_loc'] = None

# -------------------------------
# MAIN
# -------------------------------
if s_file and d_file and lt_file:
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

    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    stats['Local_Std'] = stats['Local_Std'].fillna(stats['Local_Mean'] * 0.2)

    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)
    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']],
                       on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')
    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': 0, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})

    # Safety Stock (Method 5)
    results['SS_Raw'] = z * np.sqrt(
        (results['Agg_Std_Hist']**2 / float(days_per_month)) * results['LT_Mean'] +
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / float(days_per_month))**2
    )
    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['SS_Raw']

    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0

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

    hist = df_s[['Product', 'Location', 'Period', 'Consumption', 'Forecast']].copy()
    hist.rename(columns={'Forecast': 'Forecast_Hist'}, inplace=True)
    hist['Deviation'] = hist['Consumption'] - hist['Forecast_Hist']
    hist['Abs_Error'] = hist['Deviation'].abs()
    hist['APE_%'] = (hist['Abs_Error'] / hist['Consumption'].replace(0, np.nan)).fillna(0) * 100
    hist['Accuracy_%'] = (1 - hist['APE_%'] / 100) * 100

    hist_net = (
        df_s.groupby(['Product', 'Period'], as_index=False)
            .agg(Network_Consumption=('Consumption', 'sum'),
                 Network_Forecast_Hist=('Forecast', 'sum'))
    )

    # Create tabs (added Material Dashboard)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Inventory Corridor",
        "🕸️ Network Topology",
        "📋 Full Plan",
        "⚖️ Efficiency Analysis",
        "📉 Forecast Accuracy",
        "🧮 Calculation Trace & Sim",
        "📦 By Material",
        "📊 Material Dashboard"
    ])

    # TAB 1: Inventory Corridor — first selection becomes initial
    with tab1:
        left, right = st.columns([3, 1])
        with left:
            sku_opts = sorted(results['Product'].unique())
            # choose default: initial if locked, else first SKU
            default_sku = st.session_state['initial_sku'] if (lock_initial and st.session_state['initial_sku'] in sku_opts) else sku_opts[0]
            sku = st.selectbox("Product", sku_opts, index=sku_opts.index(default_sku) if default_sku in sku_opts else 0)
            loc_opts = sorted(results[results['Product'] == sku]['Location'].unique())
            default_loc = st.session_state['initial_loc'] if (lock_initial and st.session_state['initial_loc'] in loc_opts) else (loc_opts[0] if loc_opts else "")
            loc = st.selectbox("Location", loc_opts, index=loc_opts.index(default_loc) if default_loc in loc_opts else 0)
            # store initial selection if not already present
            if st.session_state['initial_sku'] is None:
                st.session_state['initial_sku'] = sku
            if st.session_state['initial_loc'] is None:
                st.session_state['initial_loc'] = loc
            plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')

            fig = go.Figure([
                go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=0)),
                go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
                go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Direct Forecast', line=dict(color='black', dash='dot')),
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
            ])
            fig.update_layout(legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            total_fcst = float(plot_df['Forecast'].sum()) if not plot_df.empty else 0.0
            total_net = float(plot_df['Agg_Future_Demand'].sum()) if not plot_df.empty else 0.0
            total_ss = float(plot_df['Safety_Stock'].sum()) if not plot_df.empty else 0.0
            # dynamic font sizing
            main_size = font_size_for_value(max(total_fcst, total_net, total_ss), base=20)
            small_size = max(10, main_size - 6)
            badge_html = f"""
            <div style="background:#0b3d91;padding:12px;border-radius:8px;color:white;text-align:right;">
                <div style="font-size:{small_size}px;opacity:0.9">Selected</div>
                <div style="font-size:{main_size}px;font-weight:700">{sku} — {loc}</div>
                <div style="margin-top:6px;font-size:{small_size}px;opacity:0.95">
                    Fcst (Local): <strong>{euro_format(total_fcst, True)}</strong><br>
                    Net Demand: <strong>{euro_format(total_net, True)}</strong><br>
                    SS (Current): <strong>{euro_format(total_ss, True)}</strong>
                </div>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric("Total SS (sku/loc)", euro_format(total_ss, True))
            s2.metric("Total Net Demand", euro_format(total_net, True))

    # TAB 2: Network Topology — improved visuals, unused nodes light brown
    with tab2:
        # default SKU = initial if available else first
        sku_net_opts = sorted(results['Product'].unique())
        default_net_sku = st.session_state['initial_sku'] if (st.session_state['initial_sku'] in sku_net_opts) else sku_net_opts[0]
        sku_net = st.selectbox("Product for Network View", sku_net_opts, index=sku_net_opts.index(default_net_sku))
        next_month = sorted(results['Period'].unique())[0]
        label_data = results[results['Period'] == next_month].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku_net]

        net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        all_nodes = sorted(set(sku_lt['From_Location']).union(set(sku_lt['To_Location'])))

        demand_lookup = {loc: label_data.get((sku_net, loc), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0}) for loc in all_nodes}
        for n in all_nodes:
            m = demand_lookup.get(n, {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            used = (m['Agg_Future_Demand'] > 0) or (m['Forecast'] > 0)
            lbl = f"{n}\\nFcst: {euro_format(m['Forecast'])}\\nNet: {euro_format(m['Agg_Future_Demand'])}\\nSS: {euro_format(m['Safety_Stock'], True)}"
            if not used:
                # very light brown, muted text
                bg = '#f7efe6'  # very light brown/cream
                text_col = '#8b5a2b'  # brown text
                net.add_node(
                    n,
                    label=lbl,
                    title=lbl,
                    color={'background': bg, 'border': '#e0d3c4'},
                    shape='box',
                    font={'color': text_col, 'size': 10}
                )
            else:
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

        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
            to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
            # if both ends unused -> light brown arrow
            if (not from_used) and (not to_used):
                edge_color = '#d2b48c'  # tan/light brown
                font_col = '#8b5a2b'
            else:
                edge_color = '#666666'
                font_col = '#333333'
            net.add_edge(from_n, to_n, label=f"{int(r['Lead_Time_Days'])}d", color=edge_color, font={'color': font_col})

        net.set_options("""
        var options = {
          "physics": {"barnesHut": {"gravitationalConstant": -20000, "centralGravity": 0.3, "springLength": 95, "springConstant": 0.04}},
          "edges": {"smooth": true, "arrows": { "to": { "enabled": true } } }
        }
        """)
        net.save_graph("net.html")
        components.html(open("net.html").read(), height=950)

    # TAB 3: Full Plan — default filters tied to initial selection if locked
    with tab3:
        st.subheader("📋 Global Inventory Plan")
        col1, col2, col3 = st.columns(3)
        f_prod = col1.multiselect("Filter Product", sorted(results['Product'].unique()),
                                  default=[st.session_state['initial_sku']] if (lock_initial and st.session_state['initial_sku']) else [])
        f_loc = col2.multiselect("Filter Location", sorted(results['Location'].unique()),
                                 default=[st.session_state['initial_loc']] if (lock_initial and st.session_state['initial_loc']) else [])
        f_period = col3.multiselect("Filter Period", sorted(results['Period'].unique()))
        filtered = results.copy()
        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]
        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
        disp = df_format_for_display(filtered[display_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'], two_decimals_cols=['Forecast'])
        st.dataframe(disp, use_container_width=True, height=700)
        csv_buf = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # TAB 4: Efficiency Analysis
    with tab4:
        st.subheader("⚖️ Efficiency & Policy Analysis")
        sku_eff_opts = sorted(results['Product'].unique())
        default_eff = st.session_state['initial_sku'] if (st.session_state['initial_sku'] in sku_eff_opts) else sku_eff_opts[0]
        sku_eff = st.selectbox("Material", sku_eff_opts, index=sku_eff_opts.index(default_eff))
        next_month = sorted(results['Period'].unique())[0]
        eff = results[(results['Product'] == sku_eff) & (results['Period'] == next_month)].copy()
        eff['SS_to_FCST_Ratio'] = (eff['Safety_Stock'] / eff['Agg_Future_Demand'].replace(0, np.nan)).fillna(0)
        total_ss_sku = eff['Safety_Stock'].sum()
        total_net_demand_sku = eff['Agg_Future_Demand'].sum()
        sku_ratio = total_ss_sku / total_net_demand_sku if total_net_demand_sku > 0 else 0
        all_res = results[results['Period'] == next_month]
        global_ratio = all_res['Safety_Stock'].sum() / all_res['Agg_Future_Demand'].replace(0, np.nan).sum()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Network Ratio ({sku_eff})", f"{sku_ratio:.2f}")
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

    # TAB 5: Forecast Accuracy
    with tab5:
        st.subheader("📉 Historical Forecast vs Actuals")
        h_sku_opts = sorted(results['Product'].unique())
        default_hsku = st.session_state['initial_sku'] if (st.session_state['initial_sku'] in h_sku_opts) else h_sku_opts[0]
        h_sku = st.selectbox("Select Product", h_sku_opts, index=h_sku_opts.index(default_hsku))
        h_loc_opts = sorted(results[results['Product'] == h_sku]['Location'].unique())
        default_hloc = st.session_state['initial_loc'] if (st.session_state['initial_loc'] in h_loc_opts) else (h_loc_opts[0] if h_loc_opts else "")
        h_loc = st.selectbox("Select Location", h_loc_opts, index=h_loc_opts.index(default_hloc) if default_hloc in h_loc_opts else 0)
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
            st.subheader("🌐 Aggregated Network History (Selected Product)")
            net_table = (
                hist_net[hist_net['Product'] == h_sku]
                        .merge(hdf[['Period']].drop_duplicates(), on='Period', how='inner')
                        .sort_values('Period')
                        .drop(columns=['Product'])
            )
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

    # TAB 6: Calculation Trace & Sim
    with tab6:
        st.header("🧮 Transparent Calculation Engine")
        c1, c2, c3 = st.columns(3)
        calc_sku_opts = sorted(results['Product'].unique())
        default_calc_sku = st.session_state['initial_sku'] if (st.session_state['initial_sku'] in calc_sku_opts) else calc_sku_opts[0]
        calc_sku = c1.selectbox("Select Product", calc_sku_opts, index=calc_sku_opts.index(default_calc_sku))
        avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique())
        default_calc_loc = st.session_state['initial_loc'] if (st.session_state['initial_loc'] in avail_locs) else (avail_locs[0] if avail_locs else "")
        calc_loc = c2.selectbox("Select Location", avail_locs, index=avail_locs.index(default_calc_loc) if default_calc_loc in avail_locs else 0)
        avail_periods = sorted(results['Period'].unique())
        calc_period = c3.selectbox("Select Period", avail_periods)
        row = results[
            (results['Product'] == calc_sku) &
            (results['Location'] == calc_loc) &
            (results['Period'] == calc_period)
        ].iloc[0]
        st.markdown("---")
        st.subheader("1. Actual Inputs (Frozen)")
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Service Level", f"{service_level*100:.2f}%", help=f"Z-Score: {z:.4f}")
        i2.metric("Network Demand (D, monthly)", euro_format(row['Agg_Future_Demand'], True))
        i3.metric("Network Std Dev (σ_D, monthly)", euro_format(row['Agg_Std_Hist'], True))
        i4.metric("Avg Lead Time (L)", f"{row['LT_Mean']} days")
        i5.metric("LT Std Dev (σ_L)", f"{row['LT_Std']} days")
        st.subheader("2. Statistical Calculation (Actual)")
        term1_demand_var = (row['Agg_Std_Hist']**2 / float(days_per_month)) * row['LT_Mean']
        term2_supply_var = (row['LT_Std']**2) * ((row['Agg_Future_Demand'] / float(days_per_month))**2)
        combined_sd = np.sqrt(term1_demand_var + term2_supply_var)
        raw_ss_calc = z * combined_sd
        st.markdown("Using Safety Stock Method 5 (daily form):")
        st.latex(r"SS_{\text{raw}} = Z \times \sqrt{\,\sigma_D^2 \times L \;+\; \sigma_L^2 \times D^2\,}")
        st.markdown("**Step-by-Step Substitution (values used):**")
        st.code(f"""
1. σ_D_daily^2 = ({euro_format(row['Agg_Std_Hist'], True)})^2 / {days_per_month}
   Demand Component = {euro_format(term1_demand_var, True)}
2. Supply Component = {euro_format(term2_supply_var, True)}
3. Combined Variance = {euro_format(term1_demand_var + term2_supply_var, True)}
4. Combined Std Dev = {euro_format(combined_sd, True)}
5. Raw SS = {z:.4f} * {euro_format(combined_sd, True)} = {euro_format(raw_ss_calc, True)}
""")
        st.info(f"🧮 **Resulting Statistical SS (Method 5):** {euro_format(raw_ss_calc, True)} units")
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
                    st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) > Max Cap ({euro_format(upper_limit, True)}).")
                elif raw_ss_calc < lower_limit and row['Agg_Future_Demand'] > 0:
                    st.warning(f"⚠️ Raw SS ({euro_format(raw_ss_calc, True)}) < Min Cap ({euro_format(lower_limit, True)}).")
                else:
                    st.success("✅ Raw SS is within efficient boundaries.")
            else:
                st.write("Capping logic disabled.")
        st.markdown("---")
        st.subheader("4. What-If Simulation")
        sim_cols = st.columns(3)
        sim_sl = sim_cols[0].slider("Simulated Service Level (%)", min_value=50.0, max_value=99.9, value=service_level*100)
        sim_lt = sim_cols[1].slider("Simulated Avg Lead Time (Days)", min_value=0.0, max_value=max(30.0, row['LT_Mean']*2), value=float(row['LT_Mean']))
        sim_lt_std = sim_cols[2].slider("Simulated LT Variability (Days)", min_value=0.0, max_value=max(10.0, row['LT_Std']*2), value=float(row['LT_Std']))
        sim_z = norm.ppf(sim_sl / 100.0)
        sim_ss = sim_z * np.sqrt(
            (row['Agg_Std_Hist']**2 / float(days_per_month)) * sim_lt +
            (sim_lt_std**2) * (row['Agg_Future_Demand'] / float(days_per_month))**2
        )
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Original SS (Actual)", euro_format(raw_ss_calc, True))
        res_col2.metric("Simulated SS (New)", euro_format(sim_ss, True), delta=euro_format(sim_ss - raw_ss_calc, True))
        if sim_ss < raw_ss_calc:
            st.success(f"📉 Reducing uncertainty could lower inventory by **{euro_format(raw_ss_calc - sim_ss, True)}** units.")
        elif sim_ss > raw_ss_calc:
            st.warning(f"���� Increasing service or lead time requires **{euro_format(sim_ss - raw_ss_calc, True)}** more units.")

    # TAB 7: By Material (general)
    with tab7:
        st.header("📦 View by Material (No Locations)")
        t_period = st.selectbox("Select Period", sorted(results['Period'].unique()), key="mat_period")
        mat_agg = (
            results[results['Period'] == t_period]
            .groupby('Product', as_index=False)
            .agg(Forecast=('Forecast', 'sum'), Net_Demand=('Agg_Future_Demand', 'sum'),
                 Safety_Stock=('Safety_Stock', 'sum'), Nodes=('Location', lambda s: s.nunique()))
            .sort_values('Net_Demand', ascending=False)
        )
        st.plotly_chart(
            px.bar(mat_agg.melt(id_vars=['Product'], value_vars=['Net_Demand','Safety_Stock']),
                   x='Product', y='value', color='variable', barmode='group',
                   labels={'value': 'Units'}, title=f"Material Totals — {t_period.strftime('%Y-%m')}"),
            use_container_width=True)
        st.subheader("Table — Aggregated by Material")
        st.dataframe(df_format_for_display(mat_agg.copy(), cols=['Forecast','Net_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=450)
        st.download_button("📥 Download Material Aggregation (CSV)", data=mat_agg.to_csv(index=False), file_name="material_aggregation.csv", mime="text/csv")

    # TAB 8: Material Dashboard — focused on initial SKU and full detail
    with tab8:
        st.header("📊 Material Dashboard — Focus on initial selection")
        if st.session_state['initial_sku'] is None:
            st.warning("No initial material selected yet. Go to 'Inventory Corridor' and pick one.")
        else:
            sku_focus = st.session_state['initial_sku']
            st.subheader(f"Material: {sku_focus}")
            # aggregate across periods and locations
            mat_all = results[results['Product'] == sku_focus].copy()
            # timeseries: net demand / safety stock / forecast
            ts = mat_all.groupby('Period', as_index=False).agg(Net_Demand=('Agg_Future_Demand','sum'), Safety_Stock=('Safety_Stock','sum'), Forecast=('Forecast','sum'))
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(x=ts['Period'], y=ts['Net_Demand'], name='Net Demand', marker_color='#7fb3d5'))
            fig_ts.add_trace(go.Line(x=ts['Period'], y=ts['Safety_Stock'], name='Total Safety Stock', marker_color='#00cc96'))
            fig_ts.add_trace(go.Line(x=ts['Period'], y=ts['Forecast'], name='Total Forecast', marker_color='#636EFA', line=dict(dash='dot')))
            st.plotly_chart(fig_ts, use_container_width=True)

            # breakdown by location for the next period
            next_period = sorted(results['Period'].unique())[0]
            by_loc = mat_all[mat_all['Period'] == next_period].sort_values('Agg_Future_Demand', ascending=False)
            st.markdown(f"### Snapshot — {next_period.strftime('%Y-%m')}")
            st.plotly_chart(px.treemap(by_loc, path=['Location'], values='Agg_Future_Demand', title='Net Demand by Location'), use_container_width=True)

            # table with detailed metrics
            detail_cols = ['Location','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','LT_Mean','LT_Std']
            st.subheader("Detailed nodes for material")
            st.dataframe(df_format_for_display(by_loc[detail_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=450)
            st.download_button("📥 Download Material Detail (CSV)", data=by_loc[detail_cols].to_csv(index=False), file_name=f"{sku_focus}_detail.csv", mime="text/csv")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
