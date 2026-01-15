# Multi-Echelon Inventory Optimizer — Enhanced Version (Reviewed & Improved)
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

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Multi-Echelon Inventory Optimizer (Method 5)", layout="wide")
st.title("📊 MEIO for Raw Materials - Method 5 SS formula")

# -------------------------------
# HELPERS / FORMATTING
# -------------------------------
def clean_numeric(series):
    """
    Robust numeric cleaning:
    - Accepts numbers with thousand separators and decimals.
    - Converts "(123)" -> -123
    - Converts empty strings or '-' to NaN (not 0).
    - Leaves negative signs intact.
    Returns numeric (float) with NaN for unparsable values.
    """
    s = series.astype(str).str.strip()
    s = s.replace({'': np.nan, '-': np.nan, '—': np.nan, 'na': np.nan, 'n/a': np.nan, 'None': np.nan})
    # parentheses to negative
    paren_mask = s.str.startswith('(') & s.str.endswith(')')
    s.loc[paren_mask] = '-' + s.loc[paren_mask].str[1:-1]
    s = s.str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    s = s.str.replace(r'[^\d\.\-]+', '', regex=True)
    out = pd.to_numeric(s, errors='coerce')
    return out

def euro_format(x, always_two_decimals=True):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        neg = float(x) < 0
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

def aggregate_network_stats(df_forecast, df_stats, df_lt):
    """
    Propagates demand and variance up the supply chain network.
    """
    results = []
    months = df_forecast['Period'].unique()
    products = df_forecast['Product'].unique()
    for month in months:
        df_month = df_forecast[df_forecast['Period'] == month]
        for prod in products:
            p_stats = df_stats[df_stats['Product'] == prod].set_index('Location').to_dict('index')
            p_fore = df_month[df_month['Product'] == prod].set_index('Location').to_dict('index')
            p_lt = df_lt[df_lt['Product'] == prod] if 'Product' in df_lt.columns else df_lt.copy()

            nodes = set(df_month[df_month['Product'] == prod]['Location']).union(
                set(p_lt.get('From_Location', pd.Series([]))) if not p_lt.empty else set()
            ).union(
                set(p_lt.get('To_Location', pd.Series([]))) if not p_lt.empty else set()
            )
            if not nodes:
                continue

            agg_demand = {n: float(p_fore.get(n, {'Forecast': 0})['Forecast']) for n in nodes}
            agg_var = {}
            for n in nodes:
                local_std = p_stats.get(n, {}).get('Local_Std', np.nan)
                agg_var[n] = np.nan if pd.isna(local_std) else float(local_std)**2

            children = {}
            if not p_lt.empty:
                for _, row in p_lt.iterrows():
                    children.setdefault(row['From_Location'], []).append(row['To_Location'])

            for _ in range(30):
                changed = False
                for parent in nodes:
                    child_list = children.get(parent, [])
                    if child_list:
                        new_d = float(p_fore.get(parent, {'Forecast': 0})['Forecast']) + sum(agg_demand.get(c, 0) for c in child_list)
                        child_vars = [agg_var.get(c, np.nan) for c in child_list]
                        if any(pd.isna(v) for v in child_vars) or pd.isna(agg_var.get(parent, np.nan)):
                            new_v = np.nan
                        else:
                            new_v = float(p_stats.get(parent, {'Local_Std': 0})['Local_Std'])**2 + sum(child_vars)
                        if abs(new_d - agg_demand[parent]) > 0.0001 or (pd.isna(agg_var[parent]) and not pd.isna(new_v)):
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
                    'Agg_Future_Demand': agg_demand.get(n, 0.0),
                    'Agg_Std_Hist': np.sqrt(agg_var[n]) if (n in agg_var and not pd.isna(agg_var[n])) else np.nan
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
st.sidebar.subheader("📂 Data Sources (CSV)")
DEFAULT_FILES = {"sales": "sales.csv", "demand": "demand.csv", "lt": "leadtime.csv"}

s_upload = st.sidebar.file_uploader("1. Sales Data (Historical: sales.csv)", type="csv")
d_upload = st.sidebar.file_uploader("2. Demand Data (Future Forecast: demand.csv)", type="csv")
lt_upload = st.sidebar.file_uploader("3. Lead Time Data (Network Routes: leadtime.csv)", type="csv")

s_file = s_upload if s_upload is not None else (DEFAULT_FILES["sales"] if os.path.exists(DEFAULT_FILES["sales"]) else None)
d_file = d_upload if d_upload is not None else (DEFAULT_FILES["demand"] if os.path.exists(DEFAULT_FILES["demand"]) else None)
lt_file = lt_upload if lt_upload is not None else (DEFAULT_FILES["lt"] if os.path.exists(DEFAULT_FILES["lt"]) else None)

if s_file:
    st.sidebar.success(f"✅ Sales Loaded: {getattr(s_file,'name', s_file)}")
if d_file:
    st.sidebar.success(f"✅ Demand Loaded: {getattr(d_file,'name', d_file)}")
if lt_file:
    st.sidebar.success(f"✅ Lead Time Loaded: {getattr(lt_file,'name', lt_file)}")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if s_file and d_file and lt_file:
    try:
        df_s = pd.read_csv(s_file)
        df_d = pd.read_csv(d_file)
        df_lt = pd.read_csv(lt_file)
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
        st.stop()

    for df in [df_s, df_d, df_lt]:
        df.columns = [c.strip() for c in df.columns]

    needed_sales_cols = {'Product', 'Location', 'Period', 'Consumption', 'Forecast'}
    needed_demand_cols = {'Product', 'Location', 'Period', 'Forecast'}
    needed_lt_cols = {'Product', 'From_Location', 'To_Location', 'Lead_Time_Days', 'Lead_Time_Std_Dev'}

    if not needed_sales_cols.issubset(set(df_s.columns)):
        st.error(f"sales.csv missing columns: {needed_sales_cols - set(df_s.columns)}")
        st.stop()
    if not needed_demand_cols.issubset(set(df_d.columns)):
        st.error(f"demand.csv missing columns: {needed_demand_cols - set(df_d.columns)}")
        st.stop()
    if not needed_lt_cols.issubset(set(df_lt.columns)):
        st.error(f"leadtime.csv missing columns: {needed_lt_cols - set(df_lt.columns)}")
        st.stop()

    df_s['Period'] = pd.to_datetime(df_s['Period'], errors='coerce')
    df_d['Period'] = pd.to_datetime(df_d['Period'], errors='coerce')
    df_s['Period'] = df_s['Period'].dt.to_period('M').dt.to_timestamp()
    df_d['Period'] = df_d['Period'].dt.to_period('M').dt.to_timestamp()

    df_s['Consumption'] = clean_numeric(df_s['Consumption'])
    df_s['Forecast'] = clean_numeric(df_s['Forecast'])
    df_d['Forecast'] = clean_numeric(df_d['Forecast'])
    df_lt['Lead_Time_Days'] = clean_numeric(df_lt['Lead_Time_Days'])
    df_lt['Lead_Time_Std_Dev'] = clean_numeric(df_lt['Lead_Time_Std_Dev'])

    stats = df_s.groupby(['Product', 'Location'])['Consumption'].agg(['mean', 'std']).reset_index()
    stats.columns = ['Product', 'Location', 'Local_Mean', 'Local_Std']
    global_median_std = stats['Local_Std'].median(skipna=True)
    if pd.isna(global_median_std) or global_median_std == 0:
        global_median_std = 1.0
    prod_medians = stats.groupby('Product')['Local_Std'].median().to_dict()
    def fill_local_std(row):
        if not pd.isna(row['Local_Std']) and row['Local_Std'] > 0:
            return row['Local_Std']
        pm = prod_medians.get(row['Product'], np.nan)
        return pm if not pd.isna(pm) else global_median_std
    stats['Local_Std'] = stats.apply(fill_local_std, axis=1)

    network_stats = aggregate_network_stats(df_forecast=df_d, df_stats=stats, df_lt=df_lt)

    node_lt = df_lt.groupby(['Product', 'To_Location'])[['Lead_Time_Days', 'Lead_Time_Std_Dev']].mean().reset_index()
    node_lt.columns = ['Product', 'Location', 'LT_Mean', 'LT_Std']

    results = pd.merge(network_stats, df_d[['Product', 'Location', 'Period', 'Forecast']],
                       on=['Product', 'Location', 'Period'], how='left')
    results = pd.merge(results, node_lt, on=['Product', 'Location'], how='left')

    results = results.fillna({'Forecast': 0, 'Agg_Std_Hist': np.nan, 'LT_Mean': 7, 'LT_Std': 2, 'Agg_Future_Demand': 0})
    product_median_localstd = stats.groupby('Product')['Local_Std'].median().to_dict()
    results['Agg_Std_Hist'] = results.apply(
        lambda r: product_median_localstd.get(r['Product'], global_median_std) if pd.isna(r['Agg_Std_Hist']) else r['Agg_Std_Hist'],
        axis=1
    )

    # -------------------------------
    # Initial global selection modal (asks product + location on first run)
    # The selection is stored in st.session_state['global_product'] and ['global_location'].
    # It is applied as default across tabs, but users can change per-tab using the small "Change global selection" button (opens same modal).
    # -------------------------------
    def open_global_selection_modal(preselect_product=None):
        # Use modal UI if available
        try:
            modal_ctx = st.modal("Primary selection — Apply globally")
        except Exception:
            # fallback: use expander if modal not available
            modal_ctx = st.expander("Primary selection — Apply globally (fallback)")

        with modal_ctx:
            st.write("Select a primary Product and (optionally) a Location to use as default across tabs.")
            products = sorted(results['Product'].unique())
            sel_prod = st.selectbox("Product (primary)", options=[""] + products, index=0 if not preselect_product else (1 + products.index(preselect_product) if preselect_product in products else 0))
            sel_loc = ""
            if sel_prod:
                locs = sorted(results[results['Product'] == sel_prod]['Location'].unique())
                sel_loc = st.selectbox("Location (primary, optional)", options=[""] + locs, index=0)
            if st.button("Confirm selection"):
                st.session_state['global_product'] = sel_prod
                st.session_state['global_location'] = sel_loc if sel_loc else ""
                st.experimental_rerun()

    # Auto open modal on first execution (when session state not yet set)
    if 'global_product' not in st.session_state:
        # Show modal automatically to force initial selection
        open_global_selection_modal()

    # Helper to expose change button in pages
    def change_global_selection_button(preselect=None):
        col1, col2 = st.columns([1, 9])
        with col1:
            if st.button("🔁", help="Change global Product/Location (applies across tabs)"):
                open_global_selection_modal(preselect_product=preselect)

    # -------------------------------
    # SAFETY STOCK — SS METHOD 5
    # -------------------------------
    results['Pre_Rule_SS'] = z * np.sqrt(
        (results['Agg_Std_Hist']**2 / float(days_per_month)) * results['LT_Mean'] +
        (results['LT_Std']**2) * (results['Agg_Future_Demand'] / float(days_per_month))**2
    )

    results['Adjustment_Status'] = 'Optimal (Statistical)'
    results['Safety_Stock'] = results['Pre_Rule_SS']
    results['Pre_Zero_SS'] = results['Safety_Stock']
    if zero_if_no_net_fcst:
        zero_mask = (results['Agg_Future_Demand'] <= 0)
        results.loc[zero_mask, 'Adjustment_Status'] = 'Forced to Zero'
        results.loc[zero_mask, 'Safety_Stock'] = 0

    results['Pre_Cap_SS'] = results['Safety_Stock']
    if apply_cap:
        l_cap, u_cap = cap_range[0] / 100.0, cap_range[1] / 100.0
        l_lim = results['Agg_Future_Demand'] * l_cap
        u_lim = results['Agg_Future_Demand'] * u_cap

        high_mask = (results['Safety_Stock'] > u_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)')
        results.loc[high_mask, 'Adjustment_Status'] = 'Capped (High)'
        low_mask = (results['Safety_Stock'] < l_lim) & (results['Adjustment_Status'] == 'Optimal (Statistical)') & (results['Agg_Future_Demand'] > 0)
        results.loc[low_mask, 'Adjustment_Status'] = 'Capped (Low)'

        results['Safety_Stock'] = results['Safety_Stock'].clip(lower=l_lim, upper=u_lim)

    results['Safety_Stock'] = results['Safety_Stock'].round(0)
    results.loc[results['Location'] == 'B616', 'Safety_Stock'] = 0
    results['Max_Corridor'] = results['Safety_Stock'] + results['Forecast']

    # ACCURACY DATA (LOCAL)
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

    # -------------------------------
    # TABS
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
    # TAB 1: Inventory Corridor
    # -------------------------------
    with tab1:
        st.markdown("### Inventory Corridor — Selection & context")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        col_left, col_right = st.columns([3, 1])
        with col_left:
            products = sorted(results['Product'].unique())
            default_prod = st.session_state.get('global_product', products[0] if products else "")
            prod_index = products.index(default_prod) if default_prod in products else 0
            sku = st.selectbox("Product", products, index=prod_index, key='tab1_sku')

            locs = sorted(results[results['Product'] == sku]['Location'].unique())
            default_loc = st.session_state.get('global_location', locs[0] if locs else "")
            loc_index = locs.index(default_loc) if default_loc in locs else 0
            loc = st.selectbox("Location", locs, index=loc_index, key='tab1_loc')

            st.markdown(f"**Global selection (applied by default):** {st.session_state.get('global_product','(none)')} — {st.session_state.get('global_location','(none)')}")
            plot_df = results[(results['Product'] == sku) & (results['Location'] == loc)].sort_values('Period')

            fig = go.Figure([
                go.Scatter(x=plot_df['Period'], y=plot_df['Max_Corridor'], name='Max Corridor (SS + Forecast)', line=dict(width=1, color='rgba(0,0,0,0.1)')),
                go.Scatter(x=plot_df['Period'], y=plot_df['Safety_Stock'], name='Safety Stock', fill='tonexty', fillcolor='rgba(0,176,246,0.2)'),
                go.Scatter(x=plot_df['Period'], y=plot_df['Forecast'], name='Local Direct Forecast', line=dict(color='black', dash='dot')),
                go.Scatter(x=plot_df['Period'], y=plot_df['Agg_Future_Demand'], name='Total Network Demand', line=dict(color='blue', dash='dash'))
            ])
            fig.update_layout(legend=dict(orientation="h"), xaxis_title='Period', yaxis_title='Units')
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            badge_html = f"""
            <div style="background:#0b3d91;padding:14px;border-radius:8px;color:white;text-align:right;">
                <div style="font-size:12px;opacity:0.8">Selected</div>
                <div style="font-size:16px;font-weight:700">{sku} — {loc}</div>
                <div style="margin-top:8px;font-size:12px;opacity:0.95">
                    Fcst (Local): <strong>{euro_format(float(plot_df['Forecast'].sum()))}</strong><br>
                    Net Demand: <strong>{euro_format(float(plot_df['Agg_Future_Demand'].sum()))}</strong><br>
                    SS (Current): <strong>{euro_format(float(plot_df['Safety_Stock'].sum()), True)}</strong>
                </div>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            s1.metric("Total SS (sku/loc)", euro_format(float(plot_df['Safety_Stock'].sum()), True))
            s2.metric("Total Net Demand", euro_format(float(plot_df['Agg_Future_Demand'].sum()), True))

    # -------------------------------
    # TAB 2: Network Topology (rolled back to physics layout with new colors)
    # -------------------------------
    with tab2:
        st.markdown("### Network Topology")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        sku_default = st.session_state.get('global_product', None)
        sku = sku_default if sku_default else st.selectbox("Product for Network View", sorted(results['Product'].unique()), key="network_sku")
        period_choices = sorted(results['Period'].unique())
        chosen_period = st.selectbox("Period", period_choices, index=len(period_choices)-1 if period_choices else 0, key="network_period")

        label_data = results[results['Period'] == chosen_period].set_index(['Product', 'Location']).to_dict('index')
        sku_lt = df_lt[df_lt['Product'] == sku] if 'Product' in df_lt.columns else df_lt.copy()

        net = Network(height="1200px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")

        # Color rules:
        # - B616 very light green
        # - BEEX light blue (first hub after B616)
        # - LUEX light blue
        # - Other ACTIVE nodes: very light yellow
        # - INACTIVE nodes: greyed out
        hubs = {"B616", "BEEX", "LUEX"}

        all_nodes = set(sku_lt['From_Location']).union(set(sku_lt['To_Location']))
        all_nodes = set(all_nodes).union(hubs)

        demand_lookup = {}
        for n in all_nodes:
            demand_lookup[n] = label_data.get((sku, n), {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})

        for n in sorted(all_nodes):
            m = demand_lookup.get(n, {'Forecast': 0, 'Agg_Future_Demand': 0, 'Safety_Stock': 0})
            used = (m['Agg_Future_Demand'] > 0) or (m['Forecast'] > 0)
            if n == 'B616':
                bg = '#e8f5e9'  # very light green
                border = '#8bc34a'
                font_color = '#0b3d91'
                size = 14
            elif n in {'BEEX', 'LUEX'}:
                bg = '#e3f2fd'  # light blue
                border = '#64b5f6'
                font_color = '#0b3d91'
                size = 14
            else:
                if used:
                    bg = '#fffde7'  # very light yellow
                    border = '#fbc02d'
                    font_color = '#222222'
                    size = 12
                else:
                    bg = '#f0f0f0'  # inactive grey
                    border = '#cccccc'
                    font_color = '#9e9e9e'
                    size = 10
            lbl = f"{n}\nFcst: {euro_format(m['Forecast'])}\nNet: {euro_format(m['Agg_Future_Demand'])}\nSS: {euro_format(m['Safety_Stock'], True)}"
            net.add_node(n, label=lbl, title=lbl,
                         color={'background': bg, 'border': border},
                         shape='box', font={'color': font_color, 'size': size})

        for _, r in sku_lt.iterrows():
            from_n, to_n = r['From_Location'], r['To_Location']
            from_used = (demand_lookup.get(from_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(from_n, {}).get('Forecast', 0) > 0)
            to_used = (demand_lookup.get(to_n, {}).get('Agg_Future_Demand', 0) > 0) or (demand_lookup.get(to_n, {}).get('Forecast', 0) > 0)
            if not from_used and not to_used:
                edge_color = '#dddddd'
            else:
                edge_color = '#888888'
            label = f"{int(r.get('Lead_Time_Days', 0))}d" if not pd.isna(r.get('Lead_Time_Days', 0)) else ""
            net.add_edge(from_n, to_n, label=label, color=edge_color)

        net.set_options("""
        var options = {
          "physics": {"stabilization": {"iterations": 200}},
          "nodes": {"borderWidthSelected":2},
          "interaction": {"hover":true}
        }
        """)
        tmpfile = "net.html"
        net.save_graph(tmpfile)
        with open(tmpfile, 'r', encoding='utf-8') as f:
            html = f.read()
        components.html(html, height=900, scrolling=True)

    # -------------------------------
    # TAB 3: Full Plan
    # -------------------------------
    with tab3:
        st.markdown("### Full Plan — Global snapshot (defaults to primary selection)")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        col1, col2, col3 = st.columns(3)
        f_prod = col1.multiselect("Filter Product", sorted(results['Product'].unique()))
        f_loc = col2.multiselect("Filter Location", sorted(results['Location'].unique()))
        f_period = col3.multiselect("Filter Period", sorted(results['Period'].unique()))

        filtered = results.copy()
        if st.session_state.get('global_product') and not f_prod:
            filtered = filtered[filtered['Product'] == st.session_state.get('global_product')]
        if st.session_state.get('global_location') and not f_loc:
            filtered = filtered[filtered['Location'] == st.session_state.get('global_location')]

        if f_prod: filtered = filtered[filtered['Product'].isin(f_prod)]
        if f_loc: filtered = filtered[filtered['Location'].isin(f_loc)]
        if f_period: filtered = filtered[filtered['Period'].isin(f_period)]

        display_cols = ['Product','Location','Period','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status','Max_Corridor']
        disp = df_format_for_display(filtered[display_cols].copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock','Max_Corridor'], two_decimals_cols=['Forecast'])
        st.dataframe(disp, use_container_width=True, height=700)

        csv_buf = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Filtered Plan (CSV)", data=csv_buf, file_name="filtered_plan.csv", mime="text/csv")

    # -------------------------------
    # TAB 4: Efficiency Analysis
    # -------------------------------
    with tab4:
        st.markdown("### Efficiency & Policy — defaults from primary selection")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        all_products = sorted(results['Product'].unique())
        sku_default = st.session_state.get('global_product', all_products[0] if all_products else "")
        sku = st.selectbox("Material", all_products, index=all_products.index(sku_default) if sku_default in all_products else 0, key="eff_sku")
        next_month = sorted(results['Period'].unique())[-1]
        eff = results[(results['Product'] == sku) & (results['Period'] == next_month)].copy()

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
            eff['Gap'] = (eff['Safety_Stock'] - eff['Pre_Rule_SS']).abs()
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
        st.markdown("### Forecast Accuracy")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        all_products = sorted(results['Product'].unique())
        h_sku_default = st.session_state.get('global_product', all_products[0] if all_products else "")
        h_sku = st.selectbox("Select Product", all_products, index=all_products.index(h_sku_default) if h_sku_default in all_products else 0, key="h1")
        h_loc_opts = sorted(results[results['Product'] == h_sku]['Location'].unique())
        h_loc_default = st.session_state.get('global_location', h_loc_opts[0] if h_loc_opts else "")
        h_loc = st.selectbox("Select Location", h_loc_opts, index=h_loc_opts.index(h_loc_default) if h_loc_default in h_loc_opts else 0, key="h2")

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
                net_wape = (net_table['Net_Abs_Error'].sum() / net_table['Network_Consumption'].replace(0, np.nan).sum() * 100)
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
            st.warning("⚠️ No historical sales data found for this selection. Accuracy metrics cannot be calculated.")

    # -------------------------------
    # TAB 6: Calculation Trace & Simulation
    # -------------------------------
    with tab6:
        st.markdown("### Calculation Trace & Simulation")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        all_products = sorted(results['Product'].unique())
        calc_sku_default = st.session_state.get('global_product', all_products[0] if all_products else "")
        calc_sku = st.selectbox("Select Product", all_products, index=all_products.index(calc_sku_default) if calc_sku_default in all_products else 0, key="c_sku")
        avail_locs = sorted(results[results['Product'] == calc_sku]['Location'].unique())
        calc_loc_default = st.session_state.get('global_location', avail_locs[0] if avail_locs else "")
        calc_loc = st.selectbox("Select Location", avail_locs, index=avail_locs.index(calc_loc_default) if calc_loc_default in avail_locs else 0, key="c_loc")
        avail_periods = sorted(results['Period'].unique())
        calc_period = st.selectbox("Select Period", avail_periods, index=len(avail_periods)-1 if avail_periods else 0, key="c_period")

        row = results[
            (results['Product'] == calc_sku) &
            (results['Location'] == calc_loc) &
            (results['Period'] == calc_period)
        ]
        if row.empty:
            st.warning("Selection not found in results.")
        else:
            row = row.iloc[0]
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

            st.latex(r"SS_{\text{raw}} = Z \times \sqrt{\,\sigma_D^2 \times L \;+\; \sigma_L^2 \times D^2\,}")
            st.code(f"""
1. Demand component (variance term) = {euro_format(term1_demand_var, True)}
2. Lead-time component (variance term) = {euro_format(term2_supply_var, True)}
3. Combined Std Dev = {euro_format(combined_sd, True)}
4. Raw SS = {euro_format(raw_ss_calc, True)}
""")
            st.info(f"🧮 Statistical SS (Method 5, raw): {euro_format(raw_ss_calc, True)} units")

            st.subheader("3. Business Rules Application")
            col_rule_1, col_rule_2 = st.columns(2)
            with col_rule_1:
                st.markdown("**Zero Demand Rule**")
                if zero_if_no_net_fcst and row['Agg_Future_Demand'] <= 0:
                    st.error("❌ Network Demand is 0. SS Forced to 0.")
                else:
                    st.success("✅ Network Demand exists. Keep Statistical SS.")
            with col_rule_2:
                st.markdown("**Capping (Min/Max)**")
                if apply_cap:
                    lower_limit = row['Agg_Future_Demand'] * (cap_range[0]/100)
                    upper_limit = row['Agg_Future_Demand'] * (cap_range[1]/100)
                    st.write(f"Constraint: {int(cap_range[0])}% to {int(cap_range[1])}% of Demand")
                    st.write(f"Range: [{euro_format(lower_limit, True)}, {euro_format(upper_limit, True)}]")
                    if raw_ss_calc > upper_limit:
                        st.warning("⚠️ Raw SS > Max Cap. Capping downwards.")
                    elif raw_ss_calc < lower_limit and row['Agg_Future_Demand'] > 0:
                        st.warning("⚠️ Raw SS < Min Cap. Buffering upwards.")
                    else:
                        st.success("✅ Raw SS is within boundaries.")
                else:
                    st.write("Capping disabled.")

            st.subheader("4. What-If Simulation")
            sim_cols = st.columns(3)
            sim_sl = sim_cols[0].slider("Simulated Service Level (%)", 50.0, 99.9, value=service_level*100)
            sim_lt = sim_cols[1].slider("Simulated Avg Lead Time (Days)", 0.0, max(30.0, row['LT_Mean']*2), value=float(row['LT_Mean']))
            sim_lt_std = sim_cols[2].slider("Simulated LT Variability (Days)", 0.0, max(10.0, row['LT_Std']*2), value=float(row['LT_Std']))

            sim_z = norm.ppf(sim_sl / 100.0)
            sim_ss = sim_z * np.sqrt(
                (row['Agg_Std_Hist']**2 / float(days_per_month)) * sim_lt +
                (sim_lt_std**2) * (row['Agg_Future_Demand'] / float(days_per_month))**2
            )
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Original SS (Actual)", euro_format(row['Pre_Rule_SS'], True))
            res_col2.metric("Simulated SS (New)", euro_format(sim_ss, True), delta=euro_format(sim_ss - row['Pre_Rule_SS'], True))
            if sim_ss < row['Pre_Rule_SS']:
                st.success(f"📉 Possible inventory reduction: {euro_format(row['Pre_Rule_SS'] - sim_ss, True)} units")
            elif sim_ss > row['Pre_Rule_SS']:
                st.warning(f"📈 Additional inventory needed: {euro_format(sim_ss - row['Pre_Rule_SS'], True)} units")

    # -------------------------------
    # TAB 7: By Material (single-material focus + 8R whose sum equals total SS)
    # -------------------------------
    with tab7:
        st.markdown("### By Material — single-material focus. Defaults to primary selection.")
        change_global_selection_button(preselect=st.session_state.get('global_product', None))
        all_products = sorted(results['Product'].unique())
        selected_product = st.session_state.get('global_product', all_products[0] if all_products else "")
        # allow override
        selected_product = st.selectbox("Material (default = primary)", all_products, index=all_products.index(selected_product) if selected_product in all_products else 0, key="mat_sel")

        period_choices = sorted(results['Period'].unique())
        selected_period = st.selectbox("Select Period to Snapshot", period_choices, index=len(period_choices)-1 if period_choices else 0, key="mat_period")

        mat_period_df = results[(results['Product'] == selected_product) & (results['Period'] == selected_period)].copy()
        total_forecast = mat_period_df['Forecast'].sum()
        total_net = mat_period_df['Agg_Future_Demand'].sum()
        total_ss = mat_period_df['Safety_Stock'].sum()
        nodes_count = mat_period_df['Location'].nunique()
        avg_ss_per_node = (mat_period_df['Safety_Stock'].mean() if nodes_count > 0 else 0)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Local Forecast", euro_format(total_forecast, True))
        k2.metric("Total Network Demand", euro_format(total_net, True))
        k3.metric("Total Safety Stock (sum nodes)", euro_format(total_ss, True))
        k4.metric("Nodes", f"{nodes_count}")
        k5.metric("Avg SS per Node", euro_format(avg_ss_per_node, True))

        st.markdown("### 8 Reasons decomposition — drivers that sum to Total Safety Stock")
        if mat_period_df.empty:
            st.warning("No data for this material/period.")
        else:
            mat = mat_period_df.copy()
            mat['LT_Mean'] = mat['LT_Mean'].fillna(0)
            mat['LT_Std'] = mat['LT_Std'].fillna(0)
            mat['Agg_Std_Hist'] = mat['Agg_Std_Hist'].fillna(0)
            # variance components
            mat['term1'] = (mat['Agg_Std_Hist']**2 / float(days_per_month)) * mat['LT_Mean']
            mat['term2'] = (mat['LT_Std']**2) * (mat['Agg_Future_Demand'] / float(days_per_month))**2
            # Pre-rule SS
            mat['Pre_Rule_SS'] = mat['Pre_Rule_SS'].fillna(0)
            # Allocate Pre_Rule_SS into demand and lt components proportionally to variance
            mat['var_sum'] = mat['term1'] + mat['term2']
            # avoid division by zero
            mat['demand_alloc'] = np.where(mat['var_sum'] > 0, mat['Pre_Rule_SS'] * (mat['term1'] / mat['var_sum']), 0.0)
            mat['lt_alloc'] = np.where(mat['var_sum'] > 0, mat['Pre_Rule_SS'] * (mat['term2'] / mat['var_sum']), 0.0)
            # policy deltas
            mat['cap_delta'] = mat['Safety_Stock'] - mat['Pre_Rule_SS']  # can be negative (reduction) or positive (increase)
            # forced zero and B616: capture and exclude from general cap_reduction to avoid double counting
            mat['forced_zero'] = np.where((mat['Adjustment_Status'] == 'Forced to Zero') & (mat['Pre_Rule_SS'] > 0), mat['Pre_Rule_SS'], 0.0)
            mat['b616_override'] = np.where((mat['Location'] == 'B616') & (mat['Safety_Stock'] == 0) & (mat['Pre_Rule_SS'] > 0), mat['Pre_Rule_SS'], 0.0)
            # Make forced_zero and b616_override exclusive: if a row was forced_zero and also B616, prefer B616 to show explicit hub policy
            mat['forced_zero'] = np.where((mat['forced_zero'] > 0) & (mat['b616_override'] > 0), 0.0, mat['forced_zero'])

            # Now compute cap increases and cap reductions excluding forced_zero and b616_override
            mat['cap_increase'] = mat['cap_delta'].apply(lambda x: x if x > 0 else 0.0)
            mat['cap_reduction_total'] = mat['cap_delta'].apply(lambda x: -x if x < 0 else 0.0)
            # exclude forced_zero and b616 from cap_reduction_total for separate reporting
            mat['cap_reduction_excl'] = mat['cap_reduction_total'] - mat['forced_zero'] - mat['b616_override']
            mat['cap_reduction_excl'] = mat['cap_reduction_excl'].clip(lower=0.0)

            # Aggregate driver totals
            demand_total = mat['demand_alloc'].sum()
            lt_total = mat['lt_alloc'].sum()
            cap_increase_total = mat['cap_increase'].sum()
            cap_reduction_excl_total = mat['cap_reduction_excl'].sum()
            forced_zero_total = mat['forced_zero'].sum()
            b616_total = mat['b616_override'].sum()

            # Compose drivers list such that their signed sum equals total_ss:
            # total_ss = (demand_total + lt_total) + (cap_increase_total - cap_reduction_excl_total - forced_zero_total - b616_total) + residual
            net_pre = demand_total + lt_total
            net_policy = cap_increase_total - cap_reduction_excl_total - forced_zero_total - b616_total
            net_total = net_pre + net_policy
            residual = total_ss - net_total
            # Put residual into small driver to make perfect balance (this handles rounding, minor differences)
            residual = float(residual)

            drivers = [
                ("Demand Uncertainty (allocated)", float(demand_total)),
                ("Lead-time Uncertainty (allocated)", float(lt_total)),
                ("Caps — Increases (policy)", float(cap_increase_total)),
                ("Caps — Reductions (policy, excl forced/B616)", float(-cap_reduction_excl_total)),  # show as negative (reduces SS)
                ("Forced Zero Overrides (policy)", float(-forced_zero_total)),
                ("B616 Override (policy)", float(-b616_total)),
                ("Residual / Rounding (to match Total SS)", float(residual)),
            ]

            drv_df = pd.DataFrame(drivers, columns=['Driver', 'Signed_Units'])
            # For display, also show absolute contribution and percent of total_ss
            drv_df['Units'] = drv_df['Signed_Units']
            # For percent, use total_ss as denominator (if zero then denom=1)
            denom = total_ss if total_ss != 0 else 1.0
            drv_df['Pct_of_total_SS'] = drv_df['Units'] / denom * 100

            # Plot signed bars (positive bars up, reductions negative)
            fig_drv = go.Figure()
            colors = []
            for val in drv_df['Units']:
                if val >= 0:
                    colors.append('#6baed6')  # blue-ish for positive contributors
                else:
                    colors.append('#fdae6b')  # orange-ish for negative/reductions
            fig_drv.add_trace(go.Bar(x=drv_df['Driver'], y=drv_df['Units'], marker_color=colors))
            # annotate percent
            annotations = []
            for idx, r in drv_df.iterrows():
                annotations.append(dict(x=r['Driver'], y=r['Units'], text=f"{r['Pct_of_total_SS']:.1f}%", showarrow=False, yshift=8 if r['Units'] >= 0 else -16))
            fig_drv.update_layout(title=f"{selected_product} — Drivers of Total Safety Stock (signed components)", xaxis_title="Driver", yaxis_title="Units", annotations=annotations, height=420)
            st.plotly_chart(fig_drv, use_container_width=True)

            # Table view with clear sign
            display_table = drv_df[['Driver', 'Units', 'Pct_of_total_SS']].copy()
            st.dataframe(df_format_for_display(display_table.round(2), cols=['Units','Pct_of_total_SS']), use_container_width=True, height=240)

            # Sanity check: show sums
            st.markdown(f"**Sanity check:** Sum of driver-signed units = {euro_format(drv_df['Units'].sum(), True)}; Total Safety Stock = {euro_format(total_ss, True)}")

            if abs(drv_df['Units'].sum() - total_ss) > 1e-6:
                st.warning("Driver decomposition does not exactly match Total Safety Stock — a residual has been added to balance. Investigate rounding/policy overlaps if the discrepancy is large.")

        st.markdown("---")
        st.subheader("Top Locations by Safety Stock (snapshot)")
        top_nodes = mat_period_df.sort_values('Safety_Stock', ascending=False)[['Location','Forecast','Agg_Future_Demand','Safety_Stock','Adjustment_Status']]
        st.dataframe(df_format_for_display(top_nodes.head(25).copy(), cols=['Forecast','Agg_Future_Demand','Safety_Stock'], two_decimals_cols=['Forecast']), use_container_width=True, height=300)

        st.markdown("---")
        st.subheader("Export — Material Snapshot")
        st.download_button("📥 Download Material Snapshot (CSV)", data=mat_period_df.to_csv(index=False), file_name=f"material_{selected_product}_{selected_period.strftime('%Y-%m')}.csv", mime="text/csv")

else:
    st.info("No data found. Please place 'sales.csv', 'demand.csv', and 'leadtime.csv' in the script folder OR upload them via the sidebar.")
