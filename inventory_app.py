# --- 1. INITIALIZE SESSION STATE (Place this before Sidebar) ---
# This keeps the data in memory even when you interact with the app
if 'df_s' not in st.session_state: st.session_state.df_s = None
if 'df_d' not in st.session_state: st.session_state.df_d = None
if 'df_lt' not in st.session_state: st.session_state.df_lt = None

# --- 2. RESTRICTURED SIDEBAR ---
st.sidebar.header("⚙️ Parameters")
# ... (keep your existing service_level and z code here) ...

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Input")

# THE BUTTON: Now outside of any conditional blocks so it always shows
if st.sidebar.button("🚀 Load Sample Data"):
    try:
        # These filenames must match the files in your local directory
        st.session_state.df_s = pd.read_csv("sales.csv")
        st.session_state.df_d = pd.read_csv("demand.csv")
        st.session_state.df_lt = pd.read_csv("leadtime.csv")
        st.sidebar.success("✅ Samples Loaded!")
    except Exception as e:
        st.sidebar.error(f"Files not found. Ensure sales.csv, demand.csv, and leadtime.csv are in the app folder.")

st.sidebar.write("---")
# Manual Uploads: They will save directly to session state
s_file = st.sidebar.file_uploader("1. Sales Data", type="csv")
if s_file: st.session_state.df_s = pd.read_csv(s_file)

d_file = st.sidebar.file_uploader("2. Demand Data", type="csv")
if d_file: st.session_state.df_d = pd.read_csv(d_file)

lt_file = st.sidebar.file_uploader("3. Lead Time Data", type="csv")
if lt_file: st.session_state.df_lt = pd.read_csv(lt_file)

# --- 3. UPDATED MAIN LOGIC TRIGGER ---
# Instead of checking the file_uploader variables, we check Session State
if st.session_state.df_s is not None and \
   st.session_state.df_d is not None and \
   st.session_state.df_lt is not None:

    # Assign local variables from session state to keep the rest of your logic the same
    df_s = st.session_state.df_s.copy()
    df_d = st.session_state.df_d.copy()
    df_lt = st.session_state.df_lt.copy()
    
    # ... (The rest of your existing calculation and Tab logic continues here) ...

else:
    st.info("Please upload files or click 'Load Sample Data' in the sidebar to begin.")
