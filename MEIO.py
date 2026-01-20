# Multi-Echelon Inventory Optimizer — Raw Materials
# Developed by mat635418 — JAN 2026


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
from datetime import datetime
import re

# -------------------------------
# LOGO helper (moved up to show before title)
# -------------------------------
LOGO_FILENAME = "GY_logo.jpg"
DEFAULT_LOGO_WIDTH = 330  # 1.5x bigger than 220

def display_logo(width=DEFAULT_LOGO_WIDTH):
    """
    Display the Goodyear logo if present in the same folder as MEIO.py.
    Keeps the same width across tabs for consistent appearance.
    If image is missing, the function is silent (keeps layout).
    """
    if os.path.exists(LOGO_FILENAME):
        try:
            st.image(LOGO_FILENAME, use_column_width=False, width=width)
        except Exception:
            # fail silently to avoid breaking the app if image rendering fails
            st.write("")
    else:
        # small spacer to keep layout similar if image missing
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="MEIO for RM", layout="wide")
display_logo()
st.title("🧭 MEIO for Raw Materials — v0.69 — Jan 2026")
