import streamlit as st
import requests
import os
import sys
import socket

# Add src directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import API_PORT

# Auto-detect backend hostname (Docker or local)
def get_backend_host():
    """
    Auto-detect if running in Docker or locally.

    Returns:
        str: 'backend' if in Docker, '127.0.0.1' if local
    """
    try:
        socket.gethostbyname('backend')
        return 'backend'  # Running in Docker
    except socket.gaierror:
        return '127.0.0.1'  # Running locally

# Construct backend URL: hostname + port from config
BACKEND_HOST = get_backend_host()
BACKEND_URL = f"http://{BACKEND_HOST}:{API_PORT}"

# Display connection info (optional - remove if you don't want it)
st.sidebar.info(f"🔗 Connected to: {BACKEND_URL}")

purchase_value = st.number_input("Purchase Value", min_value=0)
source = st.selectbox(
    "Source",
    ("SEO", "Direct", "Ads")
)
browser = st.selectbox(
    "Browser",
    ("Chrome", "Firefox", "IE", "Opera", "Safari")
)
sex = st.selectbox(
    "Sex",
    ("M", "F")
)
age = st.number_input("Age", min_value=0)

if st.button("Make Transaction"):
    try:
        payload = {
            "purchase_value": purchase_value,
            "source": source,
            "browser": browser,
            "sex": sex,
            "age": age
        }
        response = requests.post(f"{BACKEND_URL}/predict/single", json=payload)
        if response.status_code == 200:
            result = response.json()
            predicted_class = result["predicted_class"][0]
            probability = result["probability"][0]

            if predicted_class:
                st.error("❌ Transaction Declined")
            elif probability >= 0.3: # warning verify
                st.error("Transaction to be reviewed")
            else:
                st.success("✅ Transaction Approved!")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")