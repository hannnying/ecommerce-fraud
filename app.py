import streamlit as st
import requests
import pandas as pd
from src.config import API_PORT, BACKEND_HOST, RANDOM_STATE, RAW_TEST_PATH
from streamlit_autorefresh import st_autorefresh

BACKEND_URL = f"http://{BACKEND_HOST}:{API_PORT}"

st.title("E-commerce Transactions Dashboard")

k = st.sidebar.slider("Number of recent transactions to show", min_value=5, max_value=50, value=20, step=5)
REFRESH_INTERVAL = st.sidebar.number_input("Refresh interval (seconds)", min_value=1, max_value=60, value=5, step=1)

# auto-refresh counter
st_autorefresh(interval=REFRESH_INTERVAL * 1000, limit=None)

placeholder = st.empty()

try:
    response = requests.get(f"{BACKEND_URL}/results/recent?k={k}")
    data = response.json().get("results", [])

    if data:
        df = pd.DataFrame(data)
        # convert numeric columns if needed
        for col in ["fraud_probability", "predicted_class", "txn_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        placeholder.dataframe(df)
    else:
        placeholder.text("No transactions processed yet.")

except Exception as e:
    placeholder.text(f"Error fetching transactions: {e}")