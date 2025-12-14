import streamlit as st
import requests
import os
import sys
import socket
import pandas as pd
import time
from src.config import API_PORT, BACKEND_HOST, RANDOM_STATE, RAW_TEST_PATH

BACKEND_URL = f"http://{BACKEND_HOST}:{API_PORT}"

df = pd.read_csv(RAW_TEST_PATH)
sample_df = df.sample(n=100, random_state=RANDOM_STATE)
st.write("Transactions to simulate:")
st.dataframe(sample_df)

if st.button("Start Simulation"):
    pred_class = []
    probability = []
    for idx, row in sample_df.iterrows():
        transaction = row.to_dict()

        submit_response = requests.post(f"{BACKEND_URL}/submit", json=transaction)
        transaction_id = submit_response.json()["transaction_id"]

        result = None
        while result is None or result.get("status") == "pending":
            time.sleep(1)
            result_response = requests.get(f"{BACKEND_URL}/result/{transaction_id}")
            result = result_response.json()

        pred_class.append(result["predicted_class"])
        probability.append(result["fraud_probability"])

    sample_df["pred_class"] = pred_class
    sample_df["fraud_prob"] = probability

    st.dataframe(sample_df)




