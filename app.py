import streamlit as st
import requests

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
        response = requests.post("http://127.0.0.1:8000/predict/single", json=payload)
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