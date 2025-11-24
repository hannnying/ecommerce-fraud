import streamlit as st
import pandas as pd
import json
import src.config as config
from src.evaluation import ModelEvaluator
from src.models import FraudDetectionModel
from src.preprocessing import load_and_preprocess_data

st.title("Fraud Detection Model Dashboard")

@st.cache_data
def load_test_data(path):
    return pd.read_csv(path, index_col=0)

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
    str(config.TRAIN_DATA_PATH),
    str(config.TEST_DATA_PATH),
    target_col=config.TARGET_COL
)

# Features
columns = [col for col in X_train.columns if col != config.TARGET_COL]
feature_cols = st.multiselect(
    label="Select Features for Fraud Detection Model",
    options=columns,
    default=columns
)

# Resampler
resampler = st.selectbox(
    label="Resampling Method",
    options=["random_undersampling", "enn", "smote", "smoteenn"]
)

# Model
model_type = st.selectbox(
    label="Model",
    options=["logistic_regression", "random_forest", "xgboost"]
)

# Hyperparameters
custom_params = st.text_area(
    label="Hyperparameters (JSON)",
    value=""
)

params = {}
if custom_params:
    try:
        params = json.loads(custom_params)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for hyperparameters.")

if st.button("Evaluate My Model"):
    model = FraudDetectionModel(
        columns=columns,
        resampling_type=resampler,
        model_type=model_type,
        custom_params=custom_params
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluator = ModelEvaluator(model_name=config.MODEL_CONFIGS[model_type]['display_name'])

    st.markdown(evaluator.evaluate_business(y_pred))