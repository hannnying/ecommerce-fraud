from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException
from api.schemas import (
    FraudPrediction,
    Transaction,
    TransactionDTO,
    Transactions
)
from api.inference import (
    get_feature_engineer,
    generate_device_id,
    generate_ip_address,
    generate_signup_time,
    generate_user_id,
    get_model,
    get_preprocessor,
    run_inference
)
import json
import pandas as pd
from pandera.typing import DataFrame

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_TEST_PATH

app = FastAPI()

@app.get("/health")
def get_health():
    return {"message": "app is running!"}

@app.post("/predict", response_model=FraudPrediction)
def predict(
    transaction_path: str = str(RAW_TEST_PATH),
    feature_engineer = Depends(get_feature_engineer),
    preprocessor = Depends(get_preprocessor),
    model = Depends(get_model)
):
    # Read and validate the DataFrame against the schema
    raw_df = pd.read_csv(transaction_path)
    transaction: DataFrame[Transactions] = Transactions.validate(raw_df)

    pred_class, prob = run_inference(transaction, feature_engineer, preprocessor, model)
    return FraudPrediction(
        predicted_class=pred_class.tolist(),
        probability=prob.tolist()
    )


@app.post("/predict/single", response_model=FraudPrediction)
def predict_single(
    transaction_dto: TransactionDTO,
    feature_engineer = Depends(get_feature_engineer),
    preprocessor = Depends(get_preprocessor),
    model = Depends(get_model)
):
    transaction = Transaction(
        **transaction_dto.model_dump(),
        user_id=generate_user_id(),
        signup_time=generate_signup_time(),
        purchase_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        device_id=generate_device_id(),
        ip_address=generate_ip_address()
    )
    transaction_dict = transaction.model_dump()
    pred_class, prob = run_inference(
        pd.DataFrame(transaction_dict, index=[0]),
        feature_engineer,
        preprocessor,
        model
    )
    return FraudPrediction(
        predicted_class=pred_class.tolist(),
        probability=prob.tolist()
    )