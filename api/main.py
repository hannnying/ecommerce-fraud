from datetime import datetime
import json
from uuid import uuid4

from api.schemas import (
    FraudPrediction,
    Transaction,
    TransactionDTO,
    Transactions
)
from api.dependencies import(
    get_feature_engineer,
    get_model,
    get_preprocessor,
    get_redis_client
)
from api.inference import (
    generate_device_id,
    generate_ip_address,
    generate_signup_time,
    generate_user_id,
    run_inference
)
from fastapi import Depends, FastAPI, HTTPException
import pandas as pd
from pandera.typing import DataFrame
from redis import Redis

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    RAW_TEST_PATH,
    RESULT_STREAM,
    TRANSACTION_STREAM
)

app = FastAPI()

@app.get("/health")
async def get_health():
    return {"message": "app is running!"}


# fastAPI only pushes transactionDTO to redis
@app.post("/submit")
async def submit_transaction(
    transaction_dto: TransactionDTO,
    redis_client: Redis = Depends(get_redis_client)
):
    try:
        transaction_id = str(uuid4())
        transaction = Transaction(
        **transaction_dto.model_dump(),
        user_id=generate_user_id(),
        signup_time=generate_signup_time(),
        purchase_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        device_id=generate_device_id(),
        ip_address=generate_ip_address()
    )
        transaction_dict = transaction.model_dump() 
        redis_client.xadd(
            TRANSACTION_STREAM, {"id": transaction_id, **transaction_dict}
        )
        return {"status": "queued", "transaction_id": transaction_id}
    except Exception as e:
        return {"status": f"An error occurred: {e}"}
    
@app.post("/predict/single")
async def predict_single(
    transaction_dto: TransactionDTO,
    redis_client: Redis = Depends(get_redis_client)
):
    """Enqueue single transaction for asynchronous worker processing."""
    return await submit_transaction(transaction_dto, redis_client)

@app.get("/result/{transaction_id}")
async def get_result(
    transaction_id: str,
    redis_client: Redis = Depends(get_redis_client)
):
    """
    Poll Redis RESULT_STREAM for a transaction result.
    Returns the first matching result or status "pending".
    """
    try:
        messages = redis_client.xrevrange(RESULT_STREAM, count=100)
        print(f"messages received: {messages}")
        for message_id, data in messages:
            if data.get("transaction_id") == transaction_id:
                return {
                    "transaction_id": transaction_id,
                    "predicted_class": int(data["predicted_class"]),
                    "fraud_probability": float(data["fraud_probability"])
                }
        
        raise HTTPException(status_code=202, detail="still processing")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))