from fastapi import Depends, FastAPI, HTTPException
import numpy as np
import pandas as pd
from redis import Redis
from sklearn.metrics import accuracy_score, f1_score, precision_score, average_precision_score, recall_score

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    RESULT_STREAM
)

def decode_redis_value(val):
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val

def get_redis_client():
    return Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

app = FastAPI()

@app.get("/health")
async def get_health():
    return {"message": "app is running!"}

    
@app.get("/results/recent")
async def get_k_results(
    k: int = 20,
    redis_client: Redis = Depends(get_redis_client)
):
    """
    Poll Redis RESULT_STREAM for the most recent k processed transactions from RESULT_STREAM.
    """
    try:
        entries = redis_client.xrevrange(RESULT_STREAM, count=k)

        results = []
        for stream_id, data in reversed(entries):
            results.append({
                "stream_id": stream_id,
                **data
            })

        return {
            "count": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/result/{transaction_id}")
async def get_result(
    transaction_id: str,
    redis_client: Redis = Depends(get_redis_client)
):
    """
    Poll Redis prediction hash for a transaction result.
    Returns the first matching result or status "pending".
    """
    try:
        prediction_key = f"prediction:{transaction_id}"
        prediction_record = redis_client.hgetall(prediction_key)
        return {
            "test": prediction_record
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate")
async def evaluate(
    threshold: float = 0.45,
    redis_client: Redis = Depends(get_redis_client)
):
    """
    Evaluate the current model performance using labeled transactions stored in Redis.

    This endpoint polls Redis for processed transactions whose ground-truth labels
    are available, compares model predictions against these labels, and returns
    up-to-date performance metrics for the deployed model.
    """
    y_true = []
    y_proba = []

    try:
        for key in redis_client.scan_iter("prediction:*", count=100):
            key_str = key if isinstance(key, str) else key.decode('utf-8')

            true_label = redis_client.hget(key_str, "true_label")
            if true_label in (None, ""):
                continue
            y_true.append(int(true_label))

            fraud_proba = decode_redis_value(redis_client.hget(key_str, "fraud_probability"))
            y_proba.append(float(fraud_proba))

        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        y_pred = y_proba >= threshold
        return {
            "count": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "pr_auc": average_precision_score(y_true, y_proba)
    }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

