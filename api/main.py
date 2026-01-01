from datetime import datetime
import json
from uuid import uuid4
from api.producer import TransactionProducer
from fastapi import Depends, FastAPI, HTTPException
import pandas as pd
from redis import Redis

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    RESULT_STREAM,
    TRANSACTION_STREAM
)

def get_redis_client():
    return Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

app = FastAPI()

@app.get("/health")
async def get_health():
    return {"message": "app is running!"}


@app.post("/stream")
async def stream():
    try:
        txn_producer = TransactionProducer()
        transaction_ids = txn_producer.start_streaming()
        return {"status": "queued", "transaction_ids": transaction_ids}
    except Exception as e:
        return {"status": f"An error occurred: {e}"}

    
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
    Poll Redis RESULT_STREAM for a transaction result.
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
    