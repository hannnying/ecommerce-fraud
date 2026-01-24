from collections import defaultdict
from datetime import datetime, timedelta
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

            if len(results) == k:
                break

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
        for key in redis_client.scan_iter("prediction:*", count=1000):
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

@app.get("/stats/summary")
def get_summary_stats(
    threshold: float = 0.45,
    high_risk_threshold: float = 0.8,
    redis_client: Redis = Depends(get_redis_client)
):
    try:
        entries = redis_client.xrevrange(RESULT_STREAM)
        if not entries:
            return {                                                                                 
                 "total_transactions": 0,                                                             
                 "fraud_count": 0,                                                                    
                 "fraud_percentage": 0.0,                                                             
                 "avg_fraud_probability": 0.0,                                                        
                 "high_risk_count": 0,                                                                
                 "legitimate_count": 0                                                                
            } 
        
        total = len(entries)
        fraud_count = 0
        high_risk_count = 0
        fraud_prob_sum = 0.0

        for stream_id, data in entries:
            fraud_prob = float(data.get("fraud_probability", 0))
            pred_class = fraud_prob >= threshold

            fraud_prob_sum += fraud_prob

            if pred_class == 1:
                fraud_count += 1
            
            if fraud_prob >= high_risk_threshold:
                high_risk_count += 1
        
        legitimate_count = total - fraud_count
        fraud_percentage = (fraud_count / total) * 100 if total > 0 else 0
        avg_fraud_prob = fraud_prob_sum / total if total > 0 else 0

        return {                                                                                     
             "total_transactions": total,                                                             
             "fraud_count": fraud_count,                                                              
             "fraud_percentage": fraud_percentage,                                                    
             "avg_fraud_probability": avg_fraud_prob,                                                 
             "high_risk_count": high_risk_count,                                                      
             "legitimate_count": legitimate_count                                                     
         }                                                                                            
                                                                                                      
    except Exception as e:                                                                           
        raise HTTPException(status_code=500, detail=str(e))       


@app.get("/stats/hourly")
def get_hourly_stats(
    threshold: float = 0.45,
    hours: int = 24,
    redis_client: Redis = Depends(get_redis_client)
):
    """Return hourly aggregated statistics."""
    hourly_stats = []

    try:
        entries = redis_client.xrevrange(RESULT_STREAM)

        if not entries:
            return {"hourly_stats": []}
        
        latest_timestamp = datetime.fromisoformat(entries[0][1].get("purchase_time"))
        latest_hour_str =  latest_timestamp.strftime('%Y-%m-%d %H:00:00')
        latest_hour_key = datetime.strptime(latest_hour_str, '%Y-%m-%d %H:00:00')
        earliest_hour_key = latest_hour_key - timedelta(hours=hours)

        hourly_data = defaultdict(lambda: {'total': 0, 'fraud': 0, 'prob_sum': 0.0})

        for stream_id, data in entries:
            hour_str = datetime.fromisoformat(data.get("purchase_time")).strftime('%Y-%m-%d %H:00:00') 
            hour_key = datetime.strptime(hour_str, '%Y-%m-%d %H:00:00')

            if hour_key >= earliest_hour_key:
                fraud_prob = float(data.get("fraud_probability", 0))
                pred_class = fraud_prob >= threshold

                hourly_data[hour_key]["total"] += 1
                hourly_data[hour_key]["fraud"] += 1 if pred_class else 0
                hourly_data[hour_key]["prob_sum"] += fraud_prob

            else:
                break   
                
        for hour, stats in sorted(hourly_data.items()):
            total = stats['total']
            fraud = stats['fraud']
            hourly_stats.append({
                'hour': hour,
                'total_count': total,
                'fraud_count': fraud,
                'fraud_rate': fraud/total if total > 0 else 0,
                'average_fraud_probability': stats['prob_sum'] / total if total > 0 else 0.0
            })
        
        return {"hourly_stats": hourly_stats}
                                                                                                
    except Exception as e:                                                                           
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/results/high-risk")
def get_high_risk_transactions(
    high_risk_threshold: float = 0.8,
    redis_client: Redis = Depends(get_redis_client)
):
    """Returns transactions with fraud probability above a threshold."""
    entries = redis_client.xrevrange(RESULT_STREAM)
    if not entries:
        return {"results": []}

    high_risk_transactions = []
    for _, data in entries:
        fraud_prob = float(data.get("fraud_probability", 0))
        if fraud_prob >= high_risk_threshold:
            high_risk_transactions.append(data)
    
    return {
        "count": len(high_risk_transactions),
        "results": high_risk_transactions
    }

                                                                                                    
@app.get("/stats/features")
def get_feature_stats(
    redis_client: Redis = Depends(get_redis_client)
):
    """Returns aggregated feature statistics for fraud vs non-fraud."""
    entries = redis_client.xrevrange(RESULT_STREAM)
    feature_stats = {
            "fraud_features": {},
            "legitimate_features": {},
        }
    
    try:
        if not entries:
            return feature_stats

        # boolean features used by the model, analyse how the model uses boolean features to classify transactions
        transactions = 0
        
        for _, data in entries:
            pred_class = int(data.get("predicted_class"))
            transactions += 1

            key = "fraud_features" if pred_class else "legitimate_features"

            for feature in ['ip_switched', 'sex_changed', 'identity_changed', 'fast_purchase', 'device_txn_velocity_1h', 'device_txn_velocity_24h']:
                # quickfix: works for binary addiition
                value = float(data.get(feature)) 
                feature_stats[key][feature] = feature_stats[key].get(feature, 0.0) + value

        print(f"feature_stats: {feature_stats}")

        # change from sum to mean
        for key in ["fraud_features", "legitimate_features"]:
            for feature in ['ip_switched', 'sex_changed', 'identity_changed', 'fast_purchase', 'device_txn_velocity_1h', 'device_txn_velocity_24h']:
                feature_stats[key][feature] = round(feature_stats[key][feature] / transactions, 2)
        
        return feature_stats
            
    except Exception as e:                                                                           
        raise HTTPException(status_code=500, detail=str(e))
    