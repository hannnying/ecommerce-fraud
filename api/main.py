from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Depends, FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from redis import Redis
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score, 
    recall_score
)
from src.config import (
    numerical_features,
    categorical_features,
    boolean_features
)
from src.drift_monitor.drift_monitor import DriftMonitor

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

        actual_fraud_rate = float(y_true.mean())
        predicted_fraud_rate = float(y_pred.mean())
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

        return {
            "count": len(y_true),
            "actual_fraud_rate": actual_fraud_rate,
            "predicted_fraud_rate": predicted_fraud_rate,
            "fraud_rate_gap": abs(actual_fraud_rate - predicted_fraud_rate),
            "false positive rate": 100 * fp / len(y_true),
            "false negative rate": 100 * fn / len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "pr_auc": average_precision_score(y_true, y_proba)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/evaluate/rolling")
async def evaluate_rolling(
    window_size: int = 5000,
    threshold: float = 0.45,
    redis_client: Redis = Depends(get_redis_client)
):
    fetch_size = window_size * 3
    entries = redis_client.xrevrange(RESULT_STREAM, count=fetch_size)
    
    if not entries:
        return {                                                                       
            "count": 0,                                                                
            "window_size": window_size,                                                
            "message": "No transactions in RESULT_STREAM"                              
      }     

    y_true = []
    y_proba = []

    try:
        # iterate through recent transactions, collect labelled  ones
        for stream_id, data in entries:
            transaction_id = data.get("transaction_id")
            if not transaction_id:
                continue

            # check if transaction's true label is known
            prediction_key = f"prediction:{transaction_id}"
            true_label = redis_client.get(prediction_key, "true_label")

            if not true_label:
                continue

            fraud_proba = float(data.get("fraud_probability", 0))

            y_true.append(true_label)
            y_proba.append(fraud_proba)

            if len(y_true) >= window_size:
                break
            
        if len(y_true) == 0:
            return {                                                                       
                "count": 0,                                                                
                "window_size": window_size,                                                
                "message": "No labeled transactions found in recent data"                              
        }     

        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        y_pred = y_proba >= threshold

        actual_fraud_rate = float(y_true.mean())
        predicted_fraud_rate = float(y_pred.mean())
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

        return {
            "count": len(y_true),
            "window_size": window_size,
            "actual_fraud_rate": actual_fraud_rate,
            "predicted_fraud_rate": predicted_fraud_rate,
            "fraud_rate_gap": abs(actual_fraud_rate - predicted_fraud_rate),
            "false positive rate": 100 * fp / len(y_true),
            "false negative rate": 100 * fn / len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "pr_auc": average_precision_score(y_true, y_proba)
        }

    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/drift/check")
def check_drift(
    window_size: int = 5000,
    drift_threshold: float = 0.2,
    redis_client: Redis = Depends(get_redis_client)
):
    
    try:
        entries = redis_client.xrevrange(RESULT_STREAM, count=window_size)
        if not entries:
            return {                                                                       
                "message": "No transactions available",                                    
                "drift_results": {},                                                       
                "num_features_drifted": 0,                                                 
                "should_retrain": False                                                    
            }  
        
        transactions = []

        for stream_id, data in entries:
            transaction = {
                # Numerical features
                "device_txn_idx": data.get("device_txn_idx"),
                "device_time_since_last_s": data.get("device_time_since_last_s"),
                "device_age_hours": data.get("device_age_hours"),
                "identity_counts": data.get("identity_counts"),
                "device_txn_velocity_1h": data.get("device_txn_velocity_1h"),
                "device_txn_velocity_24h": data.get("device_txn_velocity_24h"),
                "time_setup_to_txn_seconds": data.get("time_setup_to_txn_seconds"),
                "global_txn_velocity_1h": data.get("global_txn_velocity_1h"),
                "global_txn_velocity_24h": data.get("global_txn_velocity_24h"),
                "device_txn_share_24h": data.get("device_txn_share_24h"),

                # Boolean features
                "first_device_txn": data.get("first_device_txn"),
                "signup_before_first_device_txn": data.get("signup_before_first_device_txn"),
                "ip_switched": data.get("ip_switched"),
                "repeated_device_purchase": data.get("repeated_device_purchase"),
                "fast_purchase": data.get("fast_purchase")
            }
            transactions.append(transaction)

        current_df = pd.DataFrame(transactions)

        drift_monitor = DriftMonitor()

        # Get production model's run_id to retrieve stored data references
        client = MlflowClient()
        try:
            model_version = client.get_model_version_by_alias("fraud_detection_model", "Production")
            run_id = model_version.run_id 
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Production model not found. Error: {e}"
            )
        
        drift_results = {}
        for feature in categorical_features:
            if feature not in current_df.columns:
                continue

            try:
                baseline_dict = mlflow.artifacts.load_dict(
                    f"runs:/{run_id}/drift_reference/{feature}.json"
                )
                current_dict = drift_monitor.build_categorical_feature_reference(
                    current_df, feature, logging=False
                )
                drift_score = drift_monitor.monitor_drift(baseline_dict, current_dict)
                drift_results[feature] = {
                    "drift_score": float(drift_score),
                    "drifted": drift_score >= drift_threshold,
                    "type": "categorical"
                }
            
            except Exception as e:
                print(f"Error checking drift for {feature}: {e}")                          
                continue

        for feature in boolean_features:
            if feature not in current_df.columns:
                continue

            try:
                baseline_dict = mlflow.artifacts.load_dict(
                    f"runs:/{run_id}/drift_reference/{feature}.json"
                )
                current_dict = drift_monitor.build_categorical_feature_reference(
                    current_df, feature, logging=False
                )
                drift_score = drift_monitor.monitor_drift(baseline_dict, current_dict)
                drift_results[feature] = {
                    "drift_score": float(drift_score),
                    "drifted": drift_score >= drift_threshold,
                    "type": "boolean"
                }
            
            except Exception as e:
                print(f"Error checking drift for {feature}: {e}")                          
                continue

        for feature in numerical_features:
            if feature not in current_df.columns:
                continue

            try:
                baseline_dict = mlflow.artifacts.load_dict(
                    f"runs:/{run_id}/drift_reference/{feature}.json"
                )
                current_dict = drift_monitor.build_categorical_feature_reference(
                    current_df, feature, logging=False
                )
                drift_score = drift_monitor.monitor_drift(baseline_dict, current_dict)
                drift_results[feature] = {
                    "drift_score": float(drift_score),
                    "drifted": drift_score >= drift_threshold,
                    "type": "numerical"
                }
            
            except Exception as e:
                print(f"Error checking drift for {feature}: {e}")                          
                continue

        return {
            "window_size": len(current_df),
            "threshold": drift_threshold,
            "run_id": run_id,
            "drift_results": drift_results,
            "num_features_drifted": sum(1 for r in drift_results if r["drifted"]),
            "total_features_monitored": len(drift_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
