from redis import Redis
from src.config import(
   REDIS_DB,
   REDIS_HOST,
   REDIS_PORT
)

class PredictionStore:
    
    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    
    def update_predictions(self, device_id, processed_transaction): # processed_transaction_dict values are already seriaized
        transaction_id = processed_transaction["transaction_id"]
        self.client.hset(
            f"prediction:{transaction_id}",
            mapping={
                "transaction_id": transaction_id,
                "device_id": device_id,
                "device_txn_idx": processed_transaction["device_txn_idx"],
                "first_device_txn": processed_transaction["first_device_txn"],
                "device_time_since_last": processed_transaction["device_time_since_last"],
                "ip_switched": processed_transaction["ip_switched"],
                "sex_changed": processed_transaction["sex_changed"],
                "age_diff": processed_transaction["age_diff"],
                "scaled_age_diff": processed_transaction["scaled_age_diff"],
                "identity_changed": processed_transaction["identity_changed"],
                "scaled_device_purchase_diff": processed_transaction["scaled_device_purchase_diff"],
                "device_txn_velocity_1m": processed_transaction["device_txn_velocity_1m"],
                "device_txn_velocity_5m": processed_transaction["device_txn_velocity_5m"],
                "device_txn_velocity_1h": processed_transaction["device_txn_velocity_1h"],
                "device_txn_velocity_24h": processed_transaction["device_txn_velocity_24h"],
                "fast_purchase": processed_transaction["fast_purchase"],
                "log_time_setup_to_txn_seconds": processed_transaction["log_time_setup_to_txn_seconds"],
                "ip_txn_idx": processed_transaction["ip_txn_idx"],
                "txn_velocity_1h": processed_transaction["txn_velocity_1h"],
                "predicted_class": processed_transaction["predicted_class"],
                "fraud_probability": processed_transaction["fraud_probability"],
                "true_label": ""
            }
        )
        

    def serialize_processed_transaction(self, transaction_id, processed_transaction):
        """Serialize and add transaction_id to processed transaction."""
        return {
            "transaction_id": transaction_id,
            "device_txn_idx": int(processed_transaction["device_txn_idx"]),
            "first_device_txn": int(processed_transaction["first_device_txn"]),
            "device_time_since_last": float(processed_transaction["device_time_since_last"]),
            "ip_switched": int(processed_transaction["ip_switched"]),
            "sex_changed": int(processed_transaction["sex_changed"]),
            "age_diff": int(processed_transaction["age_diff"]),
            "scaled_age_diff": float(processed_transaction["scaled_age_diff"]),
            "identity_changed": int(processed_transaction["identity_changed"]),
            "scaled_device_purchase_diff": float(processed_transaction["scaled_device_purchase_diff"]),
            "device_txn_velocity_1m": int(processed_transaction["device_txn_velocity_1m"]),
            "device_txn_velocity_5m": int(processed_transaction["device_txn_velocity_5m"]),
            "device_txn_velocity_1h": int(processed_transaction["device_txn_velocity_1h"]),
            "device_txn_velocity_24h": int(processed_transaction["device_txn_velocity_24h"]),
            "fast_purchase": int(processed_transaction["fast_purchase"]),
            "log_time_setup_to_txn_seconds": float(processed_transaction["log_time_setup_to_txn_seconds"]),
            "ip_txn_idx": int(processed_transaction["ip_txn_idx"]),
            "txn_velocity_1h": int(processed_transaction["txn_velocity_1h"]),
            "predicted_class": int(processed_transaction["predicted_class"]),
            "fraud_probability": float(processed_transaction["fraud_probability"])
        }

    def update_label(self, transaction_id, is_fraud):
        prediction_key = f"prediction:{transaction_id}"
        self.client.hset(prediction_key, "true_label", int(is_fraud))

