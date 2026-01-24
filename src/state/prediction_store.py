from datetime import datetime
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
                "purchase_time": processed_transaction["purchase_time"],
                "device_txn_idx": processed_transaction["device_txn_idx"],
                "first_device_txn": processed_transaction["first_device_txn"],
                "device_time_since_last_s": processed_transaction["device_time_since_last_s"],
                "device_age_hours": processed_transaction["device_age_hours"],
                "signup_before_first_device_txn": processed_transaction["signup_before_first_device_txn"],
                "ip_switched": processed_transaction["ip_switched"],
                "repeated_device_purchase": processed_transaction["repeated_device_purchase"],
                "identity_counts": processed_transaction["identity_counts"],
                "device_txn_velocity_1h": processed_transaction["device_txn_velocity_1h"],
                "device_txn_velocity_24h": processed_transaction["device_txn_velocity_24h"],
                "fast_purchase": processed_transaction["fast_purchase"],
                "time_setup_to_txn_seconds": processed_transaction["time_setup_to_txn_seconds"],
                # "ip_txn_idx": processed_transaction["ip_txn_idx"],
                "global_txn_velocity_1h": processed_transaction["global_txn_velocity_1h"],
                "global_txn_velocity_24h": processed_transaction["global_txn_velocity_24h"],
                "device_txn_share_24h": processed_transaction["device_txn_share_24h"],
                "predicted_class": processed_transaction["predicted_class"],
                "fraud_probability": processed_transaction["fraud_probability"],
                "true_label": ""
            }
        )
        

    def serialize_processed_transaction(self, transaction_id, purchase_time, processed_transaction):
        """Serialize and add transaction_id to processed transaction."""
        return {
            "transaction_id": transaction_id,
            "purchase_time": purchase_time.isoformat(),
            "device_txn_idx": int(processed_transaction["device_txn_idx"]),
            "first_device_txn": int(processed_transaction["first_device_txn"]),
            "device_time_since_last_s": float(processed_transaction["device_time_since_last_s"]),
            "device_age_hours": float(processed_transaction["device_age_hours"]),
            "signup_before_first_device_txn": int(processed_transaction["signup_before_first_device_txn"]),
            "ip_switched": int(processed_transaction["ip_switched"]),
            "repeated_device_purchase": int(processed_transaction["repeated_device_purchase"]),
            "identity_counts": int(processed_transaction["identity_counts"]),
            "device_txn_velocity_1h": int(processed_transaction["device_txn_velocity_1h"]),
            "device_txn_velocity_24h": int(processed_transaction["device_txn_velocity_24h"]),
            "fast_purchase": int(processed_transaction["fast_purchase"]),
            "time_setup_to_txn_seconds": int(processed_transaction["time_setup_to_txn_seconds"]),
            # "ip_txn_idx": processed_transaction["ip_txn_idx"],
            "global_txn_velocity_1h": int(processed_transaction["global_txn_velocity_1h"]),
            "global_txn_velocity_24h": int(processed_transaction["global_txn_velocity_24h"]),
            "device_txn_share_24h": float(processed_transaction["device_txn_share_24h"]),
            "predicted_class": int(processed_transaction["predicted_class"]),
            "fraud_probability": float(processed_transaction["fraud_probability"]),
        }

    def update_label(self, transaction_id, is_fraud):
        prediction_key = f"prediction:{transaction_id}"
        self.client.hset(prediction_key, "true_label", int(is_fraud))

