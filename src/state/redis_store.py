import datetime
import json
from typing import Dict, List
from redis import Redis
from src.config import(
   REDIS_DB,
   REDIS_HOST,
   REDIS_PORT
)
from src.state.serializers import (
    deserialize_raw_state,
    deserialize_transaction,
    serialize_processed_transaction,
    serialize_state,
    serialize_transaction
)
from uuid import uuid4


class DeviceState:
   
    def __init__(self):
      self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    def load_initial(self, device_state_dict):
        """Load initial device state dictionary into Redis hash."""
        
        for device_id, stats in device_state_dict.items():
            txn_count = stats["txn_count"]
            purchase_sum = stats["purchase_sum"]
            last_transaction = stats["last_transaction"]
            first_seen = stats["first_seen"]
            ip_addresses = stats["ip_addresses"]
            sources = stats["sources"]
            fraud_count = stats["fraud_count"]

            self.client.hset(
                f"device:{device_id}",
                mapping=serialize_state(
                    txn_count,
                    purchase_sum,
                    last_transaction,
                    first_seen,
                    ip_addresses,
                    sources,
                    fraud_count
                )
            )

    def get_device_state(self, device_id):
        key = f"device:{device_id}"

        raw = self.client.hmget(
            key,
            "txn_count",
            "purchase_sum",
            "last_transaction",
            "first_seen",
            "ip_addresses",
            "sources",
            "fraud_count"
        )

        return deserialize_raw_state(raw)

    def update_device_state(
            self,
            device_id,
            txn_count,
            purchase_sum,
            last_transaction,
            first_seen,
            ip_addresses,
            sources,
            fraud_count,
            signup_time,
            purchase_time,
            purchase_value,
            source,
            ip_address
    ):
        """Update device state."""


        key = f"device:{device_id}"
        
        ip_addresses.add(ip_address)
        sources.add(source)

        updated_state = {
            "txn_count": txn_count + 1,
            "purchase_sum": purchase_sum + purchase_value,
            "last_transaction": purchase_time,
            "first_seen": purchase_time if not first_seen else first_seen,
            "ip_addresses": ip_addresses,
            "sources": sources,
            "fraud_count": fraud_count # updated when labels arrive 
        }

        self.client.hset(
            key, 
            mapping=serialize_state(
                updated_state["txn_count"], 
                updated_state["purchase_sum"],
                updated_state["last_transaction"],
                updated_state["first_seen"],
                updated_state["ip_addresses"],
                updated_state["sources"],
                updated_state["fraud_count"]
            )
        )

    def update_fraud_count(self, device_id, is_fraud):
        device_key = f"device:{device_id}"

        if is_fraud:
            self.client.hincrby(device_key, "fraud_count", 1)


class PredictionStore:
    
    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    def batch_update_predictions(self, X: List[Dict], device_ids, y_pred, y_proba, y=None):
        print(device_ids)
        i = 0
        for idx, processed_transaction in enumerate(X):
            transaction_id = str(uuid4())
            device_id = device_ids[idx]
            predicted_class = y_pred[idx]
            fraud_probability = y_proba[idx]

            processed_transaction["transaction_id"] = transaction_id
            processed_transaction["predicted_class"] = predicted_class
            processed_transaction["fraud_probability"] = fraud_probability
            processed_transaction_dict = serialize_processed_transaction(transaction_id, processed_transaction)

            if i == 0: 
                print(processed_transaction_dict)
                print(device_id)
                i+=1
        
            self.update_predictions(transaction_id, device_id, processed_transaction_dict)
            
            if y is not None:
                self.update_label(transaction_id, y[idx])
    
    def update_predictions(self, transaction_id, device_id, processed_transaction_dict): # processed_transaction_dict values are already seriaized
        self.client.hset(
            f"prediction:{transaction_id}",
            mapping={
                "transaction_id": processed_transaction_dict["transaction_id"],
                "device_id": device_id,
                "predicted_class": processed_transaction_dict["predicted_class"],
                "fraud_probability": processed_transaction_dict["fraud_probability"],
                "txn_count": processed_transaction_dict["txn_count"],
                "device_ip_consistency": processed_transaction_dict["device_ip_consistency"],
                "device_source_consistency": processed_transaction_dict["device_source_consistency"],
                "time_setup_to_txn_seconds": processed_transaction_dict["time_setup_to_txn_seconds"],
                "time_since_last_device_txn": processed_transaction_dict["time_since_last_device_txn"],
                "purchase_deviation_from_device_mean": processed_transaction_dict["purchase_deviation_from_device_mean"],
                "device_lifespan": processed_transaction_dict["device_lifespan"],
                "device_fraud_rate": processed_transaction_dict["device_fraud_rate"],
                "true_label": ""  # initially empty, updated later when label arrives
            }
        )

    def update_label(self, transaction_id, is_fraud):
        prediction_key = f"prediction:{transaction_id}"
        self.client.hset(prediction_key, "true_label", int(is_fraud))
