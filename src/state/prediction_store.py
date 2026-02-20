from datetime import datetime
from redis import Redis
from src.config import(
   REDIS_DB,
   REDIS_HOST,
   REDIS_PORT
)
from src.state.device_schema import PREDICTION_STORE_SCHEMA

class PredictionStore:
    
    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    def _serialize_processed_transaction(self, processed_transaction: dict) -> dict:
        serialized = {}
        for field, field_type in PREDICTION_STORE_SCHEMA.items():
            if field in processed_transaction:
                serialized[field] = field_type.serialize(processed_transaction[field])
        return serialized
    
    def update_predictions(self, processed_transaction: dict) -> None:
        key = f"prediction:{processed_transaction["transaction_id"]}"
        serialized_processed_transaction = self._serialize_processed_transaction(processed_transaction)
        self.client.hset(key, mapping=serialized_processed_transaction)
        return serialized_processed_transaction

    def update_label(self, transaction_id, is_fraud):
        prediction_key = f"prediction:{transaction_id}"
        self.client.hset(prediction_key, "true_label", int(is_fraud))
        print(f"transaction after updating label: {self.client.hgetall(prediction_key)}")

