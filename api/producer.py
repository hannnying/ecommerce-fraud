import random
from redis import Redis
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from src.config import (
    LABELS_STREAM,
    RAW_DATA_PATH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    TRANSACTION_STREAM
)
from src.state.serializers import (
    serialize_transaction
)
from uuid import uuid4

class TransactionProducer:
    def __init__(self, transactions_per_second=100, start_idx=50000):
        self.df = pd.read_csv(RAW_DATA_PATH)
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.tps = transactions_per_second
        self.start_idx = start_idx


    def start_streaming(self):
        """Simulate real-time transactions arriving and return their ids."""
        transaction_ids = []
        for idx in range(self.start_idx, self.start_idx+5):
            row = self.df.iloc[idx]

            try:
                transaction_id = str(uuid4())
                transaction_dict = serialize_transaction(transaction_id, row)
                self.client.xadd(TRANSACTION_STREAM, transaction_dict)

                self.schedule_label(transaction_id, row["device_id"], row["purchase_time"], row["is_fraud"])

                time.sleep(1.0 / self.tps)

                transaction_ids.append(transaction_id)

            except Exception as e:
                print(f"Error adding entry to stream: {e}")

        return transaction_ids

    def schedule_label(self, transaction_id: str, device_id: str, purchase_time: str, is_fraud: np.int64):
        """
        Simulate label arriving after delay.
        Store in Redis sorted set for delayed processing.
        """
        delay_days = 0 # due to the nature of the dataset, it seems that many fraud take place one after another
        arrival_time = datetime.fromisoformat(purchase_time) + timedelta(days=delay_days)
        label_dict = {
            "transaction_id": transaction_id,
            "device_id": device_id,
            "is_fraud": int(is_fraud),
            "arrival_time": arrival_time.isoformat()
        }

        self.client.xadd(LABELS_STREAM, label_dict)