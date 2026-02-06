from redis import Redis
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from src.config import (
    LABELS_STREAM,
    FRAUD_WITH_COUNTRY_PATH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    START_IDX,
    TRANSACTION_STREAM
)
from src.state.serializers import (
    serialize_transaction
)
from uuid import uuid4

class TransactionProducer:
    def __init__(self,  start_date, end_date, transactions_per_second=100):
        self.df = None
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.tps = transactions_per_second
        self.start_date = start_date
        self.end_date = end_date

        self._initialise_data(start_date, end_date)

    def _initialise_data(self, start_date, end_date):
        df = pd.read_csv(FRAUD_WITH_COUNTRY_PATH)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df = df.sort_values(by="purchase_time")
        df = df[(df["purchase_time"] >= start_date) & (df["purchase_time"] <= end_date)]

        self.df = df

    def start_streaming(self):
        """Simulate real-time transactions arriving and return their ids."""
        transaction_ids = []

        for idx, row in self.df.iterrows():
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
        arrival_time = purchase_time + timedelta(days=delay_days)
        label_dict = {
            "transaction_id": transaction_id,
            "device_id": device_id,
            "is_fraud": int(is_fraud),
            "arrival_time": arrival_time.isoformat()
        }

        self.client.xadd(LABELS_STREAM, label_dict)

if __name__=="__main__":
    txn_producer = TransactionProducer(start_date=datetime(2015, 7, 1), end_date=datetime(2015, 7, 31))
    transaction_ids = txn_producer.start_streaming()