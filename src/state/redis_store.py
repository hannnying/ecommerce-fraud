import datetime
import json
from redis import Redis
from src.config import(
   REDIS_DB,
   REDIS_HOST,
   REDIS_PORT
)
from src.state.serializers import (
    deserialize_raw_state,
    deserialize_transaction,
    serialize_state,
    serialize_transaction
)


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