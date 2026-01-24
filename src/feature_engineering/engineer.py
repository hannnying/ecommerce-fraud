import numpy as np
from redis import Redis
from src.config import(
    BUCKET_SIZE_SECONDS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT
)
from src.state.device_state import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState
from src.state.serializers import deserialize_transaction
from uuid import uuid4


class TransactionFeatureEngineer:

    def __init__(self):
        self.device_state = DeviceState()
        self.global_velocity = GlobalVelocity()
        self.ip_state = IPState()

    def compute_device_features(
            self,
            device_id,
            purchase_value,
            purchase_time,
            signup_time,
            ip_address,
            txn_count,
            first_seen_signup,
            first_seen,
            last_seen,
            prev_ip,
            identities,
            purchase_values
    ):
        device_features = {}

        device_features["device_txn_idx"] = txn_count + 1 if txn_count else 1
        device_features["first_device_txn"] = True if not txn_count else False
        device_features["device_time_since_last_s"] = -1 if not last_seen else (purchase_time - last_seen).total_seconds()
        device_features["device_age_hours"] = -1 if not first_seen else (purchase_time - first_seen).total_seconds() / 3600
        device_features["signup_before_first_device_txn"] = False if not first_seen_signup else signup_time < first_seen_signup

        device_features["ip_switched"] = True if prev_ip != ip_address else False

        if purchase_values and purchase_values[-1] == purchase_value:
            device_features["repeated_device_purchase"] = True
        else:
            device_features["repeated_device_purchase"] = False

        device_features["identity_counts"] = len(identities) if identities else 1
        device_features["device_txn_velocity_1h"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "1h") + 1
        device_features["device_txn_velocity_24h"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "24h") + 1

        return device_features
    

    def compute_features(self, raw_transaction, training=True, transaction_id=None):
        # unpack transaction
        if training:
            signup_time = raw_transaction["signup_time"]
            purchase_time = raw_transaction["purchase_time"]
            purchase_value = raw_transaction["purchase_value"]
            device_id = raw_transaction["device_id"]
            sex = raw_transaction["sex"]
            age = raw_transaction["age"]
            ip_address = raw_transaction["ip_address"]
            
        else:
            transaction_id, _, signup_time, purchase_time, purchase_value, device_id, _, _, sex, age, ip_address = deserialize_transaction(raw_transaction)
        
        # if device is unseen, get_device_state returns [None, .. , None]
        txn_count, first_seen_signup, first_seen, last_seen, prev_ip, identities, purchase_values = self.device_state.get_device_state(device_id)
        
        processed_transaction = self.compute_device_features(
            device_id,
            purchase_value,
            purchase_time,
            signup_time,
            ip_address,
            txn_count,
            first_seen_signup,
            first_seen,
            last_seen,
            prev_ip,
            identities,
            purchase_values
        )

        time_setup_to_txn_seconds = (purchase_time - signup_time).total_seconds()

        processed_transaction["fast_purchase"] = time_setup_to_txn_seconds <= 60
        processed_transaction["time_setup_to_txn_seconds"] = time_setup_to_txn_seconds
        # processed_transaction["ip_txn_idx"] = self.ip_state.get_ip_txn_idx(ip_address) + 1
        processed_transaction["global_txn_velocity_1h"] = self.global_velocity.get_global_txn_velocity(purchase_time, "1h") + 1
        processed_transaction["global_txn_velocity_24h"] = self.global_velocity.get_global_txn_velocity(purchase_time, "24h") + 1
        processed_transaction["device_txn_share_24h"] = processed_transaction["device_txn_velocity_24h"] / processed_transaction["global_txn_velocity_24h"]

        return processed_transaction, transaction_id, (device_id, txn_count, purchase_value, sex, age, purchase_time, signup_time, ip_address, txn_count, first_seen_signup, first_seen, identities, purchase_values)
