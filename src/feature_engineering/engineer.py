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
        txn_count,
        last_transaction,
        last_ip,
        ip_address,
        ip_addresses,
        prev_purchase_value, 
        prev_sex,
        prev_age,
        signup_time,
        purchase_time,
        purchase_value,
        sex,
        age
    ):
        device_features = {}

        device_features["device_txn_idx"] = txn_count + 1 if txn_count else 1
        device_features["first_device_txn"] = True if not txn_count else False
        device_features["device_time_since_last"] = -1 if not last_transaction else (purchase_time - last_transaction).total_seconds()
        device_features["ip_switched"] = True if last_ip != ip_address else False
        device_features["ip_switches"] = len(ip_addresses) - 1 if not device_features["ip_switched"] else len(ip_addresses)
        
        if not prev_sex:
            prev_sex = sex
        if not prev_age:
            prev_age = age
        device_features["sex_changed"] = True if prev_sex != sex else False
        device_features["age_diff"] = abs(prev_age - age) if prev_age else 0
        device_features["identity_changed"] = True if device_features["sex_changed"] and device_features["age_diff"] else False
        device_features["scaled_age_diff"] = device_features["age_diff"] / prev_age
        
        if device_features["first_device_txn"]:
            device_features["repeated_device_purchase"] = -1
        else:
            device_features["repeated_device_purchase"] = True if prev_purchase_value == purchase_value else False
        
        if device_features["first_device_txn"]:
            device_features["scaled_device_purchase_diff"] = -1
        else:
            device_features["scaled_device_purchase_diff"] = abs(prev_purchase_value - purchase_value) / prev_purchase_value

        device_features["device_txn_velocity_1m"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "1m")
        device_features["device_txn_velocity_5m"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "5m")
        device_features["device_txn_velocity_1h"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "1h")
        device_features["device_txn_velocity_24h"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "24h")

        return device_features
    

    def compute_features(self, raw_transaction, training=True, transaction_id=None):
        # unpack transaction
        if training:
            signup_time = raw_transaction["signup_time"]
            purchase_time = raw_transaction["purchase_time"]
            purchase_value = raw_transaction["purchase_value"]
            device_id = raw_transaction["device_id"]
            source = raw_transaction["source"]
            sex = raw_transaction["sex"]
            age = raw_transaction["age"]
            ip_address = raw_transaction["ip_address"]
            
        else:
            transaction_id, _, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address = deserialize_transaction(raw_transaction)
        
        # if device is unseen, get_device_state returns [None, .. , None]
        txn_count, prev_purchase_value, prev_sex, prev_age, last_transaction, last_ip, first_seen, ip_addresses, sources = self.device_state.get_device_state(device_id)
        
        processed_transaction = self.compute_device_features(
            device_id,
            txn_count,
            last_transaction,
            last_ip,
            ip_address,
            ip_addresses,
            prev_purchase_value, 
            prev_sex,
            prev_age,
            signup_time,
            purchase_time,
            purchase_value,
            sex,
            age
        )

        time_setup_to_txn_seconds = (purchase_time - signup_time).total_seconds()
        log_time_setup_to_txn_seconds = np.log1p(time_setup_to_txn_seconds)

        processed_transaction["fast_purchase"] = time_setup_to_txn_seconds <= 60
        processed_transaction["log_time_setup_to_txn_seconds"] = log_time_setup_to_txn_seconds
        processed_transaction["ip_txn_idx"] = self.ip_state.get_ip_txn_idx(ip_address) + 1
        processed_transaction["txn_velocity_1h"] = self.global_velocity.get_global_txn_velocity(purchase_time, "1h") + 1

        return processed_transaction, transaction_id, (device_id, txn_count, purchase_value, sex, age, first_seen, ip_addresses, sources, purchase_time, source, ip_address)
