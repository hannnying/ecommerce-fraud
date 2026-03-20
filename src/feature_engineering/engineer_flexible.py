"""
Flexible feature engineer that works with dict-based device state.
No more tuple unpacking or positional arguments!
"""
import numpy as np
from redis import Redis
from src.config import (
    BUCKET_SIZE_SECONDS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT
)
from src.state.device_state_flexible import DeviceState
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
            device_id: str,
            purchase_value: float,
            sex: str,
            age: int,
            purchase_time,
            signup_time,
            device_state: dict
    ):
        """
        Compute device-based features using device state dict.

        Args:
            device_id: Device identifier
            purchase_value: Transaction amount
            purchase_time: Transaction timestamp
            signup_time: Account signup timestamp
            ip_address: IP address (encoded as float)
            device_state: Dict containing device state fields

        Returns:
            Dict of computed features
        """
        device_features = {}

        # Access state fields with .get() for safe defaults
        txn_count = device_state.get("txn_count", 0)
        last_seen = device_state.get("last_seen")
        first_seen = device_state.get("first_seen")
        first_seen_signup = device_state.get("first_seen_signup")
        prev_identity = device_state.get("prev_identity")
        prev_purchase = device_state.get("prev_purchase")
        prev_is_fraud = device_state.get("prev_is_fraud")

        identity = sex + str(age)
        device_features["identity"] = identity

        # Compute features
        device_features["device_txn_idx"] = txn_count + 1 if txn_count else 1
        device_features["device_time_since_last_s"] = -1 if not last_seen else (purchase_time - last_seen).total_seconds()
        device_features["device_age_hours"] = -1 if not first_seen else (purchase_time - first_seen).total_seconds() / 3600
        device_features["signup_before_first_device_txn"] = False if not first_seen_signup else signup_time < first_seen_signup
        device_features["repeated_device_purchase"] = (prev_purchase != None) & (prev_purchase == purchase_value)
        device_features["purchase_spike"] = (prev_purchase != 0) & (prev_purchase < purchase_value)
        device_features["identity_changed"] = (prev_identity != None) & (prev_identity != identity)
        device_features["device_txn_velocity_24h"] = self.device_state.get_device_txn_velocity(device_id, purchase_time, "24h") + 1
        device_features["prev_is_fraud"] = prev_is_fraud == 1

        return device_features

    def compute_features(self, raw_transaction, training=True, transaction_id=None):
        """
        Compute all features for a transaction.

        Returns:
            Tuple of (processed_transaction, transaction_id, device_state_dict)
        """
        # Unpack transaction
        if training:
            signup_time = raw_transaction["signup_time"]
            purchase_time = raw_transaction["purchase_time"]
            purchase_value = raw_transaction["purchase_value"]
            device_id = raw_transaction["device_id"]
            source = raw_transaction["source"]
            browser = raw_transaction["browser"]
            sex = raw_transaction["sex"]
            age = raw_transaction["age"]
            _ = raw_transaction["ip_address"]
            country = raw_transaction["country"]

        else:
            # consider adding country
            transaction_id, _, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, _, country = deserialize_transaction(raw_transaction)

        # Get device state as dict (no more tuple unpacking!)
        device_state = self.device_state.get_device_state(device_id)

        # Compute device features
        processed_transaction = self.compute_device_features(
            device_id,
            purchase_value,
            sex,
            age,
            purchase_time,
            signup_time,
            device_state
        )

        # Compute other features
        time_setup_to_txn_seconds = (purchase_time - signup_time).total_seconds()

        processed_transaction["time_setup_to_txn_seconds"] = time_setup_to_txn_seconds
        processed_transaction["global_txn_velocity_24h"] = self.global_velocity.get_txn_velocity(purchase_time, "24h") + 1
        processed_transaction["country_txn_velocity_24h"] = self.global_velocity.get_txn_velocity(purchase_time, "24h", country) + 1
        
        processed_transaction["purchase_hour"] = purchase_time.hour
        processed_transaction["is_late_night"] = int(0 <= purchase_time.hour < 4)

        # age
        processed_transaction["under_18"] = age < 18
        processed_transaction["age_18_25"] = (age >= 18) & (age < 25)
        processed_transaction["age_26_35"] = (age >= 25) & (age < 36)
        processed_transaction["age_36_50"] = (age >= 36) & (age < 50)
        processed_transaction["age_50_above"] = age >= 50
        processed_transaction["amount_per_age"] = purchase_value / age

        # country
        processed_transaction["unknown_country"] = country == "Unknown"
        
        processed_transaction["source"] = source
        processed_transaction["browser"] = browser

        # Add purchase_time for filtering and storage
        processed_transaction["purchase_time"] = purchase_time

        # Prepare state updates for later
        # Instead of returning a huge tuple, return the state dict that will be updated
        state_to_update = {
            "txn_count": device_state.get("txn_count", 0) + 1,
            "first_seen_signup": signup_time if not device_state.get("first_seen_signup") else device_state["first_seen_signup"],
            "first_seen": purchase_time if not device_state.get("first_seen") else device_state["first_seen"],
            "last_seen": purchase_time,
            "prev_identity": sex + str(age),
            "prev_purchase": purchase_value,
            "prev_is_fraud": -1 # in reality, this will ot be known until a later date
        }

        # Return processed features, transaction_id, and state update dict
        return processed_transaction, transaction_id, (device_id, state_to_update)
