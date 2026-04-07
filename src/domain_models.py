from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class Transaction:
    """An entity that represents a transaction."""
    id: uuid.UUID
    device_id: str
    purchase_value: float
    purchase_time: datetime
    device_txn_idx: int
    device_time_since_last_s: float
    device_age_hours: float
    signup_before_first_device_txn: int
    repeated_device_purchase: int
    purchase_spike: int
    identity_changed: int
    device_txn_velocity_24h: int
    prev_is_fraud: int
    global_txn_velocity_24h: int
    country_txn_velocity_24h: int
    time_setup_to_txn_seconds: float
    purchase_hour: int
    is_late_night: int
    under_18: int
    age_18_25: int
    age_26_35: int
    age_36_50: int
    age_50_above: int
    amount_per_age: float
    unknown_country: int
    source: str
    browser: str
    true_label: int

@dataclass
class Prediction:
    """An entity that represents a transaction's prediction from a fraud detection model."""
    transaction_id: uuid.UUID
    model_used: str
    model_version: int
    fraud_proba: float