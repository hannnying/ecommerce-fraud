"""
Device state schema configuration.
Add/remove fields here to automatically update serialization across the system.
"""
from datetime import datetime
import json
from typing import Any, Callable

from sqlalchemy import Column, String, Float, Integer, DateTime

class FieldType:
    """Defines how a field is serialized and deserialized."""

    def __init__(self, serialize_fn: Callable, deserialize_fn: Callable, default: Any = None):
        self.serialize = serialize_fn
        self.deserialize = deserialize_fn
        self.default = default


# Reusable field type definitions
FIELD_TYPES = {
    "int": FieldType(
        serialize_fn=lambda x: int(x),
        deserialize_fn=lambda x: int(x) if x else 0,
        default=0
    ),
    "float": FieldType(
        serialize_fn=lambda x: float(x),
        deserialize_fn=lambda x: float(x) if x else None,
        default=None
    ),
    "str": FieldType(
        serialize_fn=lambda x: str(x),
        deserialize_fn=lambda x: str(x) if x else None,
        default=None
    ),
    "datetime": FieldType(
        serialize_fn=lambda x: x.isoformat() if not isinstance(x, str) else x,
        deserialize_fn=lambda x: datetime.fromisoformat(x) if x else None,
        default=None
    ),
    "set": FieldType(
        serialize_fn=lambda x: json.dumps(list(x)),
        deserialize_fn=lambda x: set(json.loads(x)) if x else set(),
        default=set()
    ),
    "list": FieldType(
        serialize_fn=lambda x: json.dumps(x),
        deserialize_fn=lambda x: list(json.loads(x)) if x else [],
        default=[]
    ),
}

DEVICE_STATE_SCHEMA = {
    "txn_count": FIELD_TYPES["int"],
    "first_seen_signup": FIELD_TYPES["datetime"],
    "first_seen": FIELD_TYPES["datetime"],
    "last_seen": FIELD_TYPES["datetime"],
    "prev_identity": FIELD_TYPES["str"],
    "prev_purchase": FIELD_TYPES["int"],
    "prev_is_fraud": FIELD_TYPES["int"]
}

# PREDICTION STORE SCHEMA (Redis hash storage)
# Stores values before scaling and encoding
PREDICTION_STORE_SCHEMA = {
    "transaction_id": FIELD_TYPES["str"],
    "device_id": FIELD_TYPES["str"],
    "purchase_value": FIELD_TYPES["float"],
    "purchase_time": FIELD_TYPES["datetime"],
    "device_txn_idx": FIELD_TYPES["int"],
    "device_time_since_last_s": FIELD_TYPES["float"],
    "device_age_hours": FIELD_TYPES["float"],
    "signup_before_first_device_txn": FIELD_TYPES["int"],
    "repeated_device_purchase": FIELD_TYPES["int"],
    "purchase_spike": FIELD_TYPES["int"],
    "identity_changed": FIELD_TYPES["int"],
    "device_txn_velocity_24h": FIELD_TYPES["int"],
    "prev_is_fraud": FIELD_TYPES["int"],
    "global_txn_velocity_24h": FIELD_TYPES["int"],
    "country_txn_velocity_24h": FIELD_TYPES["int"],

    "time_setup_to_txn_seconds": FIELD_TYPES["float"],
    "purchase_hour": FIELD_TYPES["int"],
    "is_late_night": FIELD_TYPES["int"],
    "under_18": FIELD_TYPES["int"],
    "age_18_25": FIELD_TYPES["int"],
    "age_26_35": FIELD_TYPES["int"],
    "age_36_50": FIELD_TYPES["int"],
    "over_50": FIELD_TYPES["int"],
    "amount_per_age": FIELD_TYPES["float"],
    "unknown_country": FIELD_TYPES["int"],
    "source": FIELD_TYPES["str"],
    "browser": FIELD_TYPES["str"],

    "fraud_proba": FIELD_TYPES["float"],
    "model_used": FIELD_TYPES["str"],
    "true_label": FIELD_TYPES["int"]
}

# PREDICTION STORE SCHEMA (SQLAlchemy)
PREDICTION_ATTR_DICT = {
    "__tablename__": "predictions",

    "transaction_id": Column(String, primary_key=True),

    "device_id": Column(String),
    "purchase_value": Column(Float),
    "purchase_time": Column(DateTime),

    "device_txn_idx": Column(Integer),
    "device_time_since_last_s": Column(Float),
    "device_age_hours": Column(Float),

    "signup_before_first_device_txn": Column(Integer),
    "repeated_device_purchase": Column(Integer),
    "purchase_spike": Column(Integer),
    "identity_changed": Column(Integer),

    "device_txn_velocity_24h": Column(Integer),
    "prev_is_fraud": Column(Integer),
    "global_txn_velocity_24h": Column(Integer),
    "country_txn_velocity_24h": Column(Integer),

    "time_setup_to_txn_seconds": Column(Float),
    "purchase_hour": Column(Integer),
    "is_late_night": Column(Integer),

    "under_18": Column(Integer),

    # Columns starting with numbers need safe Python attribute names
    "age_18_25": Column("18_25", Integer),
    "age_26_35": Column("26_35", Integer),
    "age_36_50": Column("36_50", Integer),

    "over_50": Column(Integer),

    "amount_per_age": Column(Float),

    "unknown_country": Column(Integer),
    "source": Column(String),
    "browser": Column(String),

    "fraud_proba": Column(Float),
    "model_used": Column(String),

    "true_label": Column(Integer),
}
def get_field_names(schema):
    """Get ordered list of field names from schema."""
    return list(schema.keys())


def get_default_state(schema):
    """Get default device state dict with all fields initialized."""
    return {field: field_type.default for field, field_type in schema.items()}
