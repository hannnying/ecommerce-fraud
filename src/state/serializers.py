from datetime import datetime
import json


def deserialize_raw_state(raw):
    txn_count = int(raw[0]) if raw[0] else 0
    prev_purchase_value = float(raw[1]) if raw[1] else None
    prev_sex = str(raw[2]) if raw[1] else ""
    prev_age = int(raw[3]) if raw[1] else None
    last_transaction = datetime.fromisoformat(raw[4]) if raw[4] else None
    first_seen = datetime.fromisoformat(raw[5]) if raw[5] else None
    ip_addresses = set(json.loads(raw[6])) if raw[6] else set()
    sources = set(json.loads(raw[7])) if raw[7] else set()

    return txn_count, prev_purchase_value, prev_sex, prev_age, last_transaction, first_seen, ip_addresses, sources


def deserialize_transaction(transaction):
    transaction_id = transaction["transaction_id"]
    user_id = transaction["user_id"]
    signup_time = datetime.fromisoformat(transaction["signup_time"])
    purchase_time = datetime.fromisoformat(transaction["purchase_time"])
    purchase_value = float(transaction["purchase_value"])
    device_id = transaction["device_id"]
    source = transaction["source"]
    browser = transaction["browser"]
    sex = transaction["sex"]
    age = int(transaction["age"])
    ip_address = float(transaction["ip_address"])

    return transaction_id, user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address


def serialize_processed_transaction(transaction_id, processed_transaction):
    return {
        "transaction_id": transaction_id,
        "txn_count": int(processed_transaction["txn_count"]),
        "log_time_setup_to_txn_seconds": float(processed_transaction["log_time_setup_to_txn_seconds"]),
        "first_device_transaction": int(processed_transaction["first_device_transaction"]),
        "scaled_device_purchase_diff": float(processed_transaction["scaled_device_purchase_diff"]),
        "repeated_device_purchase": int(processed_transaction["repeated_device_purchase"]),
        "identity_changed": int(processed_transaction["identity_changed"]),
        "predicted_class": int(processed_transaction["predicted_class"]),
        "fraud_probability": float(processed_transaction["fraud_probability"])
    }


def serialize_state(txn_count, prev_purchase_value, prev_sex, prev_age, last_transaction, first_seen, ip_addresses, sources):
    if type(last_transaction) != str:
        last_transaction = last_transaction.isoformat()

    if type(first_seen) != str:
        first_seen = first_seen.isoformat()

    return {
        "txn_count": txn_count,
        "prev_purchase_value": float(prev_purchase_value),
        "prev_sex": str(prev_sex),
        "prev_age": int(prev_age),
        "last_transaction": last_transaction,
        "first_seen": first_seen,
        "ip_addresss": json.dumps(list(ip_addresses)),
        "sources": json.dumps(list(sources)),
    }
    


def serialize_transaction(transaction_id, transaction): # serializes raw transaction
    return {
            "transaction_id": transaction_id,
            "user_id": str(transaction["user_id"]),
            "signup_time": transaction["signup_time"],  # in isoformat 
            "purchase_time": transaction["purchase_time"], # in isoformat 
            "purchase_value": float(transaction["purchase_value"]),
            "device_id": str(transaction["device_id"]),
            "source": str(transaction["source"]),
            "browser": str(transaction["browser"]),
            "sex": str(transaction["sex"]),
            "age": int(transaction["age"]),
            "ip_address": float(transaction["ip_address"]),
        }


