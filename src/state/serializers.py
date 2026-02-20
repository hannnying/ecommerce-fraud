from datetime import datetime
import json

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
    country = str(transaction["country"])

    return transaction_id, user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, country


def serialize_transaction(transaction_id, transaction): # serializes raw transaction
    return {
            "transaction_id": transaction_id,
            "user_id": str(transaction["user_id"]),
            "signup_time": transaction["signup_time"].isoformat(),  # in isoformat 
            "purchase_time": transaction["purchase_time"].isoformat(), # in isoformat 
            "purchase_value": float(transaction["purchase_value"]),
            "device_id": str(transaction["device_id"]),
            "source": str(transaction["source"]),
            "browser": str(transaction["browser"]),
            "sex": str(transaction["sex"]),
            "age": int(transaction["age"]),
            "ip_address": float(transaction["ip_address"]),
            "country": str(transaction["country"])
        }


