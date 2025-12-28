# run python -m pytest tests/test_streaming.py

import pytest
import fakeredis
from api.consumer import InferenceConsumer
from src.config import RESULT_STREAM
from src.state.redis_store import DeviceState
from uuid import uuid4

@pytest.fixture
def fake_redis():
    """Return an in-memory fake Redia client."""
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    yield client
    client.flushall()  # clean up after test


@pytest.fixture
def device_state(fake_redis):
    """Return a DeviceState backed by fake Redis."""
    ds = DeviceState()
    ds.client = fake_redis
    return ds


@pytest.fixture
def consumer(fake_redis, device_state):
    c = InferenceConsumer()
    c.client = fake_redis
    c.device_state = device_state
    return c


def make_transaction(device_id="DEV123", purchase_value=100.0):
    """Helper to generate a fake transaction dict"""
    transaction_id = str(uuid4())
    return {
        "transaction_id": transaction_id,
        "user_id": 22058, 
        "device_id": device_id,
        "signup_time": "2025-01-01T12:00:00",
        "purchase_time": "2025-01-02T12:00:00",
        "purchase_value": purchase_value,
        "source": "web",
        "browser": "chrome",
        "sex": "M",
        "age": 30,
        "ip_address": 732758368.79972
    }


def make_label(transaction_id, device_id, is_fraud=1):
    return {
        "transaction_id": transaction_id,
        "device_id": device_id,
        "is_fraud": is_fraud
    }


def test_handle_transaction_updates_device_state_and_results(consumer):
    txn = make_transaction(device_id="DEV123")
    consumer.handle_transaction(txn)

    # Device state should be updated
    state = consumer.device_state.get_device_state("DEV123")
    assert state[0] == 1  # txn_count incremented
    assert float(state[1]) == txn["purchase_value"]

    # RESULT_STREAM should have a new entry
    entries = consumer.client.xrange(RESULT_STREAM)
    print(entries)
    assert len(entries) == 1
    stored_txn = entries[0][1]
    assert stored_txn["transaction_id"] == txn["transaction_id"]

    # prediction hash should have a new entry
    prediction_key = f"prediction:{txn['transaction_id']}"
    prediction_hash = consumer.client.hgetall(prediction_key)
    assert "predicted_class" in prediction_hash
    assert "fraud_probability" in prediction_hash


def test_handle_label_updates_prediction_and_device_fraud_count(consumer):
    # Create a transaction to have a prediction hash
    txn = make_transaction(device_id="DEV123")
    consumer.handle_transaction(txn)
    prediction_key = f"prediction:{txn['transaction_id']}"

    # Prediction label should initially not exist
    initial_pred = consumer.client.hget(prediction_key, "true_label")
    assert initial_pred == ""

    # Create and handle a label
    label = make_label(txn["transaction_id"], txn["device_id"], is_fraud=1)
    consumer.handle_label(label)

    # Prediction hash updated
    pred = consumer.client.hget(prediction_key, "true_label")
    assert int(pred) == 1

    # Device state fraud_count incremented
    state = consumer.device_state.get_device_state("DEV123")
    assert state[-1] == 1  # fraud_count