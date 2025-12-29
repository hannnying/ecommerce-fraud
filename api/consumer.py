# load model and preprocessor
# reads from redis stream
# for each entry,
    # get device_state, call compute_features, update device_state
    # run model: preprocessor and model
    # append to processed transaction redis store
import pandas as pd
import pickle
from redis import Redis
from src.config import (
    LABELS_STREAM,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    RESULT_STREAM,
    TRANSACTION_STREAM
)
from src.feature_engineering.engineer import compute_features
from src.state.redis_store import (
    DeviceState
)
from src.state.serializers import (
    deserialize_transaction,
    serialize_processed_transaction,
    serialize_transaction
)

class InferenceConsumer:
    def __init__(self, preprocessor_path=PREPROCESSOR_PATH, model_path=MODEL_PATH):
        self.preprocessor = None
        self.model = None
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.device_state = DeviceState()

        self._initialize_preprocessor_model(preprocessor_path, model_path)


    def _initialize_preprocessor_model(self, preprocessor_path, model_path):
        try:
            with open(preprocessor_path, "rb") as p:
                self.preprocessor = pickle.load(p)

            print(f"Loaded preprocessor at {preprocessor_path} into InferenceConsumer")
        except Exception as e:
            print(f"Error loading preprocessor: {e}")

        try:
            with open(model_path, "rb") as m:
                self.model = pickle.load(m)
            print(f"Loaded model at {model_path} into InferenceConsumer")
        except Exception as e:
            print(f"Error loading model: {e}")

    def handle_transaction(self, transaction_dict):
        """
        Preprocess an incoming transaction, compute features, run fraud inference, 
        update device state based on transaction behavior and append to RESULTS_STREAM.
        """
        device_id = transaction_dict["device_id"]

        # if device is unseen, get_device_state returns [None, .. , None]
        txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources, fraud_count = self.device_state.get_device_state(device_id)
        transaction_id, _, signup_time, purchase_time, purchase_value, device_id, source, _, _, _, ip_address = deserialize_transaction(transaction_dict)

        processed_transaction = self.process_transaction(
            txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources, fraud_count, signup_time, purchase_time, purchase_value, source, ip_address
        )

        self.device_state.update_device_state(
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
            ip_address,
        )
        print(f"processed transaction: {transaction_id}")

        # add processed transaction df to results stream
        self.store_result(transaction_id, device_id, processed_transaction)

    
    def handle_label(self, label_dict):
        """
        Join label into hash storing past transaction predictions, and update fraud_count of device state. 
        """
        
        transaction_id = label_dict["transaction_id"]
        device_id = label_dict["device_id"]
        prediction_key = f"prediction:{transaction_id}"
        device_key = f"device:{device_id}"
        is_fraud = int(label_dict["is_fraud"])

        # look for prediction with prediction_key in hash
        self.client.hset(prediction_key, "true_label", is_fraud)

        # look for device record with device_id in hash
        if is_fraud:
            self.client.hincrby(device_key, "fraud_count", 1)
        
        print(f"processed label: {transaction_id}")

    def start_consuming(self):
        """
        Consume transactions and labels from separate Redis streams and process them in real-time.
        """
        last_txn_id = "0-0"
        last_label_id = "0-0"
        while True:
            messages = self.client.xread(
                streams={
                    TRANSACTION_STREAM: last_txn_id,
                    LABELS_STREAM: last_label_id
                },
                count=1,
                block=1
            )

            if not messages:
                continue

            # messages is a list of (stream_name, entries)
            for stream_name, entries in messages:
                if stream_name == TRANSACTION_STREAM:
                    for message_id, transaction_dict in entries:
                        self.handle_transaction(transaction_dict)
                        last_txn_id = message_id

                elif stream_name == LABELS_STREAM:
                    for message_id, label_dict in entries:
                        self.handle_label(label_dict)
                        last_label_id = message_id


    def process_transaction(self, txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources, fraud_count, signup_time, purchase_time, purchase_value, source, ip_address):
        """
        Process single transaction and return a dictionary of the processed transaction with predictions.
        """
        txn_count, device_ip_consistency, device_source_consistency, time_setup_to_txn_seconds, time_since_last_device_id_txn, purchase_deviation_from_device_mean, device_lifespan, device_fraud_rate = compute_features(txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources, fraud_count, signup_time, purchase_time, purchase_value, source, ip_address)
        processed_transaction = {
            "txn_count": txn_count,
            "device_ip_consistency": device_ip_consistency,
            "device_source_consistency": device_source_consistency,
            "time_setup_to_txn_seconds": time_setup_to_txn_seconds,
            "time_since_last_device_txn": time_since_last_device_id_txn,
            "purchase_deviation_from_device_mean": purchase_deviation_from_device_mean,
            "device_lifespan": device_lifespan,
            "device_fraud_rate": device_fraud_rate
        }
        processed_transaction_df = pd.DataFrame([processed_transaction])
        processed_transaction_df = self.preprocessor.transform(processed_transaction_df)
        processed_transaction["predicted_class"] = int(self.model.predict(processed_transaction_df)[0])
        processed_transaction["fraud_probability"] = float(self.model.predict_proba(processed_transaction_df)[0, 1])

        return processed_transaction

    
    def store_result(self, transaction_id, device_id, processed_transaction):
        """
        Store prediction results:
        1. Append immutable prediction event to RESULT_STREAM
        2. Persist mutable prediction record in Redis hash (for label join & evaluation)
        """

        processed_transaction_dict = serialize_processed_transaction(transaction_id, processed_transaction)
        
        # append immutable prediction event to RESULT_STREAM
        self.client.xadd(RESULT_STREAM, processed_transaction_dict)

        # persist mutable prediction record in Redis hash
        self.client.hset(
            f"prediction:{transaction_id}",
            mapping={
                "transaction_id": processed_transaction_dict["transaction_id"],
                "device_id": device_id,
                "predicted_class": processed_transaction_dict["predicted_class"],
                "fraud_probability": processed_transaction_dict["fraud_probability"],
                "txn_count": processed_transaction_dict["txn_count"],
                "device_ip_consistency": processed_transaction_dict["device_ip_consistency"],
                "device_source_consistency": processed_transaction_dict["device_source_consistency"],
                "time_setup_to_txn_seconds": processed_transaction_dict["time_setup_to_txn_seconds"],
                "time_since_last_device_txn": processed_transaction_dict["time_since_last_device_txn"],
                "purchase_deviation_from_device_mean": processed_transaction_dict["purchase_deviation_from_device_mean"],
                "device_lifespan": processed_transaction_dict["device_lifespan"],
                "device_fraud_rate": processed_transaction_dict["device_fraud_rate"],
                "true_label": ""  # initially empty, updated later when label arrives
            }
        )


if __name__=="__main__":
    inference_consumer = InferenceConsumer()
    inference_consumer.start_consuming()