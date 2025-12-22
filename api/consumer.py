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

        
        
    def start_consuming(self):
        """
        Consume transactions from Redis stream and make predictions.
        """
        last_id = "0-0"

        while True:

            messages = self.client.xread(
                streams={TRANSACTION_STREAM: last_id}, 
                block=0
            )

            if not messages:
                continue

            _, entries = messages[0]

            for message_id, transaction_dict in entries:
                device_id = transaction_dict["device_id"]
                # if device is unseen, get_device_state returns [None, .. , None]
                txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources = self.device_state.get_device_state(device_id)
                transaction_id, _, signup_time, purchase_time, purchase_value, device_id, source, _, _, _, ip_address = deserialize_transaction(transaction_dict)

                processed_transaction = self.process_transaction(
                    txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources , signup_time, purchase_time, purchase_value, source, ip_address
                )
                print
                self.device_state.update_device_state(
                    device_id,
                    txn_count,
                    purchase_sum,
                    last_transaction,
                    first_seen,
                    ip_addresses,
                    sources,
                    signup_time,
                    purchase_time,
                    purchase_value,
                    source,
                    ip_address,
                )
                print(f"processed transaction: {transaction_id}, {processed_transaction}")

                # add processed transaction df to results stream
                self.store_result(transaction_id, processed_transaction)
                
            
    def process_transaction(self, txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources , signup_time, purchase_time, purchase_value, source, ip_address):
        """
        Process single transaction and return a dictionary of the processed transaction with predictions.
        """
        txn_count, device_ip_consistency, device_source_consistency, time_setup_to_txn_seconds, time_since_last_device_id_txn, purchase_deviation_from_device_mean, device_lifespan = compute_features(txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources , signup_time, purchase_time, purchase_value, source, ip_address)
        processed_transaction = {
            "txn_count": txn_count,
            "device_ip_consistency": device_ip_consistency,
            "device_source_consistency": device_source_consistency,
            "time_setup_to_txn_seconds": time_setup_to_txn_seconds,
            "time_since_last_device_txn": time_since_last_device_id_txn,
            "purchase_deviation_from_device_mean": purchase_deviation_from_device_mean,
            "device_lifespan": device_lifespan
        }
        processed_transaction_df = pd.DataFrame([processed_transaction])
        processed_transaction_df = self.preprocessor.transform(processed_transaction_df)
        processed_transaction["predicted_class"] = int(self.model.predict(processed_transaction_df)[0])
        processed_transaction["fraud_probability"] = float(self.model.predict_proba(processed_transaction_df)[0, 1])

        return processed_transaction

    
    def store_result(self, transaction_id, processed_transaction):
        self.client.xadd(
            RESULT_STREAM,
            serialize_processed_transaction(transaction_id, processed_transaction)
        )


if __name__=="__main__":
    inference_consumer = InferenceConsumer()
    inference_consumer.start_consuming()