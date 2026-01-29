import os
import time
import mlflow
import pandas as pd
import pickle
from redis import Redis
from src.config import (
    MODELS_DIR,
    DEVICE_STATE_PATH,
    GLOBAL_BUCKETS_PATH,
    IP_STATE_PATH,
    LABELS_STREAM,
    PREPROCESSOR_PATH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    RESULT_STREAM,
    TRANSACTION_STREAM
)
from src.feature_engineering.engineer import TransactionFeatureEngineer
from src.models.rule_based import RuleBasedModel
from src.state.device_state import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState
from src.state.prediction_store import PredictionStore


class InferenceConsumer:
    def __init__(self, preprocessor_path, model_uri, model_path):
        self.preprocessor = None
        self.model = None
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.device_state = None
        self.feature_engineer = TransactionFeatureEngineer()
        self.global_velocity = None
        self.ip_state = None
        self.prediction_store = PredictionStore()
        self.rule_based_model = RuleBasedModel()

        self._initialize_redis()
        self._initialize_preprocessor(preprocessor_path)
        self._initialize_model(model_uri, model_path)

    def _initialize_redis(self):
        print(f"Loading device state, global buckets and ip state")
        
        self.device_state = DeviceState.load_from_file(DEVICE_STATE_PATH)
        self.global_velocity = GlobalVelocity.load_from_file(GLOBAL_BUCKETS_PATH)
        self.ip_state = IPState.load_from_file(IP_STATE_PATH)


    def _initialize_preprocessor(self, preprocessor_path):
        try:
            with open(preprocessor_path, "rb") as p:
                self.preprocessor = pickle.load(p)

            print(f"Loaded preprocessor at {preprocessor_path} into InferenceConsumer")
        except Exception as e:
            print(f"Error loading preprocessor: {e}")


    def _initialize_model(self, model_uri, model_path=MODELS_DIR / "logistic_regression.pkl"):
        print(f"Loading model from MLflow Registry: {model_uri}")
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
        except Exception as mlflow_error:
            print(f"Failed to load from MLflow: {mlflow_error}")
            print(f"Falling back to local model at {model_path}")

            try:
                with open(model_path, "rb") as m:
                    self.model = pickle.load(m)
                print(f"Loaded model at {model_path} into InferenceConsumer")
                    
            except Exception as local_error:
                raise RuntimeError(
                    f"Failed to load model from MLflow and local path.\n"
                    f"MLflow error: {mlflow_error}\n"
                    f"Local error: {local_error}"
            )


    def handle_transaction(self, transaction_dict):
        """
        Preprocess an incoming transaction, compute features, run fraud inference, 
        update device state based on transaction behavior and append to RESULTS_STREAM.
        """
        device_id = transaction_dict["device_id"]

        processed_transaction, transaction_id, device_data = self.feature_engineer.compute_features(
            transaction_dict,
            training=False,
            transaction_id=None
        )

        processed_transaction_pred = self.predict_transaction(processed_transaction)

        device_id, txn_count, purchase_value, sex, age, purchase_time, signup_time, ip_address, txn_count, first_seen_signup, first_seen, identities, purchase_values = device_data

        self.device_state.update_device_state(
            device_id,
            purchase_value,
            sex,
            age,
            purchase_time,
            signup_time,
            ip_address,
            txn_count,
            first_seen_signup,
            first_seen,
            identities,
            purchase_values
        )

        self.device_state.update_device_timestamp(device_id, transaction_id, purchase_time)
        self.global_velocity.update_global_bucket(purchase_time)
        self.ip_state.update_ip_txn_idx(ip_address)

        print(f"processed transaction: {transaction_id}")

        # add processed transaction df to results stream
        self.store_result(transaction_id, device_id, purchase_time, processed_transaction_pred)

    
    def handle_label(self, label_dict):
        """
        Join label into hash storing past transaction predictions, and update fraud_count of device state. 
        """
        
        transaction_id = label_dict["transaction_id"]
        device_id = label_dict["device_id"]
        is_fraud = label_dict["is_fraud"]

        self.prediction_store.update_label(transaction_id, is_fraud)

        # # look for device record with device_id in hash
        # self.device_state.update_fraud_count(device_id, is_fraud)
        
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

    def predict_transaction(self, processed_transaction):
        """
        Predict whether transaction is fraudulent or not.
        """
        
        rule_based_label = self.rule_based_model.predict(processed_transaction)
        
        if rule_based_label == -1: # 
            processed_transaction_df = pd.DataFrame([processed_transaction])

            # scale
            scaled_processed_transaction_df = self.preprocessor.transform(processed_transaction_df)

            # Add predictions to processed transaction dict
            processed_transaction["predicted_class"] = int(self.model.predict(scaled_processed_transaction_df)[0]) # customise this if you want a custom threshold
            processed_transaction["fraud_probability"] = float(self.model.predict_proba(scaled_processed_transaction_df)[0, 1])
        
        else:
            processed_transaction["predicted_class"] = rule_based_label
            processed_transaction["fraud_probability"] = 0.99 if rule_based_label == 1 else 0.01
        
        return processed_transaction

    
    def store_result(self, transaction_id, device_id, purchase_time, processed_transaction):
        """
        Store prediction results:
        1. Append immutable prediction event to RESULT_STREAM
        2. Persist mutable prediction record in Redis hash (for label join & evaluation)
        """

        processed_transaction_dict = self.prediction_store.serialize_processed_transaction(transaction_id, purchase_time, processed_transaction)
        
        # append immutable prediction event to RESULT_STREAM
        self.client.xadd(RESULT_STREAM, processed_transaction_dict)

        # persist mutable prediction record in Redis hash
        self.prediction_store.update_predictions(device_id, processed_transaction_dict)

        

if __name__=="__main__":

    model_path = MODELS_DIR / "logistic_regression_10K.pkl"
    # This URI always points to the latest model in Production stage.
    # When training with logging=True (training/train.py), new models are automatically
    # registered and promoted to Production (see src/models/models.py:149-158).
    # MLflow's Production stage reference is updated dynamically, ensuring this loads the latest promoted model.
    model_uri = "models:/fraud_detection_model/Production"

    inference_consumer = InferenceConsumer(
        preprocessor_path=PREPROCESSOR_PATH,
        model_uri="models:/fraud_detection_model/Production",
        model_path=model_path
    )

    # inference_consumer = InferenceConsumer(
    #     preprocessor_path=PREPROCESSOR_PATH,
    #     model_uri=None,
    #     model_path=model_path
    # )

    inference_consumer.start_consuming()