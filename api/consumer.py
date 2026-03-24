import os
import time
import mlflow
from mlflow.tracking import MlflowClient
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
    TRANSACTION_STREAM,
    db_url,
    seen_device_features,
    unseen_device_features
)
from src.feature_engineering.engineer_flexible import TransactionFeatureEngineer
from src.models.rule_based import RuleBasedModel
from src.state.device_state_flexible import DeviceState
from src.state.global_bucket import GlobalVelocity
from src.state.ip_state import IPState
# from src.state.prediction_store import PredictionStore
from src.state.prediction_store_sql import PredictionRepository

# A/B testing: compare current model with latest retrained model
#   - transactions have to be processed by both models
#   - log both versions to prediction_store_sql, add version to prediction row to track which model was used 
#   - model selection can be imeplemented later 


class InferenceConsumer:
    def __init__(self, db_url: str, seen_model_uri=None, unseen_model_uri=None):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.device_state = None
        self.feature_engineer = TransactionFeatureEngineer()
        self.global_velocity = None
        self.ip_state = None
        # self.prediction_store = PredictionStore()
        self.prediction_store = PredictionRepository(db_url=db_url)
        self.rule_based_model = RuleBasedModel()

        # Two models: one for seen devices, one for unseen
        self.seen_model = None
        self.seen_preprocessor = None
        self.unseen_model = None
        self.unseen_preprocessor = None

        self._initialize_redis()
        self._initialize_models()

    def _initialize_redis(self):
        print(f"Loading device state, global buckets and ip state")
        
        self.device_state = DeviceState.load_from_file(DEVICE_STATE_PATH)
        self.global_velocity = GlobalVelocity.load_from_file(GLOBAL_BUCKETS_PATH)
        self.ip_state = IPState.load_from_file(IP_STATE_PATH)


    def _initialize_models(self):
        """
        Load both models (seen and unseen devices) from MLflow or local files.
        """
        print("Loading fraud detection models...")

        self._load_model_pair(model_name="seen_devices")
        self._load_model_pair(model_name="unseen_devices")

        print("✓ Both models loaded successfully\n")

    def _load_model_pair(self, model_name):
        model_version = 1
        model_uri = f"models:/{model_name}/{model_version}"
        client = MlflowClient()

        try:
            model = mlflow.sklearn.load_model(model_uri)
            run_id = client.get_model_version(name=model_name, model_version=model_version).run_id
            
            preprocessor_uri = f"runs:/{run_id}/preprocessor/{model_name}_preprocessor.pkl"
            preprocessor_path = mlflow.artifacts.download_artifacts(artifact_uri=preprocessor_uri)
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
        
        except Exception as mlflow_error:
            print(f"MLflow error: {mlflow_error}")

        if model_name == "seen_devices":
            self.seen_preprocessor = preprocessor
            self.seen_model = model
        else:
            self.unseen_preprocessor = preprocessor
            self.unseen_model = model

        print(f"Successfully loaded {model.name} model and preprocessor from Mlflow.")


    def handle_transaction(self, transaction_dict):
        """
        Preprocess an incoming transaction, compute features, run fraud inference, 
        update device state based on transaction behavior and append to RESULTS_STREAM.
        """
        device_id = transaction_dict["device_id"]

        processed_transaction, transaction_id,(device_id, state_update) = self.feature_engineer.compute_features(
            transaction_dict,
            training=False,
            transaction_id=None
        )

        processed_transaction_pred = self.predict_transaction(processed_transaction)

        # add transaction_id, device_id to processed_transaction_pred
        processed_transaction_pred["transaction_id"] = transaction_dict["transaction_id"]
        processed_transaction_pred["device_id"] = device_id
        processed_transaction_pred["purchase_value"] = transaction_dict["purchase_value"]
        
        self.device_state.update_device_state(
            device_id=device_id,
            state_updates=state_update
        )


        self.device_state.update_device_timestamp(device_id, transaction_id, state_update["last_seen"])
        self.global_velocity.update_global_bucket(state_update["last_seen"])

        print(f"processed transaction: {transaction_id}")

        # add processed transaction df to results stream
        self.store_result(processed_transaction_pred)

    
    def handle_label(self, label_dict):
        """
        Update hash storing past transaction predictions with label, and update prev_is_fraud of device_state.
        Join label into hash storing past transaction predictions, and update fraud_count of device state. 
        """
        
        print(f"received label of: {label_dict}")
        transaction_id = label_dict["transaction_id"]
        device_id = label_dict["device_id"]
        is_fraud = label_dict["is_fraud"]

        self.prediction_store.update_label(transaction_id, is_fraud)
        self.device_state.update_prev_is_fraud(device_id, is_fraud)


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
        Automatically selects the correct model based on device_txn_idx.
        """

        rule_based_label = self.rule_based_model.predict(processed_transaction)

        if rule_based_label == -1:  # No rule triggered, use ML model
            # Determine which model to use based on device transaction index
            device_txn_idx = processed_transaction.get("device_txn_idx", 1)
            is_new_device = device_txn_idx == 1

            if is_new_device:
                # Use UNSEEN DEVICES model
                model = self.unseen_model
                preprocessor = self.unseen_preprocessor
                features = unseen_device_features
                model_used = "unseen_devices"
                
            else:
                # Use SEEN DEVICES model
                model = self.seen_model
                preprocessor = self.seen_preprocessor
                features = seen_device_features
                model_used = "seen_devices"

            processed_transaction_df = pd.DataFrame([processed_transaction])[features]

            # quick fix for encoding before resamplign during model training
            for col in preprocessor.feature_names:
                if col not in processed_transaction_df.columns:
                    processed_transaction_df[col] = 0

            # Scale
            scaled_processed_transaction_df = preprocessor.transform(processed_transaction_df)

            # Predict
            processed_transaction["fraud_proba"] = float(model.predict_proba(scaled_processed_transaction_df)[0, 1])
            processed_transaction["model_used"] = model_used

        else:
            # Rule-based prediction
            processed_transaction["fraud_proba"] = 0.99 if rule_based_label == 1 else 0.01
            processed_transaction["model_used"] = "rule_based"
        
        processed_transaction["true_label"] = -1 # label hasn't arrived

        return processed_transaction

    
    def store_result(self, processed_transaction):
        """
        Store prediction results:
        1. Append immutable prediction event to RESULT_STREAM
        2. Persist mutable prediction record in Redis hash (for label join & evaluation)
        """

        serialized_processed_transaction = self.prediction_store.update_predictions(processed_transaction)
        # append immutable prediction event to RESULT_STREAM
        self.client.xadd(RESULT_STREAM, serialized_processed_transaction)


if __name__=="__main__":
    # MLflow URIs for both models
    # These URIs point to the latest Production versions
    # Models are registered during training (see training/train.py)
    seen_model_uri = "models:/fraud_detection_seen_devices@Production"
    unseen_model_uri = "models:/fraud_detection_unseen_devices@Production"

    inference_consumer = InferenceConsumer(
        db_url=db_url
    )

    inference_consumer.start_consuming()