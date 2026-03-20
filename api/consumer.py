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
        self._initialize_models(seen_model_uri, unseen_model_uri)

    def _initialize_redis(self):
        print(f"Loading device state, global buckets and ip state")
        
        self.device_state = DeviceState.load_from_file(DEVICE_STATE_PATH)
        self.global_velocity = GlobalVelocity.load_from_file(GLOBAL_BUCKETS_PATH)
        self.ip_state = IPState.load_from_file(IP_STATE_PATH)


    def _initialize_models(self, seen_model_uri=None, unseen_model_uri=None):
        """
        Load both models (seen and unseen devices) from MLflow or local files.

        Args:
            seen_model_uri: MLflow URI for seen devices model (e.g., "models:/fraud_detection_seen_devices@Production")
            unseen_model_uri: MLflow URI for unseen devices model (e.g., "models:/fraud_detection_unseen_devices@Production")
        """
        print("Loading fraud detection models...")

        # Load SEEN DEVICES model
        self._load_model_pair(
            model_name="seen_devices",
            model_uri=seen_model_uri or "models:/fraud_detection_seen_devices@Production"
        )

        # Load UNSEEN DEVICES model
        self._load_model_pair(
            model_name="unseen_devices",
            model_uri=unseen_model_uri or "models:/fraud_detection_unseen_devices@Production"
        )

        print("✓ Both models loaded successfully\n")

    def _load_model_pair(self, model_name, model_uri):
        """
        Load a model and its preprocessor from MLflow or local files.

        Args:
            model_name: 'seen_devices' or 'unseen_devices'
            model_uri: MLflow URI for the model
        """
        model_path = MODELS_DIR / f"{model_name}_model.pkl"
        preprocessor_path = MODELS_DIR / f"{model_name}_preprocessor.pkl"

        # Try loading MODEL from MLflow first
        print(f"  Loading {model_name} model from MLflow: {model_uri}")
        model_from_mlflow = False
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"  ✓ Loaded {model_name} model from MLflow")
            model_from_mlflow = True

        except Exception as mlflow_error:
            print(f"  ⚠ Failed to load model from MLflow: {mlflow_error}")
            print(f"  Falling back to local model: {model_path}")

            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f"  ✓ Loaded {model_name} model from local file")

            except Exception as local_error:
                raise RuntimeError(
                    f"Failed to load {model_name} model from MLflow and local path.\n"
                    f"MLflow error: {mlflow_error}\n"
                    f"Local error: {local_error}"
                )

        # Try loading PREPROCESSOR from MLflow (if model came from MLflow)
        preprocessor = None
        if model_from_mlflow:
            try:
                # Get the run_id from the model URI
                client = MlflowClient()

                # Parse model name from URI
                registry_model_name = model_uri.split("/")[1].split("@")[0]

                # Get the latest version with the specified alias
                alias = model_uri.split("@")[1] if "@" in model_uri else "Production"
                model_version = client.get_model_version_by_alias(registry_model_name, alias)
                run_id = model_version.run_id

                # Download preprocessor artifact from that run
                artifact_path = f"preprocessor/{model_name}_preprocessor.pkl"
                preprocessor_uri = f"runs:/{run_id}/{artifact_path}"

                import tempfile
                import os
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_path = mlflow.artifacts.download_artifacts(preprocessor_uri, dst_path=tmpdir)
                    with open(local_path, "rb") as f:
                        preprocessor = pickle.load(f)

                print(f"  ✓ Loaded {model_name} preprocessor from MLflow")

            except Exception as mlflow_preprocessor_error:
                print(f"  ⚠ Failed to load preprocessor from MLflow: {mlflow_preprocessor_error}")
                print(f"  Falling back to local preprocessor")

        # Load preprocessor from local file if not loaded from MLflow
        if preprocessor is None:
            try:
                with open(preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                print(f"  ✓ Loaded {model_name} preprocessor from local file")

            except Exception as e:
                raise RuntimeError(f"Failed to load {model_name} preprocessor: {e}")

        # Assign to instance variables
        if model_name == "seen_devices":
            self.seen_model = model
            self.seen_preprocessor = preprocessor
        elif model_name == "unseen_devices":
            self.unseen_model = model
            self.unseen_preprocessor = preprocessor


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
        db_url=db_url,
        seen_model_uri=seen_model_uri,
        unseen_model_uri=unseen_model_uri
    )

    # If MLflow is not available, it will automatically fall back to local files:
    # - models/seen_devices_model.pkl + models/seen_devices_preprocessor.pkl
    # - models/unseen_devices_model.pkl + models/unseen_devices_preprocessor.pkl

    inference_consumer.start_consuming()