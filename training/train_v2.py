# Run this script to train the initial model on the first 50,000 transactions
import argparse
import json
import pickle

import pandas as pd
from src.config import (
    MODELS_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RAW_DATA_PATH,
    TARGET_COL
)
from src.models.models_v3 import FraudDetectionModel, FraudModelTuner
from src.preprocessing import FraudDataPreprocessor
from src.state.redis_store import (
    DeviceState,
    PredictionStore
)
from sklearn.metrics import classification_report

class InitialTrain:

    def __init__(self, resampling_type="smote", model_type="logistic_regression"):
        self.device_state = {}
        self.prediction_store = PredictionStore()
        self.processed_transactions = []
        self.param_grid = None
        self.resampling_type = resampling_type
        self.model_type = model_type

        self._initialize_param_grid()


    def _initialize_param_grid(self):
        """Initialize param grid used for hyperparameter tuning."""
        if self.model_type.startswith("logistic"):
            self.param_grid = {
                "C": [0.001, 0.01],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "class_weight": ["balanced"],
                "max_iter": [500, 1000],
                "random_state": [42],
            }

        if self.model_type == "svc":
            self.param_grid = {
                "C": [1],
                "kernel": ["linear", "poly"]
            }
        

    def compute_features(self, row):
        """Compute features for a transaction and update device_state."""
        device_id = row["device_id"]
        time_setup_to_txn_seconds = (row["purchase_time"] - row["signup_time"]).total_seconds()

        if device_id in self.device_state:
            device_stats = self.device_state[device_id]
            txn_count = device_stats["txn_count"]
            device_ip_consistency = len(device_stats["ip_addresses"]) == 1 \
                                        and (row["ip_address"] in device_stats["ip_addresses"])
            device_source_consistency = len(device_stats["sources"]) == 1 \
                                        and (row["source"] in device_stats["sources"])
            time_since_last_device_txn = (row["purchase_time"] - device_stats["last_transaction"]).total_seconds()
            purchase_deviation_from_device_mean = abs(row["purchase_value"] - device_stats["purchase_sum"] / txn_count)
            device_lifespan = (row["purchase_time"] - device_stats["first_seen"]).total_seconds()
            fraud_rate = device_stats["fraud_count"] / txn_count 

        else:
            txn_count = 1
            device_ip_consistency = True
            device_source_consistency = True
            time_since_last_device_txn = 99999
            purchase_deviation_from_device_mean = 0
            device_lifespan = 0
            fraud_rate = 0
        
        processed_transaction = {
            "txn_count": txn_count,
            "device_ip_consistency": device_ip_consistency,
            "device_source_consistency": device_source_consistency,
            "time_setup_to_txn_seconds": time_setup_to_txn_seconds,
            "time_since_last_device_txn": time_since_last_device_txn, 
            "purchase_deviation_from_device_mean": purchase_deviation_from_device_mean,
            "device_lifespan": device_lifespan,
            "device_fraud_rate": fraud_rate,
        }

        return processed_transaction
    
    def update_device_state(self, row):
        """When pipeline is run on unseen data, fraud_count does not get updated"""
        device_id = row["device_id"]

        if device_id in self.device_state:
            device_stats = self.device_state[device_id]
            
            device_stats["txn_count"] += 1
            device_stats["ip_addresses"].add(row["ip_address"])
            device_stats["sources"].add(row["source"])
            device_stats["last_transaction"] = row["purchase_time"]
            device_stats["purchase_sum"] += row["purchase_value"]

            if "is_fraud" in row and row["is_fraud"]:
                device_stats["fraud_count"] += 1
            

        else:
            self.device_state[device_id] = {
                    "txn_count": 1,
                    "ip_addresses": {row["ip_address"]},
                    "sources": {row["source"]},
                    "last_transaction": row["purchase_time"],
                    "purchase_sum": row["purchase_value"],
                    "first_seen": row["purchase_time"],
                }
            
            if "is_fraud" in row:
                self.device_state[device_id]["fraud_count"] = 1 if row["is_fraud"] else 0
            else: # on unseen devices in unseend data
                self.device_state[device_id]["fraud_count"] = 0


    def fit_preprocessor_model(self, df_train, params=None, transform=True):
        """
        Perform offline feature computation, fit a preprocessing pipeline, and train a fraud detection model.

        This method simulates historical transaction processing by iterating through the training data
        in chronological order, computing transaction-level features and updating device-level state
        after each transaction. The resulting feature set is then used to fit a preprocessing pipeline
        (e.g. scaling and encoding) and train a fraud detection model.

        If no model parameters are provided, hyperparameter tuning is performed via grid search and
        the best-performing model is selected. Otherwise, a model is trained directly using the
        provided parameters.

        Parameters
        ----------
        df_train : pd.DataFrame
            Historical transaction data used for offline training. Must contain all fields required
            for feature computation and the target column.
        params : dict, optional
            Custom model parameters. If None, grid search is performed to find the best parameters.

        Returns
        -------
        preprocessor : FraudDataPreprocessor
            Fitted preprocessing pipeline used to transform feature data.
        model: FraudDetectionModel
            Trained fraud detection model if `params` is provided, otherwise trained fraud detection model with best
            hyperparameters from grid search.
        """
        processed_train = []
        X_train, y_train = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL] 

        for idx, row in X_train.iterrows():
            processed_transaction = self.compute_features(row)
            self.update_device_state(row)
            processed_train.append(processed_transaction)

        processed_train_df = pd.DataFrame(processed_train)

        preprocessor = FraudDataPreprocessor()

        # fit preprocessor
        X_train = preprocessor.fit_transform(processed_train_df)

        if not params: # tune mode
            model_tuner = FraudModelTuner(
                resampling_type=self.resampling_type,
                model_type=self.model_type
            )
            model, grid_search = model_tuner.tune(X_train, y_train, self.param_grid)
            model.fit(X_train, y_train)

        else:
            model = FraudDetectionModel(
                resampling_type=self.resampling_type,
                model_type=self.model_type,
                custom_params=params
            )
            model.fit(X_train, y_train)

        if transform:
            y_pred = model.predict(X_train)
            y_proba = model.predict_proba(X_train)[:, 1]

            return preprocessor, model, processed_train, y_train, y_pred, y_proba
        
        else:
            return preprocessor, model
        
        
    def process_transactions(self, preprocessor, model, X, y=None, update_device_state=True): 
        """
        Transform raw transaction events into model-ready features and generate fraud predictions.

        This method performs stateful, time-ordered feature computation using device history,
        applies a fitted preprocessing pipeline, and runs inference with a trained model.

        It is designed to be used for both offline (training / evaluation) and online
        (inference / streaming) workflows.

        Parameters
        ----------
        preprocessor : FraudDataPreprocessor
            A fitted preprocessing pipeline (scaling, encoding).
        model : FraudDetectionModel
            A trained fraud detection model.
        X : pd.DataFrame
            
        update_training_state : bool, default=True
            Whether to update device state after each transaction.
            - True  → if running inference on unseen data.
            - False → if running inference on data that the model was trained on.

        Returns
        -------
        processed_inference_df: pd.DataFrame
            Preprocessed feature matrix used before scaling and encoding
        y_true : pd.Series or None
            Ground truth labels if present in the input data, otherwise None.
        y_pred : np.ndarray
            Predicted fraud class labels.
        y_proba : np.ndarray
            Predicted fraud probabilities.
        
        Notes
        -----
        - Feature computation only uses information available at or before each transaction.
        - True labels are never used during feature computation.
        - Device state updates triggered by labels must be handled separately.
        """

        processed_inference = []

        for idx, row in X.iterrows():
            processed_transaction = self.compute_features(row)

            # device fraud_count freezes when runnign inference on unseen data
            if update_device_state: 
                self.update_device_state(row)

            processed_inference.append(processed_transaction)
        
        processed_inference_df = pd.DataFrame(processed_inference)
        X = preprocessor.transform(processed_inference_df)

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:,1]

        return processed_inference, y_pred, y_proba


    def train_pipeline(self, initial_rows=50000, train_percentage=0.8):
        preprocessor = FraudDataPreprocessor()
        df = pd.read_csv(RAW_DATA_PATH)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df = df.sort_values(by="purchase_time")

        initial_training_rows = int(train_percentage * initial_rows)

        train = df[:initial_training_rows]
        train.reset_index(inplace=True)
        test = df[initial_training_rows:initial_rows+1]
        test.reset_index(inplace=True)

        # initial fitting to find the best hyperparameters with train set
        preprocessor, model, X_train_features, y_train, y_train_pred, y_train_proba = self.fit_preprocessor_model(
            train, params=None, transform=True
        )

        # store train predictions in prediction hash (where true labels are known too)
        self.prediction_store.batch_update_predictions(
            X_train_features, train["device_id"], y_train_pred, y_train_proba, y_train
        )

        # split test into X and y
        X_test, y_test = test.drop(columns=[TARGET_COL]), test[TARGET_COL]

        # validate performance on validation set
        X_test_features, y_test_pred, y_test_proba = self.process_transactions(preprocessor, model, X_test)

        print(classification_report(y_true=y_test, y_pred=y_test_pred))

        # store predictions in prediction hash 
        self.prediction_store.batch_update_predictions(
            X_test_features, test["device_id"], y_test_pred, y_test_proba, y_test
        )

        # load initial redis hash (fraud count not updated)
        redis_device_state = DeviceState()
        redis_device_state.load_initial(self.device_state)

        # update fraud_count
        for idx, is_fraud in enumerate(y_test):
            redis_device_state.update_fraud_count(test["device_id"][idx], is_fraud)
    
        return preprocessor, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resampling',
        type=str,
        default='smote',
        choices=['random_undersampling', 'enn', 'smote', 'smoteenn'],
        help='Resampling technique to handle class imbalance (default: random_undersampling)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='logistic_regression',
        help='Model type to train'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save trained model to disk'
    )

    args = parser.parse_args()

    # Create necessary directories
    MODELS_DIR.mkdir(exist_ok=True)

    initial_train = InitialTrain(
        resampling_type=args.resampling,
        model_type=args.model
    )
    preprocessor, model = initial_train.train_pipeline()

    if args.save:
        with open(PREPROCESSOR_PATH, "wb") as p:
            pickle.dump(preprocessor, p)
            print(f"preprocessor saved at: {PREPROCESSOR_PATH}")
        with open(MODEL_PATH, "wb") as m:
            pickle.dump(model, m)
            print(f"model saved at: {MODEL_PATH}")


if __name__=="__main__":
    main()