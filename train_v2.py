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
from src.models_v2 import FraudDetectionModel, FraudModelTuner
from src.preprocessing import FraudDataPreprocessor
from sklearn.metrics import classification_report

class InitialTrain:

    def __init__(self, resampling_type="smote", model_type="logistic_regression"):
        self.device_state = {}
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
            device_lifespan = (device_stats["first_seen"] - row["purchase_time"]).total_seconds()

            device_stats["txn_count"] += 1
            device_stats["ip_addresses"].add(row["ip_address"])
            device_stats["sources"].add(row["source"])
            device_stats["last_transaction"] = row["purchase_time"]
            device_stats["purchase_sum"] += row["purchase_value"]

        else:
            txn_count = 1
            device_ip_consistency = 1
            device_source_consistency = 1
            time_since_last_device_txn = 99999
            purchase_deviation_from_device_mean = 0
            device_lifespan = 0

            self.device_state[device_id] = {
                "txn_count": 1,
                "ip_addresses": {row["ip_address"]},
                "sources": {row["source"]},
                "last_transaction": row["purchase_time"],
                "purchase_sum": row["purchase_value"],
                "first_seen": row["purchase_time"]
            }
        
        processed_transaction = {
            "txn_count": txn_count,
            "device_ip_consistency": device_ip_consistency,
            "device_source_consistency": device_source_consistency,
            "time_setup_to_txn_seconds": time_setup_to_txn_seconds,
            "time_since_last_device_txn": time_since_last_device_txn, 
            "purchase_deviation_from_device_mean": purchase_deviation_from_device_mean,
            "device_lifespan": device_lifespan,
            "is_fraud": row["is_fraud"]
        }

        return processed_transaction
    
    def fit_preprocessor_model(self, df_train, df_test, params=None):
        preprocessor = FraudDataPreprocessor()
        # fit preprocessor
        X_train, y_train = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL] 
        X_train = preprocessor.fit_transform(X_train)
        X_test, y_test = df_test.drop(columns=[TARGET_COL]), df_test[TARGET_COL]
        X_test = preprocessor.transform(X_test) 

        if not params: # tune mode
            model_tuner = FraudModelTuner(
                resampling_type=self.resampling_type,
                model_type=self.model_type
            )
            model, grid_search = model_tuner.tune(X_train, y_train, self.param_grid)
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))
            return preprocessor, grid_search
            

        else:
            model = FraudDetectionModel(
                resampling_type=self.resampling_type,
                model_type=self.model_type,
                custom_params=params
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))
            return preprocessor, model


    def train_pipeline(self):
        df = pd.read_csv(RAW_DATA_PATH)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df = df.sort_values(by="purchase_time")

        for idx, row in df.iterrows():
            processed_transaction = self.compute_features(row)
            self.processed_transactions.append(processed_transaction)

        processed_transactions = pd.DataFrame(self.processed_transactions)

        train = processed_transactions[:40000]
        val = processed_transactions[40000:45000]
        test = processed_transactions[45000:50000]

        _, grid_search = self.fit_preprocessor_model(train, val)

        new_train = pd.concat([train, test])
        preprocessor, model = self.fit_preprocessor_model(new_train, test, grid_search.best_params_)

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
        with open(MODEL_PATH, "wb") as m:
            pickle.dump(model, p)

    # add device_state to redis
    # add transactions, their predictions, whether they were used in training into database


if __name__=="__main__":
    main()