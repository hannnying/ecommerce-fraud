from datetime import datetime, timedelta
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.state.prediction_store_sql import PredictionRepository
from training.train import Train 

class TrainingService:
    """Service for model training logic."""

    def __init__(self, db_url: str):
        self.prediction_repo = PredictionRepository(db_url)
    
    def should_retrain(self) -> bool:
        """
        Determine if retraining is needed.
        
        Returns:
            True: if there are 5000 new transactions
            False: if there is less than 5000 new transactions
        """
        return not self.prediction_repo.get_new_labeled_count() % 5000
        
    def retrain_model(self):
        # fetch offline training data, make path a parameter
        march_start = datetime(2015, 3, 1)
        offline_transactions = pd.read_csv("data/processed/processed_train.csv")
        offline_transactions = offline_transactions[offline_transactions["purchase_time"] >= march_start]
        new_transactions = self.prediction_repo.fetch_training_dataset()

        # merge transactions used in initial training (from march onwards) and new transactions
        merged_transactions = pd.concat([offline_transactions, new_transactions], axis=0)


        # start a new MLflow run for retrained model
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            train_instance = Train(processed_transaction_path="data/processed/processed_train.csv")
            train_instance.train_unknown_devices_model_pipeline(merged_transactions[merged_transactions["device_txn_idx"]==1])
            train_instance.train_seen_devices_model_pipeline(merged_transactions[merged_transactions["device_txn_idx"]>1])

            print(f"All artifacts logged to run: {run_id}")

        return run_id