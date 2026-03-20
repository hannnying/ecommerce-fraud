# run this python script, along with consumer.py
# this python script runs the retraining pipeline:
#   - checks if there are 5000 new labelled transactions
#   - calls TrainingService.retrain_model: train new model > log to mlflow under another run
#   - once model has been retrained, compare performance among all available models
import os
import time
from api.training_service import TrainingService
from src.config import db_url

class RetrainingScheduler:
    """Orchestrator for the retraining pipeline."""

    def __init__(self, db_url: str, check_interval: int):
        self.training_service = TrainingService(db_url=db_url)
        self.check_interval = check_interval

    def run_forever(self):
        while True:
            try:
                self.check_and_retrain()
            except Exception as e:
                print(f"❌ Error in retraining pipeline: {e}")
                # Continue running despite errors
            
            time.sleep(self.check_interval)

    def check_and_retrain(self):
        should_retrain = self.training_service.should_retrain()

        if not should_retrain:
            print(f"retraining not triggered")
            return
        
        try:
            retrain_run_id = self.training_service.retrain_model()
            print(f"retraining complete, artifacts logged to run: {retrain_run_id}")

            # compare new model performance with old
            # deploy if performance is better

        except Exception as e:
            print(f"retraining failed: {e}")
            raise

if __name__=="__main__":
    scheduler = RetrainingScheduler(
        db_url=db_url,
        check_interval=60
    )
    scheduler.run_forever()