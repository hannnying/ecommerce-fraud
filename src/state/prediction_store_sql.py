# Initially, predictions were stored in a Redis hash, now stroed in postgres
# prereq: create db
import pandas as pd
from src.state.device_schema import PREDICTION_ATTR_DICT
from sqlalchemy import create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

PredictionModel = type("Prediction", (Base,), PREDICTION_ATTR_DICT)

class PredictionRepository:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        Base.metadata.create_all(self.engine)
    
    def update_predictions(self, processed_transaction: dict):
        """
        Insert a new prediction row, with unknown actual label.
        """
        with self.Session() as session:
            processed_transaction_instance = PredictionModel(**processed_transaction)
            session.add(processed_transaction_instance)
            session.commit()

    def update_label(self, transaction_id: str, is_fraud: int) :
        """
        Update the true_label field for a given transaction when true label becomes known.
        """
        with self.Session() as session:
            processed_transaction_instance = session.query(PredictionModel).filter_by(
                transaction_id=transaction_id
            ).one_or_none

            if processed_transaction_instance:
                processed_transaction_instance.label = is_fraud
            else:
                raise ValueError(f"Transaction with id: {transaction_id} has not been processed.")

    def get_new_labeled_count(self):
        with self.Session() as session:
            count = session.query(func.count(PredictionModel.transaction_id)).scalar()
            return count
        

    def fetch_training_dataset(self):
        """Return all labelled transactions for retraining."""

        with self.Session() as session:
            transactions = session.query(PredictionModel).all()
        
        # this returns a Query object
        return pd.DataFrame(transactions)

