from dataclasses import asdict
import uuid
import pandas as pd
from src.config import seen_device_features, unseen_device_features
from src.domain_models import Prediction, Transaction
from src.infrastructure.mlflow.model_manager import ModelManager
from src.repositories import PredictionRepository
from src.services.transaction_service import TransactionService

class PredictionService:
    """
    Service responsible for handling fraud prediction workflows.

    This includes:
    - Selecting the appropriate model based on transaction characteristics
    - Running inference using production and optional shadow models
    - Persisting prediction results to the database

    Attributes
    ----------
    pred_repo : PredictionRepository
        Repository used to store prediction records.
    model_manager : ModelManager
        Manages retrieval of production and shadow models along with their metadata.
    """

    def __init__(self, pred_repo: PredictionRepository):
        """
        Initialize PredictionService.

        Parameters
        ----------
        pred_repo : PredictionRepository
            Repository for persisting prediction results.
        """
        self.pred_repo = pred_repo
        self.model_manager = ModelManager()

    def predict_transaction(self, transaction: Transaction):
        """
        Generate fraud probability predictions for a given transaction.

        This method:
        1. Determines which model to use (rule-based or ML-based)
        2. Applies preprocessing using the appropriate preprocessor
        3. Runs inference using the production model
        4. Optionally runs inference using a shadow model for comparison

        Parameters
        ----------
        transaction : Transaction
            The transaction entity containing raw and engineered features.

        Returns
        -------
        dict
            A dictionary containing:
            - model_name : str
                Name of the model used ("rule_based", "seen_devices", etc.)
            - fraud_proba_prod : float
                Fraud probability predicted by the production model
            - fraud_proba_shadow : float or None
                Fraud probability predicted by the shadow model (if available),
                otherwise None
        """
        transaction_service  = TransactionService()
        model_name, rule_label = transaction_service.select_model(transaction)
        transaction_df = pd.DataFrame([asdict(transaction)])
        fraud_proba_shadow = None

        if model_name == "rule_based":
            fraud_proba_prod = 0.99 if rule_label == 1 else 0
        
        else:
            prod_preprocessor, prod_model, shadow_preprocessor, shadow_model = self.model_manager.get_model(model_name)
            
            transaction_df_prod = prod_preprocessor.transform(transaction_df)
            fraud_proba_prod = float(prod_model.predict_proba(transaction_df_prod)[0, 1])

            if shadow_preprocessor and shadow_model:
                transaction_df_shadow = shadow_preprocessor.transform(transaction_df)
                fraud_proba_shadow = float(shadow_model.predict_proba(transaction_df_shadow)[0, 1])

        return {
            "model_name": model_name,
            "fraud_proba_prod": fraud_proba_prod,
            "fraud_proba_shadow": fraud_proba_shadow
        }
    
    def save_prediction(self, transaction_id: uuid.UUID, prediction_dict: dict):
        """
        Persist prediction results for a transaction to the database.

        This method creates and stores:
        - One record for the production model prediction
        - One record for the shadow model prediction (if available)

        Parameters
        ----------
        transaction_id : uuid.UUID
            Unique identifier of the transaction associated with the predictions.
        predictions_dict : dict
            Dictionary containing prediction outputs from `predict_transaction`,
            with keys:
            - model_name : str
            - fraud_proba_prod : float
            - fraud_proba_shadow : float or None
        """
        
        model_name = prediction_dict["model_name"]
        fraud_proba_prod = prediction_dict["fraud_proba_prod"]
        fraud_proba_shadow = prediction_dict["fraud_proba_shadow"]

        prod_prediction = Prediction(
            transaction_id=transaction_id,
            model_used=model_name,
            model_version=self.model_manager.get_model_version(
                model_name=model_name,
                prod=True
            ),
            fraud_proba=fraud_proba_prod
        )

        self.pred_repo.add(prod_prediction)

        if fraud_proba_shadow is not None:
            shadow_prediction = Prediction(
                transaction_id=transaction_id,
                model_used=model_name,
                model_version=self.model_manager.get_model_version(
                    model_name=model_name,
                    prod=False
                ),
                fraud_proba=fraud_proba_shadow
            )
            self.pred_repo.add(shadow_prediction)

        self.pred_repo.session.commit()
        
