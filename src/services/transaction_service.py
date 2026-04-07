from dataclasses import asdict
import pandas as pd
from src.config import seen_device_features, unseen_device_features
from src.repositories import TransactionRepository
from src.domain_models import Transaction
from src.model.rule_based import RuleBasedModel

class TransactionService:
    """
    Service responsible for handling transaction-related business logic.

    This includes:
    - Updating transaction labels (ground truth)
    - Selecting the appropriate model for inference
    - Running model-specific prediction logic (legacy / transitional)

    Attributes
    ----------
    txn_repo : TransactionRepository
        Repository used for retrieving and persisting transaction data.
    """

    def __init__(self, txn_repo: TransactionRepository):
        """
        Initialize TransactionService.

        Parameters
        ----------
        txn_repo : TransactionRepository
            Repository for accessing and updating transactions.
        """
        self.txn_repo = txn_repo

    def update_label(self, transaction_id, is_fraud: int):
        """
        Update the ground truth label of a transaction.

        This method retrieves a transaction by its ID and updates its
        `true_label` field to reflect whether it is fraudulent.

        Parameters
        ----------
        transaction_id : Any
            Unique identifier of the transaction.
        is_fraud : int
            Label indicating fraud (typically 0 for non-fraud, 1 for fraud).

        Returns
        -------
        Transaction
            The updated transaction entity.

        Raises
        ------
        ValueError
            If the transaction cannot be found.

        Notes
        -----
        - This represents a business action (labeling), not just a DB update.
        - May be extended in the future to trigger retraining or logging.
        """
        transaction = self.txn_repo.get_by_id(transaction_id)

        if transaction is None:
            raise ValueError("Transaction not found.")

        transaction.true_label = is_fraud
        self.txn_repo.session.commit()
        return transaction
    
    def select_model(self, transaction: Transaction):
        """
        Determine which model should be used for a given transaction.

        This method applies:
        1. A rule-based model for immediate classification (if applicable)
        2. Fallback logic to choose between 'seen_devices' and 'unseen_devices'

        Parameters
        ----------
        transaction : Transaction
            The transaction entity to evaluate.

        Returns
        -------
        tuple
            A tuple containing:
            - model_name : str
                Name of the selected model ("rule_based", "seen_devices", or "unseen_devices")
            - rule_label : int
                Output of the rule-based model (1 or 0), or -1 if not applicable

        Notes
        -----
        - Rule-based model takes precedence if it produces a valid label.
        - 'unseen_devices' is used for first-time device transactions.
        - 'seen_devices' is used otherwise.
        """
        rule_based_label = RuleBasedModel().predict(transaction)

        if rule_based_label != -1:
            return "rule_based", rule_based_label
        
        elif transaction.device_txn_idx == 1:
            return "unseen_devices", -1

        else:
            return "seen_devices", -1
        