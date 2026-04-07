import pickle
import subprocess

import mlflow
from mlflow.tracking import MlflowClient
from src.state.prediction_store_sql import PredictionRepository

class MlflowModelRegistry:
    """
    Handles interaction with the MLflow Model Registry for retrieving,
    managing, and deleting model versions and associated artifacts.

    This class is responsible for:
    - Fetching trained models and their preprocessors from MLflow
    - Retrieving the latest model version for a given model name
    - Deleting model versions from the registry

    Attributes
    ----------
    client : MlflowClient
        MLflow client used to interact with the model registry and runs.
    """

    def __init__(self):
        """
        Initialize MlflowModelRegistry with an MLflow client.
        """
        self.client = MlflowClient()

    def fetch_model(self, name: str, version: int):
        """
        Fetch a model and its associated preprocessor from MLflow.

        This method:
        - Loads the sklearn model from the MLflow Model Registry
        - Retrieves the corresponding run ID
        - Downloads the preprocessor artifact from the run
        - Deserializes the preprocessor using pickle

        Parameters
        ----------
        name : str
            Name of the registered model.
        version : int
            Version number of the model to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - model : object
                The loaded ML model.
            - preprocessor : object
                The fitted preprocessing pipeline used during training.

        Raises
        ------
        Exception
            If there is an issue retrieving the model or artifacts from MLflow.

        Notes
        -----
        - Assumes the preprocessor is stored under:
          'preprocessor/{name}_preprocessor.pkl' within the MLflow run artifacts.
        - Uses MLflow's artifact store to download preprocessing objects.
        """
        model_uri = f"models:/{name}/{str(version)}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            run_id = self.client.get_model_version(
                name=name,
                model_version=version
            ).run_id

            preprocessor_uri = f"runs:/{run_id}/preprocessor/{name}_preprocessor.pkl"
            preprocessor_path = mlflow.artifacts.download_artifacts(artifact_uri=preprocessor_uri)
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
        
        except Exception as mlflow_error:
            print(f"MLflow error: {mlflow_error}")

        return model, preprocessor
    
    def get_latest_model(self, name: str) -> int:
        """
        Retrieve the latest version number of a registered model.

        Parameters
        ----------
        name : str
            Name of the registered model.

        Returns
        -------
        int
            The version number of the latest model.

        Notes
        -----
        - Assumes that the first result returned by MLflow corresponds
          to the latest version.
        - This may depend on MLflow's internal ordering of model versions.
        """
        model_versions = self.client.search_model_versions(filter="name='{name}'")
        latest_version = model_versions[0]
        version = latest_version.version
        return version 
    
    def delete_model(self, name: str, version: int):
        """
        Delete a specific version of a registered model from MLflow.

        This performs a soft delete, meaning the model version is marked
        as deleted but may still be recoverable depending on MLflow configuration.

        Parameters
        ----------
        name : str
            Name of the registered model.
        version : int
            Version number of the model to delete.

        Notes
        -----
        - Used for cleaning up outdated shadow models.
        - Errors are caught and printed instead of raised.
        """
        try:
            self.client.delete_model_version(name=name, version=version) # soft delete
            print(f"Deleted {name} version {str(version)}")

        except Exception as e:
            print(f"Unable to delete {name} version {str(version)}: {e}")

