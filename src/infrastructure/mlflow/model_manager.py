from src.infrastructure.mlflow.mlflow_model_repository import MlflowModelRegistry

class ModelManager:
    """
    Manages in-memory ML models for inference, including both production
    and shadow (test) versions for different model types.

    Attributes
    ----------
    mlflow_repo : MlflowModelRegistry
        Repository used to fetch, version, and delete models from MLflow.
    models : dict
        Nested dictionary storing model artifacts and metadata in the format:
        {
            model_name: {
                "production": {
                    "version": int,
                    "model": object,
                    "preprocessor": object
                },
                "shadow": {
                    "version": int,
                    "model": object,
                    "preprocessor": object
                }
            }
        }
    """

    def __init__(self, mlflow_repo: MlflowModelRegistry):
        """
        Initialize ModelManager.

        Parameters
        ----------
        mlflow_repo : MlflowModelRegistry
            Repository for interacting with MLflow model registry.
        """
        self.mlflow_repo = mlflow_repo

        self.models = {
            "seen_devices": {
                "production": {},
                "shadow": {}
            },
            "unseen_devices": {
                "production": {},
                "shadow": {}
            }
        }

    def _initialize_model(self, model_name: str):
        """
        Initialize the production model for a given model type.

        This method:
        - Sets the initial production version (default = 1)
        - Fetches the corresponding model and preprocessor from MLflow
        - Stores them in memory

        Parameters
        ----------
        model_name : str
            Name of the model type (e.g., "seen_devices", "unseen_devices").

        Notes
        -----
        Assumes version 1 exists in MLflow at initialization time.
        """
        self.models[model_name]["production"]["version"] = 1
        model, preprocessor = self.mlflow_repo.fetch_model(model_name, self.models.model_name.production.version)
        self.models[model_name]["production"]["model"] = model
        self.models[model_name]["production"]["preprocessor"] = preprocessor

    def _initialize_models(self):
        """
        Initialize all production models.

        This method initializes both:
        - seen_devices model
        - unseen_devices model
        """
        self._initialize_model("seen_devices")
        self._initialize_model("unseen_devices")
        print(f"Iniitalized models: {self.models}")

    def load_test_model(self, model_name: str):
        """
        Load or update the shadow (test) model for a given model type.

        This method:
        - Checks the latest available model version from MLflow
        - Compares it with the currently loaded shadow model
        - Loads the new version if:
            * no shadow model exists, OR
            * a newer version is available
        - Deletes the old shadow model from MLflow if replaced

        Parameters
        ----------
        model_name : str
            Name of the model type (e.g., "seen_devices", "unseen_devices").

        Notes
        -----
        - Shadow models are used for evaluation and comparison against
          production models.
        - If the current shadow model is already up-to-date, no action is taken.
        """
        latest_version = self.mlflow_repo.get_latest_model(model_name)
        has_shadow_model = self.models[model_name]["shadow"] != {}
        cur_version = self.models[model_name]["shadow"]["version"] if has_shadow_model else 0

        if (not has_shadow_model) or (cur_version < latest_version):
            self.models["model_name"]["shadow"]["version"] = latest_version
            model, preprocessor = self.mlflow_repo.fetch_model(model_name, latest_version)
            self.models[model_name]["shadow"]["model"] = model
            self.models[model_name]["shadow"]["preprocessor"] = preprocessor

            if has_shadow_model:
                self.mlflow_repo.delete(model_name, cur_version)
    
        else:
            print(f"Current {model_name} shadow model is the most updated model: {latest_version}")
    
    def get_model(self, model_name: str):
        """
        Retrieve production and shadow models along with their preprocessors.

        Parameters
        ----------
        model_name : str
            Name of the model type.

        Returns
        -------
        tuple
            A tuple containing:
            - prod_preprocessor : object or None
            - prod_model : object or None
            - shadow_preprocessor : object or None
            - shadow_model : object or None

        Notes
        -----
        - Shadow model components may be None if no shadow model is loaded.
        """
        prod_preprocessor = self.models[model_name]["prod"].get("preprocessor", None)
        prod_model = self.models[model_name]["prod"].get("model", None)
        shadow_preprocessor = self.models[model_name]["shadow"].get("preprocessor", None)
        shadow_model = self.models[model_name]["shadow"].get("model", None)

        return prod_preprocessor, prod_model, shadow_preprocessor, shadow_model
    
    def get_model_version(self, model_name: str, prod: bool):
        """
        Retrieve the version of the specified model.

        Parameters
        ----------
        model_name : str
            Name of the model type.
        prod : bool
            If True, retrieves the production model version.
            If False, retrieves the shadow model version.

        Returns
        -------
        int
            Version number of the requested model.
        """
        if prod:
            version = self.models[model_name]["production"].get("version", None)
        else:
            version = self.models[model_name]["shadow"].get("version", None)

        if version is None:
            raise ValueError(f"Unable to retreive version for {model_name} model.")
        
        return version