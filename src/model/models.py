import pandas as pd
import pickle 
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
import mlflow
from mlflow.tracking import MlflowClient
from src.config import THRESHOLD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from src.preprocessing import FraudDataPreprocessor
import tempfile
import os
import joblib



class FraudDetectionModel:

    def __init__(self, resampling_type=None, scaler_path=None, model_type='logistic_regression', custom_params=None, model_name=None):
        """
        Initialize fraud detection model with resampling technique.

        Args:
            resampling_type (str): Type of model used for resampling: 'randon_undersamping', 'enn', 'smote', 'smoteenn'.
            scaler_path (str): Path of fitted scaler. If None, initialise with FraudDataPreprocessor
            model_type (str): Type of model: 'logistic_regression', 'random_forest', 'xgboost'
            custom_params (dict): Custom hyperparameters to override defaults
            model_name (str): Name for the model (e.g., 'seen_devices', 'unseen_devices')

        """

        self.resampling_type = resampling_type
        self.resampler = None
        self.scaler_path = scaler_path
        self.scaler = None
        self.model_type = model_type
        self.model = None
        self.params = None
        self.feature_importance_ = None
        self.model_name = model_name or model_type  # Default to model_type if no name given

        # Initialize resampling used
        if self.resampling_type:
            self._initialize_resampling()

        # Initialize scaler
        self._initialize_scaler()

        # Get hyperparameters
        if custom_params:
            self.params = custom_params

        # Initialize model
        self._initialize_model()

    def _initialize_resampling(self):
        """Initialize the resampler based on resampling_type."""
        if self.resampling_type == "random_undersampling":
            self.resampler = RandomUnderSampler(random_state=42, replacement=False)
        elif self.resampling_type == "enn":
            self.resampler = EditedNearestNeighbours(n_neighbors=3)
        elif self.resampling_type == "smote":
            self.resampler = SMOTE(random_state=42)
        elif self.resampling_type == "smoteenn":
            self.resampler = SMOTEENN(
                smote=SMOTE(k_neighbors=5, sampling_strategy="auto"),
                enn=EditedNearestNeighbours(n_neighbors=3, kind_sel='all')
            )
        else:
            raise ValueError(f"Unknown resampling_type: {self.resampling_type}")
    
    def _initialize_scaler(self):
        if self.scaler_path:
            try:
                with open(self.scaler_path, "rb") as p:
                    self.scaler = pickle.load(p)

                print(f"Loaded preprocessor at {self.scaler_path} into FraudDetectionModel")
            except Exception as e:
                print(f"Error loading preprocessor: {e}")
        else: 
            self.scaler = FraudDataPreprocessor()

    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type.startswith('logistic'):
            if self.params:
                self.model = LogisticRegression(**self.params)
            else:
                self.model = LogisticRegression()     
        elif self.model_type == 'svc':
            if self.params:
                 self.model = SVC(**self.params)
            else:
                self.model = SVC()
        elif self.model_type == "dt":
            if self.params:
                self.model = DecisionTreeClassifier(**self.params)
            else:
                self.model = DecisionTreeClassifier()
        elif self.model_type == 'random_forest':
            if self.params:
                self.model = RandomForestClassifier(**self.params)
            else:
                self.model = RandomForestClassifier()
        elif self.model_type == 'xgboost':
            if self.params:
                self.model = XGBClassifier(**self.params)
            else:
                self.model = XGBClassifier()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
    def _log_to_mlflow(self, cv_pr_auc):

        # log training metrics and params
        mlflow.sklearn.autolog(log_models=False)

        # log cv score
        mlflow.log_metric(key="cv_pr_auc", value=cv_pr_auc)
        print(f"Logged average CV PR-AUC: {cv_pr_auc} to MLFlow.")

        # log preprocessor artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor_path = os.path.join(tmpdir, f"{self.model_name}_preprocessor.pkl")
            
            with open(preprocessor_path, "wb") as f:
                pickle.dump(self.scaler, f)
            
            mlflow.log_artifact(local_path=preprocessor_path, artifact_path="preprocessor")
            print(f"Preprocessor logged to MLflow for '{self.model_name} model'")

        # log model artifact
        mlflow.sklearn.log_model(
            sk_model=self.model,
            name="model",
            registered_model_name=f"{self.model_name}_model"
        )


    def fit(self, X_train, y_train, logging=True):
        """
        Resample to handle class imbalance and train the model.

        Args
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training labels

        Returns
        -------
        self
        """
        if self.resampler:
            print(f"Resampling and Fitting model on {len(X_train)} samples")

            # need to encode categorical variables before resampling, dangerous way of encoding > resampling
            X_train = pd.get_dummies(X_train)
            
            X_train, y_train = self.resampler.fit_resample(X_train, y_train)
            print(f"Resampling resulted in: {len(X_train)}")

        # scale regardless of whether resampling is used
        X_train_scaled = self.scaler.fit_transform(X_train)

        if logging:
            self.model.fit(X_train_scaled, y_train)
            cv_pr_auc_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring="average_precision")

            self._log_to_mlflow(cv_pr_auc_scores.mean())

            run_id = mlflow.active_run().info.run_id
            print(f"Model {self.model_name} logged and registered with run_id: {run_id}")

        else:
            self.model.fit(X_train_scaled, y_train)

        # Extract feature importance
        self._extract_feature_importance(X_train_scaled)

        return self

    def predict(self, X, threshold=THRESHOLD):
        """
        Predict fraud labels.

        Parameters
        ----------
        X : pd.DataFrame or np.array
            Features

        Returns
        -------
        np.array
            Predicted labels (0 or 1)
        """
        y_proba = self.predict_proba(X)
        return y_proba >= threshold

    def predict_proba(self, X):
        """
        Predict fraud probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.array
            Features

        Returns
        -------
        np.array
            Predicted probabilities for each class
        """
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            print(f"{e} due to data being encoded before resampling and scaling.")
            X_scaled = self.scaler.transform(pd.get_dummies(X))
        return self.model.predict_proba(X_scaled)[:, 1]

    def _extract_feature_importance(self, X_train):
        """Extract feature importance based on model type."""

        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]

        if hasattr(self.model, 'coef_'):
            # Logistic Regression
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'coefficient': self.model.coef_[0]
            }).sort_values(by='coefficient', key=abs, ascending=False)

        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost)
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)

    def get_feature_importance(self, top_n=20):
        """
        Get top N most important features.

        Parameters
        ----------
        top_n : int
            Number of top features to return

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first")

        return self.feature_importance_.head(top_n)

    def save(self, models_dir=None):
        """
        Save model and preprocessor to disk with model_name.

        Parameters
        ----------
        models_dir : Path or str
            Directory to save models (uses MODELS_DIR from config if None)

        Returns
        -------
        dict
            Paths where model and preprocessor were saved
        """
        from pathlib import Path
        from src.config import MODELS_DIR

        models_dir = Path(models_dir) if models_dir else MODELS_DIR
        models_dir.mkdir(exist_ok=True, parents=True)

        # Save model with name
        model_path = models_dir / f"{self.model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved: {model_path}")

        # Save preprocessor with name
        preprocessor_path = models_dir / f"{self.model_name}_preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Preprocessor saved: {preprocessor_path}")

        return {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path
        }

    @classmethod
    def load(cls, model_name, models_dir=None):
        """
        Load model and preprocessor from disk by name.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'seen_devices', 'unseen_devices')
        models_dir : Path or str
            Directory to load from (uses MODELS_DIR from config if None)

        Returns
        -------
        FraudDetectionModel
            Loaded model instance with preprocessor

        Example
        -------
        >>> seen_model = FraudDetectionModel.load('seen_devices')
        >>> unseen_model = FraudDetectionModel.load('unseen_devices')
        """
        from pathlib import Path
        from src.config import MODELS_DIR

        models_dir = Path(models_dir) if models_dir else MODELS_DIR

        # Load model
        model_path = models_dir / f"{model_name}_model.pkl"
        preprocessor_path = models_dir / f"{model_name}_preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        # Create instance
        instance = cls(model_name=model_name)

        # Load model and preprocessor
        with open(model_path, "rb") as f:
            instance.model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            instance.scaler = pickle.load(f)

        print(f"✓ Loaded model '{model_name}' from {models_dir}")

        return instance


class FraudModelTuner:
    """Tune hyperparameters for fraud detection model."""

    def __init__(self, resampling_type="random_undersampling", model_type="logistic_regression", scoring="recall", cv=5):
        self.resampling_type = resampling_type
        self.model_type = model_type
        self.scoring = scoring
        self.cv = cv
        self.best_params = None
        self.best_score_ = None
        self.best_model_ = None

    def tune(self, X_train, y_train, param_grid):
        """Tune hyperparameters and return FraudDetectionModel instance with tuned hyperparameters."""
        base_model = FraudDetectionModel(
            resampling_type=self.resampling_type,
            model_type=self.model_type
        )

        X_resampled, y_resampled = base_model.resampler.fit_resample(X_train, y_train)

        print(f"Resampling resulted in: {len(X_resampled)}")

        mlflow.sklearn.autolog(max_tuning_runs=5)

        with mlflow.start_run(run_name=f"{self.model_type} Hyperparameter Tuning"):
            grid_search = GridSearchCV(
                estimator=base_model.model,
                param_grid=param_grid,
                scoring=self.scoring,
                verbose=2
            )
            grid_search.fit(X_resampled, y_resampled)

        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_model_ = grid_search.best_estimator_

        print(f"Best parameters: {self.best_params_}")
        print(f"Best CV {self.scoring} score: {self.best_score_:.4f}")

        # Create FraudDetectionModel with best parameters
        best_fraud_model = FraudDetectionModel(
            resampling_type=self.resampling_type,
            model_type=self.model_type,
            custom_params=self.best_params_
        )
        best_fraud_model.model = self.best_model_

        return best_fraud_model, grid_search

        