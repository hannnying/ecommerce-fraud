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
from sklearn.model_selection import GridSearchCV
from src.preprocessing import FraudDataPreprocessor
import joblib



class FraudDetectionModel:

    def __init__(self, resampling_type='random_undersampling', scaler_path=None, model_type='logistic_regression', custom_params=None):
        """
        Initialize fraud detection model with resampling technique.

        Args:
            resampling_type (str): Type of model used for resampling: 'randon_undersamping', 'enn', 'smote', 'smoteenn'.
            scaler_path (str): Path of fitted scaler. If None, initialise with FraudDataPreprocessor
            model_type (str): Type of model: 'logistic_regression', 'random_forest', 'xgboost'
            custom_params (dict): Custom hyperparameters to override defaults

        """

        self.resampling_type = resampling_type
        self.resampler = None
        self.scaler_path = scaler_path
        self.scaler = None
        self.model_type = model_type
        self.model = None
        self.params = None
        self.feature_importance_ = None

        # Initialize resampling used
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
        print(f"Resampling and Fitting model on {len(X_train)} samples")

        X_train_resampled, y_train_resampled = self.resampler.fit_resample(X_train, y_train)
        print(f"Resampling resulted in: {len(X_train_resampled)}")

        X_train_resampled_scaled = self.scaler.fit_transform(X_train_resampled)

        if logging:
            mlflow.sklearn.autolog(log_models=False)

            with mlflow.start_run() as run:
                self.model.fit(X_train_resampled_scaled, y_train_resampled)

                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    registered_model_name="fraud_detection_model"
                )

                run_id = run.info.run_id
                print(f"Model loged and registered with run_id: {run_id}")

            mlflow_client = MlflowClient()

            latest_version = mlflow_client.get_latest_versions("fraud_detection_model", stages=["None"])[0]
            mlflow_client.transition_model_version_stage(
                name="fraud_detection_model",
                version=latest_version.version,
                stage="Production"
            )

            print(f"Model version {latest_version.version} promoted to Production")
        
        else:
            self.model.fit(X_train_resampled_scaled, y_train_resampled)

        # Extract feature importance
        self._extract_feature_importance(X_train_resampled)

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
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def _extract_feature_importance(self, X_train):
        """Extract feature importance based on model type."""
        import pandas as pd

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

    def save(self, filepath):
        """
        Save model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, model_type='logistic_regression'):
        """
        Load model from disk.

        Parameters
        ----------
        filepath : str
            Path to load the model from
        model_type : str
            Type of model being loaded

        Returns
        -------
        FraudDetectionModel
            Loaded model instance
        """
        instance = cls(model_type=model_type)
        instance.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
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

        