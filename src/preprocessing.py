import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class FraudDataPreprocessor:
    """
    1. Drop duplicate and unused columns from train/test data from data_prep.py
    2. Separate fremaining eatures by type (numeric, categorical, boolean)
    3. Apply StandardScaler to numeric features
    4. Apply OneHotEncoder to categorical features
    5. Concatenate all features back together
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.preprocessor = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        self.boolean_features = None
        self.is_fitted = False
        self.drop_cols = [
            "user_id", "signup_time", "purchase_time",
            "device_id", "ip_address", "country",
            "sex", "transactions_by_user_id", "age_group"
        ]

    def fit(self, X_train, y_train=None):
        """Fit the preprocessor on training data."""
        self._identify_feature_types(X_train)

        def bool_to_int(X):
            X = X.fillna(False)
            return X.astype(int)

        # create Column Transformer to transform numeric and categorical columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                self.categorical_features),
                ("bool", FunctionTransformer(bool_to_int), self.boolean_features)
            ],
            remainder='drop'  # Drop everything else initially
        )
        self.preprocessor.fit(X_train)

        # Get transformed feature names
        num_features = self.numeric_features
        cat_features = self.preprocessor.named_transformers_["cat"].get_feature_names_out(
            self.categorical_features
        )
        bool_features = self.boolean_features

        self.feature_names = list(num_features) + list(cat_features) + list(bool_features)

        self.is_fitted = True

        print(f"Fitted preprocessor: {self.preprocessor}")

        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Column Transformer only transforms numeric and categorical
        X_trans = self.preprocessor.transform(X)

        num_features = self.numeric_features
        cat_features = self.preprocessor.named_transformers_["cat"].get_feature_names_out(
            self.categorical_features
        )
        bool_features = self.boolean_features
        base_features = list(num_features) + list(cat_features) + list(bool_features)
        X_scaled = pd.DataFrame(X_trans, columns=base_features, index=X.index)
       
        return X_scaled
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)
        
 
    def _identify_feature_types(self, X):
        all_features = [col for col in X.columns if col not in self.drop_cols + ["class"]]
        print(f"There are a total of {len(all_features)} features: {all_features}")

        self.numeric_features = [f for f in all_features if X[f].dtype in ["int64", "float64"]]
        print(f"There are a total of {len(self.numeric_features)}: {self.numeric_features}")

        self.boolean_features = [f for f in all_features if X[f].dtype=="bool"]
        print(f"There are a total of {len(self.boolean_features)}: {self.boolean_features}")

        self.categorical_features = [f for f in all_features if (f not in self.numeric_features) and (f not in self.boolean_features)]
        print(f"There are a total of {len(self.categorical_features)}: {self.categorical_features}")

def load_and_preprocess_data(train_path, test_path, target_col="class"):
    # load train and test data
    # split into train and test sets
    # drop unused and repeated cols > transform numerical and categorical > 
    original_idx = ["Unnamed: 0"]
    
    try:
        train_df = pd.read_csv(train_path).drop(columns=original_idx)
        test_df = pd.read_csv(test_path).drop(columns=original_idx)
    except Exception as e:
        print(f"Try running 'data_prep.py': {e}")

    # Drop additional columns from data_prep.py (repeats of txn_count_by)
    additional_drops = [
        "transactions_by_device_id", "transactions_by_ip_address",
        "purchase_count_by_device", "purchase_count_by_ip"
    ]
    for col in additional_drops:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # check if columns to drop are in train data
    drop_cols = [
            "user_id", "signup_time", "purchase_time",
            "device_id", "ip_address", "country",
            "sex", "transactions_by_user_id"
        ]
    drop_cols = [col for col in drop_cols if col in train_df.columns]

    # Separate features and target
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=drop_cols + [target_col])

    y_test = test_df[target_col]
    X_test = test_df.drop(columns=drop_cols + [target_col])

    # Initialize and fit preprocessor
    print("\nPreprocessing features...")
    preprocessor = FraudDataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")
    print(f"Number of features: {len(preprocessor.feature_names)}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    