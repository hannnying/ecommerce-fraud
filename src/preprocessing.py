import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.config import TARGET_COL


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

    def bool_to_int(self, X):
        X = X.fillna(False)
        return X.astype(int)

    def fit(self, X_train, y_train=None):
        """Fit the preprocessor on training data."""
        self._identify_feature_types(X_train)

        # create Column Transformer to transform numeric and categorical columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                self.categorical_features),
                ("bool", FunctionTransformer(self.bool_to_int), self.boolean_features)
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

        # Recognize ALL numeric dtypes (int8, int16, int32, int64, float16, float32, float64)
        import numpy as np
        self.numeric_features = [f for f in all_features if np.issubdtype(X[f].dtype, np.number)]
        print(f"There are a total of {len(self.numeric_features)}: {self.numeric_features}")

        self.boolean_features = [f for f in all_features if X[f].dtype=="bool"]
        print(f"There are a total of {len(self.boolean_features)}: {self.boolean_features}")

        self.categorical_features = [f for f in all_features if (f not in self.numeric_features) and (f not in self.boolean_features)]
        print(f"There are a total of {len(self.categorical_features)}: {self.categorical_features}")

    def preprocess(self, df):
        """Drop columns and split into X, y before further processing."""
        original_idx = "Unnamed: 0"
        print(df)

        if original_idx in df.columns:
            df = df.drop(columns=[original_idx])

        # Drop additional columns from data_prep.py (repeats of txn_count_by)
        additional_drops = [
            "transactions_by_device_id", "transactions_by_ip_address",
            "purchase_count_by_device", "purchase_count_by_ip"
        ]
        for col in additional_drops:
            if col in df.columns:
                df = df.drop(columns=[col])

        # check if columns to drop are in train data
        drop_cols = [
                "user_id", "signup_time", "purchase_time",
                "device_id", "ip_address", "country",
                "sex", "transactions_by_user_id"
            ]
        drop_cols = [col for col in drop_cols if col in df.columns]

        # Separate features and target
        if TARGET_COL in df:
            y = df[TARGET_COL]
            X = df.drop(columns=drop_cols + [TARGET_COL])
            return X, y
        else:
            X = df.drop(columns=drop_cols)
            return X, pd.DataFrame()

        
