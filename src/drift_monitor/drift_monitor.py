import bisect
import mlflow
import numpy as np
from scipy.spatial.distance import jensenshannon

class DriftMonitor:
    """                                                                                                   
    Utility class for logging reference distributions and calculating drift.                                              
                                                                                                        
    For categorical features: Uses Jensen-Shannon divergence on frequency distributions                   
    For numerical features: Bins data using Freedman-Diaconis rule, then uses Jensen-Shannon divergence   
    """   

    def __init__(self):
        pass

    def build_categorical_feature_reference(self, df, feature, logging=True):
        """
        Build reference distribution for categorical feature.

        Args:
            df: DataFrame containing the feature
            feature: Column name of categorical feature
            logging: Whether to log to MLflow (must have active run)

        Returns:
            Dictionary with counts and total observations
        """
        counts = df[feature].value_counts(normalize=False)
        categorical_dict = {
            "counts": counts.to_dict(),
            "n": int(counts.sum())
        }

        if logging:
            # Log to the currently active MLflow run (started in train.py)
            mlflow.log_dict(categorical_dict, f"drift_reference/{feature}.json")

        return categorical_dict
    

    def create_numerical_bins(self, x):
        q0 = x.min()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        bin_width = 2 * iqr * len(x) ** (-1/3)
       
        if bin_width == 0:
            bin_width = 1e-6  # avoid division by zero
        
        num_bins = max(1, int((x.max() - q0) / bin_width))

        bin_lower_bound = []

        for bin in range(num_bins):
            bin_lower_bound.append(q0 + bin*bin_width)
        
        bin_lower_bound.append(x.max() + 1e-6)
        return bin_lower_bound


    def build_numerical_feature_reference(self, df, feature, logging=True):
        """
        Build reference distribution for numerical feature.

        Args:
            df: DataFrame containing the feature
            feature: Column name of numerical feature
            logging: Whether to log to MLflow (must have active run)

        Returns:
            Dictionary with binned counts and bin edges
        """
        x = df[feature]
        bin_lower_bound = self.create_numerical_bins(x)

        def numerical_to_bin(val):
            idx = bisect.bisect_right(bin_lower_bound, val) - 1
            if idx < len(bin_lower_bound) - 1:
                lower = bin_lower_bound[idx]
                upper = bin_lower_bound[idx + 1]
                return f"{feature} >= {lower:.2f} < {upper:.2f}"
            else:
                # last bin
                lower = bin_lower_bound[idx]
                return f"{feature} >= {lower:.2f}"

        x_bin = x.apply(numerical_to_bin)
        counts = x_bin.value_counts()

        numerical_bins_dict = {
            "counts": counts.to_dict(),
            "n": int(counts.sum()),
            "bin_edges": bin_lower_bound,
            "type": "numerical"
        }

        if logging:
            # Log to the currently active MLflow run (started in train.py)
            mlflow.log_dict(numerical_bins_dict, f"drift_reference/{feature}.json")

        return numerical_bins_dict

    def _align_distributions(self, baseline_dict, current_dict):                                          
        """                                                                                               
        Align two distributions to have the same categories.                                              
        Missing categories get probability 0.                                                             
                                                                                                        
        Args:                                                                                             
            baseline_dict: Reference distribution dictionary                                              
            current_dict: Current distribution dictionary                                                 
                                                                                                        
        Returns:                                                                                          
            Tuple of (baseline_probs, current_probs) as numpy arrays                                      
        """                                                                                               
        baseline_counts = baseline_dict["counts"]                                                         
        current_counts = current_dict["counts"]                                                           
                                                                                                        
        # Get union of all categories                                                                     
        all_categories = sorted(set(baseline_counts.keys()) | set(current_counts.keys()))                 
                                                                                                        
        # Create aligned probability distributions                                                        
        baseline_n = baseline_dict["n"]                                                                   
        current_n = current_dict["n"]                                                                     
                                                                                                        
        baseline_probs = np.array([baseline_counts.get(cat, 0) / baseline_n for cat in all_categories])   
        current_probs = np.array([current_counts.get(cat, 0) / current_n for cat in all_categories])      
                                                                                                                             
        # Renormalize                                                                                     
        baseline_probs = baseline_probs / baseline_probs.sum()                                            
        current_probs = current_probs / current_probs.sum()                                               
                                                                                                        
        return baseline_probs, current_probs
    
    def monitor_drift(self, baseline_dict, current_dict, metric="jensenshannon"):                         
        """                                                                                               
        Calculate drift between baseline and current distributions.                                       
                                                                                                        
        Args:                                                                                             
            baseline_dict: Reference distribution from build_*_reference                                  
            current_dict: Current distribution from build_*_reference                                     
            metric: Drift metric to use ('jensenshannon' or 'psi')                                        
                                                                                                        
        Returns:                                                                                          
            Float representing drift magnitude                                                            
        """                                                                                               
        baseline_probs, current_probs = self._align_distributions(baseline_dict, current_dict)            
                                                                                                        
        if metric == "jensenshannon":                                                                     
            # Jensen-Shannon divergence (0 to 1, where 0 = identical)                                     
            return float(jensenshannon(baseline_probs, current_probs))                                    
                                                                                                                                                                                
        else:                                                                                             
            raise ValueError(f"Unknown metric: {metric}")
        
    def build_data_reference(self, df, numerical_features, categorical_features, boolean_features, logging=True):
        """Builds reference for all features in the dataset."""

        # Handle None values by converting to empty lists
        categorical_features = categorical_features or []
        categorical_features.append("rule_label")
        boolean_features = boolean_features or []
        numerical_features = numerical_features or []

        for cat in categorical_features:
            self.build_categorical_feature_reference(df, cat, logging)

        for bool in boolean_features:
            self.build_categorical_feature_reference(df, bool, logging)

        # for num in numerical_features:
        #     self.build_numerical_feature_reference(df, num, logging)

        # Log summary
        total_features = len(categorical_features) + len(boolean_features) + len(numerical_features)
        if logging and total_features > 0:
            print(f"Logged drift references for {total_features} features ({len(numerical_features)} numerical, {len(categorical_features)} categorical, {len(boolean_features)} boolean)")

        