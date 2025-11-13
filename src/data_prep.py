import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    """
    Comprehensive preprocessing pipeline for e-commerce fraud detection
    """
    
    def __init__(self, ip_country_path="../data/IpAddress_to_Country.csv", gdp_data_path="../data/gdp_usd.xlsx"):
        """
        Parameters:
        -----------
        gdp_data : dict, optional
            Dictionary mapping country codes to GDP per capita
        """
        self.ip_mapping = pd.read_csv(ip_country_path)
        self.gdp_data = pd.read_excel(gdp_data_path, index_col=0).loc[2015.0,].to_dict()
        self.country_map = {
            "Korea Republic of": "Korea, Rep.",
            "Taiwan; Republic of China (ROC)": "Taiwan, China",
            "Hong Kong": "Hong Kong SAR, China",
            "Egypt": "Egypt, Arab Rep.",
            "Slovakia": "Slovakia (SLOVAK Republic)",
            "Croatia": "Croatia (LOCAL Name: Hrvatska)",
            "Oman": "EMDE Middle East, North Africa, Afghanistan & Pakistan",
            "Moldova Republic of": "Moldova, Rep.",
            "Bosnia and Herzegowina": "Bosnia and Herzegovina",
            "Macedonia": "North Macedonia"
        }
        self.scaler = StandardScaler()

        # Global statistics (learned from training)
        self.global_purchase_std = None
        self.global_purchase_mean = None
        self.fraud_rate_by_month = None
        
        # Device and IP level statistics (learned from training)
        self.device_stats_dict = {}
        self.ip_stats_dict = {}
       
    
    def fit(self, df, target_col='class'):
        """
        Fit preprocessing parameters on training data
        
        CRITICAL: This learns statistics from training data that will be
        applied to test data to prevent data leakage
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe with transaction data
        target_col : str
            Name of fraud indicator column
        """
        print("Fitting preprocessing parameters on training data...")
        
        # Calculate global purchase statistics
        self.global_purchase_mean = df['purchase_value'].mean()
        self.global_purchase_std = df['purchase_value'].std()
        
        print(f"  Global purchase mean: ${self.global_purchase_mean:.2f}")
        print(f"  Global purchase std: ${self.global_purchase_std:.2f}")
        
        # Calculate fraud rate by month (for feature engineering)
        if target_col in df.columns:
            df['transaction_month'] = pd.to_datetime(df['purchase_time']).dt.month
            self.fraud_rate_by_month = df.groupby('transaction_month')[target_col].mean().to_dict()
            print(f"  Fraud rates by month calculated: {len(self.fraud_rate_by_month)} months")
        
        # Store device-level statistics for test set
        device_stats = df.groupby('device_id')['purchase_value'].agg(['std', 'mean', 'count'])
        self.device_stats_dict = device_stats.to_dict('index')
        print(f"  Stored statistics for {len(self.device_stats_dict)} unique devices")
        
        # Store IP-level statistics for test set
        ip_stats = df.groupby('ip_address')['purchase_value'].agg(['std', 'mean', 'count'])
        self.ip_stats_dict = ip_stats.to_dict('index')
        print(f"  Stored statistics for {len(self.ip_stats_dict)} unique IP addresses")
        
        return self
    
    def transform(self, df, is_training=False):
        """
        Transform dataframe with all feature engineering steps
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns:
            - device_id, ip_address, purchase_value, signup_time, purchase_time
        is_training : bool
            Whether this is training data (affects statistical calculations)
        
        Returns:
        --------
        pd.DataFrame : Processed dataframe with engineered features
        """
        df = df.copy()
    
        print("Step 1: Map IP Address to country")
        df = self._add_geo_features(df)

        # Ensure datetime columns are datetime type
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        print("Step 2: Basic aggregation features...")
        df = self._add_transaction_counts(df)
        
        print("Step 3: Statistical normalization features...")
        df = self._add_normalized_counts(df)
        
        print("Step 4: Device-IP consistency...")
        df = self._add_device_ip_consistency(df)
        
        print("Step 5: Time-based features...")
        df = self._add_time_features(df)
        
        print("Step 6: Velocity features...")
        df = self._add_velocity_features(df)
        
        print("Step 7: Purchase value statistics...")
        df = self._add_purchase_value_stats(df, is_training)
        
        print("Step 8: Temporal encoding...")
        df = self._add_temporal_encoding(df)
        
        print("Step 9: Geographic and demographic features...")
        df = self._add_geo_demographic_features(df)
        
        print("Step 10: Clustering features...")
        df = self._add_clustering_features(df)
        
        print("Feature engineering complete!")
        return df
    
    def _add_transaction_counts(self, df):
        """Number of transactions by device and IP"""
        df['txn_count_by_device'] = df.groupby('device_id')['device_id'].transform('count')
        df['txn_count_by_ip'] = df.groupby('ip_address')['ip_address'].transform('count')
        return df
    
    def _add_normalized_counts(self, df):
        """Z-scores and percentile ranks for transaction counts"""
        # Z-scores (standardized counts)
        mean_device = df['txn_count_by_device'].mean()
        std_device = df['txn_count_by_device'].std()
        df['txn_count_device_zscore'] = (df['txn_count_by_device'] - mean_device) / std_device
        
        mean_ip = df['txn_count_by_ip'].mean()
        std_ip = df['txn_count_by_ip'].std()
        df['txn_count_ip_zscore'] = (df['txn_count_by_ip'] - mean_ip) / std_ip
        
        # Percentile ranks (0-100)
        df['txn_count_device_percentile'] = df['txn_count_by_device'].rank(pct=True) * 100
        df['txn_count_ip_percentile'] = df['txn_count_by_ip'].rank(pct=True) * 100
        
        return df
    
    def _add_device_ip_consistency(self, df):
        """Check if device is tied to exactly one IP"""
        device_ip_count = df.groupby('device_id')['ip_address'].nunique().to_dict()
        df['device_ip_consistency'] = df['device_id'].map(device_ip_count)
        df['is_device_single_ip'] = (df['device_ip_consistency'] == 1).astype(int)
        
        # Reverse: IP to device consistency
        ip_device_count = df.groupby('ip_address')['device_id'].nunique().to_dict()
        df['ip_device_consistency'] = df['ip_address'].map(ip_device_count)
        df['is_ip_single_device'] = (df['ip_device_consistency'] == 1).astype(int)
        
        return df
    
    def _add_time_features(self, df):
        """Time between account setup and transaction"""
        df['time_setup_to_txn_seconds'] = (
            df['signup_time'] - df['purchase_time']
        ).dt.total_seconds()

        return df
    
    def _add_velocity_features(self, df):
        """Time since last transaction and velocity metrics"""
        # Sort by device, IP, and time

        def time_since_last_txn(df, col):
            df[f"time_since_last_{col}_txn"] = df.groupby(col)["purchase_time"].diff().dt.seconds
            
            # Fill NaN (first transaction) with large value
            df[f"time_since_last_{col}_txn"].fillna(999999, inplace=True)
            return df

        df = df.sort_values('purchase_time')
        
        # Time since last transaction for same device
        df = time_since_last_txn(df, "device_id")
        
        # Time since last transaction for same IP
        df = time_since_last_txn(df, "ip_address")
        
        
        df['time_since_last_device_id_txn'].fillna(999999, inplace=True)
        df['time_since_last_ip_address_txn'].fillna(999999, inplace=True)
        
        return df
    
    def _add_purchase_value_stats(self, df, is_training):
        """
        Standard deviation and variance of purchase value by device/IP
        
        CRITICAL DIFFERENCE BETWEEN TRAINING AND TESTING:
        - Training: Calculate stats from current training data
        - Testing: Use stats learned from training data (prevents leakage!)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        is_training : bool
            Whether this is training data
        """
        if is_training:
            print("  Calculating purchase statistics from training data...")
            # Calculate statistics from current training data
            device_stats = df.groupby('device_id')['purchase_value'].agg(['std', 'mean', 'count']).reset_index()
            device_stats.columns = ['device_id', 'purchase_std_by_device', 'purchase_mean_by_device', 'purchase_count_by_device']
            df = df.merge(device_stats, on='device_id', how='left')
            
            ip_stats = df.groupby('ip_address')['purchase_value'].agg(['std', 'mean', 'count']).reset_index()
            ip_stats.columns = ['ip_address', 'purchase_std_by_ip', 'purchase_mean_by_ip', 'purchase_count_by_ip']
            df = df.merge(ip_stats, on='ip_address', how='left')
            
        else:
            print("  Applying training statistics to test data...")
            # Use statistics learned from training (prevent leakage!)
            
            # Device statistics
            df['purchase_std_by_device'] = df['device_id'].apply(
                lambda x: self.device_stats_dict.get(x, {}).get('std', self.global_purchase_std)
            )
            df['purchase_mean_by_device'] = df['device_id'].apply(
                lambda x: self.device_stats_dict.get(x, {}).get('mean', self.global_purchase_mean)
            )
            df['purchase_count_by_device'] = df['device_id'].apply(
                lambda x: self.device_stats_dict.get(x, {}).get('count', 1)
            )
            
            # IP statistics
            df['purchase_std_by_ip'] = df['ip_address'].apply(
                lambda x: self.ip_stats_dict.get(x, {}).get('std', self.global_purchase_std)
            )
            df['purchase_mean_by_ip'] = df['ip_address'].apply(
                lambda x: self.ip_stats_dict.get(x, {}).get('mean', self.global_purchase_mean)
            )
            df['purchase_count_by_ip'] = df['ip_address'].apply(
                lambda x: self.ip_stats_dict.get(x, {}).get('count', 1)
            )
        
        # Handle missing values (single transactions) using global statistics
        df['purchase_std_by_device'].fillna(self.global_purchase_std, inplace=True)
        df['purchase_std_by_ip'].fillna(self.global_purchase_std, inplace=True)
        df['purchase_mean_by_device'].fillna(self.global_purchase_mean, inplace=True)
        df['purchase_mean_by_ip'].fillna(self.global_purchase_mean, inplace=True)
        
        # Flag for unique device/IP (useful signal)
        df['is_unique_device'] = (df['purchase_count_by_device'] == 1).astype(int)
        df['is_unique_ip'] = (df['purchase_count_by_ip'] == 1).astype(int)
        
        # Coefficient of variation (std/mean) - normalized measure of variability
        df['purchase_cv_by_device'] = df['purchase_std_by_device'] / (df['purchase_mean_by_device'] + 1)
        df['purchase_cv_by_ip'] = df['purchase_std_by_ip'] / (df['purchase_mean_by_ip'] + 1)
        
        # Deviation from user's typical purchase amount
        df['purchase_deviation_from_device_mean'] = abs(df['purchase_value'] - df['purchase_mean_by_device'])
        df['purchase_deviation_from_ip_mean'] = abs(df['purchase_value'] - df['purchase_mean_by_ip'])
        
        # Z-score of current purchase relative to device/IP history
        df['purchase_zscore_by_device'] = (df['purchase_value'] - df['purchase_mean_by_device']) / (df['purchase_std_by_device'] + 1)
        df['purchase_zscore_by_ip'] = (df['purchase_value'] - df['purchase_mean_by_ip']) / (df['purchase_std_by_ip'] + 1)
        
        return df
    
    def _add_temporal_encoding(self, df):
        """Cyclical encoding of month and fraud rate by month"""
        # Extract month from purchase_time
        df['transaction_month'] = df['purchase_time'].dt.month
        
        # Cyclical encoding (preserves cyclical nature: Dec is close to Jan)
        df['month_sin'] = np.sin(2 * np.pi * df['transaction_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['transaction_month'] / 12)
        
        # Add fraud rate by month (if available from training)
        if self.fraud_rate_by_month:
            df['fraud_rate_by_month'] = df['transaction_month'].map(self.fraud_rate_by_month)
            # Fill any missing months with overall fraud rate
            overall_fraud_rate = np.mean(list(self.fraud_rate_by_month.values()))
            df['fraud_rate_by_month'].fillna(overall_fraud_rate, inplace=True)
        
        return df
    
    def _add_geo_features(self, df):
        """Mapping of IP address to country"""
        def map_ip_to_country(ip_address, ip_mapping):
            ip_mapping_list = ip_mapping.values.tolist()
            ip_mapping_list.sort(key=lambda x: x[0])

            l, r = 0, len(ip_mapping_list)-1

            while l <= r:
                m = (l + r) // 2
                if ip_mapping_list[m][0] <= ip_address <= ip_mapping_list[m][1]:
                    return ip_mapping_list[m][2]

                elif ip_mapping_list[m][0] > ip_address:
                    r = m - 1

                elif ip_address > ip_mapping_list[m][1]:
                    l = m + 1

            logging.warning(f"IP address {ip_address} not found in any range.")
            return None

        def log_and_map_ip_to_country(ip_address, ip_mapping):
            country = map_ip_to_country(ip_address, ip_mapping)
            logging.info(f"IP Address {ip_address} mapped to country: {country}")
            return country

        filepath = os.path.join(os.getcwd(), "../data/fraud_with_country.csv")

        if os.path.exists(filepath):
            # Load the country mapping and merge it with the current dataframe
            country_mapping = pd.read_csv(filepath)[['ip_address', 'country']]
            df = df.drop('ip_address', axis=1)
            df = pd.merge(df, country_mapping, left_index=True, right_index=True)

        else:
            df["country"] = df["ip_address"].apply(lambda x: log_and_map_ip_to_country(x, self.ip_mapping))

        return df
    
    def _add_geo_demographic_features(self, df):
        """9: GDP mapping and demographic features"""
        
        df["gdp"] = df["country"].apply(
        lambda x: self.gdp_data.get(
            x, self.gdp_data.get(self.country_map.get(x))) if x in self.gdp_data or x in self.country_map else np.nan
        )
        
        # Fill missing GDP values with median
        gdp_median = df["gdp"].median()
        df["gdp"].fillna(gdp_median, inplace=True)

        # Country with highest GDP has rank 1
        df["gdp_rank"] = df["gdp"].rank(method="dense", ascending=False).astype(int)
        
        # Normalized GDP rank (0-1 scale)
        max_rank = df["gdp_rank"].max()
        df["gdp_rank_normalized"] = 1 - ((df["gdp_rank"] - 1) / max_rank)

        # Sex encoding (assuming M/F)
        if 'sex' in df.columns:
            df['sex_encoded'] = df['sex'].map({'M': 1, 'F': 0}).fillna(-1)
        
        # Age bins
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 18, 25, 35, 50, 65, 100], 
                labels=['under_18', '18_25', '26_35', '36_50', '51_65', 'over_65']
            )
            age_dummies = pd.get_dummies(df['age_group'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)
        
        return df
    
    def _add_clustering_features(self, df):
        """9: K-means clustering based on demographics and purchase behavior"""
        # Select features for clustering
        cluster_features = ['sex_encoded', 'age', 'purchase_value', 'gdp_rank']

        X_cluster = df[cluster_features].copy()
        
        # Handle any remaining missing values
        X_cluster = X_cluster.fillna(X_cluster.median())
        
        # Standardize features for clustering
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Apply K-means with multiple cluster counts
        for n_clusters in [3, 5, 8]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_scaled)
        
        return df
    
    def fit_transform(self, df, target_col='class'):
        """
        Fit on training data and transform
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe
        target_col : str
            Name of target variable
        
        Returns:
        --------
        pd.DataFrame : Transformed training data
        """
        self.fit(df, target_col)
        return self.transform(df, is_training=True)
    
    def preprocess_pipeline(self, train_df, test_df=None, target_col='class'):
        """
        Complete preprocessing pipeline for train and optionally test data
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data
        test_df : pd.DataFrame, optional
            Test data (if provided)
        target_col : str
            Name of target variable
        
        Returns:
        --------
        tuple : (train_processed, test_processed) or just train_processed
        """
        print("="*60)
        print("TRAINING DATA PREPROCESSING")
        print("="*60)
        
        # Fit and transform training data
        train_processed = self.fit_transform(train_df, target_col)
        
        if test_df is not None:
            print("\n" + "="*60)
            print("TEST DATA PREPROCESSING")
            print("="*60)
            print("Using training statistics to prevent data leakage...")
            
            # Transform test data using training parameters
            test_processed = self.transform(test_df, is_training=False)
            
            return train_processed, test_processed
        
        return train_processed


if __name__ == "__main__":
    """
    Example demonstrating proper train/test split preprocessing
    """
    
    # Initialize feature engineer
    feature_engineer = FraudFeatureEngineer(
        ip_country_path="../data/IpAddress_to_Country.csv",
        gdp_data_path="../data/gdp_usd.xlsx"
    )
    
    # Load your data
    df = pd.read_csv("../data/Fraud_Data.csv")
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

    print(f"SIZE OF TRAIN DATA: {len(train_df)}, SIZE OF TEST DATA: {len(test_df)}")

    # Process both train and test
    train_processed, test_processed = feature_engineer.preprocess_pipeline(
        train_df=train_df, 
        test_df=test_df, 
        target_col='class'
    )
    print(f"SIZE OF TRAIN DATA: {len(train_processed)}, SIZE OF TEST DATA: {len(test_processed)}")

    # Save dataset
    train_processed.to_csv("../data/train.csv")
    test_processed.to_csv("../data/test.csv")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print("\nKey points:")
    print("✓ Training data: Statistics calculated from training set")
    print("✓ Test data: Uses training statistics (NO DATA LEAKAGE)")
    print("✓ New devices/IPs in test: Use global statistics as fallback")