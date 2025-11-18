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

        # Transaction count statistics (learned from training)
        self.device_txn_counts = {}
        self.ip_txn_counts = {}

        # Normalization statistics for transaction counts
        self.txn_count_device_mean = None
        self.txn_count_device_std = None
        self.txn_count_ip_mean = None
        self.txn_count_ip_std = None

        # Device-IP consistency statistics
        self.device_ip_consistency_map = {}
        self.ip_device_consistency_map = {}

        # Velocity statistics (for time-based features)
        self.device_last_purchase_time = {}
        self.ip_last_purchase_time = {}

        # GDP ranking statistics
        self.gdp_rank_map = {}
        self.max_gdp_rank = None

        # Clustering models
        self.clustering_scaler = None
        self.kmeans_models = {}
       
    
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

        # Convert signup_time and purchase_time to Datetime object
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        

        # Calculate fraud rate by month (for feature engineering)
        if target_col in df.columns:
            df['transaction_month'] = df['purchase_time'].dt.month
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

        # Store transaction counts by device and IP
        self.device_txn_counts = df.groupby('device_id').size().to_dict()
        self.ip_txn_counts = df.groupby('ip_address').size().to_dict()
        print(f"  Stored transaction counts for devices and IPs")

        # Calculate normalization statistics for transaction counts
        device_counts = pd.Series(self.device_txn_counts.values())
        ip_counts = pd.Series(self.ip_txn_counts.values())

        self.txn_count_device_mean = device_counts.mean()
        self.txn_count_device_std = device_counts.std()
        self.txn_count_ip_mean = ip_counts.mean()
        self.txn_count_ip_std = ip_counts.std()
        print(f"  Transaction count normalization stats calculated")

        # Store device-IP consistency mappings
        self.device_ip_consistency_map = df.groupby('device_id')['ip_address'].nunique().to_dict()
        self.ip_device_consistency_map = df.groupby('ip_address')['device_id'].nunique().to_dict()
        print(f"  Device-IP consistency mappings stored")

        # Store last purchase times for velocity features
        df_sorted = df.sort_values('purchase_time')
        self.device_last_purchase_time = df_sorted.groupby('device_id')['purchase_time'].last().to_dict()
        self.ip_last_purchase_time = df_sorted.groupby('ip_address')['purchase_time'].last().to_dict()
        print(f" device_last_purchase_time: {self.device_last_purchase_time.items()}")
        print(f"  Last purchase times stored for velocity calculations")

        # Store GDP rankings (need to process geo features first to get country info)
        # This will be calculated during transform for training data
        print(f"  GDP rankings will be calculated during transform")

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
        df = self._add_transaction_counts(df, is_training)

        print("Step 3: Statistical normalization features...")
        df = self._add_normalized_counts(df, is_training)

        print("Step 4: Device-IP consistency...")
        df = self._add_device_ip_consistency(df, is_training)

        print("Step 5: Time-based features...")
        df = self._add_time_features(df)

        print("Step 6: Velocity features...")
        df = self._add_velocity_features(df, is_training)

        print("Step 7: Purchase value statistics...")
        df = self._add_purchase_value_stats(df, is_training)

        print("Step 8: Temporal encoding...")
        df = self._add_temporal_encoding(df)

        print("Step 9: Geographic and demographic features...")
        df = self._add_geo_demographic_features(df, is_training)

        print("Step 10: Clustering features...")
        df = self._add_clustering_features(df, is_training)
        
        print("Feature engineering complete!")
        return df
    
    def _add_transaction_counts(self, df, is_training):
        """
        Number of transactions by device and IP

        FIXED: Data leakage prevention
        - Training: Calculate counts from training data (will be stored in fit())
        - Testing: Use counts from training data, default to 1 for unseen devices/IPs

        Why there was leakage:
        Previously used groupby().transform('count') on entire dataframe, which would
        include test data when processing training data if they were combined.
        """
        if is_training:
            # For training, calculate from current data
            df['txn_count_by_device'] = df.groupby('device_id')['device_id'].transform('count')
            df['txn_count_by_ip'] = df.groupby('ip_address')['ip_address'].transform('count')
        else:
            # For testing, use counts learned from training
            df['txn_count_by_device'] = df['device_id'].map(self.device_txn_counts).fillna(1)
            df['txn_count_by_ip'] = df['ip_address'].map(self.ip_txn_counts).fillna(1)

        return df
    
    def _add_normalized_counts(self, df, is_training):
        """
        Z-scores and percentile ranks for transaction counts

        FIXED: Data leakage prevention
        - Training: Calculate statistics from training data
        - Testing: Use statistics learned from training data

        Why there was leakage:
        Previously calculated mean, std, and percentile ranks using statistics from
        the entire dataframe. This means test data statistics would leak into training
        features if data was combined. Percentile ranks are especially problematic as
        they depend on the entire distribution.
        """
        if is_training:
            # Z-scores (standardized counts) using training statistics
            mean_device = df['txn_count_by_device'].mean()
            std_device = df['txn_count_by_device'].std()
            df['txn_count_device_zscore'] = (df['txn_count_by_device'] - mean_device) / (std_device + 1e-8)

            mean_ip = df['txn_count_by_ip'].mean()
            std_ip = df['txn_count_by_ip'].std()
            df['txn_count_ip_zscore'] = (df['txn_count_by_ip'] - mean_ip) / (std_ip + 1e-8)

            # Percentile ranks (0-100) based on training distribution
            df['txn_count_device_percentile'] = df['txn_count_by_device'].rank(pct=True) * 100
            df['txn_count_ip_percentile'] = df['txn_count_by_ip'].rank(pct=True) * 100
        else:
            # Use training statistics for z-scores
            df['txn_count_device_zscore'] = (df['txn_count_by_device'] - self.txn_count_device_mean) / (self.txn_count_device_std + 1e-8)
            df['txn_count_ip_zscore'] = (df['txn_count_by_ip'] - self.txn_count_ip_mean) / (self.txn_count_ip_std + 1e-8)

            # For percentiles, we can't directly apply training ranks, so we approximate
            # by comparing test counts to training distribution statistics
            # This is an approximation but prevents direct leakage
            df['txn_count_device_percentile'] = df['txn_count_device_zscore'].apply(
                lambda z: min(100, max(0, 50 + 34.13 * z))  # Normal CDF approximation
            )
            df['txn_count_ip_percentile'] = df['txn_count_ip_zscore'].apply(
                lambda z: min(100, max(0, 50 + 34.13 * z))
            )

        return df
    
    def _add_device_ip_consistency(self, df, is_training):
        """
        Check if device is tied to exactly one IP (and vice versa)

        FIXED: Data leakage prevention
        - Training: Calculate consistency from training data
        - Testing: Use consistency mappings from training data

        Why there was leakage:
        Previously calculated unique IP counts per device across entire dataframe.
        A device might have 1 unique IP in training but 2 in combined train+test,
        causing the training features to be influenced by test data behavior.
        """
        if is_training:
            # Calculate from current training data
            device_ip_count = df.groupby('device_id')['ip_address'].nunique().to_dict()
            df['device_ip_consistency'] = df['device_id'].map(device_ip_count)
            df['is_device_single_ip'] = (df['device_ip_consistency'] == 1).astype(int)

            # Reverse: IP to device consistency
            ip_device_count = df.groupby('ip_address')['device_id'].nunique().to_dict()
            df['ip_device_consistency'] = df['ip_address'].map(ip_device_count)
            df['is_ip_single_device'] = (df['ip_device_consistency'] == 1).astype(int)
        else:
            # Use mappings from training data, default to 1 for unseen devices/IPs
            df['device_ip_consistency'] = df['device_id'].map(self.device_ip_consistency_map).fillna(1)
            df['is_device_single_ip'] = (df['device_ip_consistency'] == 1).astype(int)

            df['ip_device_consistency'] = df['ip_address'].map(self.ip_device_consistency_map).fillna(1)
            df['is_ip_single_device'] = (df['ip_device_consistency'] == 1).astype(int)

        return df
    
    def _add_time_features(self, df):
        """Time between account setup and transaction"""
        df['time_setup_to_txn_seconds'] = (
            df['signup_time'] - df['purchase_time']
        ).dt.total_seconds()

        return df
    
    def _add_velocity_features(self, df, is_training):
        """
        Time since last transaction and velocity metrics

        FIXED: Data leakage prevention
        - Training: Calculate from training data only
        - Testing: Use last purchase time from training as reference point

        Why there was leakage:
        Previously sorted entire dataframe and calculated time differences using
        groupby().diff(). This means if a device/IP appears in both train and test,
        the training features would see test transaction times through the diff
        calculation. For instance, if test transactions come chronologically after
        train, the last training transaction would have a small diff value (showing
        more recent activity) instead of being the actual last known transaction.
        """
        df = df.sort_values('purchase_time')

        if is_training:
            # For training, calculate time since previous transaction within training data
            def time_since_last_txn(df_local, col):
                df_local[f"time_since_last_{col}_txn"] = df_local.groupby(col)["purchase_time"].diff().dt.total_seconds()
                df_local[f"time_since_last_{col}_txn"].fillna(999999, inplace=True)
                return df_local

            df = time_since_last_txn(df, "device_id")
            df = time_since_last_txn(df, "ip_address")
        else:
            # For test data, calculate time since last known transaction from training
            def time_since_training(row, col, last_time_dict):
                if row[col] in last_time_dict:
                    # Calculate time since last training transaction
                    time_diff = (row['purchase_time'] - last_time_dict[row[col]]).total_seconds()

                    if time_diff < 0:
                        # Test transaction occurs BEFORE last training transaction
                        # This indicates random split rather than temporal split
                        # Treat as unseen pattern (temporal anomaly)
                        return 999999
                    else:
                        return time_diff
                else:
                    # New device/IP not seen in training
                    return 999999

            df['time_since_last_device_id_txn'] = df.apply(
                lambda row: time_since_training(row, 'device_id', self.device_last_purchase_time),
                axis=1
            )
            df['time_since_last_ip_address_txn'] = df.apply(
                lambda row: time_since_training(row, 'ip_address', self.ip_last_purchase_time),
                axis=1
            )

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
    
    def _add_geo_demographic_features(self, df, is_training):
        """
        GDP mapping and demographic features

        FIXED: Data leakage prevention
        - Training: Calculate GDP ranks from training data distribution
        - Testing: Use GDP rank mapping from training data

        Why there was leakage:
        Previously calculated GDP ranks using rank() on the entire dataframe. The rank
        depends on which countries appear in the dataset and their distribution. If
        test data has different country distribution, the ranks would be different,
        causing leakage. Also, the max_rank used for normalization would differ.
        """

        df["gdp"] = df["country"].apply(
            lambda x: self.gdp_data.get(
                x, self.gdp_data.get(self.country_map.get(x))) if x in self.gdp_data or x in self.country_map else np.nan
        )

        # Fill missing GDP values with median
        gdp_median = df["gdp"].median()
        df["gdp"].fillna(gdp_median, inplace=True)

        if is_training:
            # Calculate GDP ranks from training data
            df["gdp_rank"] = df["gdp"].rank(method="dense", ascending=False).astype(int)

            # Store GDP to rank mapping for test set
            self.gdp_rank_map = df.groupby("gdp")["gdp_rank"].first().to_dict()
            self.max_gdp_rank = df["gdp_rank"].max()
            print(f"GDP rank map: {self.gdp_rank_map}")

            # Normalized GDP rank (0-1 scale)
            df["gdp_rank_normalized"] = 1 - ((df["gdp_rank"] - 1) / self.max_gdp_rank)
        else:
            # Apply GDP rank mapping from training
            df["gdp_rank"] = df["gdp"].map(self.gdp_rank_map)

            # For GDP values not seen in training, assign a default rank
            # Use max_rank + 1 for unseen GDP values (lowest rank)
            df["gdp_rank"].fillna(self.max_gdp_rank + 1, inplace=True)
            df["gdp_rank"] = df["gdp_rank"].astype(int)

            # Normalized GDP rank using training max_rank
            df["gdp_rank_normalized"] = 1 - ((df["gdp_rank"] - 1) / self.max_gdp_rank)
            # Clip to [0, 1] range in case of unseen GDP values
            df["gdp_rank_normalized"] = df["gdp_rank_normalized"].clip(0, 1)

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
    
    def _add_clustering_features(self, df, is_training):
        """
        K-means clustering based on demographics and purchase behavior

        FIXED: Data leakage prevention
        - Training: Fit StandardScaler and KMeans on training data
        - Testing: Use fitted scaler and models from training

        Why there was leakage:
        Previously called fit_transform() and fit_predict() on the entire dataframe.
        This means the scaler's mean/std and the cluster centroids would be influenced
        by test data. The scaler would normalize using test statistics, and clusters
        would be formed considering test data distribution.
        """
        # Select features for clustering
        cluster_features = ['sex_encoded', 'age', 'purchase_value', 'gdp_rank']

        X_cluster = df[cluster_features].copy()

        # Handle any remaining missing values
        X_cluster = X_cluster.fillna(X_cluster.median())

        if is_training:
            # Fit scaler on training data
            self.clustering_scaler = StandardScaler()
            X_scaled = self.clustering_scaler.fit_transform(X_cluster)

            # Fit K-means with multiple cluster counts on training data
            for n_clusters in [3, 5, 8]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_scaled)
                # Store the fitted model for test set
                self.kmeans_models[n_clusters] = kmeans
        else:
            # Use fitted scaler from training
            X_scaled = self.clustering_scaler.transform(X_cluster)

            # Apply fitted K-means models from training
            for n_clusters in [3, 5, 8]:
                df[f'cluster_{n_clusters}'] = self.kmeans_models[n_clusters].predict(X_scaled)

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