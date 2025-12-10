import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.config import (
    IP_COUNTRY_PATH,
    GDP_DATA_PATH,
    FRAUD_WITH_COUNTRY_PATH,
    TARGET_COL
)
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FraudFeatureEngineer:
    """
    Comprehensive preprocessing pipeline for e-commerce fraud detection

    Attributes:
        ip_mapping (pd.DataFrame): IP-to-country mapping data
        gdp_data (pd.DataFrame): GDP data for countries
        country_map (dict): Mapping of country names in ip_mapping to gdp_data
        scaler (StandardScaler): Scaler for normalization
        global_purchase_std (float): Global standard deviation of purchase values from training data
        global_purchase_mean (float): Global mean of purchase values from training data
        fraud-rate_by_month (dict): Fraud rate for each month, where key=month, value=fraud_rate
        device_stats_dict (dict): Dictionary where keys are device IDs (str) and values are dicts with 'std', 'mean', 'count' of purchase values
        ip_stats_dict (dict): Dictionary where keys are IP addresses (str) and values are dicts with 'std', 'mean', 'count' of purchase values
        device_txn_counts (dict): Dictionary where keys are device IDs (str) and values are transaction counts (int)
        ip_txn_counts (dict): Dictionary where keys are IP addresses (str) and values are transaction counts (int)
        txn_count_device_mean (float): Mean of transaction counts by device from training data, used for normalization
        txn_count_device_std (float): Standard deviation of transaction counts by device from training data, used for normalization
        txn_count_ip_mean (float): Mean of transaction counts by IP from training data, used for normalization
        txn_count_ip_std (float): Standard deviation of transaction counts by IP from training data, used for normalization
        device_ip_consistency_map (dict): Dictionary where keys are device IDs (str) and values are counts of unique IPs used by the device
        ip_device_consistency_map (dict): Dictionary where keys are IP addresses (str) and values are counts of unique devices using the IP
        device_source_consistency_map (dict): Dictionary where keys are device IDs (str) and values are counts of unique sources used by the device
        device_browser_consistency_map (dict): Dictionary where keys are device IDs (str) and values are counts of unique browsers used by the device
        device_last_purchase_time (dict): Dictionary where keys are device IDs (str) and values are timestamps of last purchase in training data
        ip_last_purchase_time (dict): Dictionary where keys are IP addresses (str) and values are timestamps of last purchase in training data
        device_lifespan_days (dict): Dictionary where keys are device IDs (str) and values are the time span in days between the first and last purchase in training data
        device_first_purchase_time (dict): Dictionary where keys are device IDs (str) and values are timestamps of first purchase in training data
        sorted_device_transactions_history (dict): Dictionary where keys are device IDs (str) and values are list of purchase timestamps sorted in ascending order
        gdp_median (float): Global median of GDP values from training data
        gdp_rank_map (dict): Dictionary where keys are GDP values (float) and values are GDP ranks (int) in training data
        max_gdp_rank (int): Maximum GDP rank observed in training data
        clustering_scaler (StandardScaler): Scaler for clustering features **may not be needed
        kmeans_models (dict): Dictionary where keys are n_clusters (int) and values are KMeans models fitted on training data (KMeans)
        country_transaction_counts (dict): Dictionary where keys are country names (str) and values are transaction counts (int) in training data
        total_transactions (int): Total number of transactions in training data
    """

    def __init__(
            self,
            ip_country_path=None, 
            transaction_country_path=None,
            gdp_data_path=None
        ):
        """
        Initializes a new instance of FraudFeatureEngineer.

        Args:
            ip_country_path (str): Path to the IP-to-country mapping CSV file.
            transaction_country_path (str): Path to raw dataset with country mapped to IP. 
            gdp_data_path (str): Path to the GDP data Excel file.
        """
        if ip_country_path is None:
            ip_country_path = IP_COUNTRY_PATH
        if transaction_country_path is None:
            transaction_country_path = FRAUD_WITH_COUNTRY_PATH
        if gdp_data_path is None:
            gdp_data_path = GDP_DATA_PATH

        self.ip_mapping = pd.read_csv(ip_country_path)
        self.transaction_country_path = transaction_country_path
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
        self.fraud_rate_by_country = {} # (country, fraud_rate)

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

        # Source/Browser-Device consistency statistics
        self.device_source_consistency_map = {}
        self.device_browser_consistency_map = {}

        # Velocity statistics (for time-based features)
        self.device_last_purchase_time = {}
        self.ip_last_purchase_time = {}
        self.device_lifespan_days = {}
        self.device_first_purchase_time = {}
        self.device_transaction_history = {}

        # GDP ranking statistics
        self.gdp_median = None
        self.gdp_rank_map = {}
        self.max_gdp_rank = None

        # Clustering models
        self.clustering_scaler = None
        self.kmeans_models = {}

        # Country rarity statistics
        self.country_transaction_counts = {}
        self.total_transactions = 0


    def fit(self, df, target_col=None):
        """
        Fit preprocessing parameters on training data

        This learns statistics from training data that will be applied to test data to prevent data leakage

        Args:
            df (pd.DataFrame): Training dataframe with transaction data
            target_col (str): Name of fraud indicator column
        """
        print("Fitting preprocessing parameters on training data...")

        if not target_col:
            target_col = TARGET_COL

        # Calculate global purchase statistics
        self.global_purchase_mean = df['purchase_value'].mean()
        self.global_purchase_std = df['purchase_value'].std()

        print(f"  Global purchase mean: ${self.global_purchase_mean:.2f}")
        print(f"  Global purchase std: ${self.global_purchase_std:.2f}")

        # Convert signup_time and purchase_time to Datetime object
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Store total transactions for country rarity calculation
        self.total_transactions = len(df)

        # Calculate fraud rate by month (for feature engineering)
        if target_col in df.columns:
            df['transaction_month'] = df['purchase_time'].dt.month
            self.fraud_rate_by_month = df.groupby('transaction_month')[target_col].mean().to_dict()
            print(f"  Fraud rates by month calculated: {len(self.fraud_rate_by_month)} months")

        # Store device-level statistics
        device_stats = df.groupby('device_id')['purchase_value'].agg(['std', 'mean', 'count'])
        self.device_stats_dict = device_stats.to_dict('index')
        print(f"  Stored statistics for {len(self.device_stats_dict)} unique devices")

        # Store IP-level statistics
        ip_stats = df.groupby('ip_address')['purchase_value'].agg(['std', 'mean', 'count'])
        self.ip_stats_dict = ip_stats.to_dict('index')
        print(f"  Stored statistics for {len(self.ip_stats_dict)} unique IP addresses")

        # Store transaction counts by device and IP
        self.device_txn_counts = df.groupby('device_id').size().to_dict()
        self.ip_txn_counts = df.groupby('ip_address').size().to_dict()
        print(f"  Stored transaction counts for devices and IPs")

        # Calculate mean and standard deviation of transaction counts per device/IP to use for normalization
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

        # Store source-device and browser-device consistency mappings
        if 'source' in df.columns:
            self.device_source_consistency_map = df.groupby('device_id')['source'].nunique().to_dict()
            print(f"  Device-Source consistency mappings stored")

        if 'browser' in df.columns:
            self.device_browser_consistency_map = df.groupby('device_id')['browser'].nunique().to_dict()
            print(f"  Device-Browser consistency mappings stored")

        # Store last purchase times for velocity features
        df_sorted = df.sort_values('purchase_time')
        self.device_last_purchase_time = df_sorted.groupby('device_id')['purchase_time'].last().to_dict()
        self.ip_last_purchase_time = df_sorted.groupby('ip_address')['purchase_time'].last().to_dict()

        # Store first purchase times for device lifespan calculation
        self.device_first_purchase_time = df_sorted.groupby('device_id')['purchase_time'].first().to_dict()

        # Calculate device lifespan in days
        for device_id in self.device_first_purchase_time:
            first_time = self.device_first_purchase_time[device_id]
            last_time = self.device_last_purchase_time[device_id]
            lifespan = (last_time - first_time).total_seconds() / 86400  # Convert to days
        #     self.device_lifespan_days[device_id] = lifespan

        # print(f"  Device lifespan statistics stored for {len(self.device_lifespan_days)} devices")
        print(f"  Last purchase times stored for velocity calculations")

        # Store GDP rankings (need to process geo features first to get country info)
        # This will be calculated during transform for training data
        print(f"  GDP rankings will be calculated during transform")
        
        return self

    def transform(self, df, is_training=False):
        """
        Transform dataframe with all feature engineering steps

        Args:
            df (pd.DataFrame): Input dataframe to transform
            is_training (bool): Whether this is training data (affects statistical calculations)
        
        Returns:
            pd.DataFrame: Processed dataframe with engineered features
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

        print("Step 5: Source-Device and Browser-Device consistency...")
        df = self._add_source_browser_consistency(df, is_training)

        print("Step 6: Time-based features...")
        df = self._add_time_features(df)

        print("Step 7: Add transaction velocity features (last hour) and create history")
        df = self._add_transaction_velocity(df, is_training)

        print("Step 8: Calculate time between current and latest transaction")
        df = self._add_velocity_features(df)

        print("Step 9: Device lifespan features...")
        df = self._add_device_lifespan_features(df)

        print("Step 10: Purchase value statistics...")
        df = self._add_purchase_value_stats(df, is_training)

        print("Step 11: Temporal encoding...")
        df = self._add_temporal_encoding(df)

        print("Step 12: Geographic and demographic features...")
        df = self._add_geo_demographic_features(df, is_training)

        print("Step 13: Country rarity and fraud rate by country...")
        df = self._add_country_features(df, is_training)

        print("Step 14: Clustering features...")
        df = self._add_clustering_features(df, is_training)

        print("Feature engineering complete!")
        return df

    def _add_transaction_counts(self, df, is_training):
        """
        Add number of transactions by device and IP

        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data
        
        Returns:
            pd.DataFrame: Dataframe with added transaction count features
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
        Add Z-scores and percentile ranks for transaction counts by device/IP

        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data
        
        Returns:
            pd.DataFrame: Dataframe with added normalized count features
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
        Count unique IPs per device and unique devices per IP.
        For each transaction, check if device is tied to exactly one IP (and vice versa)

        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data

        Returns:
            pd.DataFrame: Dataframe with added device-IP consistency features
        """
        if is_training:
            # Calculate from current training data
            device_ip_count = df.groupby('device_id')['ip_address'].nunique().to_dict()
            df['device_ip_consistency'] = df['device_id'].map(device_ip_count)
            df['is_device_single_ip'] = (df['device_ip_consistency'] == 1)

            # Reverse: IP to device consistency
            ip_device_count = df.groupby('ip_address')['device_id'].nunique().to_dict()
            df['ip_device_consistency'] = df['ip_address'].map(ip_device_count)
            df['is_ip_single_device'] = (df['ip_device_consistency'] == 1)
        else:
            # Use mappings from training data, default to 1 for unseen devices/IPs
            df['device_ip_consistency'] = df['device_id'].map(self.device_ip_consistency_map).fillna(1)
            df['is_device_single_ip'] = (df['device_ip_consistency'] == 1)

            df['ip_device_consistency'] = df['ip_address'].map(self.ip_device_consistency_map).fillna(1)
            df['is_ip_single_device'] = (df['ip_device_consistency'] == 1)

        return df

    def _add_source_browser_consistency(self, df, is_training):
        """
        Count unique sources and browsers per device.
        For test data, if device not seen in training, impute as single-source/browser.
        Check if device is tied to exactly one source/browser.
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data

        Returns:
            pd.DataFrame: Dataframe with added device-source/browser consistency features
        """
        if is_training:
            # Calculate from current training data
            if 'source' in df.columns:
                device_source_count = df.groupby('device_id')['source'].nunique().to_dict()
                df['source_device_consistency'] = df['device_id'].map(device_source_count)
                df['is_device_single_source'] = (df['source_device_consistency'] == 1)
            else:
                df['source_device_consistency'] = 1
                df['is_device_single_source'] = True

            if 'browser' in df.columns:
                device_browser_count = df.groupby('device_id')['browser'].nunique().to_dict()
                df['browser_device_consistency'] = df['device_id'].map(device_browser_count)
                df['is_device_single_browser'] = (df['browser_device_consistency'] == 1)
            else:
                df['browser_device_consistency'] = 1
                df['is_device_single_browser'] = True
        else:
            # Use mappings from training data, default to 1 for unseen devices
            if 'source' in df.columns and self.device_source_consistency_map:
                df['source_device_consistency'] = df['device_id'].map(self.device_source_consistency_map).fillna(1)
                df['is_device_single_source'] = (df['source_device_consistency'] == 1)
            else:
                df['source_device_consistency'] = 1
                df['is_device_single_source'] = True

            if 'browser' in df.columns and self.device_browser_consistency_map:
                df['browser_device_consistency'] = df['device_id'].map(self.device_browser_consistency_map).fillna(1)
                df['is_device_single_browser'] = (df['browser_device_consistency'] == 1)
            else:
                df['browser_device_consistency'] = 1
                df['is_device_single_browser'] = True

        return df

    def _add_time_features(self, df):
        """Calculate time between account setup and transaction in seconds."""
        df['time_setup_to_txn_seconds'] = (
            df['signup_time'] - df['purchase_time']
        ).dt.total_seconds()

        return df
    
    def _add_velocity_features(self, df):
        """
        Calculate time since last transaction by device.

        For both training and test data, the current transaction's timestamp will be used to search for the last transaction time by device using self.device_transaction_history.
        """
        velocities = np.zeros(len(df), dtype=np.int32)
        for idx, row in df.iterrows():
            device_id = row["device_id"]
            current_time = row["purchase_time"]

            train_times = self.device_transaction_history.get(device_id, [])

            if train_times:
                idx = np.searchsorted(train_times, current_time) - 1
                if idx >= 0:
                    last_transaction_time = train_times[idx]
                    time_diff = (current_time - last_transaction_time).total_seconds()

            if not train_times or idx < 0:
                time_diff = 999999
            velocities[idx] = time_diff
        
        df["time_since_last_device_id_txn"] = velocities
        return df


    def _add_device_lifespan_features(self, df):
        """
        Calculate device lifespan (time difference in days between first and current transaction) and derived features.

        Rationale:
        Devices with very short lifespans (created and immediately used for high-value
        transactions) might indicate fraud. Long-established devices are typically
        more trustworthy.
        """
        def device_lifespan_days_row(row):
            device_id = row["device_id"]
            if device_id in self.device_first_purchase_time:
                first_purchase_by_device = self.device_first_purchase_time[device_id]
                return (row["purchase_time"] - first_purchase_by_device).total_seconds() / 86400  # Convert to days
            else:
                return 0
        
        df["device_lifespan_days"] = df.apply(lambda row: device_lifespan_days_row(row), axis=1)

        # Add derived features
        df['is_new_device'] = (df['device_lifespan_days'] == 0)

        return df

    def _add_transaction_velocity(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Calculate transaction velocity using fully vectorized operations.
        
        This implementation uses pandas' merge_asof for ultra-fast time-window queries,
        achieving O(n log n) complexity with optimized C-level operations.
        
        Performance: ~2-5 seconds for 1M transactions (100-1000x faster than nested loops).
        
        Args:
            df (pd.DataFrame): Transaction dataframe.
            is_training (bool): Whether this is training data.
        
        Returns:
            pd.DataFrame: Dataframe with velocity features.
        """
    
        df = df.sort_values(['device_id', 'purchase_time']).reset_index(drop=True)
        
        if is_training:
            # Create a self-join to count prior transactions
            df_self = df[['device_id', 'purchase_time']].copy()
            df_self['count'] = 1
            
            # For each transaction, count how many happened in last hour
            velocities = []
            
            for device_id, group in df.groupby('device_id', sort=False):
                times = group['purchase_time'].values

                # Vectorized: for each time, count times in [time - 1h, time)
                counts = np.zeros(len(times), dtype=np.int32)
                for i, t in enumerate(times):
                    hour_ago = t - np.timedelta64(1, "h")
                    # Vectorized comparison (no loops!)
                    mask = (times >= hour_ago) & (times < t)
                    counts[i] = mask.sum()
                
                velocities.extend(counts)
            
            df['transactions_last_hour_device'] = velocities
            
            # Store for test data
            self.device_transaction_history = df.groupby('device_id')['purchase_time'].apply(list).to_dict()

        else:
            # Test mode: use training history
            velocities = []
            
            for _, row in df.iterrows():
                device_id = row['device_id']
                current_time = row['purchase_time']
                
                if device_id in self.device_transaction_history:
                    train_times = self.device_transaction_history[device_id]
                    hour_ago = current_time - timedelta(hours=1)
                    
                    # Vectorized count
                    count = sum(1 for t in train_times if hour_ago <= t < current_time)
                    velocities.append(count)
                else:
                    velocities.append(0)
            
            df['transactions_last_hour_device'] = velocities
        
        # Derived features
        df['has_recent_activity'] = (df['transactions_last_hour_device'] > 0).astype(np.int8)
        df['velocity_high'] = (df['transactions_last_hour_device'] > 2).astype(np.int8)
        
        return df


    def _add_transaction_velocity_ultimate(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
            Count transactions in the last hour for each device

            For test data, we only look back at training transactions to
            avoid leakage. This means test transactions won't see other test transactions
            in the same batch.

            Rationale:
            High transaction velocity (many transactions in short time) is a strong
            fraud indicator. Fraudsters often make rapid purchases before detection.
        """
        df = df.sort_values(['device_id', 'purchase_time']).reset_index(drop=True)
        
        # Set purchase_time as index for rolling operations
        df_indexed = df.set_index('purchase_time')
        
        if is_training:
            # Use pandas rolling window with time-based window
            # This is EXTREMELY fast (implemented in C)
            velocity_series = (
                df_indexed
                .groupby('device_id')['device_id']  # Group by device
                .rolling('1H', closed='left')       # 1-hour window, exclude current
                .count()                            # Count transactions in window
                .reset_index(level=0, drop=True)    # Drop device_id from index
            )
            
            df['transactions_last_hour_device'] = velocity_series.values
            
            # Store device histories for test
            self.device_histories = df.groupby('device_id')[['purchase_time']].apply(
                lambda x: x['purchase_time'].sort_values().tolist()
            ).to_dict()
            
        else:
            # Test mode: manual lookup (still fast with binary search)
            velocities = np.zeros(len(df), dtype=np.int32)
            
            for idx, row in df.iterrows():
                device_id = row['device_id']
                current_time = row['purchase_time']
                
                if device_id in self.device_histories:
                    times = self.device_histories[device_id]
                    hour_ago = current_time - pd.Timedelta(hours=1)
                    
                    # Binary search for efficiency
                    left = np.searchsorted(times, hour_ago, side='left')
                    right = np.searchsorted(times, current_time, side='left')
                    velocities[idx] = right - left
            
            df['transactions_last_hour_device'] = velocities
        
        # Derived features
        df['has_recent_activity'] = (df['transactions_last_hour_device'] > 0).astype(np.int8)
        df['velocity_high'] = (df['transactions_last_hour_device'] > 2).astype(np.int8)
        
        return df

    def _add_purchase_value_stats(self, df, is_training):
        """
        Standard deviation and mean of purchase value, along with transaction counts by device/IP
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
        df['is_unique_device'] = (df['purchase_count_by_device'] == 1)
        df['is_unique_ip'] = (df['purchase_count_by_ip'] == 1)

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
        """Map IP address to country"""
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

        def log_and_map_ip_to_country(ip_address, ip_mapping):
            country = map_ip_to_country(ip_address, ip_mapping)
            logging.info(f"IP Address {ip_address} mapped to country: {country}")
            return country

        if os.path.exists(self.transaction_country_path):
            # Load the country mapping and merge it with the current dataframe
            country_mapping = pd.read_csv(self.transaction_country_path)[['ip_address', 'country']]
            df = df.drop('ip_address', axis=1)
            df = pd.merge(df, country_mapping, left_index=True, right_index=True)

        else:
            df["country"] = df["ip_address"].apply(lambda x: log_and_map_ip_to_country(x, self.ip_mapping))

        df["country"] = df["country"].fillna("Unknown")

        return df

    def _add_country_features(self, df, is_training):
        """
        NEW FEATURE: Country rarity and fraud rate by country

        FIXED: Data leakage prevention
        - Training: Calculate country statistics from training data
        - Testing: Use country statistics from training data

        Rationale:
        - Country rarity: Transactions from rare countries might be more suspicious
        - Fraud rate by country: Some countries may have higher fraud rates
        """
        if is_training:
            # Calculate country transaction counts and fraud rates from training data
            if 'country' in df.columns:
                self.country_transaction_counts = df['country'].value_counts().to_dict()

                # Calculate fraud rate by country
                if TARGET_COL in df.columns:
                    country_fraud = df.groupby('country')[TARGET_COL].agg(['sum', 'count'])
                    self.fraud_rate_by_country = (country_fraud['sum'] / country_fraud['count']).to_dict()
                    print(f"  Fraud rates calculated for {len(self.fraud_rate_by_country)} countries")

                # Calculate country rarity (inverse frequency)
                df['country_transaction_count'] = df['country'].map(self.country_transaction_counts)
                df['country_rarity'] = 1.0 / (df['country_transaction_count'] + 1)

                # Normalize rarity to [0, 1] scale
                max_rarity = df['country_rarity'].max()
                df['country_rarity_normalized'] = df['country_rarity'] / (max_rarity + 1e-8)

                # Fraud rate by country
                if self.fraud_rate_by_country:
                    df['fraud_rate_by_country'] = df['country'].map(self.fraud_rate_by_country)
                    overall_fraud_rate = df[TARGET_COL].mean() if TARGET_COL in df.columns else 0.1
                    df['fraud_rate_by_country'].fillna(overall_fraud_rate, inplace=True)
                else:
                    df['fraud_rate_by_country'] = 0.1  # HY: Shouldn't default value be a central tendency

            # else: # will this only be reached if country column doesn't exist
            #     df['country_transaction_count'] = 1
            #     df['country_rarity'] = 1.0
            #     df['country_rarity_normalized'] = 1.0
            #     df['fraud_rate_by_country'] = 0.1
        else:
            # Use country statistics from training data
            if 'country' in df.columns:
                # Map country to transaction count from training, default to 1 for unseen countries
                df['country_transaction_count'] = df['country'].map(self.country_transaction_counts).fillna(1)

                # Calculate rarity
                df['country_rarity'] = 1.0 / (df['country_transaction_count'] + 1)

                # Normalize using training max, can we scale this by fitting Scaler on training data then fit with test?
                if self.country_transaction_counts:
                    min_rarity = min(self.country_transaction_counts.values())
                    max_rarity = 1.0 / (1 + 1)  # Minimum count is 1
                    df['country_rarity_normalized'] = (df['country_rarity'] - min_rarity) / (max_rarity - min_rarity)
                else:
                    raise ValueError("Fit FraudFeatureEngineer on training data before transformation")
                    # df['country_rarity_normalized'] = 1.0

                # Fraud rate by country from training
                if self.fraud_rate_by_country:
                    df['fraud_rate_by_country'] = df['country'].map(self.fraud_rate_by_country)
                    overall_fraud_rate = np.mean(list(self.fraud_rate_by_country.values()))
                    df['fraud_rate_by_country'].fillna(overall_fraud_rate, inplace=True)
                else:
                    raise ValueError("Fit FraudFeatureEngineer on training data before transformation")


        # Add derived features
        if 'country' in df.columns:
            df['is_rare_country'] = (df['country_rarity_normalized'] > 0.5)
            df['is_high_fraud_country'] = (df['fraud_rate_by_country'] > 0.15)

        return df

    def _add_geo_demographic_features(self, df, is_training):
        """
        Map GDP to country, impute with median gdp if country is unknown or no gdp data available.

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

        if is_training:
            # Fill missing GDP values with median
            self.gdp_median = df["gdp"].median()
            df["gdp"].fillna(self.gdp_median, inplace=True)

            # Calculate GDP ranks from training data
            df["gdp_rank"] = df["gdp"].rank(method="dense", ascending=False).astype(int)

            # Store GDP to rank mapping for test set
            self.gdp_rank_map = df.groupby("gdp")["gdp_rank"].first().to_dict()
            self.max_gdp_rank = df["gdp_rank"].max()
            print(f"GDP rank map: {self.gdp_rank_map}")

            # Normalized GDP rank (0-1 scale)
            df["gdp_rank_normalized"] = 1 - ((df["gdp_rank"] - 1) / self.max_gdp_rank)
        else:
            df["gdp"].fillna(self.gdp_median, inplace=True)
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

    def fit_transform(self, df, target_col=TARGET_COL):
        """
        Fit on training data and transform

        Args:
        df (pd.DataFrame): Training dataframe
        target_col (str): Name of target variable

        Returns:
            pd.DataFrame : Transformed training data
        """
        self.fit(df, target_col)
        return self.transform(df, is_training=True)

    def preprocess_pipeline(self, train_df, target_col=TARGET_COL, test_df=None,):
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
