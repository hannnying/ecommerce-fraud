from datetime import datetime
import json
from pathlib import Path
import pickle
import pandas as pd
from typing import Dict, List
from redis import Redis
from src.config import(
   REDIS_DB,
   REDIS_HOST,
   REDIS_PORT
)


class DeviceState:

    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    def update_device_state(
            self,
            device_id,
            txn_count,
            purchase_value,
            sex,
            age,
            first_seen,
            ip_addresses,
            sources,
            purchase_time,
            source,
            ip_address
    ):
        """Update device state."""

        key = f"device:{device_id}"
        
        ip_addresses.add(ip_address)
        sources.add(source)

        updated_state = {
            "txn_count": txn_count + 1,
            "prev_purchase_value": purchase_value,
            "prev_sex": sex,
            "prev_age": age,
            "last_transaction": purchase_time,
            "last_ip": ip_address,
            "first_seen": purchase_time if not first_seen else first_seen,
            "ip_addresses": ip_addresses,
            "sources": sources,
        }

        self.client.hset(
            key, 
            mapping=self.serialize_device_state(
                updated_state["txn_count"], 
                updated_state["prev_purchase_value"],
                updated_state["prev_sex"],
                updated_state["prev_age"],
                updated_state["last_transaction"],
                updated_state["last_ip"],
                updated_state["first_seen"],
                updated_state["ip_addresses"],
                updated_state["sources"]
            )
        )
        

    def update_device_timestamp(self, device_id, transaction_id, purchase_time):
        key = f"device:{device_id}:txn_timestamp"
        
        self.client.zadd(key, mapping={transaction_id: purchase_time.timestamp()})

        # remove timestamps > 24 hours
        timestamp_threshold = (purchase_time - pd.Timedelta(days=1)).timestamp()
        self.client.zremrangebyscore(key, float("-inf"), timestamp_threshold)

    
    def serialize_device_state(self, txn_count, prev_purchase_value, prev_sex, prev_age, last_transaction, last_ip, first_seen, ip_addresses, sources):
        if type(last_transaction) != str:
            last_transaction = last_transaction.isoformat()

        if type(first_seen) != str:
            first_seen = first_seen.isoformat()

        return {
            "txn_count": txn_count,
            "prev_purchase_value": float(prev_purchase_value),
            "prev_sex": str(prev_sex),
            "prev_age": int(prev_age),
            "last_transaction": last_transaction,
            "last_ip": float(last_ip),
            "first_seen": first_seen,
            "ip_addresss": json.dumps(list(ip_addresses)),
            "sources": json.dumps(list(sources)),
        }
    

    def deserialize_device_state(self, raw):
        txn_count = int(raw[0]) if raw[0] else 0
        prev_purchase_value = float(raw[1]) if raw[1] else None
        prev_sex = str(raw[2]) if raw[1] else ""
        prev_age = int(raw[3]) if raw[1] else None
        last_transaction = datetime.fromisoformat(raw[4]) if raw[4] else None
        last_ip = float(raw[5]) if raw[5] else None
        first_seen = datetime.fromisoformat(raw[6]) if raw[6] else None
        ip_addresses = set(json.loads(raw[7])) if raw[7] else set()
        sources = set(json.loads(raw[8])) if raw[8] else set()

        return txn_count, prev_purchase_value, prev_sex, prev_age, last_transaction, last_ip, first_seen, ip_addresses, sources

        
    def get_device_state(self, device_id):
        key = f"device:{device_id}"

        raw = self.client.hmget(
            key,
            "txn_count",
            "prev_purchase_value",
            "prev_sex",
            "prev_age",
            "last_transaction",
            "last_ip",
            "first_seen",
            "ip_addresses",
            "sources"
        )

        return self.deserialize_device_state(raw)
    

    def get_device_txn_velocity(self, device_id, purchase_time, time_window):
        key = f"device:{device_id}:txn_timestamp"

        if time_window == "1m":
            timestamp_threshold = purchase_time - pd.Timedelta(1, "m")
        if time_window == "5m":
            timestamp_threshold = purchase_time - pd.Timedelta(5, "m")
        if time_window == "1h":
            timestamp_threshold = purchase_time - pd.Timedelta(1, "h")
        if time_window == "24h":
            timestamp_threshold = purchase_time - pd.Timedelta(1, "d")

        min_score = timestamp_threshold.timestamp()
        max_score = purchase_time.timestamp()

        return self.client.zcount(key, min_score, max_score)
        
    def count_devices(self) -> int:
        """Count total number of devices stored in Redis."""
        count = 0
        for _ in self.client.scan_iter("device:*", count=1000):
            # Don't count timestamp sorted sets, only device hashes
            if ":txn_timestamp" not in _:
                count += 1
        return count

    def export_to_file(self, filepath: str, export_timestamps: bool = True) -> None:
        """
        Export device state from Redis to pickle file(s).

        Creates separate files for device hashes and timestamps for efficiency:
        - Device hashes are always exported to the specified filepath
        - Timestamps are exported to a separate file (optional)

        Args:
            filepath: Path to save device hashes (e.g., 'models/device_state.pkl')
            export_timestamps: Whether to also export device timestamp sorted sets
                                to a separate file (default True)

        Returns:
            None

        Example:
            device_state = DeviceState()
            device_state.export_to_file('models/device_state.pkl', export_timestamps=True)
            # Creates: models/device_state.pkl and models/device_timestamps.pkl
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Export device hashes
        print(f"Exporting device hashes to {filepath}...")
        device_hashes = {}
        device_count = 0

        for key in self.client.scan_iter("device:*", count=1000):
            key_str = key if isinstance(key, str) else key.decode('utf-8')

            # Skip timestamp sorted sets
            if ":txn_timestamp" in key_str:
                continue

            device_id = key_str.split(":")[1]
            device_data = self.client.hgetall(key_str)
            device_hashes[device_id] = device_data
            device_count += 1

            if device_count % 10000 == 0:
                print(f"  Exported {device_count} device hashes...")

        # Save device hashes
        export_data = {
            "device_hashes": device_hashes,
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "device_count": device_count,
                "type": "device_hashes"
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported {device_count} device hashes ({file_size_mb:.2f} MB)")

        # Export device timestamps to separate file
        if export_timestamps:
            timestamp_filepath = filepath.parent / f"{filepath.stem}_timestamps{filepath.suffix}"
            print(f"Exporting device timestamps to {timestamp_filepath}...")

            device_timestamps = {}
            timestamp_count = 0

            for key in self.client.scan_iter("device:*:txn_timestamp", count=1000):
                key_str = key if isinstance(key, str) else key.decode('utf-8')
                device_id = key_str.split(":")[1]

                # Get all transaction timestamps for this device (last 24h)
                timestamps = self.client.zrange(key_str, 0, -1, withscores=True)
                if timestamps:  # Only store if device has timestamps
                    device_timestamps[device_id] = timestamps
                    timestamp_count += 1

                if timestamp_count % 10000 == 0:
                    print(f"  Exported {timestamp_count} timestamp sets...")

            # Save timestamps
            timestamp_export_data = {
                "device_timestamps": device_timestamps,
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "timestamp_count": timestamp_count,
                    "type": "device_timestamps"
                }
            }

            with open(timestamp_filepath, 'wb') as f:
                pickle.dump(timestamp_export_data, f)

            timestamp_size_mb = timestamp_filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ Exported {timestamp_count} timestamp sets ({timestamp_size_mb:.2f} MB)")

        print(f"✓ Device state export complete")

    @classmethod
    def load_from_file(cls, filepath: str, load_timestamps: bool = True, flush_existing: bool = False) -> 'DeviceState':
        """
        Load device state from pickle file(s) and restore to Redis.

        Loads device hashes from the specified filepath and optionally loads
        timestamps from the companion timestamp file.

        Args:
            filepath: Path to device hashes file (e.g., 'models/device_state.pkl')
            load_timestamps: Whether to also load device timestamps from companion file
                           (default True - loads from 'models/device_timestamps.pkl')
            flush_existing: Whether to flush existing device state in Redis before loading
                          (default True - ensures clean state)

        Returns:
            DeviceState instance with loaded state

        Example:
            # Load only device hashes (fast startup)
            device_state = DeviceState.load_from_file('models/device_state.pkl', load_timestamps=False)

            # Load device hashes and timestamps (complete state)
            device_state = DeviceState.load_from_file('models/device_state.pkl', load_timestamps=True)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")

        print(f"Loading device state from {filepath}...")

        # Create instance
        instance = cls()

        # Flush existing device state if requested
        if flush_existing:
            print("  Flushing existing device state from Redis...")
            deleted_count = 0
            for key in instance.client.scan_iter("device:*", count=1000):
                instance.client.delete(key)
                deleted_count += 1
            if deleted_count > 0:
                print(f"  Deleted {deleted_count} existing device keys")

        # Load device hashes
        with open(filepath, 'rb') as f:
            export_data = pickle.load(f)

        device_hashes = export_data.get("device_hashes", {})
        device_count = 0

        for device_id, device_data in device_hashes.items():
            key = f"device:{device_id}"
            instance.client.hset(key, mapping=device_data)
            device_count += 1

            if device_count % 10000 == 0:
                print(f"  Loaded {device_count} device hashes...")

        metadata = export_data.get("metadata", {})
        export_time = metadata.get("export_time", "unknown")
        print(f"  ✓ Loaded {device_count} device hashes (exported at: {export_time})")

        # Load device timestamps from separate file if requested
        if load_timestamps:
            timestamp_filepath = filepath.parent / f"{filepath.stem}_timestamps{filepath.suffix}"

            if timestamp_filepath.exists():
                print(f"Loading device timestamps from {timestamp_filepath}...")

                with open(timestamp_filepath, 'rb') as f:
                    timestamp_export_data = pickle.load(f)

                device_timestamps = timestamp_export_data.get("device_timestamps", {})
                timestamp_count = 0

                for device_id, timestamps in device_timestamps.items():
                    key = f"device:{device_id}:txn_timestamp"

                    # timestamps is a list of (transaction_id, score) tuples
                    if timestamps:
                        mapping = {txn_id: score for txn_id, score in timestamps}
                        instance.client.zadd(key, mapping=mapping)
                        timestamp_count += 1

                    if timestamp_count % 10000 == 0:
                        print(f"  Loaded {timestamp_count} timestamp sets...")

                timestamp_metadata = timestamp_export_data.get("metadata", {})
                timestamp_export_time = timestamp_metadata.get("export_time", "unknown")
                print(f"  ✓ Loaded {timestamp_count} timestamp sets (exported at: {timestamp_export_time})")
            else:
                print(f"  ⚠ Timestamp file not found: {timestamp_filepath}")
                print(f"  Continuing without timestamps (velocity features will start fresh)")

        print(f"✓ Device state loaded successfully")

        return instance

    def clear_all_devices(self) -> int:
        """
        Clear all device state from Redis.

        Returns:
            Number of keys deleted

        WARNING: This deletes all device data. Use with caution.
        """
        deleted_count = 0
        for key in self.client.scan_iter("device:*", count=1000):
            self.client.delete(key)
            deleted_count += 1

        print(f"Cleared {deleted_count} device keys from Redis")
        return deleted_count
