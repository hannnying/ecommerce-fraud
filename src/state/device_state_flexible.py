"""
Flexible, schema-driven DeviceState implementation.
When you change DEVICE_STATE_SCHEMA, serialization automatically adapts.
"""
from datetime import datetime
from pathlib import Path
import pickle
import pandas as pd
from redis import Redis
from src.config import REDIS_DB, REDIS_HOST, REDIS_PORT
from src.state.device_schema import DEVICE_STATE_SCHEMA, get_field_names, get_default_state


class DeviceState:

    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    def update_device_state(self, device_id: str, state_updates: dict):
        """
        Update device state with provided fields.

        Args:
            device_id: Device identifier
            state_updates: Dict of fields to update (e.g., {"txn_count": 5, "last_seen": datetime.now()})

        Example:
            device_state.update_device_state("device123", {
                "txn_count": txn_count + 1,
                "last_seen": purchase_time,
                "prev_ip": ip_address
            })
        """
        key = f"device:{device_id}"

        # Get current state
        current_state = self.get_device_state(device_id)

        # Merge updates
        current_state.update(state_updates)

        # Serialize and save
        self.client.hset(key, mapping=self._serialize_state(current_state))

    def update_device_timestamp(self, device_id: str, transaction_id: str, purchase_time: datetime):
        """Update device transaction velocity tracking."""
        key = f"device:{device_id}:txn_timestamp"

        self.client.zadd(key, mapping={transaction_id: purchase_time.timestamp()})

        # Remove timestamps > 24 hours
        timestamp_threshold = (purchase_time - pd.Timedelta(days=1)).timestamp()
        self.client.zremrangebyscore(key, float("-inf"), timestamp_threshold)

    def update_prev_is_fraud(self, device_id: str, label: int) -> None:
        """Update prev_is_fraud of device_state"""
        
        key = f"device:{device_id}"
        self.client.hset(key, "prev_is_fraud", label)

    def _serialize_state(self, state: dict) -> dict:
        """
        Serialize state dict for Redis storage.
        Uses DEVICE_STATE_SCHEMA to determine how each field is serialized.
        """
        serialized = {}
        for field, field_type in DEVICE_STATE_SCHEMA.items():
            if field in state:
                serialized[field] = field_type.serialize(state[field])
        return serialized

    def _deserialize_state(self, raw: dict) -> dict:
        """
        Deserialize state dict from Redis.
        Uses DEVICE_STATE_SCHEMA to determine how each field is deserialized.
        """
        deserialized = {}
        for field, field_type in DEVICE_STATE_SCHEMA.items():
            raw_value = raw.get(field)
            deserialized[field] = field_type.deserialize(raw_value)
        return deserialized

    def get_device_state(self, device_id: str) -> dict:
        """
        Get device state as a dict.

        Returns:
            Dict with all fields from DEVICE_STATE_SCHEMA.
            For new devices, returns default values.

        Example:
            state = device_state.get_device_state("device123")
            txn_count = state["txn_count"]
            last_seen = state.get("last_seen")
        """
        key = f"device:{device_id}"

        # Get all fields from Redis
        raw = self.client.hgetall(key)

        if not raw:
            # New device - return defaults
            return get_default_state(DEVICE_STATE_SCHEMA)

        return self._deserialize_state(raw)

    def get_device_txn_velocity(self, device_id: str, purchase_time: datetime, time_window: str) -> int:
        """Get transaction count within time window."""
        key = f"device:{device_id}:txn_timestamp"

        if time_window == "1m":
            timestamp_threshold = purchase_time - pd.Timedelta(1, "m")
        elif time_window == "5m":
            timestamp_threshold = purchase_time - pd.Timedelta(5, "m")
        elif time_window == "1h":
            timestamp_threshold = purchase_time - pd.Timedelta(1, "h")
        elif time_window == "24h":
            timestamp_threshold = purchase_time - pd.Timedelta(1, "d")
        else:
            raise ValueError(f"Invalid time_window: {time_window}")

        min_score = timestamp_threshold.timestamp()
        max_score = purchase_time.timestamp()

        return self.client.zcount(key, min_score, max_score)

    def count_devices(self) -> int:
        """Count total number of devices stored in Redis."""
        count = 0
        for key in self.client.scan_iter("device:*", count=1000):
            if ":txn_timestamp" not in key:
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
                "type": "device_hashes",
                "schema_fields": get_field_names(DEVICE_STATE_SCHEMA)
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
                if timestamps:
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

        Args:
            filepath: Path to device hashes file (e.g., 'models/device_state.pkl')
            load_timestamps: Whether to also load device timestamps from companion file
            flush_existing: Whether to flush existing device state in Redis before loading

        Returns:
            DeviceState instance with loaded state

        Example:
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
        schema_fields = metadata.get("schema_fields", [])

        # Warn if schema has changed
        current_fields = set(get_field_names(DEVICE_STATE_SCHEMA))
        saved_fields = set(schema_fields) if schema_fields else current_fields

        if saved_fields != current_fields:
            print(f"  ⚠ Schema mismatch detected:")
            added = current_fields - saved_fields
            removed = saved_fields - current_fields
            if added:
                print(f"    New fields (will use defaults): {added}")
            if removed:
                print(f"    Removed fields (will be ignored): {removed}")

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
