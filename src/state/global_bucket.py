from datetime import datetime
from pathlib import Path
import pickle
from redis import Redis
from src.config import(
    BUCKET_SIZE_SECONDS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT
)

class GlobalVelocity:
    
    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.bucket_size_seconds = BUCKET_SIZE_SECONDS

    def update_bucket(self, purchase_time: datetime, country: str) -> None:
        current_bucket = int(purchase_time.timestamp() // self.bucket_size_seconds)

        self.update_country_bucket(country, current_bucket)
        self.update_global_bucket(current_bucket)

    def update_country_bucket(self, country: str, current_bucket: int) -> None:
        country_bucket_key = f"{country}:txn_count:bucket:{current_bucket}"
        self.client.incr(country_bucket_key)
        self.client.expire(country_bucket_key, 25 * 3600)

    def update_global_bucket(self, current_bucket: int) -> None:
        bucket_key = f"global:txn_count:bucket:{current_bucket}"
        self.client.incr(bucket_key)
        self.client.expire(bucket_key, 25 * 3600)

    def get_txn_velocity(self, purchase_time: datetime, time_window: str, country: str = None):
        if time_window == "1h":
            num_buckets = 3600 // self.bucket_size_seconds

        elif time_window == "24h":
            num_buckets = 86400 // self.bucket_size_seconds

        else:
            raise ValueError(f"{time_window} not supported")
        
        current_bucket = int(purchase_time.timestamp() // self.bucket_size_seconds)
        total = 0

        for i in range(num_buckets):
            bucket_id = current_bucket - i
            if country:
                bucket_key = f"{country}:txn_count:bucket:{bucket_id}"
            else:
                bucket_key = f"global:txn_count:bucket:{bucket_id}"
            count = self.client.get(bucket_key)
            total += int(count) if count else 0 

        return total

    def count_buckets(self) -> int:
        """Count total number of buckets (global and country-specific) stored in Redis."""
        count = 0
        for _ in self.client.scan_iter("*:txn_count:bucket:*", count=1000):
            count += 1
        return count

    def export_to_file(self, filepath: str) -> None:
        """
        Export all velocity buckets (global and country-specific) from Redis to a pickle file.

        Args:
            filepath: Path to save the state file (e.g., 'models/global_buckets.pkl')

        Returns:
            None

        Example:
            global_velocity = GlobalVelocity()
            global_velocity.export_to_file('models/global_buckets.pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        print(f"Exporting velocity buckets to {filepath}...")

        # Export all bucket keys and their counts
        # Store as full keys to preserve global vs country distinction
        all_buckets = {}
        bucket_count = 0

        for key in self.client.scan_iter("*:txn_count:bucket:*", count=1000):
            key_str = key if isinstance(key, str) else key.decode('utf-8')

            # Get the count for this bucket
            count = self.client.get(key_str)
            if count:
                all_buckets[key_str] = int(count)
                bucket_count += 1

            if bucket_count % 1000 == 0:
                print(f"  Exported {bucket_count} buckets...")

        # Save to file
        export_data = {
            "buckets": all_buckets,
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "bucket_count": bucket_count,
                "bucket_size_seconds": self.bucket_size_seconds,
                "type": "velocity_buckets"
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported {bucket_count} velocity buckets ({file_size_mb:.2f} MB)")
        print(f"✓ Velocity buckets export complete")

    @classmethod
    def load_from_file(cls, filepath: str, flush_existing: bool = False) -> 'GlobalVelocity':
        """
        Load velocity buckets (global and country-specific) from a pickle file and restore to Redis.

        Args:
            filepath: Path to the state file (e.g., 'models/global_buckets.pkl')
            flush_existing: Whether to flush existing buckets in Redis before loading
                          (default False - preserves existing state)

        Returns:
            GlobalVelocity instance with loaded state

        Example:
            # On app startup
            global_velocity = GlobalVelocity.load_from_file('models/global_buckets.pkl')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")

        print(f"Loading velocity buckets from {filepath}...")

        # Create instance
        instance = cls()

        # Flush existing buckets if requested
        if flush_existing:
            print("  Flushing existing buckets from Redis...")
            deleted_count = 0
            for key in instance.client.scan_iter("*:txn_count:bucket:*", count=1000):
                instance.client.delete(key)
                deleted_count += 1
            if deleted_count > 0:
                print(f"  Deleted {deleted_count} existing bucket keys")

        # Load data from file
        with open(filepath, 'rb') as f:
            export_data = pickle.load(f)

        # Try new format first (with full keys), then fall back to old format
        buckets = export_data.get("buckets")
        if buckets is None:
            # Old format - only had global buckets
            print("  ⚠ Loading from old format (global buckets only)")
            buckets = {}
            for bucket_id, count in export_data.get("global_buckets", {}).items():
                buckets[f"global:txn_count:bucket:{bucket_id}"] = count

        bucket_count = 0

        for bucket_key, count in buckets.items():
            instance.client.set(bucket_key, count)

            # Set TTL to 25 hours (same as in update methods)
            instance.client.expire(bucket_key, 25 * 3600)

            bucket_count += 1

            if bucket_count % 1000 == 0:
                print(f"  Loaded {bucket_count} buckets...")

        metadata = export_data.get("metadata", {})
        export_time = metadata.get("export_time", "unknown")
        bucket_size = metadata.get("bucket_size_seconds", "unknown")

        print(f"  ✓ Loaded {bucket_count} velocity buckets")
        print(f"    Exported at: {export_time}")
        print(f"    Bucket size: {bucket_size} seconds")
        print(f"✓ Velocity state loaded successfully")

        return instance

    def clear_all_buckets(self) -> int:
        """
        Clear all velocity buckets (global and country-specific) from Redis.

        Returns:
            Number of keys deleted

        WARNING: This deletes all velocity data. Use with caution.
        """
        deleted_count = 0
        for key in self.client.scan_iter("*:txn_count:bucket:*", count=1000):
            self.client.delete(key)
            deleted_count += 1

        print(f"Cleared {deleted_count} bucket keys from Redis")
        return deleted_count
