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

    def update_global_bucket(self, purchase_time):
        current_bucket = int(purchase_time.timestamp() // 60)
        bucket_key = f"global:txn_count:bucket:{current_bucket}"
        self.client.incr(bucket_key)
        self.client.expire(bucket_key, 25 * 3600)

    def get_global_txn_velocity(self, purchase_time, time_window):
        if time_window == "1h":
            num_buckets = 3600 // self.bucket_size_seconds

        current_bucket = int(purchase_time.timestamp() // 60)
        total = 0

        for i in range(num_buckets):
            bucket_id = current_bucket - i
            bucket_key = f"global:txn_count:bucket:{bucket_id}"
            count = self.client.get(bucket_key)
            total += int(count) if count else 0

        return total

    def count_buckets(self) -> int:
        """Count total number of global buckets stored in Redis."""
        count = 0
        for _ in self.client.scan_iter("global:txn_count:bucket:*", count=1000):
            count += 1
        return count

    def export_to_file(self, filepath: str) -> None:
        """
        Export all global velocity buckets from Redis to a pickle file.

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

        print(f"Exporting global velocity buckets to {filepath}...")

        # Export all bucket keys and their counts
        global_buckets = {}
        bucket_count = 0

        for key in self.client.scan_iter("global:txn_count:bucket:*", count=1000):
            key_str = key if isinstance(key, str) else key.decode('utf-8')

            # Extract bucket_id from key (format: global:txn_count:bucket:{bucket_id})
            bucket_id = key_str.split(":")[-1]

            # Get the count for this bucket
            count = self.client.get(key_str)
            if count:
                global_buckets[bucket_id] = int(count)
                bucket_count += 1

            if bucket_count % 1000 == 0:
                print(f"  Exported {bucket_count} buckets...")

        # Save to file
        export_data = {
            "global_buckets": global_buckets,
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "bucket_count": bucket_count,
                "bucket_size_seconds": self.bucket_size_seconds,
                "type": "global_velocity"
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported {bucket_count} global velocity buckets ({file_size_mb:.2f} MB)")
        print(f"✓ Global velocity export complete")

    @classmethod
    def load_from_file(cls, filepath: str, flush_existing: bool = False) -> 'GlobalVelocity':
        """
        Load global velocity buckets from a pickle file and restore to Redis.

        Args:
            filepath: Path to the state file (e.g., 'models/global_buckets.pkl')
            flush_existing: Whether to flush existing global buckets in Redis before loading
                          (default True - ensures clean state)

        Returns:
            GlobalVelocity instance with loaded state

        Example:
            # On app startup
            global_velocity = GlobalVelocity.load_from_file('models/global_buckets.pkl')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")

        print(f"Loading global velocity buckets from {filepath}...")

        # Create instance
        instance = cls()

        # Flush existing global buckets if requested
        if flush_existing:
            print("  Flushing existing global buckets from Redis...")
            deleted_count = 0
            for key in instance.client.scan_iter("global:txn_count:bucket:*", count=1000):
                instance.client.delete(key)
                deleted_count += 1
            if deleted_count > 0:
                print(f"  Deleted {deleted_count} existing bucket keys")

        # Load data from file
        with open(filepath, 'rb') as f:
            export_data = pickle.load(f)

        global_buckets = export_data.get("global_buckets", {})
        bucket_count = 0

        for bucket_id, count in global_buckets.items():
            bucket_key = f"global:txn_count:bucket:{bucket_id}"
            instance.client.set(bucket_key, count)

            # Set TTL to 25 hours (same as in update_global_bucket)
            instance.client.expire(bucket_key, 25 * 3600)

            bucket_count += 1

            if bucket_count % 1000 == 0:
                print(f"  Loaded {bucket_count} buckets...")

        metadata = export_data.get("metadata", {})
        export_time = metadata.get("export_time", "unknown")
        bucket_size = metadata.get("bucket_size_seconds", "unknown")

        print(f"  ✓ Loaded {bucket_count} global velocity buckets")
        print(f"    Exported at: {export_time}")
        print(f"    Bucket size: {bucket_size} seconds")
        print(f"✓ Global velocity state loaded successfully")

        return instance

    def clear_all_buckets(self) -> int:
        """
        Clear all global velocity buckets from Redis.

        Returns:
            Number of keys deleted

        WARNING: This deletes all global velocity data. Use with caution.
        """
        deleted_count = 0
        for key in self.client.scan_iter("global:txn_count:bucket:*", count=1000):
            self.client.delete(key)
            deleted_count += 1

        print(f"Cleared {deleted_count} global bucket keys from Redis")
        return deleted_count
