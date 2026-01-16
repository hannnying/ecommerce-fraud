from datetime import datetime
from pathlib import Path
import pickle
from redis import Redis
from src.config import (
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT
)

class IPState:

    def __init__(self):
        self.client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    def get_ip_txn_idx(self, ip_address):
        key = f"ip:{ip_address}"

        curr = self.client.hget(key, "count")
        return int(curr) if curr else 0
    
    def update_ip_txn_idx(self, ip_address):
        key = f"ip:{ip_address}"

        prev_count = self.client.hget(key, "count")

        if prev_count:
            self.client.hincrby(key, "count", 1)
        else:
            self.client.hset(key, mapping={"count": 1})
    
    def export_to_file(self, filepath: str):
        ip_address_hash = {}
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        for key in self.client.scan_iter("ip:*", count=1000):
            key_str = key if isinstance(key, str) else key.decode('utf-8')

            ip_address = key_str.split(":")[1]
            ip_address_count = self.client.hget(key_str, "count")
            ip_address_hash[ip_address] = ip_address_count

        export_data = {
            "ip_address_hash": ip_address_hash,
            "metadata": {
                "export_time": datetime.now().isoformat()
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
    
    @classmethod
    def load_from_file(cls, filepath: str, flush_existing: bool = False):
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"IP Address Hash file not found: {filepath}")
    
        print(f"Loading device state from {filepath}...")

        instance = cls()

        if flush_existing:
            print(f"Flushing existing IP state from Redis...")
            deleted_count = 0
            for key in instance.client.scan_iter("device:*", count=1000):
                instance.client.delete(key)
                deleted_count += 1
            if deleted_count > 0:
                print(f"  Deleted {deleted_count} existing device keys")

        with open(filepath, 'rb') as f:
            export_data = pickle.load(f)

        ip_hash = export_data.get("ip_address_hash", {})
        ip_counts = 0
        for ip_address, count in ip_hash.items():
            key = f"ip:{ip_address}"
            instance.client.hset(key, mapping={"count":count})
    
        print(f"IP Address state loaded successfully")

        return instance

    def clear_all_ip(self):
        deleted_count = 0
        for key in self.client.scan_iter("ip:*", count=1000):
            self.client.delete(key)
            deleted_count += 1
        
        print(f"Cleared {deleted_count} IP keys from Redis")
        return deleted_count
    