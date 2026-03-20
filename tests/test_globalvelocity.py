import pandas as pd
from src.state.global_bucket import GlobalVelocity

class TestGlobalVelocity:

    def test_update_bucket(self, global_velocity):
        global_velocity = global_velocity 

        txn_us_1 = {
            "transaction_id": 1,
            "purchase_time": pd.to_datetime("2015-06-08 01:38:54"),
            "country": "United States"
        }
        global_velocity.update_bucket(txn_us_1["transaction_id"], txn_us_1["purchase_time"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 01:38:54"), "1h") == 1

        global_velocity.update_bucket(txn_us_1["transaction_id"], txn_us_1["purchase_time"], txn_us_1["country"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 01:38:54"), "1h", "United States") == 1

        txn_us_2 = {
            "transaction_id": 2,
            "purchase_time": pd.to_datetime("2015-06-08 01:48:54"),
            "country": "United States"
        }

        global_velocity.update_bucket(txn_us_2["transaction_id"], txn_us_2["purchase_time"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 01:48:54"), "1h") == 2

        global_velocity.update_bucket(txn_us_2["transaction_id"], txn_us_2["purchase_time"], txn_us_1["country"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 01:48:54"), "1h", "United States") == 2

        txn_us_3 = {
            "transaction_id": 3,
            "purchase_time": pd.to_datetime("2015-06-08 02:40:54"),
            "country": "United States"
        }

        global_velocity.update_bucket(txn_us_3["transaction_id"], txn_us_3["purchase_time"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:54"), "1h")  == 2
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:54"), "24h") == 3

        global_velocity.update_bucket(txn_us_3["transaction_id"], txn_us_3["purchase_time"], txn_us_1["country"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:54"), "1h", "United States") == 2
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:54"), "24h", "United States") == 3

        txn_japan_1 = {
            "transaction_id": 4,
            "purchase_time": pd.to_datetime("2015-06-08 02:40:55"),
            "country": "Japan"
        }

        global_velocity.update_bucket(txn_japan_1["transaction_id"], txn_japan_1["purchase_time"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:55"), "1h") == 3
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:55"), "24h") == 4

        global_velocity.update_bucket(txn_japan_1["transaction_id"], txn_japan_1["purchase_time"], txn_japan_1["country"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:55"), "1h", "Japan") == 1
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:55"), "24h", "Japan") == 1
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:55"), "1h", "United States") == 2
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-08 02:40:55"), "24h", "United States") == 3

        txn_us_4 = {
            "transaction_id": 5,
            "purchase_time": pd.to_datetime("2015-06-09 01:38:55"),
            "country": "United States"
        }

        global_velocity.update_bucket(txn_us_4["transaction_id"], txn_us_4["purchase_time"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-09 01:38:55"), "24h") == 4

        global_velocity.update_bucket(txn_us_4["transaction_id"], txn_us_4["purchase_time"], txn_us_4["country"])
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-09 01:38:55"), "24h", "United States") == 3
        assert global_velocity.get_txn_velocity(pd.to_datetime("2015-06-09 01:38:55"), "24h", "Japan") == 1
        