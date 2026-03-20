import pandas as pd
import pytest
from src.feature_engineering.engineer_flexible import TransactionFeatureEngineer

class TestFeatureEngineer:

    def test_compute_device_features(self, device_state, global_velocity, ip_state):
        feature_engineer = TransactionFeatureEngineer()

        feature_engineer.device_state = device_state
        feature_engineer.global_velocity = global_velocity
        feature_engineer.ip_state = ip_state

        # test compute_device_features for unseen device
        unseen_device_features = feature_engineer.compute_device_features(
            device_id="BBPACGBUVJUXF",
            purchase_value=14,
            sex="F",
            age=38,
            purchase_time=pd.to_datetime("2015-01-01 00:00:44"),
            signup_time=pd.to_datetime("2015-01-01 00:00:43"),
            device_state=feature_engineer.device_state.get_device_state("BBPACGBUVJUXF")
        )

        assert unseen_device_features["device_txn_idx"] == 1
        assert unseen_device_features["device_time_since_last_s"] == -1
        assert unseen_device_features["device_age_hours"] == -1
        assert unseen_device_features["signup_before_first_device_txn"] is False
        assert unseen_device_features["repeated_device_purchase"] is False
        assert unseen_device_features["purchase_spike"] is False
        assert unseen_device_features["identity_changed"] is False
        assert unseen_device_features["device_txn_velocity_24h"] == 1
        assert unseen_device_features["prev_is_fraud"] is False
                
        # manual update of device_state, global_velocity
        feature_engineer.device_state.update_device_state(
            device_id="BBPACGBUVJUXF",
            state_updates={
                "txn_count": 1,
                "first_seen_signup": pd.to_datetime("2015-01-01 00:00:43"),
                "first_seen": pd.to_datetime("2015-01-01 00:00:44"),
                "last_seen": pd.to_datetime("2015-01-01 00:00:44"),
                "prev_identity": "F38",
                "prev_purchase": 14,
                "prev_is_fraud": 1
            }
        )

        feature_engineer.device_state.update_device_timestamp(
            device_id="BBPACGBUVJUXF",
            transaction_id=1, # random transaction_id
            purchase_time=pd.to_datetime("2015-01-01 00:00:44")
        )

        feature_engineer.global_velocity.update_bucket(
            purchase_time=pd.to_datetime("2015-01-01 00:00:44"),
            country="Korea Republic of"
        )
        
        # test compute_device_features for seen device (must update previous values correctly)
        seen_device_features = feature_engineer.compute_device_features(
            device_id="BBPACGBUVJUXF",
            purchase_value=14,
            sex="F",
            age=38,
            purchase_time=pd.to_datetime("2015-01-01 00:00:45"),
            signup_time=pd.to_datetime("2015-01-01 00:00:44"),
            device_state=feature_engineer.device_state.get_device_state("BBPACGBUVJUXF")
        )

        assert seen_device_features["device_txn_idx"] == 2
        assert seen_device_features["device_time_since_last_s"] == 1
        assert seen_device_features["device_age_hours"] == 1 / 3600
        assert seen_device_features["repeated_device_purchase"] is True
        assert seen_device_features["device_txn_velocity_24h"] == 2
        assert seen_device_features["prev_is_fraud"] is True

        # test compute_device_features for seen device but different purchase value (increase/decrease)
        seen_different_purchase_features = feature_engineer.compute_device_features(
            device_id="BBPACGBUVJUXF",
            purchase_value=15,
            sex="F",
            age=38,
            purchase_time=pd.to_datetime("2015-01-01 00:00:45"),
            signup_time=pd.to_datetime("2015-01-01 00:00:44"),
            device_state=feature_engineer.device_state.get_device_state("BBPACGBUVJUXF")
        )

        assert seen_different_purchase_features["repeated_device_purchase"] is False
        assert seen_different_purchase_features["purchase_spike"] is True

        # test compute_device_features for seen device but different identity
        seen_different_identity_features = feature_engineer.compute_device_features(
            device_id="BBPACGBUVJUXF",
            purchase_value=14,
            sex="M",
            age=38,
            purchase_time=pd.to_datetime("2015-01-01 00:00:45"),
            signup_time=pd.to_datetime("2015-01-01 00:00:44"),
            device_state=feature_engineer.device_state.get_device_state("BBPACGBUVJUXF")
        )

        print(seen_different_identity_features)

        assert seen_different_identity_features["identity_changed"] is True 

        # test compute_device_features for seen device but significantly later in the future
        seen_distant_features = feature_engineer.compute_device_features(
            device_id="BBPACGBUVJUXF",
            purchase_value=14,
            sex="F",
            age=38,
            purchase_time=pd.to_datetime("2015-01-03 00:00:45"),
            signup_time=pd.to_datetime("2015-01-03 00:00:44"),
            device_state=feature_engineer.device_state.get_device_state("BBPACGBUVJUXF")
        )

        assert seen_distant_features["device_txn_idx"] == 2
        assert seen_distant_features["device_time_since_last_s"] == 172801
        assert seen_distant_features["device_txn_velocity_24h"] == 1

        # test compute_device_features for seen device but signed up before fist
        seen_before_features = feature_engineer.compute_device_features(
            device_id="BBPACGBUVJUXF",
            purchase_value=14,
            sex="F",
            age=38,
            purchase_time=pd.to_datetime("2015-01-01 00:00:45"),
            signup_time=pd.to_datetime("2015-01-01 00:00:42"),
            device_state=feature_engineer.device_state.get_device_state("BBPACGBUVJUXF")
        )

        assert seen_before_features["signup_before_first_device_txn"] is True

    # def test_compute_features(self, device_state, global_velocity, ip_state):
    #     feature_engineer = TransactionFeatureEngineer()

    #     feature_engineer.device_state = device_state
    #     feature_engineer.global_velocity = global_velocity
    #     feature_engineer.ip_state = ip_state

    #     # 1: new device: processed_transaction
    #     # 1.5: updated state_to_update
    #     # manually update device_state, global_velocity
    #     # 2: new device within 24h, check global_txn_velocity_24h, coutnry_txn_velocity_24h