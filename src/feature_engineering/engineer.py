import numpy as np

def compute_features(
        txn_count,
        prev_purchase_value, 
        prev_sex,
        prev_age,
        signup_time,
        purchase_time,
        purchase_value,
        sex,
        age
):

    time_setup_to_txn_seconds = (purchase_time - signup_time).total_seconds()
    log_time_setup_to_txn_seconds = np.log1p(time_setup_to_txn_seconds)

    if txn_count == 0:
        first_device_transaction = True
    else:
        first_device_transaction = False

    if prev_purchase_value:
        scaled_device_purchase_diff = abs(purchase_value - prev_purchase_value) / prev_purchase_value
        repeated_device_purchase = purchase_value == prev_purchase_value
    else:
        scaled_device_purchase_diff = -1
        repeated_device_purchase = False

    if prev_sex and prev_age:
        if sex == prev_sex and age == prev_age:
            identity_changed = False
        else:
            identity_changed = True 
    else:
        identity_changed = False

    return txn_count + 1, log_time_setup_to_txn_seconds, first_device_transaction, scaled_device_purchase_diff, repeated_device_purchase, identity_changed
