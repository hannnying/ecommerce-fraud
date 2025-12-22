def compute_features(txn_count, purchase_sum, last_transaction, first_seen, ip_addresses, sources , signup_time, purchase_time, purchase_value, source, ip_address):

    time_setup_to_txn_seconds = (
        purchase_time - signup_time
    ).total_seconds()

    device_ip_consistency = (not ip_addresses) or \
                            (len(ip_addresses) == 1 and ip_address in ip_addresses)
    
    device_source_consistency = (not sources) or \
                                (len(sources) == 1 and source in sources)
        
    time_since_last_device_id_txn = (
        purchase_time - last_transaction 
    ).total_seconds() if last_transaction else 99999
    
    device_lifespan = (
        purchase_time - first_seen
    ).total_seconds() if first_seen else 0

    purchase_deviation_from_device_mean = abs(purchase_value - purchase_sum / txn_count) if txn_count else 0

    return txn_count + 1, device_ip_consistency, device_source_consistency, time_setup_to_txn_seconds, time_since_last_device_id_txn, purchase_deviation_from_device_mean, device_lifespan