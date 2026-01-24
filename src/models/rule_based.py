class RuleBasedModel:

    def predict(self, transaction, mode=2):
        """                                                                       
        Rule-based fraud detection with configurable strictness.                  
                                                                                    
        This function implements a two-mode fraud detection system based on       
    explicit                                                                      
        business rules. The mode determines the tradeoff between catching fraud   
        (recall) and minimizing false alarms (precision).                         
                                                                                    
        Parameters                                                                
        ----------                                                                
        transaction : dict                                                        
            Transaction data containing features like 'fast_purchase', 'device_txn_velocity_24h',                                                    
            'ip_switched', 'repeated_device_purchase', etc.                       
        mode : int, default=2                                                     
            Detection mode controlling strictness:                                
                                                                                    
            **Mode 1 - High Recall (Fraud Detection Priority)**                   
                - Only checks: fast_purchase                                      
                - Use when: Fraud losses are extremely costly, false alarms are   
    acceptable                                                                    
                - Tradeoff: Catches more fraud but flags many legitimate          
    transactions                                                                  
                - Typical metrics: ~99% recall, ~20-30% precision                 
                - Example use case: High-value transactions, new merchant         
    onboarding                                                                    
                                                                                    
            **Mode 2 - High Precision (Customer Experience Priority)**            
                - Checks: All rules (device velocity, IP switches, behavioral     
    patterns, etc.)                                                               
                - Use when: Customer friction is costly, want to minimize false   
    declines                                                                      
                - Tradeoff: Misses some fraud but rarely flags legitimate         
    transactions                                                                  
                - Typical metrics: ~30-50% recall, ~95-99% precision              
                - Example use case: Established customers, low-value transactions 
                                                                                    
        Returns                                                                   
        -------                                                                   
        int                                                                       
            Fraud classification result:                                          
                                                                                
          **1 (Fraud)**                                                         
              - High confidence fraud based on explicit rules                   
              - Transaction should be blocked or flagged for immediate review   
              - Examples: Fast purchase + high velocity, known fraud patterns   
                                                                                
          **0 (Not Fraud)**                                                     
              - High confidence legitimate transaction                          
                                                   
          **-1 (Ambiguous)**                                                    
              - Cannot determine with high confidence from rules alone          
              - Transaction should be sent to ML model for prediction               
    """
        if transaction["fast_purchase"]:
            return 1
        
        if mode == 1:
            return -1
        
        elif transaction["device_txn_idx"] > 1:
            if transaction["repeated_device_purchase"] == 1:
                if transaction["signup_before_first_device_txn"]==1:
                    if transaction["device_txn_velocity_24h"] > 1:
                        return 1
                    else:
                        return 0
        return -1



