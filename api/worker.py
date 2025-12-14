import json
import time
from api.dependencies import (
    get_feature_engineer,
    get_model,
    get_preprocessor
)
from api.inference import run_inference
import pandas as pd
from redis import Redis
from src.config import (
    TRANSACTION_STREAM,
    REDIS_BLOCK_TIMEOUT,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    RESULT_STREAM
)

def process_transaction(transaction_dict, feature_engineer, preprocessor, model):
    """Run inference and push result to RESULT_STREAM."""
    print(f"processing transaction for: {transaction_dict}")
    try:
        transaction_df = pd.DataFrame([transaction_dict])

        predicted_class, probability = run_inference(
            transaction=transaction_df,
            feature_engineer=feature_engineer,
            preprocessor=preprocessor,
            model=model
        )

        # Prepare result dict
        result = {
            "transaction_id": str(transaction_dict["id"]),
            "predicted_class": int(predicted_class[0]),
            "fraud_probability": float(probability[0])
        }
        print(f"prediction: {result}")
        return result

    except Exception as e:
        print(f"Failed to process transaction {transaction_dict.get('id')}: {e}")


# async def main():
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
feature_engineer = get_feature_engineer()
preprocessor = get_preprocessor()
model = get_model()

print("Worker started. Listening for transactions")
last_id = "0-0"

while True:
    try:
        messages = redis_client.xread(
            {TRANSACTION_STREAM: last_id},
            block=0
        )
        if not messages:
            continue

        print(f"this is the current message: {messages}")
        _, entries = messages[0]

        for message_id, transaction_dict in entries:
            last_id = message_id

            transaction_id = transaction_dict["id"]
            transaction_dict["age"] = int(transaction_dict["age"])
            transaction_dict["purchase_value"] = int(transaction_dict["purchase_value"])
            transaction_dict["ip_address"] = float(transaction_dict["ip_address"])

            res = process_transaction(
                transaction_dict,
                feature_engineer,
                preprocessor,
                model
            )
            redis_client.xadd(
                RESULT_STREAM,
                res
            )
        
    except Exception as e:
        print(f"Worker loop failed: {e}")
        time.sleep(1)

# if __name__ == "__main__":
#     main()