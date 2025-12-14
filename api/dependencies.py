from fastapi import Request
from pathlib import Path
import pickle
from redis import Redis

# Import configuration from centralized config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    FEATURE_ENGINEER_PATH,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT
)

def load_feature_engineer():
    with open(FEATURE_ENGINEER_PATH, "rb") as f:
        feature_engineer = pickle.load(f)
    return feature_engineer

def get_feature_engineer():
    return load_feature_engineer()

def load_preprocessor():
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor

def get_preprocessor():
    return load_preprocessor()

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def get_model():
    return load_model()

def get_redis_client():
    return Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)