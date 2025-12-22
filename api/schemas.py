# currently, the schema will be the input features of models/logistic_regression_models.pkl
from datetime import datetime
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from pydantic import BaseModel

class Transactions(pa.DataFrameModel):
    """Pandera schema for raw transaction data"""
    
    user_id: Series[int]
    signup_time: Series[str]
    purchase_time: Series[str]
    purchase_value: Series[int]
    device_id: Series[str]
    source: Series[str]
    browser: Series[str]
    sex: Series[str]
    age: Series[int]
    ip_address: Series[float]
    is_fraud: Series[int]

    class Config:
        coerce = True  # Allow type coercion

class Transaction(BaseModel):
    user_id: int
    signup_time: str
    purchase_time: str
    purchase_value: int
    device_id: str
    source: str
    browser: str
    sex: str
    age: int
    ip_address: float

class TransactionDTO(BaseModel):

    purchase_value: int
    source: str
    browser: str
    sex: str
    age: int
    

# API Schema for fraud prediction (output)
class FraudPrediction(BaseModel):
    predicted_class: list[int]
    probability: list[float]
