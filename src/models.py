import uuid
from sqlalchemy import (
    Column,
    DateTime,
    String,
    Float,
    ForeignKey,
    Integer,
    Uuid
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class TransactionModel(Base):
    __tablename__ = 'transactions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(13))
    country = Column(String)
    signup_time = Column(DateTime)
    purchase_value = Column(Float)
    purchase_time = Column(DateTime)
    device_txn_idx = Column(Integer)
    device_time_since_last_s = Column(Float)
    device_age_hours = Column(Float)
    signup_before_first_device_txn = Column(Integer)
    repeated_device_purchase = Column(Integer)
    purchase_spike = Column(Integer)
    identity_changed = Column(Integer)
    device_txn_velocity_24h = Column(Integer)
    prev_is_fraud = Column(Integer)
    global_txn_velocity_24h = Column(Integer)
    country_txn_velocity_24h = Column(Integer)
    time_setup_to_txn_seconds = Column(Float)
    purchase_hour = Column(Integer)
    is_late_night = Column(Integer)
    under_18 = Column(Integer)
    age_18_25 = Column(Integer)
    age_26_35 = Column(Integer)
    age_36_50 = Column(Integer)
    age_50_above = Column(Integer)
    amount_per_age = Column(Float)
    unknown_country = Column(Integer)
    source = Column(String)
    browser = Column(String)
    true_label = Column(Integer)
    predictions = relationship("PredictionModel", back_populates="trnasaction")

class PredictionModel(Base):
    __tablename__ = "predictions"

    transaction_id = Column(
        UUID(as_uuid=True),
        ForeignKey("transactions.id"),
        primary_key=True
    )
    model_used = Column(String, primary_key=True)
    model_version = Column(Integer, primary_key=True)
    fraud_proba = Column(Float)

    transaction = relationship("TransactionModel", back_populates="predictions")