import pandas as pd
from sqlalchemy import Session, func
from src.domain_models import Prediction, Transaction
from src.mappers import entity_to_orm, orm_to_entity
from src.models import PredictionModel, TransactionModel
from typing import Type, TypeVar, Generic, List, Optional

T = TypeVar("T")

class BaseRepository(Generic[T]):
    def __init__(self, session: Session, model: Type[T]):
        self.session = session
        self.model = model # PredictionModel or TransactionModel

    def add(self, entity: T):
        instance = entity_to_orm(entity, self.model)
        self.session.add(instance)
        
    def remove(self, entity: T):
        model = self.session.query(self.model).get(entity.id)
        self.session.delete(model)

    def get_by_id(self, id_):
        instance = self.session.get(self.model, id_)
        return self._get_entity(instance)

    def get_all(self) -> List[T]:
        return self.session.query(self.model).all()

    def filter_by(self, **kwargs) -> List[T]:
        return self.session.query(self.model).filter_by(**kwargs).all()
    
    def _get_entity(self, instance):
        if instance is None:
            return None
        
        entity = orm_to_entity(instance, self.model)
        return entity
    
    def bulk_insert(self, df: pd.DataFrame):
        records = df.to_dict(orient="records")

        with self.session() as session:
            session.bulk_insert_mappings(self.model, records)

    

class PredictionRepository(BaseRepository):
    def __init__(self, session):
        super().__init__(session, PredictionModel)

    def get_by_model(self, model_used: str):
        return self.session.query(self.model).filter_by(model_used=model_used).all()


class TransactionRepository(BaseRepository):
    def _init__(self, session):
        super().__init__(session, TransactionModel)

    def get_new_labeled_count(self):
        return self.session.query(func.count(TransactionModel.transaction_id)).scalar()
    
    def get_training_dataset(self):
        """Return all transactions with known labels."""
        return self.session.query(self.model).all()
    