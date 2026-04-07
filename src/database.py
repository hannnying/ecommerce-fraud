from src.config import db_url
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(db_url)
session = sessionmaker(bind=engine)