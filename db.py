from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = 'prediction_records'
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_type = Column(String)  # 例如 'pose' 或 'audio'
    result = Column(String)    # 推論結果字串
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
