import logging
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timezone
from src.backend.config import settings

logger = logging.getLogger("db_service")
Base = declarative_base()

class TransactionRecord(Base):
    __tablename__ = "transactions"
    
    id = Column(String, primary_key=True)
    description = Column(String, nullable=False)
    category = Column(String, nullable=False)
    amount = Column(Float, default=0.0)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class DBService:
    def __init__(self):
        pg_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        
        try:
            engine = create_engine(pg_url, connect_args={"connect_timeout": 1})
            engine.connect()
            self.engine = engine
            logger.info("Connected to PostgreSQL database.")
        except Exception:
            logger.warning("PostgreSQL server unavailable. Initializing SQLite local fallback database.")
            self.engine = create_engine(
                "sqlite:///smartwealth_fallback.db",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )

        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def save_transaction(self, tx_id: str, desc: str, cat: str, amount: float, confidence: float):
        session = self.SessionLocal()
        try:
            record = TransactionRecord(
                id=tx_id,
                description=desc,
                category=cat,
                amount=amount,
                confidence=confidence
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving transaction: {e}")
            return False
        finally:
            session.close()

db_service = DBService()
