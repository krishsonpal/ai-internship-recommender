from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from utils.config import DATABASE_URL
from urllib.parse import urlparse

Base = declarative_base()

# Configure engine depending on driver (sqlite vs postgres/etc.)
_url = urlparse(DATABASE_URL)
if _url.scheme.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()