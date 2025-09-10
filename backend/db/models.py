from sqlalchemy import Column, Integer, String, Text, LargeBinary, DateTime
from db.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
import pickle
from datetime import datetime

class Internship(Base):
    __tablename__ = "internships"

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    description = Column(Text)
    skills_required = Column(Text)
    location = Column(String(100))
    stipend = Column(String(50))
    duration = Column(String(50))


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    company_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ResumeSummary(Base):
    __tablename__ = "resume_summaries"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, unique=True, index=True)  # Changed to String to match session student_id
    summary_text = Column(Text, nullable=False)
    embedding_vector = Column(LargeBinary)  # Store the embedding as binary data
    created_at = Column(String)  # Simple timestamp

    def set_embedding(self, embedding_list):
        """Store embedding vector as pickle binary data."""
        self.embedding_vector = pickle.dumps(embedding_list)
    
    def get_embedding(self):
        """Retrieve embedding vector from pickle binary data."""
        if self.embedding_vector:
            return pickle.loads(self.embedding_vector)
        return None