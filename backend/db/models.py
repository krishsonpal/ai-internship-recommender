from sqlalchemy import Column, Integer, String, Text, LargeBinary
from db.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
import pickle

class Internship(Base):
    __tablename__ = "internships"

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    description = Column(Text)
    skills_required = Column(Text)
    location = Column(String(100))
    stipend = Column(String(50))
    duration = Column(String(50))


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)


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