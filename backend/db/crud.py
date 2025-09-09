from db.models import Internship, ResumeSummary
from db.database import SessionLocal
from datetime import datetime

def add_internship(title, description, skills, location, stipend, duration):
    db = SessionLocal()
    internship = Internship(
        title=title,
        description=description,
        skills_required=skills,
        location=location,
        stipend=stipend,
        duration=duration
    )
    db.add(internship)
    db.commit()
    db.refresh(internship)
    db.close()
    return internship

def get_all_internships():
    db = SessionLocal()
    internships = db.query(Internship).all()
    db.close()
    return internships


from sqlalchemy.orm import Session
from . import models

def save_resume_summary_with_embedding(db: Session, student_id: str, summary_text: str, embedding_vector: list):
    """
    Save or update a student's resume summary with embedding vector.
    """
    summary = db.query(ResumeSummary).filter(ResumeSummary.student_id == student_id).first()
    if summary:
        summary.summary_text = summary_text
        summary.set_embedding(embedding_vector)
        summary.created_at = datetime.now().isoformat()
    else:
        summary = ResumeSummary(
            student_id=student_id, 
            summary_text=summary_text,
            created_at=datetime.now().isoformat()
        )
        summary.set_embedding(embedding_vector)
        db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


def get_resume_summary_with_embedding(db: Session, student_id: str):
    """
    Get stored resume summary and embedding for a student.
    """
    return db.query(ResumeSummary).filter(ResumeSummary.student_id == student_id).first()


def save_resume_summary(db: Session, student_id: str, summary_text: str):
    """
    Save or update a student's resume summary (legacy function).
    """
    summary = db.query(ResumeSummary).filter(ResumeSummary.student_id == student_id).first()
    if summary:
        summary.summary_text = summary_text
    else:
        summary = ResumeSummary(student_id=student_id, summary_text=summary_text)
        db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


def get_resume_summary(db: Session, student_id: str):
    """
    Get stored resume summary for a student (legacy function).
    """
    return db.query(ResumeSummary).filter(ResumeSummary.student_id == student_id).first()
