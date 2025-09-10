from db.models import Internship, ResumeSummary, User, Company
from db.database import SessionLocal
from datetime import datetime
from utils.auth import get_password_hash, verify_password

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

# User authentication functions
def create_user(db: Session, student_id: str, email: str, password: str, name: str):
    """Create a new user (student)."""
    hashed_password = get_password_hash(password)
    db_user = User(
        student_id=student_id,
        email=email,
        password_hash=hashed_password,
        name=name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_student_id(db: Session, student_id: str):
    """Get user by student_id."""
    return db.query(User).filter(User.student_id == student_id).first()

def get_user_by_email(db: Session, email: str):
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, student_id: str, password: str):
    """Authenticate user with student_id and password."""
    user = get_user_by_student_id(db, student_id)
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user

# Company authentication functions
def create_company(db: Session, company_id: str, email: str, password: str, company_name: str):
    """Create a new company."""
    hashed_password = get_password_hash(password)
    db_company = Company(
        company_id=company_id,
        email=email,
        password_hash=hashed_password,
        company_name=company_name
    )
    db.add(db_company)
    db.commit()
    db.refresh(db_company)
    return db_company

def get_company_by_company_id(db: Session, company_id: str):
    """Get company by company_id."""
    return db.query(Company).filter(Company.company_id == company_id).first()

def get_company_by_email(db: Session, email: str):
    """Get company by email."""
    return db.query(Company).filter(Company.email == email).first()

def authenticate_company(db: Session, company_id: str, password: str):
    """Authenticate company with company_id and password."""
    company = get_company_by_company_id(db, company_id)
    if not company:
        return False
    if not verify_password(password, company.password_hash):
        return False
    return company

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
