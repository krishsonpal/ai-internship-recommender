from fastapi import APIRouter
from db.crud import add_internship
from db.vectorstore import build_vectorstore
from pydantic import BaseModel
from db.schemas import InternshipCreate
router = APIRouter()

@router.post("/add_internship/")
def create_internship(internship: InternshipCreate):
    new_internship = add_internship(
        title=internship.title,
        description=internship.description,
        skills=internship.skills_required,
        location=internship.location,
        stipend=internship.stipend,
        duration=internship.duration
    )
    build_vectorstore()  # update FAISS
    return {"message": "Internship added", "job_id": new_internship.job_id}