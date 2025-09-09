from pydantic import BaseModel

class InternshipCreate(BaseModel):
    title: str
    description: str
    skills_required: str
    location: str
    stipend: str
    duration: str
