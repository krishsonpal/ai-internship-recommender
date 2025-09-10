from pydantic import BaseModel, EmailStr
from typing import Optional

class InternshipCreate(BaseModel):
    title: str
    description: str
    skills_required: str
    location: str
    stipend: str
    duration: str

# User authentication schemas
class UserSignup(BaseModel):
    student_id: str
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    student_id: str
    password: str

class UserResponse(BaseModel):
    student_id: str
    email: str
    name: str
    
    class Config:
        from_attributes = True

# Company authentication schemas
class CompanySignup(BaseModel):
    company_id: str
    email: str
    password: str
    company_name: str

class CompanyLogin(BaseModel):
    company_id: str
    password: str

class CompanyResponse(BaseModel):
    company_id: str
    email: str
    company_name: str
    
    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    user_type: str  # "student" or "company"
    user_id: str
