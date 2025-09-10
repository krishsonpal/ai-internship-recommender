from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.crud import add_internship
from db.database import get_db
from db.vectorstore import build_vectorstore
from db.schemas import InternshipCreate, CompanySignup, CompanyLogin, CompanyResponse, Token
from db import crud
from utils.auth import create_access_token

router = APIRouter()

# Company authentication routes
@router.post("/signup", response_model=CompanyResponse)
async def company_signup(company_data: CompanySignup, db: Session = Depends(get_db)):
    """Sign up a new company."""
    # Check if company_id already exists
    if crud.get_company_by_company_id(db, company_data.company_id):
        raise HTTPException(status_code=400, detail="Company ID already registered")
    
    # Check if email already exists
    if crud.get_company_by_email(db, company_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create company
    company = crud.create_company(
        db=db,
        company_id=company_data.company_id,
        email=company_data.email,
        password=company_data.password,
        company_name=company_data.company_name
    )
    
    return CompanyResponse(
        company_id=company.company_id,
        email=company.email,
        company_name=company.company_name
    )

@router.post("/login", response_model=Token)
async def company_login(login_data: CompanyLogin, db: Session = Depends(get_db)):
    """Login a company."""
    company = crud.authenticate_company(db, login_data.company_id, login_data.password)
    if not company:
        raise HTTPException(status_code=401, detail="Invalid company ID or password")
    
    # Create access token
    access_token = create_access_token(
        data={"sub": company.company_id, "user_type": "company"}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_type="company",
        user_id=company.company_id
    )

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