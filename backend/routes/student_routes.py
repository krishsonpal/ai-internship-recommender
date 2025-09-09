from fastapi import APIRouter, UploadFile, Request, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import (get_db, Base, engine)
from db import crud
from services.resume_parser import extract_text_from_pdf, extract_resume_summary
from db.vectorstore import build_or_update_student_vectorstore
from utils.config import get_embeddings_with_fallback
from services.recommendation import get_internship_recommendations_by_vector
from services.recommendation import (
    get_internship_recommendations,
    get_internship_recommendations_by_vector,
    answer_with_context
)

# Make sure tables exist
Base.metadata.create_all(bind=engine)

router = APIRouter()


@router.post("/login")
async def login(request: Request, student_id: str):
    request.session["student_id"] = str(student_id)
    return {"message": "Logged in", "student_id": student_id}


@router.get("/session-status")
async def check_session(request: Request):
    """Check what's stored in the current session."""
    student_id = request.session.get("student_id")
    resume_summary = request.session.get("resume_summary", "")
    recommendations = request.session.get("recommendations", [])
    
    return {
        "student_id": student_id,
        "has_resume_summary": bool(resume_summary),
        "resume_summary_length": len(resume_summary) if resume_summary else 0,
        "recommendations_count": len(recommendations) if recommendations else 0,
        "all_session_keys": list(request.session.keys())
    }


@router.post("/analyze_resume/")
async def analyze_resume(
    request: Request,
    file: UploadFile,
    db: Session = Depends(get_db)   # ✅ Inject DB session
):
    student_id = request.session.get("student_id")
    if not student_id:
        raise HTTPException(
            status_code=401,
            detail="Not logged in: student_id missing in session"
        )

    # Save file locally
    file_path = f"{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    cv_text = extract_text_from_pdf(file_path)

    # ✅ Check if summary and embedding already exist in DB
    stored_data = crud.get_resume_summary_with_embedding(db, student_id)
    if stored_data and stored_data.get_embedding():
        # Use cached data - no API calls needed!
        resume_summary = {"summary": stored_data.summary_text}
        resume_embedding = stored_data.get_embedding()
        print(f"✅ Using cached resume data for student {student_id}")
    else:
        # Extract new summary from Gemini (only if not cached)
        resume_summary = extract_resume_summary(cv_text)
        
        # Compute embedding (only if not cached) with fallback
        try:
            embeddings_model, embedding_type = get_embeddings_with_fallback()
            resume_embedding = embeddings_model.embed_query(resume_summary["summary"])
            print(f"✅ Generated new embedding for student {student_id} using {embedding_type}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
        
        # Save to database for future use
        crud.save_resume_summary_with_embedding(db, student_id, resume_summary["summary"], resume_embedding)
        print(f"✅ Saved resume data to database for student {student_id}")

    # ✅ Get recommendations by vector (no additional embedding calls)
    recs = get_internship_recommendations_by_vector(resume_embedding)

    # ✅ Store in session for chat use
    request.session["resume_summary"] = resume_summary["summary"]
    request.session["recommendations"] = recs

    # Debug: log what we're storing
    print(f"Storing in session for student {student_id}:")
    print(f"- resume_summary length: {len(resume_summary['summary'])}")
    print(f"- recommendations count: {len(recs)}")
    print(f"- session keys: {list(request.session.keys())}")

    return {
        "resume_summary": resume_summary,
        "recommendations": recs,
        "message": "Resume processed successfully. Use /chat to ask questions.",
        "session_stored": True
    }


@router.post("/chat")
async def chat_with_ai(request: Request, question: str):
    """Chat with AI using stored resume summary and recommendations."""
    student_id = request.session.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Not logged in")

    # Get stored data from session
    resume_summary = request.session.get("resume_summary", "")
    recommendations = request.session.get("recommendations", [])

    # Debug: log session data
    print(f"Session data for student {student_id}:")
    print(f"- resume_summary: {bool(resume_summary)}")
    print(f"- recommendations: {len(recommendations) if recommendations else 0}")

    if not resume_summary:
        raise HTTPException(
            status_code=400, 
            detail="No resume processed. Please upload resume first via /analyze_resume/"
        )

    try:
        # Get AI response with limited context
        answer = answer_with_context(student_id, question, resume_summary, recommendations)
        
        return {
            "question": question,
            "answer": answer,
            "model_used": "gemini-1.5-flash"
        }
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
