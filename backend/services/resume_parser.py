from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from google.api_core.exceptions import ResourceExhausted
import re

# Use lighter model to avoid free-tier quota issues
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def extract_resume_summary(cv_text: str) -> dict:
    """
    Extract structured summary (skills, education, projects, experience) from resume text.
    """
    # Truncate to keep token usage low
    truncated = cv_text[:4000]

    prompt = f"""
    Extract the following information from this resume:
    - Skills (list)
    - Education (degrees, universities, years if mentioned)
    - Work/Project Experience (short bullet summary)
    - Strengths or specialties

    Resume:
    {truncated}
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"summary": response.content}
    except ResourceExhausted:
        # Fallback: return a naive summary slice to keep flow working
        return {"summary": truncated[:800]}
