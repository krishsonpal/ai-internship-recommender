# AI Internship Recommender

An intelligent internship recommendation system that uses AI to match students with relevant internship opportunities based on their resume and skills.

## Features

- **Resume Analysis**: Extract and analyze resume content using AI
- **Smart Recommendations**: Get personalized internship recommendations based on resume
- **Chat Interface**: Ask questions about internships and get AI-powered responses
- **Session Management**: Secure student login and data persistence
- **Dual Embedding Support**: Uses Gemini API with Hugging Face fallback

## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd sih
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
# Copy the example file
cp env.example .env

# Edit .env and add your API keys:
# - GOOGLE_API_KEY: Get from https://makersuite.google.com/app/apikey
# - HUGGINGFACE_API_TOKEN: Get from https://huggingface.co/settings/tokens
# - SESSION_SECRET: Any random string for session security
```

### 5. Run the Application
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Deploy

- Use Postgres for `DATABASE_URL` in production
- Set `CORS_ORIGINS` to your frontend URLs
- Mount persistent storage for FAISS indexes via `VECTORSTORE_DIR` and `STUDENT_VECTORSTORE_DIR`
- Start command example: `uvicorn main:app --host 0.0.0.0 --port 8080`

## API Endpoints

### Student Routes
- `POST /student/login?student_id=123` - Login with student ID
- `POST /student/analyze_resume/` - Upload and analyze resume (PDF)
- `POST /student/chat?question=your_question` - Chat with AI about internships
- `GET /student/session-status` - Check current session status

### Company Routes
- `POST /company/add_internship` - Add new internship (for companies)

## Usage Flow

1. **Login**: `POST /student/login?student_id=123`
2. **Upload Resume**: `POST /student/analyze_resume/` (upload PDF file)
3. **Chat**: `POST /student/chat?question=What skills should I learn?`

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy
- **AI/ML**: LangChain, Google Gemini, Hugging Face
- **Vector Database**: FAISS
- **Database**: SQLite
- **File Processing**: PyPDF2

## Project Structure

```
sih/
├── main.py                 # FastAPI application
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore rules
├── env.example            # Environment variables template
├── db/                    # Database models and operations
├── routes/                # API route handlers
├── services/              # Business logic (AI, parsing)
└── utils/                 # Configuration and utilities
```

## Notes

- The system automatically falls back to Hugging Face embeddings when Gemini quota is exceeded
- Resume summaries and embeddings are cached in the database for performance
- Separate FAISS indexes are maintained for different embedding models
