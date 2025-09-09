from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
import os
from db.database import Base, engine
from db.models import Internship
from routes import internship_routes, student_routes

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Internship Recommender")

# Session middleware for storing student_id after login
SESSION_SECRET = os.environ.get("SESSION_SECRET", "change-this-secret-in-production")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

app.include_router(internship_routes.router, prefix="/company", tags=["Company"])
app.include_router(student_routes.router, prefix="/student", tags=["Student"])
