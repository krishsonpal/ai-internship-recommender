import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from google.api_core.exceptions import ResourceExhausted
import logging

# API Keys - Load from environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")

# Database URL
DATABASE_URL = "sqlite:///internships.db"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Primary embedding model (Gemini)
primary_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Fallback embedding model (Hugging Face API)
# Set the token in environment before creating the embeddings object
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACE_API_TOKEN")
fallback_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_embeddings_with_fallback():
    """Get embeddings with automatic fallback to Hugging Face API if Gemini fails."""
    try:
        # Test Gemini first
        test_embedding = primary_embeddings.embed_query("test")
        logger.info("✅ Using Gemini embeddings")
        return primary_embeddings, "gemini"
    except ResourceExhausted:
        logger.warning("⚠️ Gemini quota exceeded, switching to Hugging Face API embeddings")
        return fallback_embeddings, "huggingface"
    except Exception as e:
        logger.warning(f"⚠️ Gemini error, using Hugging Face API fallback: {e}")
        return fallback_embeddings, "huggingface"

# Default embeddings (will be set dynamically)
embeddings, embedding_type = get_embeddings_with_fallback()
