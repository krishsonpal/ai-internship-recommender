from db.vectorstore import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama  # ✅ fallback LLaMA
from google.api_core.exceptions import ResourceExhausted
import os
import logging

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Use gemini-1.5-flash instead of pro (fewer quota errors)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Fallback model (runs locally via Ollama)
fallback_llm = ChatOllama(model="llama3")

# Base folder for storing per-student chat histories
CHAT_HISTORY_DIR = os.path.join("data", "chat_histories")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def _get_history(student_id: str) -> FileChatMessageHistory:
    """Get or create conversation history for a student."""
    path = os.path.join(CHAT_HISTORY_DIR, f"{student_id}.json")
    return FileChatMessageHistory(path)


def save_conversation_context(student_id: str, role: str, content: str):
    """Save a message to the student's conversation history."""
    history = _get_history(student_id)

    if role == "user":
        history.add_message(HumanMessage(content=content))
    elif role == "ai":
        history.add_message(AIMessage(content=content))
    else:
        raise ValueError("Role must be either 'user' or 'ai'")


def _invoke_with_fallback(messages):
    """Try Gemini first, then fallback to LLaMA if quota exceeded."""
    try:
        response = llm.invoke(messages)
        return {"model": "gemini", "content": response.content}
    except ResourceExhausted as e:
        logger.warning(f"⚠️ Gemini quota exceeded, switching to LLaMA: {e}")
        response = fallback_llm.invoke(messages)
        return {"model": "llama", "content": response.content}
    except Exception as e:
        logger.error(f"⚠️ Unexpected error with Gemini, using LLaMA: {e}")
        response = fallback_llm.invoke(messages)
        return {"model": "llama", "content": response.content}


def get_llm_recommendation_reason(student_id: str, recommendations: list) -> str:
    """Generate a concise explanation using prior conversation context and the recommended internships."""
    history = _get_history(student_id)

    # Format recommendations into readable text
    rec_text = "\n\n".join([
        f"Recommendation {idx+1}:\n{rec}" if isinstance(rec, str)
        else f"Recommendation {idx+1}:\n{rec}"
        for idx, rec in enumerate(recommendations)
    ])

    # ✅ Truncate to avoid token overload
    rec_text = rec_text[:3000]

    # Instruction prompt
    prompt = (
        "Given the student's conversation context and the following recommended internships, "
        "explain briefly why these were chosen and how they match the student's profile. "
        "Be helpful and specific."
    )

    # ✅ Only keep last 5 messages to avoid quota issues
    messages = history.messages[-5:] + [
        HumanMessage(content=f"Context recall and recommendations:\n\n{rec_text}\n\n{prompt}")
    ]

    response = _invoke_with_fallback(messages)
    logger.info(f"LLM ({response['model']}) Response: {response['content']}")

    # Save AI's response for continuity
    save_conversation_context(student_id, "ai", response["content"])

    return response["content"]


def get_internship_recommendations(student_summary: str, top_k: int = 5):
    """Get internship recommendations without saving to conversation history."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(student_summary, k=top_k)

    recs = [
        {
            "job_id": res.metadata.get("job_id"),
            "internship": res.page_content
        }
        for res in results
    ]
    return recs


def get_internship_recommendations_by_vector(query_embedding: list, top_k: int = 5):
    """Get internship recommendations using a precomputed embedding vector."""
    vectorstore = load_vectorstore()
    
    try:
        results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    except AssertionError as e:
        print(f"⚠️ Embedding dimension mismatch error: {e}")
        print("⚠️ Falling back to text-based search instead of vector search")
        # Fall back to text search if dimensions don't match
        return get_internship_recommendations("resume text placeholder", top_k)
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        print("⚠️ Falling back to text-based search")
        return get_internship_recommendations("resume text placeholder", top_k)

    recs = [
        {
            "job_id": res.metadata.get("job_id"),
            "internship": res.page_content
        }
        for res in results
    ]
    return recs

def answer_with_context(student_id: str, user_query: str, resume_summary: str = "", recommendations: list = None) -> str:
    """Answer a user question with resume context and recommendations, no conversation history."""
    # Save user's question
    save_conversation_context(student_id, "user", user_query)

    # Build context with resume and recommendations
    context_parts = []
    if resume_summary:
        context_parts.append(f"Student Resume Summary: {resume_summary[:1000]}")  # Limit resume text
    
    if recommendations:
        rec_text = "\n".join([f"- {rec.get('internship', '')[:200]}" for rec in recommendations[:3]])  # Top 3, limited
        context_parts.append(f"Recommended Internships:\n{rec_text}")

    context = "\n\n".join(context_parts)
    
    # Single message with all context
    messages = [
        HumanMessage(content=f"{context}\n\nUser Question: {user_query}\n\nPlease provide a helpful answer based on the context above.")
    ]

    response = _invoke_with_fallback(messages)

    # Save AI's answer
    save_conversation_context(student_id, "ai", response["content"])

    return response["content"]
