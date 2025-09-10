from db.vectorstore import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama  # ✅ fallback LLaMA
from google.api_core.exceptions import ResourceExhausted
from utils.config import get_embeddings_with_fallback
import os
import logging
from typing import List, Dict, Tuple
import re

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


def embed_text(text: str) -> Tuple[List[float], str]:
    """Embed text using current embeddings with fallback. Returns (vector, provider)."""
    embeddings_model, provider = get_embeddings_with_fallback()
    vector = embeddings_model.embed_query(text)
    return vector, provider


def average_embeddings(vectors: List[List[float]]) -> List[float]:
    """Compute element-wise average of multiple embedding vectors."""
    if not vectors:
        return []
    length = len(vectors[0])
    # Guard: ensure uniform length
    filtered = [v for v in vectors if len(v) == length]
    if not filtered:
        return []
    sums = [0.0] * length
    for v in filtered:
        for i, val in enumerate(v):
            sums[i] += val
    count = float(len(filtered))
    return [s / count for s in sums]


def detect_intent(user_query: str) -> str:
    """Classify intent: 'recommend_internships', 'suggest_skills', or 'general'."""
    prompt = (
        "Classify the user's intent into one of: recommend_internships, suggest_skills, general.\n"
        "- recommend_internships: asking for internships, matches, best roles, opportunities.\n"
        "- suggest_skills: asking what skills to learn/pursue/improve.\n"
        "- general: anything else.\n"
        f"User: {user_query}\n"
        "Answer with only the label."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        label = resp.content.strip().lower()
        if "recommend" in label or "intern" in label or label == "recommend_internships":
            return "recommend_internships"
        if "skill" in label or label == "suggest_skills":
            return "suggest_skills"
        return "general"
    except Exception:
        return "general"


def score_skill_match(resume_summary: str, internship_text: str) -> Dict:
    """Compute skill match using rule-based extraction first, then LLM, then naive token overlap."""

    def extract_skills_rule_based(text: str) -> List[str]:
        # Prefer line starting with "Skills:" then comma-separated list
        m = re.search(r"(?im)^\s*skills\s*:\s*(.+)$", text)
        skills_line = m.group(1) if m else ""
        skills = []
        if skills_line:
            skills = [s.strip().lower() for s in re.split(r",|/|\||;", skills_line) if s.strip()]
        # Also capture common tech tokens present as standalone words (basic heuristic)
        common = [
            "python","java","javascript","typescript","c++","c#","sql","nosql","mongodb","postgres","mysql",
            "react","angular","vue","node","express","django","flask","fastapi","spring","dotnet",
            "html","css","tailwind","bootstrap","nextjs","nestjs",
            "aws","gcp","azure","docker","kubernetes","terraform","git","linux",
            "ml","machine learning","deep learning","pytorch","tensorflow","sklearn","opencv",
            "nlp","llm","langchain","faiss","retrieval","rag","airflow","spark","hadoop",
            "firebase","supabase","graphql","rest","kafka","redis","rabbitmq"
        ]
        norm_text = text.lower()
        for tok in common:
            if tok in norm_text and tok not in skills:
                skills.append(tok)
        # Dedup and limit
        seen = set()
        out = []
        for s in skills:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:50]

    # Rule-based extraction
    resume_skills = extract_skills_rule_based(resume_summary)
    internship_skills = extract_skills_rule_based(internship_text)

    if internship_skills:
        rs = set(resume_skills)
        iset = set(internship_skills)
        matched = sorted(list(rs.intersection(iset)))
        missing = sorted(list(iset - rs))
        denom = max(1, len(iset))
        percent = int(100 * len(matched) / denom)
        # If we have at least some internship skills, trust this computation
        if percent > 0 or len(matched) + len(missing) > 0:
            return {"percent_match": percent, "matched_skills": matched[:10], "missing_skills": missing[:10]}

    # LLM-based scoring as secondary option
    prompt = (
        "You are a precise evaluator. Given a student's resume summary and an internship description,"
        " return a strict JSON with: percent_match (0-100 integer), matched_skills (array of strings),"
        " missing_skills (array of strings). Keep arrays concise and deduplicated.\n\n"
        f"Resume Summary:\n{resume_summary[:2000]}\n\n"
        f"Internship Text:\n{internship_text[:2000]}\n\n"
        "JSON only: {\"percent_match\": 0-100, \"matched_skills\": [], \"missing_skills\": []}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        import json
        data = json.loads(content)
        pm = int(max(0, min(100, int(data.get("percent_match", 0)))))
        matched = [s.strip().lower() for s in data.get("matched_skills", []) if isinstance(s, str)]
        missing = [s.strip().lower() for s in data.get("missing_skills", []) if isinstance(s, str)]
        return {"percent_match": pm, "matched_skills": matched[:10], "missing_skills": missing[:10]}
    except Exception:
        # Final fallback: naive token overlap
        rs = set([t.lower() for t in re.findall(r"[a-zA-Z+#\.]+", resume_summary)])
        it = set([t.lower() for t in re.findall(r"[a-zA-Z+#\.]+", internship_text)])
        overlap = rs.intersection(it)
        denom = max(1, len(it))
        percent = int(100 * len(overlap) / denom)
        return {
            "percent_match": percent,
            "matched_skills": sorted(list(overlap))[:10],
            "missing_skills": sorted(list(it - rs))[:10]
        }


def augment_recommendations_with_scoring(resume_summary: str, recs: List[Dict], top_n: int = 3) -> List[Dict]:
    """For top_n recs, add percent_match and skill_gap fields (missing_skills)."""
    enhanced = []
    for idx, rec in enumerate(recs):
        if idx < top_n:
            internship_text = rec.get("internship", "")
            score = score_skill_match(resume_summary, internship_text)
            rec = {
                **rec,
                "percent_match": score.get("percent_match", 0),
                "matched_skills": score.get("matched_skills", []),
                "skill_gap": score.get("missing_skills", []),
            }
        enhanced.append(rec)
    return enhanced


def suggest_skills_to_pursue(user_query: str, resume_summary: str) -> str:
    """LLM suggests concise, prioritized skills to learn based on resume and query."""
    prompt = (
        "Given the student's resume summary and question, suggest the most impactful 5-8 skills to pursue."
        " Group by theme if helpful. Keep it concise, actionable, and tailored to the profile.\n\n"
        f"Resume Summary:\n{resume_summary[:2000]}\n\n"
        f"Question:\n{user_query}\n\n"
        "Respond in bullets, each with a 1-line rationale."
    )
    resp = _invoke_with_fallback([HumanMessage(content=prompt)])
    return resp["content"]

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
