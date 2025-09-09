import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from db.crud import get_all_internships
from utils.config import embeddings, embedding_type  # current embeddings and type identifier

# Keep separate FAISS stores per embedding type to avoid dimension mismatch
VECTORSTORE_PATH = f"faiss_index_{embedding_type}"  # Folder, not .pkl
STUDENT_VECTORSTORE_DIR = f"student_faiss_{embedding_type}"  # Base folder for per-student FAISS indexes

def build_vectorstore():
    internships = get_all_internships()
    docs = []

    for i in internships:
        text = f"JobID: {i.job_id}\nTitle: {i.title}\nDescription: {i.description}\nSkills: {i.skills_required}\nLocation: {i.location}"
        docs.append(Document(page_content=text, metadata={"job_id": i.job_id}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)  # ✅ Save using FAISS
    print("✅ Vector store built/updated")

def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        build_vectorstore()
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)  # ✅ Load using FAISS





def build_or_update_student_vectorstore(student_id: str, resume_text: str):
	"""Create or update a FAISS vectorstore for a specific student.

	- Splits the provided resume text into chunks
	- Embeds and stores them in a per-student FAISS index under STUDENT_VECTORSTORE_DIR/{student_id}
	- Re-usable later via load_student_vectorstore(student_id)
	"""
	# Ensure base directory exists
	os.makedirs(STUDENT_VECTORSTORE_DIR, exist_ok=True)

	student_store_path = os.path.join(STUDENT_VECTORSTORE_DIR, str(student_id))

	# Prepare documents from resume text
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	chunks = splitter.split_documents([Document(page_content=resume_text, metadata={"student_id": str(student_id)})])

	# If the student's index exists, load and append. Otherwise, create new
	if os.path.exists(student_store_path):
		vectorstore = FAISS.load_local(student_store_path, embeddings, allow_dangerous_deserialization=True)
		vectorstore.add_documents(chunks)
	else:
		vectorstore = FAISS.from_documents(chunks, embeddings)

	# Persist to disk
	vectorstore.save_local(student_store_path)
	print(f"✅ Student vector store built/updated for student_id={student_id}")


def load_student_vectorstore(student_id: str):
	"""Load the FAISS vectorstore for a specific student. Returns None if not found."""
	student_store_path = os.path.join(STUDENT_VECTORSTORE_DIR, str(student_id))
	if not os.path.exists(student_store_path):
		return None
	return FAISS.load_local(student_store_path, embeddings, allow_dangerous_deserialization=True)

