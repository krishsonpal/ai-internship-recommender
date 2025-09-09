from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle, os
from db.crud import get_all_internships


os.environ["GOOGLE_API_KEY"] = "AIzaSyAejeADh8KAbsDbj_pZFNtg6PRMzmLUL7M"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def build_vectorstore(session):
    internships = get_all_internships()
    docs = []
    for i in internships:
        text = f"JobID: {i.job_id}\nTitle: {i.title}\nDescription: {i.description}\nSkills: {i.skills_required}\nLocation: {i.location}"
        docs.append(Document(page_content=text, metadata={"job_id": i.job_id}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # ✅ Save FAISS vectorstore using built-in method
    vectorstore.save_local("faiss_index")

    print("✅ Vector store built from internships table")
