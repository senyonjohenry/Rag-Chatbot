import os
import chromadb
from dotenv import load_dotenv

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings


# --------------------------------
# Load Environment Variables
# --------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_BASE = os.getenv("AZURE_EMBEDDING_BASE")

# --------------------------------
# 1. Get Embeddings Model (Azure)
# --------------------------------
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # Best balance of speed & quality
        model_kwargs={"device": "cpu"},       # Use "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}  # Crucial for similarity search
    )


# --------------------------------
# 2. Get Groq LLM
# --------------------------------
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL
    )

# --------------------------------
# 3. Load and Chunk Document
# --------------------------------
def load_and_chunk(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return chunks

# --------------------------------
# 4. Create Vector Store
# --------------------------------
def create_vector_store(chunks, persist_directory="./db"):
    embeddings = get_embeddings_model() 
    vectordb = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# --------------------------------
# 5. Load Existing Store
# --------------------------------
def load_vector_store(persist_directory="./db"):
    embeddings = get_embeddings_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

# --------------------------------
# 6. Build Retrieval QA Chain
# --------------------------------
def build_qa_chain(vectordb):
    llm = get_llm()
    
    prompt_template = """
You are a helpful assistant answering questions about a document.
Use only the provided context to answer accurately and clearly.

Context:
{context}

Question:
{question}

Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def get_chroma_client():
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_SERVER_HOST", "chromadb"),  # From env
        port=int(os.getenv("CHROMA_SERVER_HTTP_PORT", "8000")),
        settings=Settings(allow_reset=True)
    )
