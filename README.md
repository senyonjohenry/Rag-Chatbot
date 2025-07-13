Document Intelligence RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot that lets users upload text documents and ask questions about their contents.

This project demonstrates building an interactive question-answering system over custom documents, using Streamlit for the UI, LangChain for retrieval orchestration, and ChromaDB as the vector store.

Project Purpose
   This chatbot helps users interrogate unstructured text documents by turning them into searchable knowledge bases.

Chosen document type: 
   Plain-text .txt files.

Typical use cases:

   Research papers

   Manuals and policy docs

   Log files

Architecture Overview
Components:

Streamlit Frontend

  Uploads text documents.

  Provides a chat interface.

  Displays answers and sources.

LangChain Pipeline

  Splits uploaded text into chunks.

  Generates embeddings (HuggingFace or Azure).

  Stores/retrieves embeddings from ChromaDB.

  Uses Groq LLM to answer questions with retrieved context.
  

ChromaDB

  Persistent vector database.

  Runs as a separate service in Docker Compose.


Flow Diagram

  User -> Streamlit -> LangChain -> ChromaDB
                           |
                        Groq LLM

Chunking Strategy

 Splitter: RecursiveCharacterTextSplitter

Settings:

  chunk_size=1000

  chunk_overlap=100

This balances context coverage with retrieval precision, ensuring the model has enough context to answer questions accurately.

Embedding Model
  Default: HuggingFace model BAAI/bge-small-en-v1.5

  Free and fast

  Supports CPU

  Normalized embeddings for better semantic search


Optional: Azure OpenAI Embeddings

  Can be enabled via environment variables

Assumptions
  Input documents are plain-text .txt.

  User provides their own API keys (Groq, optionally Azure).

  The vector store is local on-disk, mounted via Docker.

  Chunk size and retrieval parameters are configurable via the UI.

🚀 Getting Started

1. Clone the Repository

   git clone https://github.com/senyonjohenry/Rag-Chatbot.git
 
   cd Rag-Chatbot


2. Create a .env File

  GROQ_API_KEY=your_groq_api_key
  GROQ_MODEL=llama3-70b-8192

  AZURE_EMBEDDING_API_KEY=your_azure_openai_key
  AZURE_EMBEDDING_BASE=your_azure_openai_base_url

  CHROMA_SERVER_HOST=chroma
  CHROMA_SERVER_HTTP_PORT=8000

Note: Do not commit your .env file to GitHub!

3. Build the Docker Images
   
   docker compose build

   This command:

  Builds the Streamlit app image with all dependencies.

  Pulls the official ChromaDB image.

4. Start the App

   docker compose up
   
 This will:

  Start ChromaDB on port 8000.

  Launch the Streamlit frontend on port 8501.

5. Access the App
  Open your browser and go to:

http://localhost:8501


🗂️ Example Project Structure

.
├── app/
│   ├── main_app.py         # Streamlit UI
│   └── rag_pipeline.py     # LangChain + Chroma integration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml

🧪 Notes for Development

   The app persists embeddings to ./db.

   Uploading new documents rebuilds the vector store.

   You can tune chunk size and temperature from the Streamlit sidebar.

✅ Requirements

  Docker and Docker Compose

  Groq API key

  (Optional) Azure OpenAI API key for embeddings

Contributing
  Contributions welcome! Please open an issue to discuss improvements or fixes.

✨ Author
Senyonjo Henry






