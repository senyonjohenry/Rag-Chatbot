import streamlit as st
import os
from rag_pipeline import get_chroma_client, load_and_chunk, create_vector_store, load_vector_store, build_qa_chain

    
# --- Config ---
st.set_page_config(
    page_title="RAG Chatbot", 
    layout="wide",
    page_icon="🤖" 
)
st.title("📚 Document Intelligence Chatbot") 

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    
    # Model settings expander
    with st.expander("Advanced Options"):
        chunk_size = st.slider("Chunk Size", 200, 2000, 1000, help="Smaller chunks = more precise retrieval")
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, help="Higher values = more creative answers")
    
    # Document Upload
    uploaded_file = st.file_uploader(
        "Upload a .txt document", 
        type=["txt"],
        help="Supported formats: Plain text (.txt)"  # Help text
    )
    
    # System status indicator
    if os.path.exists("./db"):
        st.success("✅ Database ready")
    else:
        st.warning("⚠️ Upload a document to begin")

# --- Document Processing ---
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        # Temp file handling with context manager
        with open("temp_upload.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        chunks = load_and_chunk("temp_upload.txt")
        create_vector_store(chunks, persist_directory="./db")
        os.remove("temp_upload.txt")  # NEW: Cleanup
        
    st.toast("✅ Document processed!", icon="✅")

# --- Chat Interface ---
st.header("Chat with Your Document")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (preserved across reruns)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"): 
            with st.expander("Source Excerpts"):
                for doc in message["sources"]:
                    st.code(doc.page_content, language="text")

# Enhancing input with on-submit
if prompt := st.chat_input("Ask anything about the document..."):
    if not os.path.exists("./db"):
        st.error("Please upload a document first!")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vectordb = load_vector_store(persist_directory="./db")
            qa_chain = build_qa_chain(vectordb)
            response = qa_chain({"query": prompt})  # Dict input for source docs
        
        # Display answer
        st.markdown(response["result"])
        
        #  Store sources if available
        sources = response.get("source_documents", [])
        if sources:
            with st.expander("View Sources"):
                for doc in sources:
                    st.caption(f"Source (score: {doc.metadata.get('score', 'N/A')})")  #  Metadata
                    st.code(doc.page_content, language="text")
        
        # Update session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["result"],
            "sources": sources
        })

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Delete Database"):
        if os.path.exists("./db"):
            import shutil
            shutil.rmtree("./db")
            st.rerun()