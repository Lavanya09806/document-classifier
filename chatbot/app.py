import streamlit as st
import os
import shutil
from document_processor import process_documents
from rag_engine import RAGEngine
from langchain_core.messages import HumanMessage, AIMessage

# Define directories
DB_PATH = "faiss_db"
TEMP_DIR = "temp_uploads"

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Cognitive RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS styling
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "vector_db_ready" not in st.session_state:
    st.session_state.vector_db_ready = os.path.exists(os.path.join(DB_PATH, "index.faiss"))

# Sidebar Configuration
st.sidebar.markdown('<h2 style="color: #818cf8; margin-bottom: 1.5rem;">⚙️ Settings</h2>', unsafe_allow_html=True)

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Google Gemini", "OpenAI"],
    help="Select the LLM and Embedding provider."
)

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder=f"Enter your {provider} API Key",
    help="Your key is only kept in memory and never saved to disk."
)

# Model selection based on provider
if provider == "Google Gemini":
    model_name = st.sidebar.selectbox(
        "Model",
        ["Gemini 2.5 Flash (Fast & Efficient)", "Gemini 2.5 Pro (High Reasoning)"],
        index=0
    )
else:
    model_name = st.sidebar.selectbox(
        "Model",
        ["GPT-4o Mini (Cost-Effective)", "GPT-4o (High Performance)"],
        index=0
    )

# RAG Hyperparameters Expander
with st.sidebar.expander("🛠️ Advanced RAG Parameters"):
    chunk_size = st.slider("Chunk Size (characters)", 200, 3000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 1000, 200, 50)
    k_retrieved = st.slider("Chunks to Retrieve (K)", 1, 15, 5, 1)
    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.3, 0.1)

st.sidebar.markdown("---")

# Reset button in Sidebar
if st.sidebar.button("🗑️ Clear Vector Database"):
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            st.sidebar.error(f"Error clearing db: {str(e)}")
    st.session_state.indexed_files = []
    st.session_state.vector_db_ready = False
    st.session_state.chat_history = []
    st.sidebar.success("Vector database cleared successfully!")
    st.rerun()

# Main Application Layout
st.markdown('<h1 class="gradient-text">🤖 Cognitive RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; margin-top: -0.5rem; margin-bottom: 2rem;">Upload documents (PDF, DOCX, TXT) and chat with them using semantic search and advanced retrieval-augmented generation.</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["💬 Chatbot Interface", "📂 Document Manager & Search"])

# Tab 2: Document Manager & Search
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop or browse files to add them to your vector store",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents"):
        if not api_key:
            st.error("Please enter a valid API Key in the sidebar before indexing documents.")
        elif not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Processing documents and generating embeddings..."):
                try:
                    # Clean temporary uploads directory
                    for f in os.listdir(TEMP_DIR):
                        try:
                            os.remove(os.path.join(TEMP_DIR, f))
                        except:
                            pass
                        
                    file_paths = []
                    file_names = []
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(temp_path)
                        file_names.append(uploaded_file.name)
                        
                    # Process and chunk documents
                    chunks = process_documents(file_paths, chunk_size, chunk_overlap)
                    
                    if not chunks:
                        st.error("No extractable text found in the uploaded documents.")
                    else:
                        # Generate embeddings and create vector store
                        engine = RAGEngine(provider, api_key)
                        engine.create_vector_store(chunks, DB_PATH)
                        
                        st.session_state.indexed_files = file_names
                        st.session_state.vector_db_ready = True
                        st.success(f"Successfully processed {len(file_names)} files into {len(chunks)} chunks and updated the vector database!")
                except Exception as e:
                    st.error(f"Failed to index documents: {str(e)}")
                finally:
                    # Cleanup temp directory
                    for f in os.listdir(TEMP_DIR):
                        try:
                            os.remove(os.path.join(TEMP_DIR, f))
                        except:
                            pass
    st.markdown('</div>', unsafe_allow_html=True)

    # Document Status and Search section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📋 Indexed Documents")
        if st.session_state.vector_db_ready:
            if st.session_state.indexed_files:
                for file_name in st.session_state.indexed_files:
                    st.markdown(f"📄 `{file_name}`")
            else:
                st.markdown("⚠️ Vector database is ready, but indexed file metadata is empty (database loaded from disk).")
        else:
            st.info("No documents indexed yet. Upload files above to get started.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🔍 Vector Document Search")
        search_query = st.text_input("Enter a query to search document chunks directly", placeholder="Type keywords or question...")
        if search_query:
            if not api_key:
                st.error("Please configure your API Key in the sidebar.")
            elif not st.session_state.vector_db_ready:
                st.error("Vector database is not ready. Please index documents first.")
            else:
                with st.spinner("Searching chunks..."):
                    try:
                        engine = RAGEngine(provider, api_key)
                        results = engine.search_documents(DB_PATH, search_query, k=k_retrieved)
                        if results:
                            st.write(f"Found {len(results)} matching chunks:")
                            for idx, res in enumerate(results):
                                score = res['score']
                                relevance = f"Score: {score:.4f}"
                                source = res['metadata'].get('source_display', res['metadata'].get('source', 'Unknown'))
                                with st.expander(f"Chunk {idx+1} | {source} ({relevance})"):
                                    st.markdown(f"**Content:**\n{res['content']}")
                        else:
                            st.info("No matching document chunks found.")
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 1: Chatbot Interface
with tab1:
    # Check if vector DB is ready
    if not st.session_state.vector_db_ready:
        st.warning("⚠️ **Get Started**: Please go to the **Document Manager & Search** tab, upload documents, and click **Process & Index Documents** to enable the chatbot.")
    
    # Display Chat History
    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if role == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Source Documents Used"):
                    for idx, doc_info in enumerate(message["sources"]):
                        st.markdown(f"**Source [{idx+1}]:** `{doc_info['source']}`")
                        st.markdown(f"*{doc_info['content']}*")
                        st.markdown("---")

    # User Chat Input
    if user_query := st.chat_input("Ask a question about your uploaded documents...", disabled=not st.session_state.vector_db_ready):
        if not api_key:
            st.error("Please enter a valid API Key in the sidebar to start chatting.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display assistant streaming response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                context_docs = []
                
                try:
                    # Convert session state history to LangChain messages for history-awareness
                    langchain_history = []
                    # Keep only the last 10 messages for context to avoid token bloat
                    for msg in st.session_state.chat_history[:-1][-10:]:
                        if msg["role"] == "user":
                            langchain_history.append(HumanMessage(content=msg["content"]))
                        else:
                            langchain_history.append(AIMessage(content=msg["content"]))
                    
                    engine = RAGEngine(provider, api_key)
                    rag_chain = engine.get_chat_chain(
                        DB_PATH, 
                        model_name, 
                        temperature=temperature, 
                        k=k_retrieved
                    )
                    
                    # Stream the response
                    response_stream = rag_chain.stream({
                        "input": user_query,
                        "chat_history": langchain_history
                    })
                    
                    for chunk in response_stream:
                        if "context" in chunk:
                            context_docs = chunk["context"]
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                            placeholder.markdown(full_response + "▌")
                    
                    placeholder.markdown(full_response)
                    
                    # Process sources
                    sources_list = []
                    if context_docs:
                        for doc in context_docs:
                            sources_list.append({
                                "source": doc.metadata.get("source_display", doc.metadata.get("source", "Unknown")),
                                "content": doc.page_content
                            })
                            
                        # Show sources in UI
                        with st.expander("📚 View Source Documents Used"):
                            for idx, doc_info in enumerate(sources_list):
                                st.markdown(f"**Source [{idx+1}]:** `{doc_info['source']}`")
                                st.markdown(f"*{doc_info['content']}*")
                                st.markdown("---")
                                
                    # Add response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources_list
                    })
                    
                except Exception as e:
                    placeholder.markdown("⚠️ An error occurred during response generation.")
                    st.error(f"Error details: {str(e)}")
