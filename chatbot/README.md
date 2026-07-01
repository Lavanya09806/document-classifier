# Cognitive RAG Chatbot

An advanced, production-ready Retrieval-Augmented Generation (RAG) Chatbot built with Python, Streamlit, LangChain, and FAISS. It supports multiple document formats, local semantic chunking, stateful chat history, and flexible LLM configuration (Google Gemini or OpenAI).

---

## 🌟 Key Features

* **Multi-Format Document Upload**: Support for PDF, DOCX, TXT, MD, CSV, JSON, XML, and PY files.
* **Intelligent Document Chunking**: Leverages `RecursiveCharacterTextSplitter` from LangChain to build optimal overlapping chunks.
* **Vector Store Management**: Uses `FAISS` (Facebook AI Similarity Search) locally. No external database server setup required.
* **Dual-Tab Interface**:
  * **💬 Chatbot Interface**: Sleek, streaming chatbot with visual expansions highlighting the exact document chunks and page numbers used to formulate each answer.
  * **📂 Document Manager & Search**: View loaded documents and perform direct semantic similarity queries against your vector store with precision relevance scores.
* **History-Aware Search Query Formulation**: Automatically converts multi-turn conversational history into context-independent search queries before scanning the vector database.
* **Ultra-Premium Design**: Customized glassmorphic dark mode styling using Outfit typography, smooth animations, and visual enhancements.

---

## 🚀 Getting Started

### 📋 Prerequisites

* **Python 3.9 - 3.11** installed. You can check your version with:
  ```powershell
  python --version
  ```

### 🔧 Installation Steps

1. **Clone or Navigate to the Directory**:
   ```powershell
   cd "c:\Users\lavanya\OneDrive\Desktop\aiml\chatbot"
   ```

2. **Create a Virtual Environment** (Optional but Recommended):
   ```powershell
   python -m venv venv
   # Activate on Windows:
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

Launch the Streamlit web server:
```powershell
streamlit run app.py
```

After executing the command, Streamlit will open a web browser tab pointing to the local host address (usually `http://localhost:8501`).

---

## ⚙️ Configuration & Usage Walkthrough

1. **Configure Provider & API Key**:
   * In the sidebar, select your preferred model provider: **Google Gemini** (default) or **OpenAI**.
   * Input your API Key. (The key is kept in memory only and is not logged or written to disk).
   
2. **Adjust Advanced Parameters (Optional)**:
   * **Chunk Size**: Controls the character length of each chunk.
   * **Chunk Overlap**: Overlapping characters between adjacent chunks to maintain context boundaries.
   * **Chunks to Retrieve (K)**: The number of highest-scoring chunks sent as context to the LLM.
   * **Temperature**: Control LLM response creativity (lower values are recommended for strict factuality).

3. **Upload and Index Documents**:
   * Navigate to the **Document Manager & Search** tab.
   * Drag & drop your `.pdf`, `.docx`, or `.txt` files into the uploader box.
   * Click **Process & Index Documents** to read, split, embed, and index your files into the FAISS store.

4. **Direct Vector Search (Testing & Debugging)**:
   * Use the **Vector Document Search** widget on the same tab to type query keywords.
   * It retrieves the most relevant chunks immediately along with their exact similarity score and source metadata (e.g. filename and page number).

5. **Start Chatting**:
   * Move back to the **Chatbot Interface** tab.
   * Ask questions related to your documents. The assistant will answer using only the indexed texts and list its sources directly beneath its response.

---

## 📁 Architecture & File Layout

* `app.py`: Streamlit frontend application, routing user actions, chat interface, and sidebar parameters.
* `document_processor.py`: Parses raw text, extracts text page-by-page from PDFs/DOCX, and divides files into semantic text blocks.
* `rag_engine.py`: Encapsulates embedding generation, FAISS index reading/writing, direct similarity searches, and LangChain history-aware retrieval chains.
* `styles.css`: Custom CSS styling injected into the Streamlit page for premium typography, card visuals, custom tags, and tab styling.
* `requirements.txt`: Specified Python libraries with version constraints.
