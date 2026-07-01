import os
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class RAGEngine:
    def __init__(self, provider: str, api_key: str):
        """
        Initializes the RAG Engine with the selected provider and API key.
        provider: 'Google Gemini' or 'OpenAI'
        """
        self.provider = provider
        self.api_key = api_key
        self.embeddings = self._get_embeddings()
        
    def _get_embeddings(self):
        """Initializes the embedding model based on the selected provider."""
        if self.provider == "Google Gemini":
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
        elif self.provider == "OpenAI":
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
    def _get_llm(self, model_name: str, temperature: float = 0.3):
        """Initializes the LLM based on the selected provider."""
        if self.provider == "Google Gemini":
            # Map user-friendly names to API model names
            api_model = "gemini-2.5-flash" if "flash" in model_name.lower() else "gemini-2.5-pro"
            return ChatGoogleGenerativeAI(
                model=api_model,
                google_api_key=self.api_key,
                temperature=temperature,
                streaming=True
            )
        elif self.provider == "OpenAI":
            api_model = "gpt-4o-mini" if "mini" in model_name.lower() else "gpt-4o"
            return ChatOpenAI(
                model=api_model,
                openai_api_key=self.api_key,
                temperature=temperature,
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def create_vector_store(self, documents: List[Document], save_path: str) -> FAISS:
        """Generates embeddings for document chunks and saves the FAISS index locally."""
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(save_path)
        return vectorstore

    def load_vector_store(self, load_path: str) -> Optional[FAISS]:
        """Loads a local FAISS index."""
        if not os.path.exists(os.path.join(load_path, "index.faiss")):
            return None
        return FAISS.load_local(
            load_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def search_documents(self, db_path: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs a direct similarity search to inspect or debug chunks."""
        db = self.load_vector_store(db_path)
        if not db:
            return []
        
        docs_and_scores = db.similarity_search_with_relevance_scores(query, k=k)
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        return results

    def get_chat_chain(self, db_path: str, model_name: str, temperature: float = 0.3, k: int = 5):
        """
        Creates and returns a conversational retrieval chain.
        Returns:
            The retrieval chain which can be used to stream results.
        """
        db = self.load_vector_store(db_path)
        if not db:
            raise ValueError("Vector store not found. Please upload and index documents first.")
            
        retriever = db.as_retriever(search_kwargs={"k": k})
        llm = self._get_llm(model_name, temperature)
        
        # 1. Create a history-aware retriever that rephrases the question
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", (
                "Given the above conversation, generate a search query to retrieve relevant documents. "
                "Do NOT answer the question directly, just output the search query itself. "
                "If the question doesn't require search or is generic, output the original question."
            ))
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, rephrase_prompt)
        
        # 2. Create the question-answering chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert, professional assistant specialized in RAG (Retrieval-Augmented Generation).\n"
                "Answer the user's questions using ONLY the provided context. If the answer cannot be found in the context "
                "or if the context is insufficient, state clearly that the answer is not available in the uploaded documents.\n"
                "Do not make up facts or use external knowledge unless it is minor common sense formatting (e.g., formatting tables).\n"
                "Provide detailed, structured responses when appropriate (bullet points, markdown tables, bold text).\n\n"
                "Context:\n{context}"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # 3. Create the final retrieval RAG chain
        rag_chain = create_retrieval_chain(retriever_chain, question_answer_chain)
        return rag_chain
