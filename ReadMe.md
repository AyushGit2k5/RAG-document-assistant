# 📄 RAG Document Assistant

A deployed AI-powered document assistant that enables users to upload PDFs and ask questions using Retrieval-Augmented Generation (RAG).

## 🌐 Live Demo
https://rag-document-assistant-gwnmdxrfevfdbnrqhxbums.streamlit.app/
<img width="1147" height="581" alt="Screenshot 2026-04-22 171923" src="https://github.com/user-attachments/assets/e94c9168-4686-4bfe-8df8-b3e1d7d1c7bf" />

## 🚀 Features
- Upload and process PDF/TXT documents
- Extract and clean document text
- Semantic search using embeddings (MiniLM)
- Context-aware question answering with LLM (Groq - LLaMA 3)
- Document summarization
- Optimized for deployment (chunk limiting + caching)

## 🧠 System Architecture

User Query  
↓  
Embedding (MiniLM)  
↓  
Similarity Search (cosine similarity)  
↓  
Top-K Context Retrieval  
↓  
LLM (LLaMA 3 via Groq)  
↓  
Final Answer  

## ⚙️ Key Engineering Decisions

- Limited chunks to prevent memory crashes in cloud deployment  
- Used caching (`st.cache_data`) to avoid recomputing embeddings  
- Reduced chunk size and added overlap for better retrieval quality  
- Optimized batch size for embedding generation  

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Sentence Transformers (all-MiniLM-L6-v2)  
- Groq API (LLaMA 3)  
- NumPy  

## 🖥️ Run Locally
