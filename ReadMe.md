# 📄 RAG Document Assistant

A deployed AI-powered document assistant that enables users to upload PDFs and ask questions using Retrieval-Augmented Generation (RAG).

## 🌐 Live Demo
https://rag-document-assistant-gwnmdxrfevfdbnrqhxbums.streamlit.app/
## Screenshots below give a representation of the project. Please click the link to the live demo to view the project
<img width="1121" height="774" alt="Screenshot 2026-04-22 172006" src="https://github.com/user-attachments/assets/238b14da-7683-4c97-b420-54aad141d5ad" />
<img width="1133" height="874" alt="Screenshot 2026-04-22 171841" src="https://github.com/user-attachments/assets/860df2ba-732f-4578-818c-d86e625ad880" />


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
