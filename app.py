import streamlit as st
import fitz  # PyMuPDF
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================
# 🔑 LOAD API KEY
# ============================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("❌ API key missing")
    st.stop()

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = Groq(api_key=GROQ_API_KEY)
    return embedder, client

embedder, client = load_models()

# -------------------------------
# Extract text
# -------------------------------
def extract_text(file_bytes, file_type):
    if file_type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page in doc:
            t = page.get_text("text")

            if not t.strip():
                blocks = page.get_text("blocks")
                t = " ".join([b[4] for b in blocks if len(b) > 4])

            text += t + "\n"

        return text.strip()

    elif file_type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore")

    return ""

# -------------------------------
# Chunk text (optimized)
# -------------------------------
def chunk_text(text, chunk_size=120, overlap=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks

# -------------------------------
# Build embeddings ONCE
# -------------------------------
@st.cache_data
def build_embeddings(chunks):
    embeddings = embedder.encode(
        chunks,
        batch_size=8,
        show_progress_bar=False
    )
    return embeddings

# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve(query, chunks, embeddings):
    q_emb = embedder.encode([query])[0]

    scores = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
    )

    top_idx = np.argsort(scores)[-3:][::-1]

    return " ".join([chunks[i] for i in top_idx])

# -------------------------------
# Call LLM (with error handling)
# -------------------------------
def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Answer ONLY using the provided context. If unsure, say you don't know."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error from LLM: {str(e)}"

# -------------------------------
# Summary
# -------------------------------
def summarize_text(text):
    text = text[:2500]

    prompt = f"""
Summarize this document clearly:

- Main points
- Key facts
- Important details

Document:
{text}
"""
    return ask_llm(prompt)

# -------------------------------
# RAG Answer
# -------------------------------
def answer_query(query, chunks, embeddings):
    context = retrieve(query, chunks, embeddings)

    prompt = f"""
Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer clearly:
"""
    return ask_llm(prompt)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="RAG Assistant", layout="centered")

st.title("📄 RAG Document Assistant (Groq Powered)")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    text = extract_text(uploaded_file.getvalue(), uploaded_file.type)
    text = " ".join(text.split())

    st.subheader("📌 Extracted Text Preview")
    st.write(text[:500])

    if text.strip():
        # 🔥 Build chunks safely
        chunks = chunk_text(text)

        # 🔥 HARD LIMIT (critical for cloud)
        if len(chunks) > 40:
            chunks = chunks[:40]

        st.info(f"Chunks used: {len(chunks)}")

        # 🔥 Build embeddings ONCE (cached)
        embeddings = build_embeddings(chunks)

        # -------------------
        # Summary
        # -------------------
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)

            st.subheader("📝 Summary")
            st.write(summary)

        # -------------------
        # Q&A
        # -------------------
        st.subheader("💬 Ask Questions")
        query = st.text_input("Enter your question")

        if query:
            with st.spinner("Thinking..."):
                answer = answer_query(query, chunks, embeddings)

            st.subheader("📌 Answer")
            st.write(answer)

    else:
        st.error("❌ No text could be extracted. Try a different PDF.")
