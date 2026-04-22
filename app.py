import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================
# 🔑 PASTE YOUR GROQ API KEY HERE (ONLY HERE)
# ============================================
GROQ_API_KEY = "GROQ_API_KEY"

if GROQ_API_KEY == "PASTE_YOUR_API_KEY_HERE":
    st.error("⚠️ Please paste your Groq API key at the top of the code.")
    st.stop()

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = Groq(api_key=GROQ_API_KEY)
    return embedder, client

embedder, client = load_models()

# -------------------------------
# Extract text (FIXED VERSION)
# -------------------------------
def extract_text(file_bytes, file_type):
    if file_type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page in doc:
            t = page.get_text("text")

            # fallback if empty
            if not t.strip():
                blocks = page.get_text("blocks")
                t = " ".join([b[4] for b in blocks if len(b) > 4])

            text += t + "\n"

        return text.strip()

    elif file_type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore")

    return ""

# -------------------------------
# Chunk text
# -------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve(query, chunks, embeddings):
    q_emb = embedder.encode([query])[0]

    scores = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
    )

    top_idx = np.argsort(scores)[-3:]
    return " ".join([chunks[i] for i in top_idx])

# -------------------------------
# Call LLM (Groq)
# -------------------------------
def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ WORKING MODEL
        messages=[
            {"role": "system", "content": "You are a helpful document assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

# -------------------------------
# Structured summary
# -------------------------------
def summarize_text(text):
    text = text[:3000]

    prompt = f"""
    Summarize this document in a structured way:

    - Main points
    - Key rules
    - Important clauses

    Document:
    {text}
    """

    return ask_llm(prompt)

# -------------------------------
# Answer query (RAG)
# -------------------------------
def answer_query(query, chunks, embeddings, full_text):
    if any(q in query.lower() for q in ["summary", "overview", "main idea"]):
        context = full_text[:2000]
    else:
        context = retrieve(query, chunks, embeddings)

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}

    Give a clear and accurate answer:
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
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks)

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)
            st.subheader("📝 Summary")
            st.write(summary)

        st.subheader("💬 Ask Questions")
        query = st.text_input("Enter your question")

        if query:
            with st.spinner("Thinking..."):
                answer = answer_query(query, chunks, embeddings, text)
            st.subheader("📌 Answer")
            st.write(answer)
    else:
        st.error("❌ No text could be extracted. Try a different PDF.")