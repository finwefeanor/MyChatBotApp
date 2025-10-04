import os, io, traceback, numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from time import sleep

st.set_page_config(page_title="PDF Q&A (robust)", page_icon="🤖")
st.title("🤖 PDF Embedding Test (robust)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def chunk_text(text, chunk_chars=1000, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if i >= len(text): break
    return chunks

def extract_pdf_text(files):
    out = []
    for f in files:
        data = f.read()
        r = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in r.pages]
        out.append("\n".join(pages).replace("Unity Programmer Task", ""))
    return "\n\n".join(out)

def embed_texts(texts, batch=16, retries=3):
    all_vecs = []
    total = len(texts); done = 0
    pbar = st.progress(0.0) if total > batch else None
    for i in range(0, total, batch):
        b = texts[i:i+batch]
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model="text-embedding-3-small", input=b)
                vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
                all_vecs.append(vecs)
                done += len(b)
                if pbar: pbar.progress(done/total)
                break
            except Exception:
                if attempt == retries - 1: raise
                sleep(1 + attempt)
    arr = np.vstack(all_vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms

uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
overlap = st.slider("Overlap (chars)", 0, 1000, 200, 50)

if uploaded and st.button("Embed (max 48 chunks)"):
    if not client:
        st.error("OPENAI_API_KEY missing in Settings → Secrets.")
    else:
        try:
            st.info("Reading PDF…")
            text = extract_pdf_text(uploaded)
            if not text.strip():
                st.error("No text extracted.")
            else:
                st.info("Chunking…")
                chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
                if len(chunks) > 48:
                    chunks = chunks[:48]
                    st.warning("Limiting to first 48 chunks for stability.")
                st.write(f"Chunks: {len(chunks)}")

                st.info("Embedding…")
                vecs = embed_texts(chunks)
                st.success(f"Embeddings OK. Shape: {vecs.shape}")
        except Exception:
            st.error("Embedding failed:")
            st.code(traceback.format_exc())
