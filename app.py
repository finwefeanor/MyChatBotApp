import streamlit as st
import numpy as np
from time import sleep
from pypdf import PdfReader
from openai import OpenAI
import io
import os

st.set_page_config(page_title="PDF Smoke", page_icon="📄")
st.title("📄 PDF upload test")

up = st.file_uploader("Upload a PDF", type=["pdf"])
if up and st.button("Parse"):
    data = up.read()
    r = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in r.pages]
    st.success(f"Pages: {len(pages)}")
    st.write((pages[0] or "")[:1000])

key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
st.write("OpenAI key present?" , bool(key))
client = OpenAI(api_key=key) if key else None

def chunk_text(text: str, chunk_chars=1000, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if i >= len(text): break
    return chunks

if "rag" not in st.session_state:
    st.session_state.rag = {"chunks": [], "vecs": None}

st.subheader("RAG settings")
chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
overlap = st.slider("Overlap (chars)", 0, 1000, 200, 50)

st.subheader("Build index")
if up and st.button("Split into chunks"):
    text = "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(up.getvalue())).pages])
    text = text.replace("Unity Programmer Task", "")  # optional cleanup
    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
    st.session_state.rag["chunks"] = chunks[:60]  # keep first 60 while testing
    st.success(f"Prepared {len(st.session_state.rag['chunks'])} chunks.")
    with st.expander("Preview first 2 chunks"):
        for i, c in enumerate(st.session_state.rag["chunks"][:2], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(c[:1000] + ("…" if len(c) > 1000 else ""))
