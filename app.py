import os, io, traceback
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="PDF Chunk Test", page_icon="📄")
st.title("📄 Minimal PDF → Chunks test (no OpenAI)")

def chunk_text(text: str, chunk_chars=1000, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if i >= len(text): break
    return chunks

def extract_pdf_text(uploaded_files):
    full_texts = []
    for file in uploaded_files:
        data = file.read()
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages).replace("Unity Programmer Task", "")
        full_texts.append(text)
    return "\n\n".join(full_texts)

uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
overlap = st.slider("Overlap (chars)", 0, 1000, 200, 50)

if uploaded and st.button("Build index (no OpenAI)"):
    try:
        st.info("Reading PDF…")
        text = extract_pdf_text(uploaded)
        if not text.strip():
            st.error("No text extracted (are these scanned PDFs?).")
        else:
            st.info("Splitting…")
            chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
            st.success(f"OK. Chunks created: {len(chunks)}")
            with st.expander("Preview first 2 chunks"):
                for i, c in enumerate(chunks[:2], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(c[:1000] + ("…" if len(c) > 1000 else ""))
    except Exception:
        st.error("Exception occurred:")
        st.code(traceback.format_exc())
