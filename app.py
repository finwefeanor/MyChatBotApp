import os, io, traceback, json
import numpy as np
import streamlit as st
import requests
from pypdf import PdfReader
from openai import OpenAI
from time import sleep

st.set_page_config(page_title="🤖📄 MiniChatBot + PDF Q&A", page_icon="🤖")
st.title("Mini Chatbot")

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- Helpers ----------
def extract_pdf_text(uploaded_files):
    """Read uploaded PDFs and return combined text."""
    full_texts = []
    for file in uploaded_files:
        data = file.getvalue() if hasattr(file, "getvalue") else file.read()
        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        full_texts.append("\n".join(pages))
    return "\n\n".join(full_texts)

def chunk_text(text: str, chunk_chars=1000, overlap=200):
    """Split into overlapping character chunks (whitespace collapsed)."""
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunks.append(text[i:j])
        i = j - overlap
        if j >= len(text):
            break
    return chunks

def embed_texts(texts, batch=8, retries=3, timeout=30):
    """
    Requests-based embeddings to avoid SDK/runtime crashes.
    Returns L2-normalized numpy array of shape (n, d).
    """
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY missing.")
        st.stop()

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    all_vecs = []
    total = len(texts)
    done = 0
    pbar = st.progress(0.0) if total > batch else None

    for i in range(0, total, batch):
        chunk = texts[i:i+batch]
        payload = {
            "model": "text-embedding-3-small",
            "input": chunk
        }
        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    st.error(f"Embeddings HTTP {resp.status_code}: {resp.text[:500]}")
                    raise RuntimeError(f"Embeddings HTTP {resp.status_code}")
                data = resp.json()
                vecs = np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
                all_vecs.append(vecs)
                done += len(chunk)
                if pbar:
                    pbar.progress(done / total)
                break
            except Exception as e:
                if attempt == retries - 1:
                    st.error(f"Embedding error after {retries} retries: {str(e)}")
                    raise
                sleep(1 + attempt)

    if pbar:
        pbar.empty()
    
    arr = np.vstack(all_vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms

def cosine_top_k(query_vec, doc_matrix, k=5):
    """Return indices + scores of the top-k most similar vectors."""
    sims = doc_matrix @ query_vec  # (n,d) @ (d,) -> (n,)
    k = int(min(max(1, k), sims.shape[0]))
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# ---------- Session state ----------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "rag_state" not in st.session_state:
    st.session_state.rag_state = {
        "doc_chunks": [],
        "doc_vectors": None
    }

# ---------- UI ----------
st.title("📄 Chat + PDF Q&A")
st.caption(f"OpenAI key present: {bool(OPENAI_API_KEY)}")

tab1, tab2 = st.tabs(["💬 Chat", "📚 Ask your PDF"])

# ====== Chat tab ======
with tab1:
    for m in st.session_state.chat_messages:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask me anything...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not client:
            st.error("No OPENAI_API_KEY found. Add it in Settings → Secrets.")
        else:
            try:
                with st.spinner("Thinking..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.chat_messages,
                    )
                reply = resp.choices[0].message.content
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.chat_message("assistant").write(reply)
            except Exception:
                st.error("Chat failed:")
                st.code(traceback.format_exc())

# ====== PDF Q&A tab ======
with tab2:
    st.write("Upload a PDF (or a few), then ask questions. I'll search the document and answer using the most relevant parts.")
    
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    
    colA, colB = st.columns(2)
    with colA:
        chunk_chars = st.slider("Chunk size (chars)", 500, 2000, 1000, 100)
    with colB:
        top_k = st.slider("Top-K chunks", 2, 10, 5)

    if uploaded and client:
        if st.button("Build index"):
            try:
                with st.spinner("Reading & embedding..."):
                    text = extract_pdf_text(uploaded)
                    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=200)
                    
                    # Limit to prevent accidental huge PDFs
                    max_chunks = 1200
                    if len(chunks) > max_chunks:
                        chunks = chunks[:max_chunks]
                        st.warning(f"Too large; using first {max_chunks} chunks.")
                    
                    vecs = embed_texts(chunks, batch=8)
                    st.session_state.rag_state["doc_chunks"] = chunks
                    st.session_state.rag_state["doc_vectors"] = vecs
                    st.success(f"Indexed {len(chunks)} chunks.")
            except Exception:
                st.error("Failed to build index:")
                st.code(traceback.format_exc())

    elif uploaded and not client:
        st.error("No OPENAI_API_KEY set. Add it to secrets to build the index.")

    # Query section
    if st.session_state.rag_state["doc_chunks"] and st.session_state.rag_state["doc_vectors"] is not None:
        q = st.text_input("Your question about the PDF(s)")
        if q and client:
            try:
                with st.spinner("Searching..."):
                    qv = embed_texts([q])[0]  # normalized single vector
                    idx, sims = cosine_top_k(qv, st.session_state.rag_state["doc_vectors"], k=top_k)
                    context_parts = [st.session_state.rag_state["doc_chunks"][int(i)] for i in idx]
                    context = "\n\n---\n\n".join(context_parts)

                sys_prompt = (
                    "You are a helpful assistant. Answer the user's question strictly using the provided document context. "
                    "If the answer is not in the context, say you don't see it in the document."
                )
                user_prompt = (
                    f"Question:\n{q}\n\n"
                    f"Document context (may be partial):\n{context}\n\n"
                    "Answer clearly and cite key phrases from the context when possible."
                )

                with st.spinner("Answering..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                
                st.markdown("### Answer")
                st.write(resp.choices[0].message.content)
                
                with st.expander("Show retrieved chunks"):
                    for i, (ii, sim) in enumerate(zip(idx, sims), 1):
                        st.markdown(f"**Chunk {i} (score {float(sim):.3f})**")
                        st.write(st.session_state.rag_state["doc_chunks"][int(ii)])
            except Exception:
                st.error("Search/answer failed:")
                st.code(traceback.format_exc())
