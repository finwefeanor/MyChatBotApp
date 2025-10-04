import os
import io
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from time import sleep

st.set_page_config(page_title="Chat + PDF Q&A", page_icon="📄")

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- Helpers ----------
def chunk_text(text: str, chunk_chars=1000, overlap=200):
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())  # collapse whitespace
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunk = text[i:j]
        chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break
    return chunks

def embed_texts(texts, batch_size=100, retries=3):
    """Embed texts with batching + retries, return L2-normalized vectors."""
    all_vecs = []
    pbar = st.progress(0) if len(texts) > batch_size else None
    total = len(texts)
    done = 0
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
                vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
                all_vecs.append(vecs)
                done += len(batch)
                if pbar:
                    pbar.progress(done / total)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                sleep(1 + attempt)  # backoff & retry
    arr = np.vstack(all_vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms

def cosine_top_k(query_vec, doc_matrix, k=5):
    """Return indices + scores of top-k most similar vectors."""
    sims = doc_matrix @ query_vec  # doc_matrix (n,d), query_vec (d,)
    k = min(k, sims.shape[0])
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def extract_pdf_text(uploaded_files):
    """Extract plain text from uploaded PDFs."""
    full_texts = []
    for file in uploaded_files:
        reader = PdfReader(io.BytesIO(file.read()))
        pages = [p.extract_text() or "" for p in reader.pages]
        # optional cleanup (remove repeated headers/footers)
        text = "\n".join(pages).replace("Unity Programmer Task", "")
        full_texts.append(text)
    return "\n\n".join(full_texts)

# ---------- UI ----------
st.title("📄 Chat + PDF Q&A")

tab1, tab2 = st.tabs(["💬 Chat", "📚 Ask your PDF"])

# ====== Chat tab ======
with tab1:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for m in st.session_state.chat_messages:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask me anything...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not client:
            st.error("No OPENAI_API_KEY found. Add it to .streamlit/secrets.toml or env.")
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
            except Exception as e:
                st.error(f"Error: {e}")

# ====== PDF Q&A tab ======
with tab2:
    st.write("Upload a PDF (or multiple), then ask questions. I’ll search and answer from the document.")

    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    colA, colB = st.columns(2)
    with colA:
        chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
    with colB:
        top_k = st.slider("Top-K chunks", 2, 10, 5)

    if "rag_state" not in st.session_state:
        st.session_state.rag_state = {"doc_chunks": [], "doc_vectors": None}

    if uploaded and client:
        if st.button("Build index"):
            with st.spinner("Reading & embedding..."):
                text = extract_pdf_text(uploaded)
                if not text.strip():
                    st.error("No text could be extracted. Are these scanned PDFs?")
                    st.stop()
                chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=200)
                st.write(f"Preparing {len(chunks)} chunks…")
                max_chunks = 300
                if len(chunks) > max_chunks:
                    chunks = chunks[:max_chunks]
                    st.warning(f"Too large; only using first {max_chunks} chunks.")
                vecs = embed_texts(chunks)
                st.session_state.rag_state["doc_chunks"] = chunks
                st.session_state.rag_state["doc_vectors"] = vecs
                st.success(f"Indexed {len(chunks)} chunks.")

    if st.session_state.rag_state["doc_chunks"] and st.session_state.rag_state["doc_vectors"] is not None:
        q = st.text_input("Your question about the PDF(s)")
        if q and client:
            with st.spinner("Searching..."):
                qv = embed_texts([q])[0]  # (d,)
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

            try:
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
            except Exception as e:
                st.error(f"Error: {e}")
    elif uploaded and not client:
        st.error("No OPENAI_API_KEY set. Add it in Settings → Secrets to build the index.")
