import os, io, traceback
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from time import sleep

st.set_page_config(page_title="Chat + PDF Q&A", page_icon="📄")

# ---------- OpenAI ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- Helpers ----------
def chunk_text(text: str, chunk_chars=1000, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if i >= len(text): break
    return chunks

def embed_texts(texts, batch_size=32, retries=3):
    all_vecs = []
    total = len(texts); done = 0
    pbar = st.progress(0.0) if total > batch_size else None
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
                vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
                all_vecs.append(vecs); done += len(batch)
                if pbar: pbar.progress(done / total)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                sleep(1 + attempt)  # simple backoff
    arr = np.vstack(all_vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms

def cosine_top_k(query_vec, doc_matrix, k=5):
    sims = doc_matrix @ query_vec  # (n,d) @ (d,) -> (n,)
    k = int(min(max(1, k), sims.shape[0]))
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def extract_pdf_text(uploaded_files):
    full_texts = []
    for file in uploaded_files:
        data = file.read()
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
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

    prompt = st.chat_input("Ask me anything…")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if not client:
            st.error("No OPENAI_API_KEY found (Settings → Secrets).")
        else:
            try:
                with st.spinner("Thinking…"):
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
    st.write("Upload a PDF (or multiple), build the index, then ask questions based on the document.")

    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    colA, colB, colC = st.columns(3)
    with colA:
        chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
    with colB:
        overlap = st.slider("Overlap (chars)", 0, 1000, 200, 50)
    with colC:
        top_k = st.slider("Top-K chunks", 1, 10, 4)

    if "rag_state" not in st.session_state:
        st.session_state.rag_state = {"doc_chunks": [], "doc_vectors": None}

    if uploaded and client:
        if st.button("Build index"):
            try:
                st.info("Step 1/4: Reading PDF…")
                text = extract_pdf_text(uploaded)
                if not text.strip():
                    st.error("No text extracted (are these scanned PDFs?).")
                    st.stop()

                st.info("Step 2/4: Splitting into chunks…")
                chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
                st.write(f"Chunks created: **{len(chunks)}**")
                max_chunks = 60  # keep conservative; raise later
                if len(chunks) > max_chunks:
                    chunks = chunks[:max_chunks]
                    st.warning(f"Large file: only indexing first {max_chunks} chunks.")

                st.info("Step 3/4: Embedding chunks…")
                vecs = embed_texts(chunks)  # L2-normalized

                st.info("Step 4/4: Saving in memory…")
                st.session_state.rag_state["doc_chunks"] = chunks
                st.session_state.rag_state["doc_vectors"] = vecs
                st.success(f"Indexed {len(chunks)} chunks. You can ask questions now.")
            except Exception:
                st.error("Indexing failed with an exception:")
                st.code(traceback.format_exc())

    # Q&A
    if st.session_state.rag_state["doc_chunks"] and st.session_state.rag_state["doc_vectors"] is not None:
        q = st.text_input("Your question about the PDF(s)")
        if q and client:
            try:
                st.info("Retrieving relevant chunks…")
                qv = embed_texts([q])[0]
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

                st.info("Generating answer…")
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
    elif uploaded and not client:
        st.error("No OPENAI_API_KEY set. Add it in Settings → Secrets to build the index.")
