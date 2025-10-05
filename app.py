import os, io, json, math, traceback
from time import sleep

import requests
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="📄 Chat + PDF Q&A", page_icon="📄")
st.title("📄 Chat + PDF Q&A (robust minimal)")

# ---------------- OpenAI keys/clients ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
st.caption(f"OpenAI key present: {bool(OPENAI_API_KEY)}")

chat_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
EMBED_MODEL = "text-embedding-3-small"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------- Helpers ----------------
def extract_pdf_text(file):
    """Read a single uploaded PDF and return text."""
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in reader.pages]
    # optional cleanup for sample doc headers/footers
    text = "\n".join(pages).replace("Unity Programmer Task", "")
    return text

def chunk_text(text: str, chunk_chars=1000, overlap=200):
    """Split into overlapping character chunks (whitespace collapsed)."""
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(i + chunk_chars, len(text))
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if i >= len(text):
            break
    return chunks

def l2_normalize(vec):
    s = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/s for v in vec]

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cosine_top_k(query_vec, doc_matrix, k=4):
    """doc_matrix: list[list[float]] (already normalized), query_vec normalized list[float]"""
    sims = [dot(query_vec, dv) for dv in doc_matrix]
    k = max(1, min(int(k), len(sims)))
    idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
    return idx, [sims[i] for i in idx]

def embed_texts(texts, batch=8, retries=3, timeout=30):
    """
    Requests-based embeddings (SDK-free for this call).
    Returns list of L2-normalized vectors (list[list[float]]).
    """
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY missing. Add it in Settings → Secrets.")
        st.stop()

    out = []
    total = len(texts)
    done = 0
    pbar = st.progress(0.0) if total > batch else None

    for i in range(0, total, batch):
        chunk = texts[i:i+batch]
        payload = {"model": EMBED_MODEL, "input": chunk}
        for attempt in range(retries):
            try:
                r = requests.post(EMBEDDINGS_URL, headers=HEADERS, data=json.dumps(payload), timeout=timeout)
                if r.status_code != 200:
                    st.error(f"Embeddings HTTP {r.status_code}: {r.text[:500]}")
                    raise RuntimeError(f"HTTP {r.status_code} during embeddings")
                data = r.json()["data"]
                for item in data:
                    out.append(l2_normalize(item["embedding"]))
                done += len(chunk)
                if pbar:
                    pbar.progress(done/total)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                sleep(1 + attempt)  # simple backoff
    return out

# ---------------- Session state ----------------
if "rag" not in st.session_state:
    st.session_state.rag = {"chunks": [], "vecs": None}

# ---------------- 1) Upload & preview ----------------
st.subheader("1) Upload a PDF")
up = st.file_uploader("Upload a PDF", type=["pdf"])

if up and st.button("Parse PDF (preview)"):
    try:
        text = extract_pdf_text(up)
        if not text.strip():
            st.error("No text extracted (scanned PDF?).")
        else:
            st.success("PDF parsed.")
            st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
    except Exception:
        st.error("Parsing failed:")
        st.code(traceback.format_exc())

# ---------------- 2) Build index ----------------
st.subheader("2) Build index (chunk + embed)")
colA, colB, colC = st.columns(3)
with colA:
    chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
with colB:
    overlap = st.slider("Overlap (chars)", 0, 1000, 200, 50)
with colC:
    max_chunks = st.number_input("Max chunks (testing)", 10, 500, 24, 2)

if up and chat_client and st.button("Build index"):
    try:
        text = extract_pdf_text(up)
        if not text.strip():
            st.error("No text extracted; cannot index.")
        else:
            st.info("Splitting into chunks…")
            chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
            st.write(f"Total chunks found: **{len(chunks)}**")

            # keep very small during testing to avoid any timeouts
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
                st.warning(f"Limiting to first {max_chunks} chunks for stability.")

            st.info("Embedding chunks…")
            vecs = embed_texts(chunks, batch=8)  # list[list[float]]
            st.session_state.rag["chunks"] = chunks
            st.session_state.rag["vecs"] = vecs
            st.success(f"Indexed {len(chunks)} chunks.")
    except Exception:
        st.error("Exception while building index:")
        st.code(traceback.format_exc())

elif up and not chat_client:
    st.warning("OPENAI_API_KEY is missing in Settings → Secrets.")

# ---------------- 3) Ask ----------------
st.subheader("3) Ask your PDF")
q = st.text_input("Your question")
top_k = st.slider("Top-K retrieved chunks", 1, 10, 4)

if q and chat_client and st.session_state.rag["vecs"] is not None:
    try:
        with st.spinner("Retrieving relevant chunks…"):
            qv = embed_texts([q])[0]  # normalized query vector
            idx, sims = cosine_top_k(qv, st.session_state.rag["vecs"], k=top_k)
            parts = [st.session_state.rag["chunks"][int(i)] for i in idx]
            context = "\n\n---\n\n".join(parts)

        sys_prompt = (
            "You are a helpful assistant. Answer the user's question strictly using the provided document context. "
            "If the answer is not in the context, say you don't see it in the document."
        )
        user_prompt = (
            f"Question:\n{q}\n\n"
            f"Document context (may be partial):\n{context}\n\n"
            "Answer clearly and cite key phrases from the context when possible."
        )

        with st.spinner("Generating answer…"):
            resp = chat_client.chat.completions.create(
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
                st.write(st.session_state.rag["chunks"][int(ii)])
    except Exception:
        st.error("Search/answer failed:")
        st.code(traceback.format_exc())
