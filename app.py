import os, io, traceback, numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from time import sleep

st.set_page_config(page_title="📄 Chat + PDF Q&A", page_icon="📄")
st.title("📄 Chat + PDF Q&A (minimal & robust)")

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
st.caption(f"OpenAI key present: {bool(OPENAI_API_KEY)}")

# ---------- Helpers ----------
def extract_pdf_text(file):
    """Read a single uploaded PDF and return text."""
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in reader.pages]
    # Optional cleanup for repeated headers/footers in your sample PDF:
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
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
                if resp.status_code != 200:
                    # Show exact API error so we don't get a generic "Oh no"
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
                    raise
                # brief backoff then retry
                sleep(1 + attempt)

    arr = np.vstack(all_vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms

def cosine_top_k(query_vec, doc_matrix, k=4):
    """Return indices + scores of the top-k most similar vectors."""
    sims = doc_matrix @ query_vec  # (n,d) @ (d,) -> (n,)
    k = int(min(max(1, k), sims.shape[0]))
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# ---------- Session state ----------
if "rag" not in st.session_state:
    st.session_state.rag = {"chunks": [], "vecs": None}

# ---------- UI: Upload & parse ----------
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

# ---------- UI: Build index ----------
st.subheader("2) Build index (chunk + embed)")
colA, colB, colC = st.columns(3)
with colA:
    chunk_chars = st.slider("Chunk size (chars)", 500, 4000, 1000, 100)
with colB:
    overlap = st.slider("Overlap (chars)", 0, 1000, 200, 50)
with colC:
    max_chunks = st.number_input("Max chunks (testing)", 10, 500, 60, 10)

if up and client and st.button("Build index"):
    try:
        text = extract_pdf_text(up)
        if not text.strip():
            st.error("No text extracted; cannot index.")
        else:
            st.info("Splitting into chunks…")
            chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
            st.write(f"Total chunks: **{len(chunks)}**")
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
                st.warning(f"Index limited to first {max_chunks} chunks for stability.")
            st.info("Embedding chunks…")
            vecs = embed_texts(chunks)  # (n,d)
            st.session_state.rag["chunks"] = chunks
            st.session_state.rag["vecs"] = vecs
            st.success(f"Indexed {len(chunks)} chunks. Embeddings shape: {vecs.shape}")
    except Exception:
        st.error("Indexing failed:")
        st.code(traceback.format_exc())

elif up and not client:
    st.warning("OPENAI_API_KEY is missing in Settings → Secrets.")

# ---------- UI: Ask ----------
st.subheader("3) Ask your PDF")
q = st.text_input("Your question")
top_k = st.slider("Top-K retrieved chunks", 1, 10, 4)

if q and client and st.session_state.rag["vecs"] is not None:
    try:
        with st.spinner("Retrieving relevant chunks…"):
            qv = embed_texts([q])[0]  # (d,)
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
                st.write(st.session_state.rag["chunks"][int(ii)])
    except Exception:
        st.error("Search/answer failed:")
        st.code(traceback.format_exc())
