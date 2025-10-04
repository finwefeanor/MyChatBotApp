import streamlit as st
from pypdf import PdfReader
import io

st.set_page_config(page_title="PDF Smoke", page_icon="📄")
st.title("📄 PDF upload test")

up = st.file_uploader("Upload a PDF", type=["pdf"])
if up and st.button("Parse"):
    data = up.read()
    r = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in r.pages]
    st.success(f"Pages: {len(pages)}")
    st.write((pages[0] or "")[:1000])
