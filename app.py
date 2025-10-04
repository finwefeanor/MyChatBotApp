import streamlit as st
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
