import streamlit as st
import pandas as pd
from docx import Document
import PyPDF2

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="DocuMind AI", page_icon="📄")

load_css("style.css")

st.markdown('<div class="main-title">📄 DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload your document and ask questions only based on that file.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx", "csv", "xlsx"]
)

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    st.write("File type:", uploaded_file.type)

    text = ""

    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        text = df.to_string()

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        text = df.to_string()

    st.subheader("Extracted Text")
    st.text_area("Document content", text, height=300)

    if text.strip():
        chunks = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embeddings = model.encode(chunks)

        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings).astype("float32"))

        st.subheader("Ask a question about your file")

        question = st.text_input("Enter your question", placeholder="Type your question and press Enter")

        if question:
            question_embedding = model.encode([question]).astype("float32")
            D, I = index.search(question_embedding, k=1)

            distance = D[0][0]

            if distance < 1.5:
                answer = chunks[I[0][0]]

                st.markdown(f"""
                <div class="answer-box">
                <b>Answer:</b><br><br>
                {answer}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                This information is not available in the uploaded file.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No readable text was found in this file.")
else:
    st.info("Please upload a document to start.")