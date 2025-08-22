import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

# Load summarization model
summarizer = pipeline("summarization",
                     model="facebook/bart-large-cnn",
                     revision="main")

# 📄 Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ✂️ Chunk text into manageable pieces
def chunk_text(text, max_tokens=500):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

# 🧠 Summarize long text in chunks
def summarize_long_text(text):
    chunks = chunk_text(text)
    summaries = []
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append("[Error summarizing chunk]")
        progress.progress((i + 1) / len(chunks))

    return "\n\n".join(summaries)

# 🚀 Streamlit UI
st.title("📚 Chapter Summarizer")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.info(f"Processing: {uploaded_file.name}")
    text = extract_text_from_pdf(uploaded_file)

    if len(text.strip()) == 0:
        st.error("No readable text found. Is this a scanned PDF?")
    else:
        st.write("Extracted text preview:", text[:500])
        summary = summarize_long_text(text)
        st.subheader("📝 Summary")
        st.write(summary)
