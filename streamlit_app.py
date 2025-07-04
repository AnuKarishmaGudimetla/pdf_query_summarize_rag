import streamlit as st
from pdf_loader import load_and_chunk_pdfs
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from llm_response import get_llm_response, get_summary_response
import os
import tempfile

st.set_page_config(page_title="📄 Chat with Your PDF", layout="wide")
st.title("📚 RAG PDF QA with Streamlit")

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunks_with_meta" not in st.session_state:
    st.session_state.chunks_with_meta = []

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info("Processing PDFs...")
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        pdf_paths.append(file_path)

    # Load & chunk
    chunks_with_meta = load_and_chunk_pdfs(temp_dir)
    st.session_state.chunks_with_meta = chunks_with_meta
    st.success(f"Loaded {len(chunks_with_meta)} chunks from {len(uploaded_files)} PDFs.")

    # Embed
    embedder = Embedder()
    texts = [c["text"] for c in chunks_with_meta]
    embeddings = embedder.embed(texts)

    # Store
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks_with_meta)
    st.session_state.store = store
    st.session_state.retriever = Retriever(vector_store=store)

# Show TOC and Q&A
if st.session_state.chunks_with_meta:
    tabs = st.tabs(["Ask a Question", "Summarize PDF"])

    # Ask a question
    with tabs[0]:
        st.subheader("Ask a question across all uploaded PDFs")
        user_q = st.text_input("Question:", placeholder="e.g. What is the purpose of this project?")
        if user_q:
            ctx = st.session_state.retriever.retrieve(user_q)
            st.write(f"Retrieved {len(ctx)} chunks from: {[c['source'] for c in ctx]}")
            answer = get_llm_response(user_q, ctx)
            st.markdown(f"**🤖 Answer:**\n{answer}")

    # TOC or Summary
    with tabs[1]:
        st.subheader("📄 Summarize a specific PDF")
        filenames = list({c['source'] for c in st.session_state.chunks_with_meta})
        selected_file = st.selectbox("Select a PDF", filenames)
        mode = st.radio("Summary type", ["One-line", "Full paragraph"])

        if st.button("Generate Summary"):
            chunks = [c for c in st.session_state.chunks_with_meta if c['source'] == selected_file]
            summary = get_summary_response(chunks, mode="paragraph" if mode == "Full paragraph" else "one-line")
            st.markdown(f"**Summary of {selected_file}:**\n{summary}")
