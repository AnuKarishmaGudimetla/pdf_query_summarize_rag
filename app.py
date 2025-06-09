from pdf_loader import load_pdf_text, chunk_text
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from llm_response import get_llm_response
import os

def main():
    pdf_path = r"C:\Users\ANUKARISHMA\rag_pdf_qa\pdfs\HUMAN STRESS LEVEL DETECTION.pdf"

    print("Loading and chunking PDF...")
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"Loaded {len(chunks)} chunks.")

    print("Embedding chunks...")
    embedder = Embedder()
    embeddings = embedder.embed(chunks)
    print(f"Generated embeddings of shape: {embeddings.shape}")

    print("Saving embeddings to FAISS index...")
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save()
    print("FAISS index and metadata saved successfully.")

    retriever = Retriever()

    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        relevant_chunks = retriever.retrieve(user_query)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks.")

        answer = get_llm_response(user_query, relevant_chunks)
        print("\n🤖 Answer:\n")
        print(answer)

if __name__ == "__main__":
    main()
