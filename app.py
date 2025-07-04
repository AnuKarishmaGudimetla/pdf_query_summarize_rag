from pdf_loader import load_and_chunk_pdfs
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from llm_response import get_llm_response, get_summary_response  # Combine both here

def main():
    pdf_folder = r"C:\Users\ANUKARISHMA\rag_pdf_qa\pdfs"
    print("Loading & chunking all PDFs…")
    chunks_with_meta = load_and_chunk_pdfs(pdf_folder)
    print(f"→ Got {len(chunks_with_meta)} chunks from {pdf_folder}.")

    print("Embedding chunks…")
    embedder = Embedder()
    texts = [c["text"] for c in chunks_with_meta]
    embeddings = embedder.embed(texts)
    print(f"→ Embeddings shape: {embeddings.shape}")

    print("Saving to FAISS…")
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks_with_meta)
    store.save()
    print("✓ FAISS index + metadata saved.")

    retriever = Retriever()

    while True:
        print("\nOptions:")
        print("  1. Ask a question")
        print("  2. Generate mini‑TOC (one‑line summaries per PDF)")
        print("  3. Exit")
        choice = input("Choose [1/2/3]: ").strip()

        if choice == "3":
            print("Goodbye!")
            break

        elif choice == "2":
            # Group all chunks by filename
            docs = {}
            for chunk in chunks_with_meta:
                fname = chunk["source"]
                docs.setdefault(fname, []).append(chunk)

            print("\nAvailable PDFs:")
            for i, fname in enumerate(docs.keys(), 1):
                print(f"  {i}. {fname}")

            selection = input("Enter the number of the PDF you want to summarize: ").strip()
            if not selection.isdigit() or int(selection) not in range(1, len(docs) + 1):
                print("Invalid selection.")
                continue

            fname = list(docs.keys())[int(selection) - 1]
            chunks = docs[fname]

            summary_type = input("One-liner or full summary? [1/2]: ").strip()
            if summary_type == "1":
                mode = "one-line"
            elif summary_type == "2":
                mode = "paragraph"
            else:
                print("Invalid input. Defaulting to one-line.")
                mode = "one-line"

            print(f"\n📄 Summary for {fname}:\n")
            try:
                summary = get_summary_response(chunks, mode=mode)
                print(summary)
            except Exception as e:
                print(f"[Error generating summary] {e}")

        elif choice == "1":
            q = input("\nEnter your question (or 'exit'): ")
            if q.lower() == "exit":
                break
            ctx = retriever.retrieve(q)
            print(f"Retrieved {len(ctx)} chunks from: {[c['source'] for c in ctx]}")
            ans = get_llm_response(q, ctx)
            print("\n🤖 Answer:\n", ans)

if __name__ == "__main__":
    main()
