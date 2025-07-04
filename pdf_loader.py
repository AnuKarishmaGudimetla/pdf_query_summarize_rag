import fitz  # PyMuPDF
import os

def load_pdf_text(pdf_path: str) -> str:
    """
    Extracts all text from a single PDF file.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks (no metadata).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_and_chunk_pdfs(folder_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Walk through all PDFs in folder_path, extract & chunk each,
    returning a list of dicts with 'text' and 'source' (filename).
    """
    all_chunks = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".pdf"):
            continue
        full_text = load_pdf_text(os.path.join(folder_path, fname))
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk = full_text[start:end].strip()
            all_chunks.append({
                "text": chunk,
                "source": fname
            })
            start += chunk_size - overlap
    return all_chunks
