import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def format_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"From **{chunk['source']}**:\n{chunk['text']}"
        for chunk in context_chunks
    )
    return (
        "You are a helpful assistant. Use the following information from multiple PDFs to answer.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    )

def get_llm_response(query: str, context_chunks: list[dict], model: str = "llama3-8b-8192") -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": format_prompt(query, context_chunks)}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def get_summary_response(context_chunks: list[dict], mode: str = "one-line", model: str = "llama3-8b-8192") -> str:
    combined = "\n\n".join(chunk["text"] for chunk in context_chunks)

    if mode == "one-line":
        prompt = (
            "Summarize the following document in **one concise sentence**:\n\n"
            f"{combined}\n\nSummary:"
        )
        max_tokens = 80

    elif mode == "paragraph":
        prompt = (
            "Please write a **detailed paragraph summary** of the following document:\n\n"
            f"{combined}\n\nSummary:"
        )
        max_tokens = 300

    else:
        raise ValueError("Invalid mode. Use 'one-line' or 'paragraph'.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
