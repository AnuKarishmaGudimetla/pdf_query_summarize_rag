import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following information from a PDF to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""
    return prompt

def get_llm_response(query: str, context_chunks: list[str], model: str = "gpt-3.5-turbo") -> str:
    prompt = format_prompt(query, context_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
