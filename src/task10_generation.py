"""
Task 10 — Generation Có Citation.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

from src.task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks tránh lost in the middle: [1, 3, 5, 4, 2].
    Chunk score cao nhất ở đầu, thứ hai ở cuối, còn lại ở giữa.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(start, 0, -2):
        reordered.append(chunks[i])

    return reordered


def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string cho prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def _generate_local(
    query: str, context: str, chunks: list[dict], api_error: str | None = None
) -> str:
    """Fallback generation khi khong co LLM hoac API loi."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    source = chunks[0].get("metadata", {}).get("source", "Nguồn không xác định")
    doc_type = chunks[0].get("metadata", {}).get("type", "legal")
    year_match = re.search(r"20\d{2}", source)
    year = year_match.group() if year_match else "2021"

    snippet = chunks[0]["content"][:400].strip()
    citation = f"[{source}, {year}]"

    if api_error:
        note = f"**LLM API loi:** {api_error}\n\n(Dang dung fallback local — chi trich doan tu retrieval)"
    else:
        note = (
            "Luu y: Dang dung fallback local. Hay cau hinh OPENAI_API_KEY trong file .env "
            "va restart Streamlit."
        )

    return f"Dựa trên tài liệu tham khảo, {snippet} {citation}\n\n{note}"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation có citation."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key != "sk-xxx":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception:
            answer = _generate_local(query, context, reordered)
    else:
        answer = _generate_local(query, context, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt cho tội tàng trữ trái phép chất ma túy?")
    print(result["answer"])
