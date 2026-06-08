"""
RAG Pipeline wrapper — tích hợp Task 9 + Task 10 cho chatbot và evaluation.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from group_project.env_config import (  # noqa: E402
    get_openai_api_key,
    get_openai_base_url,
    get_openai_model,
    load_project_env,
)

load_project_env()

from src.task10_generation import (  # noqa: E402
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    _generate_local,
    format_context,
    reorder_for_llm,
)
from src.task9_retrieval_pipeline import retrieve  # noqa: E402

FOLLOW_UP_SIGNALS = (
    "thế", "còn", "vậy", "đó", "này", "điều đó", "điều này", "họ", "anh ấy",
    "cô ấy", "ông ấy", "sao", "không", "bao nhiêu", "ở đâu", "khi nào", "ai",
    "tại sao", "thì", "và", "nếu", "trên", "dưới",
)


def contextualize_query(query: str, chat_history: list[dict] | None) -> str:
    """
    Viết lại câu hỏi follow-up thành câu đủ ngữ cảnh để retrieval tìm đúng.
    Ví dụ: 'thế còn trên 100 gram?' -> 'Hình phạt tàng trữ trên 100 gram ma túy?'
    """
    if not chat_history:
        return query

    q_lower = query.lower().strip()
    is_followup = len(query) < 100 or any(s in q_lower for s in FOLLOW_UP_SIGNALS)
    if not is_followup:
        return query

    last_user = None
    last_assistant = None
    for msg in reversed(chat_history):
        if msg["role"] == "user" and last_user is None:
            last_user = msg["content"]
        elif msg["role"] == "assistant" and last_assistant is None:
            last_assistant = msg["content"][:400]
        if last_user and last_assistant:
            break

    if not last_user:
        return query

    return (
        f"Ngữ cảnh hội thoại trước:\n"
        f"- Câu hỏi trước: {last_user}\n"
        f"- Trả lời trước (tóm tắt): {last_assistant[:250]}\n"
        f"- Câu hỏi tiếp theo (follow-up): {query}\n"
        f"Hãy hiểu câu follow-up này liên quan đến chủ đề trên."
    )


def build_chat_messages(
    chat_history: list[dict] | None,
    context: str,
    query: str,
) -> list[dict]:
    """Xây messages array cho LLM — giữ lịch sử hội thoại."""
    system = (
        SYSTEM_PROMPT
        + "\n\nBạn đang trong cuộc hội thoại nhiều lượt. "
        "Trả lời câu hỏi follow-up dựa trên lịch sử chat VÀ context retrieval. "
        "Nếu câu hỏi tham chiếu 'thế', 'còn', 'đó' — hiểu theo ngữ cảnh trước đó."
    )
    messages = [{"role": "system", "content": system}]

    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"][:800],
                })

    messages.append({
        "role": "user",
        "content": (
            f"Context từ knowledge base:\n{context}\n\n"
            f"---\n\nCâu hỏi hiện tại: {query}"
        ),
    })
    return messages


class RAGPipeline:
    """Unified RAG interface cho chatbot và eval."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        use_reranking: bool = True,
        score_threshold: float = 0.3,
        top_k: int | None = None,
    ) -> list[dict]:
        return retrieve(
            query,
            top_k=top_k or self.top_k,
            score_threshold=score_threshold,
            use_reranking=use_reranking,
        )

    def generate_with_citation(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        use_memory: bool = True,
        use_reranking: bool = True,
        score_threshold: float = 0.3,
        top_k: int | None = None,
    ) -> dict:
        load_project_env()
        k = top_k or self.top_k
        history = chat_history if use_memory else None
        retrieval_query = contextualize_query(query, history)

        chunks = self.retrieve(
            retrieval_query,
            use_reranking=use_reranking,
            score_threshold=score_threshold,
            top_k=k,
        )
        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)

        api_key = get_openai_api_key()
        placeholder_keys = {"", "sk-xxx", "your-api-key", "xxx"}
        llm_error = None
        used_llm = False
        used_memory = bool(history)

        if api_key and api_key not in placeholder_keys:
            try:
                from openai import OpenAI

                base_url = get_openai_base_url()
                model = get_openai_model()
                client = OpenAI(api_key=api_key, base_url=base_url)
                messages = build_chat_messages(history, context, query)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                answer = response.choices[0].message.content
                used_llm = True
            except Exception as e:
                llm_error = f"{type(e).__name__}: {e}"
                answer = _generate_local(query, context, reordered, api_error=llm_error)
        else:
            llm_error = "OPENAI_API_KEY trong .env chua duoc cau hinh"
            answer = _generate_local(query, context, reordered)

        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
            "llm_error": llm_error,
            "used_llm": used_llm,
            "used_memory": used_memory,
            "retrieval_query": retrieval_query if retrieval_query != query else None,
        }
