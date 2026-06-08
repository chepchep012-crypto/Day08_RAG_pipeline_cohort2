"""
RAG Chatbot — Streamlit UI cho bài tập nhóm.

Chạy:
    streamlit run group_project/app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from group_project.env_config import (  # noqa: E402
    ENV_PATH,
    get_openai_api_key,
    get_openai_base_url,
    get_openai_model,
    load_project_env,
)

load_project_env()

import importlib
import group_project.rag_pipeline as _rag_module

importlib.reload(_rag_module)
from group_project.chat_storage import (  # noqa: E402
    clear_chat_history,
    load_chat_history,
    save_chat_history,
)
from group_project.rag_pipeline import RAGPipeline  # noqa: E402

st.set_page_config(
    page_title="RAG Chatbot - Pháp luật Ma tuý",
    page_icon="⚖️",
    layout="wide",
)

st.title("RAG Chatbot — Pháp luật & Tin tức Ma tuý")
st.caption("Hybrid Search + Reranking + Citation | Day 08 Group Project")

with st.sidebar:
    st.header("Cấu hình")
    top_k = st.slider("Top K chunks", 3, 10, 5)
    use_rerank = st.checkbox("Bật Reranking", value=True)
    score_threshold = st.slider("Score threshold", 0.0, 0.9, 0.3, 0.05)
    show_scores = st.checkbox("Hiển thị relevance score", value=True)
    use_memory = st.checkbox("Conversation memory (follow-up)", value=True)
    memory_turns = st.slider("Số lượt nhớ", 2, 10, 6) if use_memory else 0
    persist_chat = st.checkbox("Lưu lịch sử khi reload trang", value=True)

    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = []
        clear_chat_history()
        st.rerun()

    load_project_env()
    api_key = get_openai_api_key()
    if not api_key or api_key == "sk-xxx":
        st.warning("OPENAI_API_KEY chua cau hinh")
        st.caption(f".env: {ENV_PATH} | exists: {ENV_PATH.exists()}")
    else:
        masked = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "(key ngan?)"
        base_url = get_openai_base_url() or "https://api.openai.com/v1"
        model = get_openai_model()
        st.success(f"LLM: {model}")
        st.caption(f"Key: {masked} | len={len(api_key)}")
        st.caption(f"Base URL: {base_url}")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history() if persist_chat else []
    st.session_state.chat_loaded = True

# Tao pipeline moi moi lan rerun de tranh cache class cu
pipeline = RAGPipeline(top_k=top_k)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "unknown")
                    score = src.get("score", 0)
                    st.markdown(f"**[{i}]** `{source_name}` — score: `{score:.3f}`")
                    st.text(src["content"][:300] + "...")

if prompt := st.chat_input("Hỏi về pháp luật ma tuý hoặc tin tức nghệ sĩ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if persist_chat:
        save_chat_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = (
        st.session_state.messages[:-1][-memory_turns:]
        if use_memory and st.session_state.messages[:-1]
        else None
    )

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp..."):
            result = pipeline.generate_with_citation(
                prompt,
                chat_history=chat_history,
                use_memory=use_memory,
                use_reranking=use_rerank,
                score_threshold=score_threshold,
                top_k=top_k,
            )

        tags = []
        if result.get("used_llm"):
            tags.append("LLM API")
        if result.get("used_memory"):
            tags.append(f"Memory ({len(chat_history or [])} msgs)")
        if tags:
            st.success(" | ".join(tags))
        elif result.get("llm_error"):
            st.error(f"LLM API loi: {result['llm_error']}")

        if result.get("retrieval_query"):
            with st.expander("Query đã contextualize cho retrieval"):
                st.text(result["retrieval_query"])

        st.markdown(result["answer"])
        st.caption(f"Retrieval: **{result['retrieval_source']}**")

        if result["sources"]:
            with st.expander(f"Nguồn tham khảo ({len(result['sources'])} chunks)"):
                for i, src in enumerate(result["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "unknown")
                    doc_type = meta.get("type", "?")
                    score = src.get("score", 0)
                    header = f"**[{i}]** `{source_name}` ({doc_type})"
                    if show_scores:
                        header += f" — score: `{score:.3f}` | source: `{src.get('source', '?')}`"
                    st.markdown(header)
                    st.text(src["content"][:400])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

    if persist_chat:
        save_chat_history(st.session_state.messages)
