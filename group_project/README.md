# Bài Tập Nhóm — RAG Chatbot + Evaluation

## Sản phẩm

**Option B: RAG Chatbot** — chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức nghệ sĩ, có citation và conversation memory.

## Kiến Trúc

```
User (Streamlit UI)
       │
       ▼
group_project/app.py          ← Chat UI + conversation memory
       │
       ▼
group_project/rag_pipeline.py ← Wrapper
       │
       ├── Task 9: retrieve()     hybrid (semantic + BM25) → RRF → rerank → PageIndex fallback
       └── Task 10: generate()    reorder chunks → format context → GPT/local → citation
       │
       ▼
data/index/store.pkl          ← Vector store local
data/standardized/            ← Markdown corpus
```

## Phân Công

| Thành viên | Nhiệm vụ | Trạng thái |
|-----------|----------|------------|
| Member 1 | Task 1-3: Data collection + convert | Done |
| Member 2 | Task 4-6: Indexing + search modules | Done |
| Member 3 | Task 7-9: Rerank + retrieval pipeline | Done |
| Member 4 | Task 10 + Chatbot UI (Streamlit) | Done |
| Cả nhóm | Evaluation pipeline + báo cáo | Done |

## Chạy Chatbot

```bash
# Cài dependencies (tối thiểu)
pip install streamlit python-dotenv sentence-transformers rank-bm25 langchain-text-splitters

# Index data (nếu chưa chạy)
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing

# Chạy app
streamlit run group_project/app.py
```

Tùy chọn: thêm `OPENAI_API_KEY` vào `.env` để câu trả lời chất lượng hơn.

## Chạy Evaluation

```bash
python group_project/evaluation/eval_pipeline.py
```

Output: `group_project/evaluation/results.md`

- Golden dataset: 16 cặp Q&A (`golden_dataset.json`)
- Metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision
- A/B: hybrid+rerank vs hybrid không rerank

## Tính năng Chatbot

- Giao diện chat Streamlit
- Trả lời có citation (Task 10)
- Conversation memory (2 turn gần nhất)
- Hiển thị source documents + relevance score
- Cấu hình top_k, reranking, threshold trong sidebar
