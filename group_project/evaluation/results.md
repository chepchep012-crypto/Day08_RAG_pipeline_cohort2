# RAG Evaluation Results

## Framework su dung

**Local heuristic metrics** (token overlap) — chay duoc khong can API key.
Neu co `OPENAI_API_KEY`, co the nang cap sang DeepEval.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Delta |
|--------|---------------------------|------------------------------|-------|
| Faithfulness | 0.26 | 0.196 | +0.064 |
| Answer Relevance | 0.113 | 0.102 | +0.011 |
| Context Recall | 0.028 | 0.024 | +0.004 |
| Context Precision | 0.85 | 0.662 | +0.188 |
| **Average** | **0.313** | **0.246** | **+0.067** |

---

## A/B Comparison Analysis

**Config A (hybrid + rerank):** Semantic + BM25 merge (RRF) + keyword reranking.

**Config B (hybrid, no rerank):** Semantic + BM25 merge (RRF), khong rerank.

**Ket luan:** Config A tot hon Config B (avg 0.313 vs 0.246). Reranking giup sap xep lai ket qua theo do lien quan keyword.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Config |
|---|----------|-------------|-----------|--------|--------|
| 1 | Nghị định 105/2021 hướng dẫn thi hành luật nào?... | 0.184 | 0.014 | 0.008 | hybrid_rerank |
| 2 | Nghị định 105/2021 hướng dẫn thi hành luật nào?... | 0.184 | 0.014 | 0.008 | hybrid_no_rerank |
| 3 | Luật Phòng chống ma tuý 2021 quy định những hình t... | 0.135 | 0.1 | 0.024 | hybrid_no_rerank |

---

## Recommendations

### Cai tien 1
**Action:** Tang CHUNK_OVERLAP hoac dung MarkdownHeaderTextSplitter cho van ban phap luat.
**Expected impact:** Context Recall tang vi giu nguyen cau truc Dieu/Khoan.

### Cai tien 2
**Action:** Bat OPENAI_API_KEY de dung GPT-4o-mini cho generation thay vi local fallback.
**Expected impact:** Faithfulness va Answer Relevance tang ro ret.

### Cai tien 3
**Action:** Convert lai PDF phap luat bang MarkItDown (hien tai mot so file bi loi encoding).
**Expected impact:** Lexical search tim dung Dieu/Khoan hon.
