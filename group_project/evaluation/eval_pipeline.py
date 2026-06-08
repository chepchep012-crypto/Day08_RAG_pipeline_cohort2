"""
RAG Evaluation Pipeline — local metrics + DeepEval (nếu có OPENAI_API_KEY).
"""

import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from group_project.rag_pipeline import RAGPipeline  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _overlap_score(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _local_metrics(item: dict, result: dict) -> dict:
    """Heuristic metrics khi không có DeepEval/OpenAI."""
    answer = result["answer"]
    contexts = [c["content"] for c in result["sources"]]
    context_text = " ".join(contexts)
    question = item["question"]
    expected = item["expected_answer"]
    expected_ctx = item.get("expected_context", "")

    faithfulness = _overlap_score(answer, context_text)
    relevance = 0.6 * _overlap_score(answer, question) + 0.4 * _overlap_score(answer, expected)
    recall = _overlap_score(expected_ctx, context_text) if expected_ctx else _overlap_score(expected, context_text)

    if contexts:
        useful = sum(1 for c in contexts if _overlap_score(c, expected) > 0.05 or _overlap_score(c, expected_ctx) > 0.05)
        precision = useful / len(contexts)
    else:
        precision = 0.0

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(relevance, 3),
        "context_recall": round(recall, 3),
        "context_precision": round(precision, 3),
    }


def evaluate_config(
    pipeline: RAGPipeline,
    golden_dataset: list[dict],
    config_name: str,
    use_reranking: bool,
) -> list[dict]:
    rows = []
    for item in golden_dataset:
        result = pipeline.generate_with_citation(
            item["question"],
            use_reranking=use_reranking,
        )
        metrics = _local_metrics(item, result)
        rows.append({
            "question": item["question"],
            "answer": result["answer"][:200],
            "config": config_name,
            **metrics,
        })
    return rows


def compare_configs(pipeline: RAGPipeline, golden_dataset: list[dict]) -> dict:
    configs = {
        "hybrid_rerank": True,
        "hybrid_no_rerank": False,
    }
    all_rows = {}
    for name, use_rerank in configs.items():
        all_rows[name] = evaluate_config(pipeline, golden_dataset, name, use_rerank)
    return all_rows


def _avg_metrics(rows: list[dict]) -> dict:
    keys = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    return {
        k: round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0.0
        for k in keys
    }


def export_results(comparison: dict):
    config_a = "hybrid_rerank"
    config_b = "hybrid_no_rerank"
    avg_a = _avg_metrics(comparison[config_a])
    avg_b = _avg_metrics(comparison[config_b])

    all_rows = comparison[config_a] + comparison[config_b]
    worst = sorted(all_rows, key=lambda r: r["faithfulness"] + r["answer_relevance"])[:3]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework su dung",
        "",
        "**Local heuristic metrics** (token overlap) — chay duoc khong can API key.",
        "Neu co `OPENAI_API_KEY`, co the nang cap sang DeepEval.",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Delta |",
        "|--------|---------------------------|------------------------------|-------|",
    ]

    for metric, label in [
        ("faithfulness", "Faithfulness"),
        ("answer_relevance", "Answer Relevance"),
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
    ]:
        delta = round(avg_a[metric] - avg_b[metric], 3)
        lines.append(f"| {label} | {avg_a[metric]} | {avg_b[metric]} | {delta:+} |")

    avg_a_all = round(sum(avg_a.values()) / 4, 3)
    avg_b_all = round(sum(avg_b.values()) / 4, 3)
    lines += [
        f"| **Average** | **{avg_a_all}** | **{avg_b_all}** | **{avg_a_all - avg_b_all:+.3f}** |",
        "",
        "---",
        "",
        "## A/B Comparison Analysis",
        "",
        "**Config A (hybrid + rerank):** Semantic + BM25 merge (RRF) + keyword reranking.",
        "",
        "**Config B (hybrid, no rerank):** Semantic + BM25 merge (RRF), khong rerank.",
        "",
        f"**Ket luan:** Config A {'tot hon' if avg_a_all >= avg_b_all else 'kem hon'} Config B "
        f"(avg {avg_a_all} vs {avg_b_all}). Reranking giup sap xep lai ket qua theo do lien quan keyword.",
        "",
        "---",
        "",
        "## Worst Performers (Bottom 3)",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Config |",
        "|---|----------|-------------|-----------|--------|--------|",
    ]

    for i, row in enumerate(worst, 1):
        q = row["question"][:50] + "..."
        lines.append(
            f"| {i} | {q} | {row['faithfulness']} | {row['answer_relevance']} "
            f"| {row['context_recall']} | {row['config']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "### Cai tien 1",
        "**Action:** Tang CHUNK_OVERLAP hoac dung MarkdownHeaderTextSplitter cho van ban phap luat.",
        "**Expected impact:** Context Recall tang vi giu nguyen cau truc Dieu/Khoan.",
        "",
        "### Cai tien 2",
        "**Action:** Bat OPENAI_API_KEY de dung GPT-4o-mini cho generation thay vi local fallback.",
        "**Expected impact:** Faithfulness va Answer Relevance tang ro ret.",
        "",
        "### Cai tien 3",
        "**Action:** Convert lai PDF phap luat bang MarkItDown (hien tai mot so file bi loi encoding).",
        "**Expected impact:** Lexical search tim dung Dieu/Khoan hon.",
        "",
    ]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported: {RESULTS_PATH}")


if __name__ == "__main__":
    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} test cases")

    pipeline = RAGPipeline(top_k=5)
    comparison = compare_configs(pipeline, golden)
    export_results(comparison)

    for name, rows in comparison.items():
        avg = _avg_metrics(rows)
        print(f"\n{name}: {avg}")
