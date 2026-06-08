"""
Task 4 — Chunking & Indexing vào Vector Store (local pickle store).
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.store import get_store, save_store

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# RecursiveCharacterTextSplitter: an toàn với cả legal lẫn news, không cần heading
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# all-MiniLM-L6-v2: nhẹ (90MB), nhanh, đủ tốt cho demo; fallback TF-IDF nếu chưa cài
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "local_pickle"


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents bằng RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def _embed_texts(texts: list[str]) -> tuple[list, str]:
    """Embed texts; fallback TF-IDF nếu sentence-transformers chưa có."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist(), EMBEDDING_MODEL
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=EMBEDDING_DIM)
        matrix = vectorizer.fit_transform(texts).toarray()
        return matrix.tolist(), "tfidf-fallback"


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks."""
    texts = [c["content"] for c in chunks]
    embeddings, model_name = _embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks + embeddings vào local pickle store."""
    texts = [c["content"] for c in chunks]
    embeddings, model_name = _embed_texts(texts)

    store = {
        "chunks": [
            {"content": c["content"], "metadata": c["metadata"]}
            for c in chunks
        ],
        "embeddings": embeddings,
        "model_name": model_name,
    }
    save_store(store)


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
