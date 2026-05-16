from rag.embedding import embed_query
from rag.store import _get_collection
from core.config import config
from utils.logger import get_logger

logger = get_logger("rag.retriever")


def search(user_id: str, query: str, top_k: int | None = None) -> list[dict]:
    """Search the user's knowledge base and return relevant chunks."""
    if top_k is None:
        top_k = config.RAG_TOP_K

    query_embedding = embed_query(query)
    collection = _get_collection(user_id)

    doc_count = collection.count()
    if doc_count == 0:
        logger.debug("Search skipped: collection is empty")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, doc_count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = round(1.0 - distance, 4)
            meta = results["metadatas"][0][i]

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "heading": meta.get("heading", ""),
                "content": results["documents"][0][i],
                "score": score,
                "image_count": meta.get("image_count", 0),
            })

    logger.debug(
        "Search query=%.80s top_k=%d results=%d scores=%s",
        query, top_k, len(chunks),
        [c["score"] for c in chunks],
    )
    return chunks
