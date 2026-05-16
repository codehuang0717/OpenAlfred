from core.config import config
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger("rag.embedding")

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", config.RAG_EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(config.RAG_EMBEDDING_MODEL)
        dim = _embedding_model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. model=%s dim=%d", config.RAG_EMBEDDING_MODEL, dim)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    logger.debug("Embedding %d texts...", len(texts))
    embeddings = model.encode(texts, normalize_embeddings=True)
    logger.debug("Embedded %d texts -> %d vectors of dim=%d", len(texts), len(embeddings), embeddings.shape[1])
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    logger.debug("Embedding query: %.100s...", query)
    return embed_texts([query])[0]
