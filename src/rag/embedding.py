import os
import threading

# Never phone Hugging Face on load — BGE-M3 is already in the local hub cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from core.config import config
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger("rag.embedding")

_embedding_model: SentenceTransformer | None = None
_load_lock = threading.Lock()


def _embedding_device() -> str:
    """Use CUDA only when this wheel actually contains the GPU's arch (e.g. sm_120)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability(0)
        tag = f"sm_{major}{minor}"
        arches = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
        if tag in arches or f"sm_{major}0" in arches:
            return "cuda"
        logger.warning(
            "GPU %s not in this PyTorch wheel (%s); embeddings will use CPU",
            tag,
            arches,
        )
        return "cpu"
    except Exception:
        return "cpu"


def embedding_ready() -> bool:
    return _embedding_model is not None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _load_lock:
        if _embedding_model is not None:
            return _embedding_model
        device = _embedding_device()
        logger.info("Loading embedding model: %s device=%s (local cache only)", config.RAG_EMBEDDING_MODEL, device)
        _embedding_model = SentenceTransformer(
            config.RAG_EMBEDDING_MODEL, device=device, local_files_only=True
        )
        dim = _embedding_model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded. model=%s dim=%d device=%s",
            config.RAG_EMBEDDING_MODEL,
            dim,
            device,
        )
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
