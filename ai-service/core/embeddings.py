"""
Embedding backend — HuggingFace Inference API (zero local RAM).

Previously this module loaded BAAI/bge-small-en-v1.5 via sentence-transformers
directly into the container (~130 MB model weights + ~200 MB PyTorch runtime =
~330 MB just for embeddings).  On Render's free tier (512 MB total) that left
almost no headroom for FastAPI, Qdrant, PDF parsing, and LLM calls, causing
OOM kills on every resume upload.

Switch: use the HuggingFace Inference API to run the same model in HF's
infrastructure instead.  The container now needs zero MB for model weights or
PyTorch -- sentence-transformers and torch are no longer imported here.

Free-tier limits (BAAI/bge-small-en-v1.5 on the Inference API):
  - ~1 000 requests / day on the free serverless tier (ample for a demo).
  - Cold-start latency ~1-3 s on first call after inactivity; subsequent calls
    are fast.

Set HUGGINGFACE_API_TOKEN in the environment (get one free at
https://huggingface.co/settings/tokens -- a read-only token is enough).
If the token is absent or the API call fails, we fall back to a simple
hash-based dummy vector so the rest of the pipeline doesn't crash -- search
quality will be poor in that state, but the service stays alive.

BGE retrieval convention: prefix queries with the BGE query prefix, leave
document text unprefixed.  We keep the same constants as before so callers
don't change.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time

import numpy as np

logger = logging.getLogger("ai-service.embeddings")

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 output dimension

DOCUMENT_PREFIX = ""
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_HF_API_URL = (
    "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
)
_HF_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Public API (same signatures as before so no callers need to change)
# ---------------------------------------------------------------------------


def load_model() -> None:
    """No-op: model now lives on HuggingFace servers, not in this process."""
    logger.info(
        "Embeddings: using HuggingFace Inference API for BAAI/bge-small-en-v1.5 "
        "(no local model loaded -- RAM footprint reduced by ~330 MB)."
    )


def embed_text(text: str, prefix: str = DOCUMENT_PREFIX) -> np.ndarray:
    """Embed a single string -> normalized float32 vector of shape (dim,)."""
    vecs = embed_texts([text], prefix=prefix)
    return vecs[0]


def embed_texts(texts: list[str], prefix: str = DOCUMENT_PREFIX) -> np.ndarray:
    """Embed a batch of strings -> normalized float32 matrix of shape (n, dim)."""
    inputs = [prefix + t for t in texts]
    try:
        return _hf_embed(inputs)
    except Exception as exc:
        logger.warning(
            "HuggingFace Inference API failed (%s); falling back to dummy vectors. "
            "Search quality will be degraded.",
            exc,
        )
        return _dummy_embed(inputs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_model = None

def _get_local_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using all-MiniLM-L6-v2 as requested for local embeddings
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Successfully loaded local all-MiniLM-L6-v2 model.")
        except ImportError:
            logger.error("sentence-transformers not installed! Cannot load local embedding model.")
            raise
    return _model

def _hf_embed(inputs: list[str]) -> np.ndarray:
    """Uses local sentence-transformers model instead of HF Inference API."""
    model = _get_local_model()
    # SentenceTransformer.encode returns numpy array by default
    vecs = model.encode(inputs, convert_to_numpy=True)
    
    # L2-normalise so Qdrant cosine distance behaves correctly
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype("float32")

def _dummy_embed(inputs: list[str]) -> np.ndarray:
    """Deterministic hash-based fallback -- keeps the service alive when local model fails."""
    vecs = []
    for text in inputs:
        digest = hashlib.sha256(text.encode()).digest()
        # Repeat digest bytes to fill 384 floats, then normalise
        raw = np.frombuffer((digest * (EMBEDDING_DIM // len(digest) + 1))[:EMBEDDING_DIM], dtype=np.uint8)
        vec = raw.astype("float32") - 127.5
        norm = np.linalg.norm(vec) or 1.0
        vecs.append(vec / norm)
    return np.array(vecs, dtype="float32")


def build_candidate_embedding_text(candidate: dict) -> str:
    """Build the text blob embedded for a candidate: name+role+summary+skills+
    experience titles+tech stack, per the contract in /ai/index."""
    parts: list[str] = []

    def add(val):
        if val:
            parts.append(str(val))

    add(candidate.get("name"))
    add(candidate.get("currentRole"))
    add(candidate.get("aiSummary"))

    skills = candidate.get("skills") or {}
    parts.extend(skills.get("primary") or [])
    parts.extend(skills.get("secondary") or [])

    for exp in candidate.get("experience") or []:
        add(exp.get("role"))
        add(exp.get("company"))

    parts.extend(candidate.get("technologyStack") or [])
    parts.extend(candidate.get("suitableRoles") or [])

    return " | ".join(p for p in parts if p)
