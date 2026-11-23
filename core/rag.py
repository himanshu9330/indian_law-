"""
RAG (Retrieval-Augmented Generation) layer for LexAI.
Uses SentenceTransformers ('all-MiniLM-L6-v2') and Pinecone serverless vector index.
Implements the 2-stage cluster-aware semantic search from the original pipeline.
"""

import os
from typing import Any, Optional

from core.config import (
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    RAG_TOP_K,
)

_EMBEDDING_MODEL_CACHE = None
_PINECONE_CLIENT_CACHE = None


def get_embedding_model():
    """Lazy-load and cache the SentenceTransformer model."""
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL_CACHE = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error loading embedding model {EMBEDDING_MODEL}: {e}")
            return None
    return _EMBEDDING_MODEL_CACHE


def check_pinecone_availability(
    api_key: Optional[str] = None,
    index_name: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Check if Pinecone API key is provided and index is reachable.
    Returns (is_available, status_message).
    """
    key = api_key or PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")
    if not key or key.strip() in ("", "your_pinecone_api_key_here"):
        return False, "Pinecone API key is not configured"

    try:
        idx = get_pinecone_index(api_key=key, index_name=index_name)
        if idx is None:
            return False, "Unable to initialize Pinecone index client"
        return True, "Pinecone index reachable"
    except Exception as e:
        return False, f"Pinecone connection error: {e}"


def get_pinecone_index(api_key: Optional[str] = None, index_name: Optional[str] = None):
    """Lazy-load and return the Pinecone Index handle."""
    global _PINECONE_CLIENT_CACHE
    key = api_key or PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")
    if not key or key.strip() in ("", "your_pinecone_api_key_here"):
        return None

    target_index = index_name or PINECONE_INDEX
    try:
        from pinecone import Pinecone
        if _PINECONE_CLIENT_CACHE is None:
            _PINECONE_CLIENT_CACHE = Pinecone(api_key=key.strip())
        return _PINECONE_CLIENT_CACHE.Index(target_index)
    except Exception as e:
        print(f"Error accessing Pinecone index '{target_index}': {e}")
        return None


def search_rag(
    query: str,
    top_k: int = RAG_TOP_K,
    api_key: Optional[str] = None,
    index_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute 2-stage cluster-aware retrieval:
      Stage 1: Find best-matching cluster from query embedding.
      Stage 2: Retrieve top_k chunks filtered by that cluster.

    Returns:
      {
        "context": str,
        "score": float,
        "confidence_score": float, # calibrated in [0.0, 1.0]
        "cluster_id": int | None,
        "chunks": list[dict],
        "chunk_count": int,
        "source": "pinecone_rag",
        "status": "success" | "key_missing" | "connection_error" | "no_documents_found"
      }
    """
    key = api_key or PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")
    if not key or key.strip() in ("", "your_pinecone_api_key_here"):
        return {
            "context": "",
            "score": 0.0,
            "confidence_score": 0.0,
            "cluster_id": None,
            "chunks": [],
            "chunk_count": 0,
            "source": "pinecone_rag",
            "status": "key_missing",
            "message": "Pinecone API key is not configured.",
        }

    model = get_embedding_model()
    index = get_pinecone_index(api_key=key, index_name=index_name)

    if model is None or index is None:
        return {
            "context": "",
            "score": 0.0,
            "confidence_score": 0.0,
            "cluster_id": None,
            "chunks": [],
            "chunk_count": 0,
            "source": "pinecone_rag",
            "status": "connection_error",
            "message": "Embedding model or Pinecone index unreachable.",
        }

    for attempt in range(2):
        try:
            query_embedding = model.encode(query).tolist()

            # Stage 1: Find best cluster
            first_search = index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True,
            )

            best_cluster_id = None
            if first_search.get("matches") and len(first_search["matches"]) > 0:
                best_cluster_id = first_search["matches"][0]["metadata"].get("cluster_id")

            contexts = []
            chunks_info = []
            best_score = 0.0

            if best_cluster_id is not None:
                # Stage 2: Deep dive with cluster filter
                deep_search = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter={"cluster_id": {"$eq": best_cluster_id}},
                    include_metadata=True,
                )

                if deep_search.get("matches"):
                    best_score = float(deep_search["matches"][0]["score"])
                    for match in deep_search["matches"]:
                        text = match["metadata"].get("text", "")
                        chunk_score = float(match.get("score", 0.0))
                        if text:
                            contexts.append(text)
                            chunks_info.append({
                                "text": text,
                                "score": round(chunk_score, 3),
                                "cluster_id": best_cluster_id,
                            })

            combined_context = "\n\n---\n\n".join(contexts)
            calibrated_score = round(max(0.0, min(1.0, best_score)), 3)
            status = "success" if contexts else "no_documents_found"

            return {
                "context": combined_context,
                "score": calibrated_score,
                "confidence_score": calibrated_score,
                "cluster_id": best_cluster_id,
                "chunks": chunks_info,
                "chunk_count": len(chunks_info),
                "source": "pinecone_rag",
                "status": status,
            }
        except Exception as e:
            if attempt == 0:
                import time
                time.sleep(0.5)
                continue
            print(f"RAG search error after retry: {e}")
            return {
                "context": "",
                "score": 0.0,
                "confidence_score": 0.0,
                "cluster_id": None,
                "chunks": [],
                "chunk_count": 0,
                "source": "pinecone_rag",
                "status": "connection_error",
                "error": str(e),
                "fallback_reason": f"Pinecone vector service temporarily unreachable: {str(e)}",
            }
