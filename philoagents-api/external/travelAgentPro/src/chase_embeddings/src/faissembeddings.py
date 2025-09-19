import os,sys
import numpy as np
from sentence_transformers import SentenceTransformer

import faiss
# add project root (two levels up) so you can import adk_functions from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from .embeddings import load_embeddings_from_meta   # adjust if your path differs

#if 'call_agent_async' not in globals():
  #from adk_functions import call_agent_async           # async runner caller you already have
# conditional import for call_agent_async
if 'call_agent_async' not in globals():
    try:
        #from adk_functions import call_agent_async
        imp=0
    except ImportError:
        pass

from typing import List, Tuple

def ensure_2d(emb: np.ndarray) -> np.ndarray:
    """
    Ensure embeddings is a 2D numpy array (N, D).
    If a 1D array is passed, try to treat it as a single vector (reshape to 1, D).
    If object-dtype list-of-lists is passed, stack into 2D.
    """
    arr = np.asarray(emb)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:      
    # try to treat elements as rows (list of rows) if object dtype
        if arr.dtype == object:
            try:
                stacked = np.vstack(arr)
                return stacked
            except Exception:
                pass
        # otherwise treat as a single vector -> reshape to (1, D)
        return arr.reshape(1, -1)
    # higher / zero-dim fallback
    raise ValueError(f"Unsupported embeddings shape: {arr.shape}")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def build_faiss_index(embeddings: np.ndarray, index_path: str, use_gpu: bool = False) -> None:
    """
    Build an IndexFlatIP FAISS index on L2-normalized vectors and save to index_path.
    embeddings: np.ndarray shape (N, D)
    index_path: path to write faiss index file
    """
    if not _HAS_FAISS:
        raise RuntimeError("faiss not available; pip install faiss-cpu")

    #emb = np.asarray(embeddings, dtype=np.float32)
    emb = ensure_2d(embeddings).astype(np.float32)
    # Fix transposed embeddings if needed
    if emb.shape[0] == 1 and emb.shape[1] > 8:  # likely (1, D)
        print("[faiss] Warning: only one embedding found. Did you transpose?")
    if emb.shape[1] == 1 and emb.shape[0] > 8:  # likely (N, 1)
        print("[faiss] Warning: embeddings appear to be (N, 1). Transposing to (1, N).")
        emb = emb.T

    print(f"[faiss] embeddings shape after ensure_2d: {emb.shape}")
    emb = _normalize_rows(emb)
    
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    # optionally move to GPU (requires faiss-gpu & proper setup)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, index_path)
    print(f"[faiss] index saved -> {index_path}")

def load_faiss_index(index_path: str):
    if not _HAS_FAISS:
        raise RuntimeError("faiss not available; pip install faiss-cpu")
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    return faiss.read_index(index_path)

def query_embedding(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    vec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec

def faiss_search(query: str, meta_path: str, index_path: str, top_k: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[Tuple[int, float, str]]:
    """
    Load meta (embeddings + chunks), load FAISS index, search top_k, return list of (idx, score, chunk).
    """
    embeddings, chunks, meta = load_embeddings_from_meta(meta_path)
    if embeddings is None or len(chunks) == 0:
        return []

    index = load_faiss_index(index_path)
    q = query_embedding(query, model_name).astype(np.float32).reshape(1, -1)
    scores, idxs = index.search(q, top_k)
    scores = scores[0]
    idxs = idxs[0]
    results = []
    for i, s in zip(idxs, scores):
        if i == -1:
            continue
        results.append((int(i), float(s), chunks[i]))
    return results

async def ask_with_faiss_retrieval(query: str, meta_path: str, index_path: str, runner, user_id: str, session_id: str, top_k: int = 5):
    """
    Do FAISS retrieval, assemble context, and call agent via call_agent_async.
    """
    results = faiss_search(query, meta_path, index_path, top_k=top_k)
    if not results:
        print("[retrieval] no results found, calling agent directly")
        return await call_agent_async(query, runner, user_id, session_id)

    context_parts = []
    for idx, score, chunk in results:
        context_parts.append(f"[doc {idx} | score={score:.4f}]\n{chunk.strip()}\n")

    context = "\n\n".join(context_parts)
    prompt_with_context = (
        "Use the following retrieved documents as context to answer the user's question.\n\n"
        "RETRIEVED DOCUMENTS:\n"
        f"{context}\n\n"
        f"USER QUERY: {query}"
    )
    return await call_agent_async(prompt_with_context, runner, user_id, session_id)

# Example helper to build & save index given a meta file
def build_index_from_meta(meta_path: str, index_path: str, use_gpu: bool = False):
    embeddings, chunks, meta = load_embeddings_from_meta(meta_path)
    if embeddings is None:
        raise RuntimeError("No embeddings found in meta")
    build_faiss_index(embeddings, index_path, use_gpu=use_gpu)