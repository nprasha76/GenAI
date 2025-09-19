from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import time
from typing import Tuple, List, Dict


def create_embeddings(texts):
    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Create embeddings
    embeddings = model.encode(texts)
    print("Embeddings shape:", embeddings.shape)
    return embeddings

    # Save embeddings to a file
    #output_file = os.path.join(os.getcwd(), 'embeddings.npy')
    #np.save(output_file, embeddings)

    #return output_file
def create_chunks(text: str, max_words: int = 250, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by words.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i : i + max_words]
        chunks.append(" ".join(chunk_words))
        i += max_words - overlap
    return chunks


def save_embeddings(embeddings: np.ndarray, chunks: List[str], out_prefix: str = "embeddings") -> Dict:
    ts = int(time.time())
    emb_file = f"{out_prefix}.npy"
    meta_file = f"{out_prefix}_meta.json"
    print(f"Saving embeddings to {emb_file} and metadata to {meta_file}")

    np.save(emb_file, embeddings)
    meta = {
        "embeddings_file": os.path.abspath(emb_file),
        "chunks_file": os.path.abspath(meta_file.replace("_meta.json", "_chunks.json")),
        "model": 'all-MiniLM-L6-v2',
        "created_at": ts,
        "n_vectors": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None
    }
    # Save chunks separately (human readable)
    with open(meta["chunks_file"], "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    # Save metadata
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta

def load_embeddings_from_meta(meta_path: str) -> Tuple[np.ndarray, List[str], Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    emb_path = meta["embeddings_file"]
    chunks_path = meta.get("chunks_file")
    embeddings = np.load(emb_path)
    chunks = []
    if chunks_path and os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f).get("chunks", [])
    return embeddings, chunks, meta    