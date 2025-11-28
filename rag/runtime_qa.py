import os, json
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Qdrant opsiyonel
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None  # pip install qdrant-client yoksa local moda düşer

# ingest içinden ayarları al
from .ingest import (
    USE_QDRANT, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    EMBED_MODEL,
)

STORE_DIR = "data/store"
_local_store = None  # {"emb": np.ndarray, "meta": List[Dict]}


# ---------------- Embedding ----------------
def _embed_query(text: str) -> np.ndarray:
    """Tek sorgu embedding'i döner (float32)."""
    load_dotenv()
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return vec

def retrieve_docs(query: str, k: int = 5) -> List[Dict]:
    if USE_QDRANT:
        return _retrieve_qdrant(query, k)
    else:
        return _retrieve_local(query, k)

# --------------- Local store ---------------
def _load_local_store() -> None:
    """data/store'dan embeddings + meta'yı belleğe al."""
    global _local_store
    emb_path = os.path.join(STORE_DIR, "embeddings.npz")
    meta_path = os.path.join(STORE_DIR, "meta.jsonl")

    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Local store not found. Run:  python -m rag.ingest  (to build data/store)"
        )

    X = np.load(emb_path)["X"].astype("float32")
    meta: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    _local_store = {"emb": X, "meta": meta}


def _cosine_sim(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A:(N,D), b:(D,) -> sims:(N,)  (normalize + dot)"""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return A @ b


# ---------------- Retrieve -----------------
def _retrieve_local(query: str, k: int = 5) -> List[Dict]:
    """Yerel npz+jsonl üzerinden en benzer k chunk'ı döner."""
    global _local_store
    if _local_store is None:
        _load_local_store()

    qvec = _embed_query(query)
    sims = _cosine_sim(_local_store["emb"], qvec)  # (N,)
    top_idx = sims.argsort()[-k:][::-1]

    hits: List[Dict] = []
    for i in top_idx:
        r = _local_store["meta"][int(i)]
        hits.append({
            "text": r.get("text", ""),
            "title": r.get("title", ""),
            "page": r.get("page", 0),
            "score": float(sims[int(i)]),
        })
    return hits


def _retrieve_qdrant(query: str, k: int = 5) -> List[Dict]:
    """Qdrant koleksiyonundan arama yapar."""
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is not installed. `pip install qdrant-client`")

    client = QdrantClient(url=QDRANT_URL, api_key=(QDRANT_API_KEY or None))
    qvec = _embed_query(query).tolist()

    res = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=k,
    )

    hits: List[Dict] = []
    for p in res:
        pl = p.payload or {}
        hits.append({
            "text": pl.get("text", ""),
            "title": pl.get("title", ""),
            "page": pl.get("page", 0),
            "score": float(p.score),
        })
    return hits


def retrieve(query: str, k: int = 5) -> List[Dict]:
    """Kullanılabilir moda göre arama yapar."""
    if USE_QDRANT:
        return _retrieve_qdrant(query, k=k)
    return _retrieve_local(query, k=k)


# -------------- Answer synth ---------------
def synthesize_answer(query: str, hits: List[Dict], max_ctx_chars: int = 6000) -> Tuple[str, List[Dict]]:
    """
    Toplanan pasajlardan bağlam oluşturur ve kısa bir yanıt üretir.
    [#] biçiminde kaynak numaralarıyla referans verir.
    """
    load_dotenv()
    client = OpenAI()

    # bağlamı [1]..[k] olarak hazırla
    ctx_parts = []
    for i, h in enumerate(hits, 1):
        piece = f"[{i}] {h.get('title','')} (p.{h.get('page',0)}): {h.get('text','')}".strip()
        if not piece:
            continue
        if sum(len(x) for x in ctx_parts) + len(piece) > max_ctx_chars:
            break
        ctx_parts.append(piece)

    context = "\n\n".join(ctx_parts) if ctx_parts else "(no context)"

    prompt = (
        "You are an HHN student assistant. Use the given sources to answer briefly and accurately. "
        "Cite sources using [#] where appropriate.\n\n"
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    return answer, hits


def answer_query(q: str, k: int = 5) -> Tuple[str, List[Dict]]:
    hits = retrieve(q, k=k)
    return synthesize_answer(q, hits)
