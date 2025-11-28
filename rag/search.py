import os, json
import numpy as np
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

STORE_DIR = "rag/store"
EMBED_MODEL = "text-embedding-3-large"

# Load vector store (embeddings + meta)
def load_store():
    vectors_path = os.path.join(STORE_DIR, "vectors.npz")
    meta_path = os.path.join(STORE_DIR, "meta.jsonl")

    if not os.path.exists(vectors_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("❌ No store found. Run rag/ingest.py first.")

    X = np.load(vectors_path)["X"]
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    print(f"✅ Loaded store: {len(meta)} chunks, {X.shape[1]} dims")
    return X, meta


# Embed query using OpenAI
def embed_query(query: str) -> np.ndarray:
    load_dotenv()
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    v = np.array(resp.data[0].embedding, dtype="float32")
    return v


# Cosine similarity (semantic search)
def top_k(qvec: np.ndarray, X: np.ndarray, k=5):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sims = Xn @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


# Main retrieval function
def retrieve(query: str, k=5) -> List[Dict]:
    X, meta = load_store()
    qv = embed_query(query)
    idx, sims = top_k(qv, X, k)
    results = []
    for i, s in zip(idx, sims):
        item = meta[i].copy()
        item["_score"] = float(s)
        results.append(item)
    return results


# Test model
if __name__ == "__main__":
    q = input("Enter your test query: ")
    res = retrieve(q, k=5)
    print("\nTop results:\n")
    for r in res:
        print(f"- {r['title']} (p.{r['page']}) | score={r['_score']:.3f}")
        snippet = r['text'][:200].replace("\n", " ")
        print(f"  {snippet}...")
        print()
