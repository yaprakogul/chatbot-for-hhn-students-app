# rag/ingest.py
import os
import glob
import json
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Qdrant (opsiyonel: USE_QDRANT=1 ise devreye girer)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except Exception:
    QdrantClient = None  # type: ignore

# ---------- Ayarlar ----------
CLEAN_DIR = "data/cleaned"          # .jsonl / .txt kaynakları
STORE_DIR = "data/store"            # lokal yedek (npz + meta)
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 64
# Qdrant yapılandırması
USE_QDRANT = True  # Qdrant kullanmak istemezsen False yapabilirsin
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None
QDRANT_COLLECTION = "hhn_knowledge"

# ---------- Util ----------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _env_true(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip() in {"1", "true", "TRUE", "yes", "YES"}

def _load_env() -> None:
    load_dotenv()  # .env okumak için

def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """
    Uzun metinleri küçük parçalara ayırır (yaklaşık 2000 token civarı).
    """
    # buradaki ölçüm kaba tahmin: 1 token ≈ 4 karakter
    max_chars = max_tokens * 4
    parts = []
    for i in range(0, len(text), max_chars):
        parts.append(text[i:i+max_chars])
    return parts

# ---------- Veriyi Yükle ----------
def load_cleaned_rows() -> List[Dict]:
    """
    data/cleaned altındaki .jsonl ve .txt dosyalarını okur.
    jsonl formatı: {"title": ..., "page": int, "text": "..."}
    txt formatı: tek alan "text"
    """
    rows: List[Dict] = []

    # .jsonl
    for jf in glob.glob(os.path.join(CLEAN_DIR, "*.jsonl")):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # .txt (opsiyonel)
    for tf in glob.glob(os.path.join(CLEAN_DIR, "*.txt")):
        with open(tf, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt:
                rows.append({"title": os.path.basename(tf), "page": 0, "text": txt})

    # boş metinleri ayıkla
    rows = [r for r in rows if (r.get("text") or "").strip()]
    return rows


# ---------- Embedding ----------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    OpenAI embedding üretir. (text-embedding-3-large)
    """
    _load_env()
    client = OpenAI()
    vecs: List[List[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])

    X = np.array(vecs, dtype="float32")
    return X

# ---------- Lokal Store (yedek) ----------
def save_local_store(vectors: np.ndarray, rows: List[Dict]) -> None:
    _ensure_dir(STORE_DIR)
    np.savez_compressed(os.path.join(STORE_DIR, "embeddings.npz"), X=vectors)
    meta_path = os.path.join(STORE_DIR, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✔ local store saved → {STORE_DIR} "
          f"({vectors.shape[0]} vectors, dim={vectors.shape[1] if vectors.size else 0})")

# ---------- Qdrant ----------
def _get_qdrant_client() -> Tuple[QdrantClient, str]:
    """
    Qdrant client ve collection adını döner.
    """
    if QdrantClient is None:
        raise RuntimeError("qdrant-client yüklü değil. `pip install qdrant-client`")

    url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    api_key = os.getenv("QDRANT_API_KEY", "") or None
    coll = os.getenv("QDRANT_COLLECTION", "hhn_knowledge").strip() or "hhn_knowledge"
    client = QdrantClient(url=url, api_key=api_key)
    return client, coll

def _ensure_collection(client: "QdrantClient", collection: str, dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"✔ created collection: {collection} (dim={dim}, metric=cosine)")
    else:
        # koleksiyon var ama dim uyumsuz ise hata alırsın—gerekirse sil/oluştur yap
        pass

def upload_to_qdrant(vectors: np.ndarray, rows: List[Dict]) -> None:
    client, collection = _get_qdrant_client()
    if vectors.size == 0:
        print("⚠ nothing to upload (0 vectors)")
        return

    dim = vectors.shape[1]
    _ensure_collection(client, collection, dim)

    # nokta listesi
    points: List[PointStruct] = []
    for idx, (vec, r) in enumerate(zip(vectors, rows), start=1):
        payload = {
            "title": r.get("title", ""),
            "page": r.get("page", 0),
            "text": r.get("text", ""),
        }
        points.append(PointStruct(id=idx, vector=vec.tolist(), payload=payload))

    # batch upsert
    # büyük veride istersen 1000'lik parçalara bölebilirsin
    client.upsert(collection_name=collection, points=points)
    print(f"✔ uploaded to Qdrant → {collection}: {len(points)} points")

# ---------- Pipeline ----------
def build_store() -> None:
    """
    1) data/cleaned içeriğini oku
    2) metinleri güvenli boyutta 'chunk'lara böl
    3) embedding üret
    4) yerel yedek (npz + meta.jsonl) kaydet
    5) USE_QDRANT=1 ise Qdrant koleksiyonuna upsert et
    """
    _load_env()

    rows = load_cleaned_rows()
    if not rows:
        print(f"⚠ no data under {CLEAN_DIR}. jsonl/txt dosyalarını ekle.")
        return

    # --- chunk'lama ---
    max_tok = int(os.getenv("CHUNK_TOKENS", "2000"))  # istersen .env ile değiştirebilirsin
    texts: list[str] = []
    new_rows: list[dict] = []

    for r in rows:
        base_title = r.get("title", "")
        base_page  = r.get("page", 0)
        chunks = chunk_text(r.get("text", ""), max_tokens=max_tok)
        if not chunks:
            continue
        total = len(chunks)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            new_rows.append({
                "title": base_title,
                "page": base_page,
                "chunk_index": i,
                "chunk_total": total,
                "text": ch
            })

    rows = new_rows
    if not texts:
        print("⚠ after chunking no text remained.")
        return

    print(f"… embedding {len(texts)} chunks with {EMBED_MODEL} (chunk_tokens≈{max_tok})")
    vectors = embed_texts(texts)  # (N, dim)

    # --- yerel yedek ---
    save_local_store(vectors, rows)

    # --- Qdrant'a yükle (isteğe bağlı) ---
    if _env_true("USE_QDRANT", "1"):
        try:
            upload_to_qdrant(vectors, rows)
        except Exception as e:
            print(f"⚠ Qdrant upload failed: {e}")

    print("✔ ingest finished.")


# ---------- CLI ----------
if __name__ == "__main__":
    build_store()
