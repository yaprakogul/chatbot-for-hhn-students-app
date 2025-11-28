# tools/web_to_jsonl.py
import os, json, time, hashlib
from urllib.parse import urlparse
import tldextract
import trafilatura
import requests


OUT_DIR = "data/cleaned"
URLS_FILE = "data/raw/site_urls.txt"
os.makedirs(OUT_DIR, exist_ok=True)

def slugify_domain(url: str) -> str:
    ext = tldextract.extract(url)
    parts = [p for p in [ext.domain, ext.suffix] if p]
    return "-".join(parts)

def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]

def fetch_and_extract(url: str, timeout=20):
    # 1) requests ile indir (timeout burada)
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0 Safari/537.36"
            },
        )
        if resp.status_code != 200 or not resp.text.strip():
            return None
        html = resp.text
    except Exception:
        # requests başarısız olursa trafilatura.fetch_url ile dene (timeout’suz)
        html = trafilatura.fetch_url(url)
        if not html:
            return None

    # 2) metni çıkar
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True
    )
    if not text:
        return None

    # 3) başlık/metaveri
    try:
        meta = trafilatura.metadata.extract_metadata(html)
        title = meta.title if meta and meta.title else url
    except Exception:
        title = url

    return title, text


def main():
    if not os.path.exists(URLS_FILE):
        print(f"❌ Missing {URLS_FILE}")
        return

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip() and not u.strip().startswith("#")]

    if not urls:
        print("❌ No URLs found.")
        return

    # Çıkış dosyası: domain’e göre grupluyoruz
    dom = slugify_domain(urls[0])
    out_path = os.path.join(OUT_DIR, f"site_{dom}.jsonl")
    count_ok = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for i, url in enumerate(urls, start=1):
            try:
                res = fetch_and_extract(url)
                if not res:
                    print(f"[{i}/{len(urls)}] skip (no content): {url}")
                    continue
                title, text = res
                item = {
                    "title": title,
                    "page": 1,               # web için sembolik
                    "url": url,
                    "text": text.strip()
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                count_ok += 1
                print(f"[{i}/{len(urls)}] ✓ {title[:60]}")
                time.sleep(0.5)  # nazik gecikme
            except Exception as e:
                print(f"[{i}/{len(urls)}] error: {url} -> {e}")

    print(f"✔ Exported {count_ok} pages → {out_path}")

if __name__ == "__main__":
    main()
