import os, json
from PyPDF2 import PdfReader

RAW_PATH = "data/raw"
CLEANED_PATH = "data/cleaned"

os.makedirs(CLEANED_PATH, exist_ok=True)

def export_to_cleaned(pages: list[dict], base_filename: str):
    """
    Converts extracted PDF data into RAG-compatible JSONL format.
    Each element of 'pages' should be a dict like:
      {"title": "filename.pdf", "page": 3, "text": "..."}
    It saves the result in data/cleaned/{base_filename}.jsonl
    """
    out_dir = "data/cleaned"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_filename}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pages:
            item = {
                "title": p.get("title", base_filename),
                "page": p.get("page", 0),
                "text": p.get("text", "").strip()
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✔ Exported {len(pages)} pages to {out_path}")


for filename in os.listdir(RAW_PATH):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(RAW_PATH, filename)
        base_name = filename.replace(".pdf", "")
        text_path = os.path.join(CLEANED_PATH, f"{base_name}.txt")

        print(f" Processing: {filename} ...")

        reader = PdfReader(pdf_path)
        full_text = ""
        pages = [] 

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                text = text.replace("\n", " ").strip()
                full_text += text + "\n\n"
                pages.append({
                    "title": filename,
                    "page": i,
                    "text": text
                })

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f" Saved: {text_path}")

        export_to_cleaned(pages, base_name)
        print(f" JSONL created: {base_name}.jsonl\n")

print("All PDFs have been processed and prepared for RAG!")
