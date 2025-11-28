from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

SYS = (
    "You are a helpful assistant for first-semester students at Heilbronn University. "
    "Answer ONLY using the provided context below. "
    "If the answer is not contained in the context, say 'I’m not sure about that yet.' "
    "Cite sources in parentheses like (Source: {title}, page {page})."
)

def format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "?")
        page = c.get("page", "?")
        text = c.get("text", "").strip().replace("\n", " ")
        lines.append(f"[{i}] {title} (p.{page}) → {text}")
    return "\n\n".join(lines)


# --------------------------
# Ana cevap üretici fonksiyon
# --------------------------
def answer_with_rag(question: str, chunks: List[Dict]) -> str:
    load_dotenv()
    client = OpenAI()

    context = format_context(chunks)
    user_msg = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Use the context above to answer clearly and concisely."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": user_msg},
        ],
    )

    return resp.choices[0].message.content

# Test 
if __name__ == "__main__":
    from rag.search import retrieve
    q = input("Enter your question: ")
    chunks = retrieve(q, k=5)
    print("\n--- Retrieved context ---")
    for c in chunks:
        print(f"- {c['title']} (p.{c['page']}) | score={c['_score']:.3f}")

    print("\n--- Model answer ---")
    answer = answer_with_rag(q, chunks)
    print(answer)
