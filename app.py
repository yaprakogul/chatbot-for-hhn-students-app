from rag.runtime_qa import answer_query
import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role:"user"/"assistant", content:str, sources:list|None}]
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None

def main():
    print("RAG chatbot ready. Ask me something (q to quit).")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        try:
            answer, refs = answer_query(q, k=5)
            print("\n" + answer + "\n")
            if refs:
                print("Sources:")
                for i, h in enumerate(refs, 1):
                    title = h.get("title", "")
                    page = h.get("page", 0)
                    score = h.get("score", 0.0)
                    print(f"[{i}] {title} (p.{page})  score={score:.3f}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
