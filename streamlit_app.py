import os
import re
import streamlit as st
import os
import streamlit as st

from rag.runtime_qa import answer_query
from router.router import load_graph, rephrase_route, route_b2b 
from router.router_utils import px_to_m, px_to_min 
from router.router import load_graph, route_b2b, rephrase_route

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "QDRANT_URL" in st.secrets:
    os.environ["QDRANT_URL"] = st.secrets["QDRANT_URL"]

if "QDRANT_COLLECTION" in st.secrets:
    os.environ["QDRANT_COLLECTION"] = st.secrets["QDRANT_COLLECTION"]

if "GEN_MODEL" in st.secrets:
    os.environ["GEN_MODEL"] = st.secrets["GEN_MODEL"]

if "EMBED_MODEL" in st.secrets:
    os.environ["EMBED_MODEL"] = st.secrets["EMBED_MODEL"]

@st.cache_resource
def _boot_router():
    G, nodes = load_graph()
    return G, nodes

G_ROUTER, NODES = _boot_router()

ALIAS = {
    # --- TechCampus shortpaths ---
    "mensa": ("TechCampus", "B"),
    "canteen": ("TechCampus", "B"),
    "cafeteria": ("TechCampus", "B"),


    "library": ("BildungsCampus", "13"),

    # V, N, T, S = 10, 12, 14, 17
    "v": ("BildungsCampus", "10"),
    "building v": ("BildungsCampus", "10"),
    "hhn v": ("BildungsCampus", "10"),

    "n": ("BildungsCampus", "12"),
    "building n": ("BildungsCampus", "12"),
    "hhn n": ("BildungsCampus", "12"),

    "t": ("BildungsCampus", "14"),
    "building t": ("BildungsCampus", "14"),
    "hhn t": ("BildungsCampus", "14"),

    "s": ("BildungsCampus", "17"),
    "building s": ("BildungsCampus", "17"),
    "hhn s": ("BildungsCampus", "17"),
}


ROUTE_PATTERNS = [
    # route from A to D
    r"\broute\s+from\s+(?P<from>[A-Za-z0-9]+)\s*to\s*(?P<to>[A-Za-z0-9]+)\b",

    # how can i go from A to D
    r"\bhow\s+can\s+i\s+go\s+from\s+(?P<from>[A-Za-z0-9]+)\s*to\s*(?P<to>[A-Za-z0-9]+)\b",
    r"\bhow\s+do\s+i\s+go\s+from\s+(?P<from>[A-Za-z0-9]+)\s*to\s*(?P<to>[A-Za-z0-9]+)\b",

    # go to D from A
    r"\bgo\s+to\s*(?P<to>[A-Za-z0-9]+)\s*from\s*(?P<from>[A-Za-z0-9]+)\b",

    # genel: from A to D 
    r"\bfrom\s+(?P<from>[A-Za-z0-9]+)\s*to\s*(?P<to>[A-Za-z0-9]+)\b",
]

def normalize_token(tok: str):
    t = tok.strip().lower()

    if t in ALIAS:
        return ALIAS[t]
    
    if re.fullmatch(r"[a-z]", t):
        if t in ["a", "b", "c", "d", "e", "f", "l", "y"]:
            return ("TechCampus", t.upper())
        return None
    
    if re.fullmatch(r"\d{1,2}", t):
        return ("BildungsCampus", t)

    return None 


def parse_route_intent(text: str):
    q = text.strip().lower()
    q = re.sub(r"\s+", " ", q)

    for pat in ROUTE_PATTERNS:
        m = re.search(pat, q)
        if m:
            raw_from = m.group("from")
            raw_to = m.group("to")

            nf = normalize_token(raw_from)
            nt = normalize_token(raw_to)
            if nf and nt:
                return {"from_code": nf[1], "to_code": nt[1]}
    return None

# ================
#  Streamlit UI
# ================
st.set_page_config(page_title="HHN Student Assistant", page_icon="🎓", layout="wide")

# ---------- State ----------
if "chat" not in st.session_state:
    # [{"role":"user"/"assistant","text":..., "sources":[...]}]
    st.session_state.chat = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("Vector Store: Qdrant")
    st.write("URL:", os.environ.get("QDRANT_URL", "http://localhost:6333"))
    st.write("Collection:", os.environ.get("QDRANT_COLLECTION", "hhn_knowledge"))
    st.write("Model (gen):", os.environ.get("GEN_MODEL", "gpt-4o-mini"))
    st.write("Embedding:", os.environ.get("EMBED_MODEL", "text-embedding-3-large"))
    st.markdown("---")
    st.markdown("**Tips**")
    st.caption("Ask specific things like 'semester ticket fee', 'where is the Mensa', 'International Office', or 'route from A to D'.")
    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.success("Cleared.")

# ---------- Header ----------
st.markdown("## HHN Student Assistant")
st.caption("Ask anything about campus info, booklets, FAQ & more.")

# ---------- CHAT AREA (mesajlar formdan ÖNCE çizilir) ----------
chat_area = st.container()

def render_message(msg):
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["text"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write("- ", s)

with chat_area:
    for m in st.session_state.chat:
        render_message(m)

# ---------- INPUT FORM ----------
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input(
        " ",
        placeholder="Type your question…",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Send")

# ---------- Submit Handling ----------
if submitted and user_text.strip():
    q = user_text.strip()

    st.session_state.chat.append({"role": "user", "text": q, "sources": []})

    want_route = parse_route_intent(q)
    if want_route:
        start, end = want_route["from_code"], want_route["to_code"]
        best = None
        try:
            best = route_b2b(G_ROUTER, NODES, start, end)
        except Exception as e:
            best = None

        if not best:
            answer = f"Sorry, I couldn’t find a walkable route from {start} to {end} in the campus graph."
            refs = []
        else:
            _, path = best
            answer = rephrase_route(G_ROUTER, path)
            refs = [f"Campus map graph ({len(G_ROUTER.nodes())} nodes)"]
    else:
        try:
            answer, refs = answer_query(q)
        except Exception as e:
            answer, refs = (f"Query failed: {e}", [])

    srcs = []
    for r in refs or []:
        if isinstance(r, dict):
            title = r.get("title", "")
            page = r.get("page", "")
            score = r.get("score", None)
            s = title
            if page not in ("", None):
                s += f" (p.{page})"
            if score is not None:
                s += f" | score={score:.3f}"
            srcs.append(s or str(r))
        else:
            srcs.append(str(r))

    st.session_state.chat.append({"role": "assistant", "text": answer, "sources": srcs})

    st.rerun()
