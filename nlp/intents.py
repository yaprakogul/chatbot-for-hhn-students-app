import re

def _clean(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _pick_src_dst(src_raw: str, dst_raw: str):
    # keep "building 10" intact; otherwise take the last word (A, D, mensa, 12…)
    def norm_side(x: str):
        x = x.strip()
        if x.startswith("building "):
            return x
        return x.split()[-1]
    return norm_side(src_raw), norm_side(dst_raw)

def extract_places(text: str):
    """
    works for:
      - how do i go from A to D?
      - how can i get to D from A
      - route from 11 to 17
      - directions to mensa from B
    returns (src, dst) or (None, None)
    """
    t = _clean(text)

    # 1) from ... to ...
    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:$|\s*[?.!])", t)
    if m:
        return _pick_src_dst(m.group(1), m.group(2))

    # 2) to ... from ...   (reverse order)
    m = re.search(r"\bto\s+(.+?)\s+from\s+(.+?)(?:$|\s*[?.!])", t)
    if m:
        src, dst = _pick_src_dst(m.group(2), m.group(1))
        return src, dst

    return None, None

def is_route_query(text: str) -> bool:
    # intent = we can extract places
    s, d = extract_places(text)
    return bool(s and d)
