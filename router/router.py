import os, csv
import networkx as nx
import re

from .router_utils import px_to_m, px_to_min, PATH_WEIGHTS  # <-- PATH_WEIGHTS buradan geliyor

MAPS_DIR = "data/maps"
NODES = os.path.join(MAPS_DIR, "nodes.csv")
EDGES = os.path.join(MAPS_DIR, "edges_distances.csv")

def _f(v): 
    return (v or "").strip()

def _to_float(v):
    try:
        return float(v)
    except:
        return None

def load_graph():
    # --- nodeları oku ---
    nodes = {}
    with open(NODES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nid = _f(r.get("id"))
            if not nid:
                continue
            nodes[nid] = {
                "label": _f(r.get("label")) or nid,
                "type": _f(r.get("type")),
                "building_code": _f(r.get("building_code")),
                "x": _to_float(r.get("x")), 
                "y": _to_float(r.get("y")),
            }

    G = nx.Graph()
    for nid, data in nodes.items():
        G.add_node(nid, **data)

    # --- kenarları oku ve ağırlığı uygula ---
    with open(EDGES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u, v = _f(r.get("from")), _f(r.get("to"))
            if not u or not v:
                continue

            dist_px = _to_float(r.get("distance_px"))
            if dist_px is None:
                # compute_edge_distances.py çalışmadan eklenmiş satırlar olabilir
                # güvenli tarafta olmak için atlıyoruz
                continue

            path_type = _f(r.get("path_type")) or "path"
            accessible = (_f(r.get("accessible")).lower() != "false")

            # --- KRİTİK: AĞIRLIK UYGULA ---
            weight = dist_px * PATH_WEIGHTS.get(path_type, 1.0)

            # not: nx.Graph() çift yönlüdür; tek eklemek yeterli
            G.add_edge(
                u, v,
                weight=weight,                # rota seçiminde kullanılacak maliyet
                distance_px=dist_px,          # raporlama için ham piksel
                path_type=path_type,
                accessible=accessible,
            )

    return G, nodes


def entrances(nodes, code: str):
    """Belirli bina kodu için 'entrance' node'larını döndür."""
    return [
        nid for nid, nd in nodes.items()
        if nd.get("type") == "entrance" and nd.get("building_code") == code
    ]


def _path_raw_px(G, path):
    """Bir yolun ham piksel toplamını hesapla."""
    s = 0.0
    for a, b in zip(path, path[1:]):
        s += float(G[a][b]["distance_px"])
    return s


def _path_weight_cost(G, path):
    """Bir yolun ağırlıklı maliyetini (weight) hesapla."""
    s = 0.0
    for a, b in zip(path, path[1:]):
        s += float(G[a][b]["weight"])
    return s


def route_b2b(G, nodes, start_code: str, end_code: str):
    """
    Bina kodlarından (A, D, 10, 12...) en iyi yolu bul.
    Seçimde ağırlıklı maliyet (weight) kullanılır, kullanıcıya ham mesafe (px) raporlanır.
    Return: (distance_px, path)
    """
    S, T = entrances(nodes, start_code), entrances(nodes, end_code)
    if not S or not T:
        return None

    best = None  # (cost_weighted, raw_px, path)
    for s in S:
        for t in T:
            try:
                p = nx.shortest_path(G, s, t, weight="weight")
                cost = _path_weight_cost(G, p)   # seçim için
                raw_px = _path_raw_px(G, p)      # çıktı için
                if (best is None) or (cost < best[0]):
                    best = (cost, raw_px, p)
            except nx.NetworkXNoPath:
                pass

    return (best[1], best[2]) if best else None  # (distance_px, path)


def _entrances(nodes, code: str):
    return [
        nid for nid, nd in nodes.items()
        if nd.get("type") == "entrance" and nd.get("building_code") == code
    ]


def route_b2b_debug(G, nodes, start_code: str, end_code: str):
    """
    Debug sürümü: her giriş çifti için rota ve hem 'cost' hem 'raw_px' döner.
    Return: (best_raw_px, best_path, rows)
      rows: [(s, t, cost_weighted, raw_px, path, error), ...]
    """
    S, T = _entrances(nodes, start_code), _entrances(nodes, end_code)
    rows = []
    best = None

    for s in S:
        for t in T:
            try:
                p = nx.shortest_path(G, s, t, weight="weight")
                cost = _path_weight_cost(G, p)
                raw_px = _path_raw_px(G, p)
                rows.append((s, t, cost, raw_px, p, None))
                if (best is None) or (cost < best[0]):
                    best = (cost, raw_px, p)
            except nx.NetworkXNoPath:
                rows.append((s, t, None, None, None, "no_path"))

    return (best[1], best[2], rows) if best else (None, None, rows)


def format_directions_verbose(G, path, dist_px):
    lines = []
    for a, b in zip(path, path[1:]):
        A = G.nodes[a].get("label", a)
        B = G.nodes[b].get("label", b)
        e = G.edges[a, b]
        typ = e.get("path_type", "path")
        w = float(e["weight"])
        dp = float(e["distance_px"])
        lines.append(f"- {A} → {B} via {typ}  (px={dp:.1f}, w={w:.1f})")
    lines.append(f"\nToplam mesafe: {dist_px:.1f}px")
    return "\n".join(lines)


def format_directions(G, path, dist_px):
    steps = []
    for a, b in zip(path, path[1:]):
        A = G.nodes[a].get("label", a)
        B = G.nodes[b].get("label", b)
        t = G.edges[a, b].get("path_type", "path")
        steps.append(f"- {A} → {B} via {t}")
    return "\n".join(steps) + f"\n\n≈ {px_to_m(dist_px):.0f} m, ≈ {px_to_min(dist_px):.1f} min"

# === Pretty route description (English, step-based, with landmarks) ===

# -----------------------------------------------
# Human-friendly building / entrance descriptions
# -----------------------------------------------

# Short building descriptions for landmarks
BUILDING_DESCRIPTIONS = {
    "A": "the main building with the information desk",
    "B": "the building with the Mensa (canteen)",
    "C": "a teaching building with several lecture rooms",
    "D": "a building with larger lecture halls",
    "E": "a building with seminar rooms",
    "F": "the long glass building with labs",
    "L": "the building with project and collaboration rooms",

    # BildungsCampus
    "8":  "the Mensa building on the BildungsCampus",
    "10": "Building V on the BildungsCampus (Mitte)",
    "12": "Building N on the BildungsCampus (Nord)",
    "14": "Building T on the BildungsCampus (Nord)",
    "17": "Building S on the BildungsCampus (Nord)",
    "13": "the building with the library and CAS",
    "15": "the building with the large auditorium and administration",
}

# Letter aliases for BildungsCampus (V, N, T, S)
BC_LETTER_BY_CODE = {
    "10": "V",   # BildungsCampus 10 = Building V
    "12": "N",   # BildungsCampus 12 = Building N
    "14": "T",   # BildungsCampus 14 = Building T
    "17": "S",   # BildungsCampus 17 = Building S
}

def _pretty_building_name(code: str) -> str:
    """
    Turn a building_code into something like 'Building A' / 'Building 12'.

    For BildungsCampus letter buildings, prefer V/N/T/S instead of numbers:
    10 → V, 12 → N, 14 → T, 17 → S.
    """
    if not code:
        return "the building"

    # Önce harf alias’ı var mı bak (V, N, T, S)
    letter = BC_LETTER_BY_CODE.get(code)
    if letter:
        return f"Building {letter}"

    # Diğerleri olduğu gibi
    return f"Building {code}"



def _entrance_phrase(node: dict, include_hint: bool = False) -> str:
    """
    Make a human-friendly phrase for a node that is (usually) an entrance.
    Avoids 'north/east' directions, uses generic terms instead.
    """
    label = (node.get("label") or "").strip()
    code = (node.get("building_code") or "").strip()
    base_name = _pretty_building_name(code)

    # Default phrase
    phrase = f"the entrance of {base_name}"

    # Try to detect more specific types from the label
    lower = label.lower()
    if "main entrance" in lower:
        phrase = f"the main entrance of {base_name}"
    elif "mensa" in lower:
        phrase = f"the entrance of {base_name} next to the Mensa"
    elif "library" in lower:
        phrase = f"the entrance of {base_name} near the library"
    elif "entrance" in lower and any(d in lower for d in ["north", "south", "east", "west"]):
        # We do NOT expose direction words to the user
        phrase = f"a side entrance of {base_name}"

    # Optional landmark-style hint
    if include_hint and code in BUILDING_DESCRIPTIONS:
        phrase += f", {BUILDING_DESCRIPTIONS[code]}"

    return phrase


def _destination_phrase(G, node_id: str) -> str:
    """
    For intermediate steps: if node is an entrance, use entrance_phrase;
    otherwise fall back to its label.
    """
    nd = G.nodes[node_id]
    ntype = (nd.get("type") or "").strip()
    if ntype == "entrance":
        return _entrance_phrase(nd, include_hint=False)
    # junction / gate / etc → just label
    label = (nd.get("label") or "").strip()
    return label if label else node_id


def _sentence_for_segment(ptype: str, is_first_step: bool) -> str:
    """
    Choose a verb phrase based on path_type and whether it is the first segment.
    Returns something like 'Go inside and follow the indoor corridor'.
    """
    p = (ptype or "path").lower()

    if p == "indoor":
        if is_first_step:
            return "Go inside and follow the indoor corridor"
        else:
            return "Then continue along the indoor corridor"

    if p in ("sidewalk", "footpath", "path"):
        if is_first_step:
            return "Walk outside"
        else:
            return "Then continue outside"

    if p == "crosswalk":
        if is_first_step:
            return "Use the crosswalk and walk outside"
        else:
            return "Then use the crosswalk and continue outside"

    if p == "elevator":
        if is_first_step:
            return "Take the elevator"
        else:
            return "Then take the elevator"

    if p == "stairs":
        if is_first_step:
            return "Use the stairs"
        else:
            return "Then use the stairs"

    # Fallback
    return "Continue"


def rephrase_route(G, path):
    """
    Turn a graph path into a step-based, student-friendly English route description.
    No meters/minutes, no 'north/south', just simple walking instructions.
    """
    if not path or len(path) < 2:
        return "I couldn’t find a walkable route. Please check the building codes."

    start_id = path[0]
    end_id = path[-1]
    start_node = G.nodes[start_id]
    end_node = G.nodes[end_id]

    # Decide how to name start & end for the title
    start_code = (start_node.get("building_code") or "").strip()
    end_code = (end_node.get("building_code") or "").strip()

    if start_code:
        start_title = _pretty_building_name(start_code)
    else:
        start_title = (start_node.get("label") or start_id)

    if end_code:
        end_title = _pretty_building_name(end_code)
    else:
        end_title = (end_node.get("label") or end_id)

    # Phrases for start/end entrances
    start_phrase = _entrance_phrase(start_node, include_hint=True)
    end_phrase = _entrance_phrase(end_node, include_hint=False)

    # Collect steps
    steps_lines = []
    # Step 1: starting point
    steps_lines.append(f"1. Start at {start_phrase}.")

    indoor_edges = 0
    outdoor_edges = 0
    has_stairs = False

    step_index = 2
    for a, b in zip(path, path[1:]):
        edge = G.edges[a, b]
        ptype = edge.get("path_type", "path")

        # Count stats for summary
        p = (ptype or "path").lower()
        if p == "stairs":
            has_stairs = True
        if p == "indoor":
            indoor_edges += 1
        elif p in ("sidewalk", "footpath", "path", "crosswalk"):
            outdoor_edges += 1

        verb = _sentence_for_segment(ptype, is_first_step=(step_index == 2))
        dest_phrase = _destination_phrase(G, b)

        steps_lines.append(f"{step_index}. {verb} to {dest_phrase}.")
        step_index += 1

    # Footer: arrival + very short indoor/outdoor note
    footer_parts = [f"Once you arrive at {end_phrase}, you have reached your destination."]

    if indoor_edges > 0 and outdoor_edges == 0:
        footer_parts.append("This route is fully indoors.")
    elif outdoor_edges > 0 and indoor_edges == 0:
        footer_parts.append("This route is fully outdoors.")
    elif indoor_edges > 0 and outdoor_edges > 0:
        footer_parts.append("This route includes both indoor corridors and short outdoor segments.")

    if has_stairs:
        footer_parts.append("Note: this route includes stairs.")

    footer_text = " ".join(footer_parts).strip()

    # Title line (Markdown)
    title = f"**Route from {start_title} to {end_title}**"

    # Final markdown string
    body = "\n".join(steps_lines)
    return f"{title}\n\n{body}\n\n{footer_text}"

