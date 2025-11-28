import os, csv, math
import networkx as nx

from router_utils import px_to_meters, px_to_minutes, ALIASES

MAPS_DIR = "data/maps"
NODES_CSV = os.path.join(MAPS_DIR, "nodes.csv")
EDGES_DIST_CSV = os.path.join(MAPS_DIR, "edges_distances.csv")
EDGES_RAW_CSV = os.path.join(MAPS_DIR, "edges.csv")  # fallback

def _n(s): return (s or "").replace("\ufeff","").strip().lower()

def _to_float(v):
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    try: return float(s)
    except: return None

def load_nodes():
    nodes = {}
    with open(NODES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("id") or str(r["id"]).lstrip().startswith("#"):  # skip comments
                continue
            nid = r["id"].strip()
            nodes[nid] = {
                "label": r.get("label","").strip() or nid,
                "type":  r.get("type","").strip(),
                "building_code": (r.get("building_code") or "").strip(),
                "x": _to_float(r.get("x")),
                "y": _to_float(r.get("y")),
            }
    return nodes

def load_edges(nodes):
    # prefer pre-computed distances
    src = EDGES_DIST_CSV if os.path.exists(EDGES_DIST_CSV) else EDGES_RAW_CSV
    edges = []
    with open(src, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if not row.get("from") or not row.get("to"):  # skip empty/comment lines
                continue
            u, v = row["from"].strip(), row["to"].strip()
            if u.startswith("#") or v.startswith("#"):
                continue
            # weight: prefer distance_px; else compute from node coords if available
            w = _to_float(row.get("distance_px"))
            if w is None:
                a, b = nodes.get(u), nodes.get(v)
                if a and b and a["x"] is not None and a["y"] is not None and b["x"] is not None and b["y"] is not None:
                    w = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                else:
                    # skip if we cannot assign a weight
                    continue
            path_type = row.get("path_type","").strip() or "path"
            accessible = (str(row.get("accessible","true")).strip().lower() != "false")
            edges.append((u, v, {"weight": float(w), "path_type": path_type, "accessible": accessible}))
    return edges, os.path.basename(src)

def build_graph(accessible_only=False):
    nodes = load_nodes()
    edges, src = load_edges(nodes)
    G = nx.Graph()
    for nid, data in nodes.items():
        G.add_node(nid, **data)
    for u, v, d in edges:
        if accessible_only and not d.get("accessible", True):
            continue
        G.add_edge(u, v, **d)
    return G, nodes, src

def entrances(nodes, code: str):
    code = (code or "").strip()
    return [nid for nid, nd in nodes.items()
            if nd.get("type") == "entrance" and nd.get("building_code") == code]

def shortest_node_to_node(G, start_id, end_id):
    path = nx.shortest_path(G, start_id, end_id, weight="weight")
    dist = nx.path_weight(G, path, weight="weight")
    return dist, path

def shortest_building_to_building(G, nodes, start_code, end_code):
    S, T = entrances(nodes, start_code), entrances(nodes, end_code)
    best = None
    for s in S:
        for t in T:
            try:
                p = nx.shortest_path(G, s, t, weight="weight")
                d = nx.path_weight(G, p, weight="weight")
                if (best is None) or d < best[0]:
                    best = (d, p)
            except nx.NetworkXNoPath:
                pass
    return best  # (distance, path) or None

def directions(G, path):
    lines = []
    for a, b in zip(path, path[1:]):
        A = G.nodes[a].get("label", a)
        B = G.nodes[b].get("label", b)
        typ = G.edges[a, b].get("path_type", "path")
        lines.append(f"• {A} → {B} via {typ}")
    return "\n".join(lines)

if __name__ == "__main__":
    G, nodes, src = build_graph(accessible_only=False)
    print(f"Graph ready. Edges source: {src}. Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    # Example 1 — Node→Node (TechCampus)
    try:
        d, p = shortest_node_to_node(G, "n_tc_A_south", "n_tc_D_main")
        print("\n[Node→Node] A_south → D_main")
        print("Path:", " → ".join(p))
        print("Distance (px):", round(d, 1))
        print(directions(G, p))
    except Exception as e:
        print("\n[Node→Node] could not compute:", e)

    # Example 2 — Building→Building with multi-entrance (E → F)
    best = shortest_building_to_building(G, nodes, "E", "F")
    if best:
        d, p = best
        print("\n[Building→Building] E → F (best entrance pair)")
        print("Path:", " → ".join(p))
        print("Distance (px):", round(d, 1))
        print(directions(G, p))
    else:
        print("\n[Building→Building] E → F: no path found")

    # Example 3 — BildungsCampus: 11 → 17
    best = shortest_building_to_building(G, nodes, "11", "17")
    if best:
        d, p = best
        print("\n[Building→Building] 11 → 17")
        print("Path:", " → ".join(p))
        print("Distance (px):", round(d, 1))
        print(directions(G, p))
