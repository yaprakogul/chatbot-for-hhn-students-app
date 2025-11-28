import csv, os, math

MAPS_DIR   = "data/maps"
nodes_path = os.path.join(MAPS_DIR, "nodes.csv")
edges_path = os.path.join(MAPS_DIR, "edges.csv")
out_path   = os.path.join(MAPS_DIR, "edges_distances.csv")

def norm(s):
    return (s or "").replace("\ufeff","").strip().lower()

def to_int_or_none(v):
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    try:
        return int(float(s))
    except ValueError:
        return None

nodes = {}
with open(nodes_path, newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        raw_id = row.get("id") or row.get("\ufeffid")
        if raw_id and str(raw_id).lstrip().startswith("#"):
            continue
        nid = (raw_id or "").strip()
        x = to_int_or_none(row.get("x"))
        y = to_int_or_none(row.get("y"))
        if nid and x is not None and y is not None:
            nodes[nid] = (x, y)

print(f"Loaded nodes with coords: {len(nodes)}")

rows_out = []
updated, skipped = 0, 0

with open(edges_path, newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    fieldnames = list(rdr.fieldnames or [])
    if "distance_px" not in [norm(h) for h in fieldnames]:
        fieldnames = fieldnames + ["distance_px"]

    for row in rdr:
        fcol = (row.get("from") or "").strip()
        tcol = (row.get("to") or "").strip()
        if (not fcol and not tcol) or fcol.startswith("#") or tcol.startswith("#"):
            continue

        if None in row:
            row.pop(None, None)

        u, v = fcol, tcol
        if u in nodes and v in nodes:
            (x1, y1) = nodes[u]; (x2, y2) = nodes[v]
            row["distance_px"] = str(round(math.hypot(x2-x1, y2-y1), 1))
            updated += 1
        else:
            skipped += 1

        rows_out.append(row)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)

print(f"Updated edges with distance: {updated}")
print(f"Skipped edges (missing node coords / comments): {skipped}")
print(f"✔ Wrote: {out_path}")
