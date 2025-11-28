import csv, os, math, sys

MAPS_DIR = "data/maps"
nodes_path = os.path.join(MAPS_DIR, "nodes.csv")
edges_path = os.path.join(MAPS_DIR, "edges.csv")
out_path   = os.path.join(MAPS_DIR, "edges_distances.csv")

def normalize_key(k: str) -> str:
    return (k or "").replace("\ufeff", "").strip().lower()

def to_int_or_none(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None

nodes = {}
missing_xy_nodes = []
with open(nodes_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    field_map = {name: normalize_key(name) for name in reader.fieldnames}
    rows = []
    for row in reader:
        fixed = {}
        for k, v in row.items():
            fixed[normalize_key(k)] = v
        nid = fixed.get("id")
        x = to_int_or_none(fixed.get("x"))
        y = to_int_or_none(fixed.get("y"))
        if nid and x is not None and y is not None:
            nodes[nid] = (x, y)
        else:
            missing_xy_nodes.append(nid or "(no id)")

print(f"Loaded nodes with coords: {len(nodes)}")
if missing_xy_nodes:
    print(f"Skipped nodes without coords: {len(missing_xy_nodes)}")
    print("Examples:", missing_xy_nodes[:10])

updated = 0
skipped = 0
rows_out = []

with open(edges_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = [normalize_key(n) for n in reader.fieldnames]
    original_fieldnames = reader.fieldnames

    for row in reader:
        r = {normalize_key(k): v for k, v in row.items()}
        u = r.get("from")
        v = r.get("to")

        if u in nodes and v in nodes:
            (x1, y1) = nodes[u]
            (x2, y2) = nodes[v]
            dist = round(math.hypot(x2 - x1, y2 - y1), 1)
            row["distance_px"] = str(dist)
            updated += 1
        else:
            skipped += 1
        rows_out.append(row)

if "distance_px" not in [normalize_key(n) for n in original_fieldnames]:
    original_fieldnames = original_fieldnames + ["distance_px"]

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=original_fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Updated edges with distance: {updated}")
print(f"Skipped edges (missing node coords): {skipped}")
print(f"✔ Wrote: {out_path}")
