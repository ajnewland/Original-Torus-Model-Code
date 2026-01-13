#!/usr/bin/env python3
# extract_plaquettes.py
#
# Purpose:
#   From an existing spin-network / spinfoam model, compute plaquette (face) holonomies:
#     - SU(2): 2x2 holonomy product around the loop → tr(U_p) = 2 cos θ_p → θ_p
#     - U(1): signed sum of edge phases around the loop → φ_p wrapped to (-π, π]
#
# Inputs (choose one of the following formats):
#   (A) JSON model (single file, preferred for projects)
#       {
#         "nodes": [{"id": "n1"}, ...],  # optional for faces; not strictly needed
#         "edges": [
#           {
#             "id": "e12", "u": "n1", "v": "n2",    # undirected edge endpoints
#             "dir": +1,                            # stored orientation (+1 from u->v, -1 from v->u)
#             "su2": [[Re00,Im00, Re01,Im01],
#                     [Re10,Im10, Re11,Im11]],      # 2x2 complex matrix (spin-1/2 rep)
#             "u1_phase": 0.018                     # hypercharge phase (radians)
#           },
#           ...
#         ],
#         "faces": [                                # optional; if missing, script will discover cycles
#           {"id":"p001","edges":[["e12",+1],["e23",+1],["e31",+1]]},
#           ...
#         ]
#       }
#
#   (B) CSV pair (edges + faces):
#     edges.csv:
#       edge_id,u,v,dir,Re00,Im00,Re01,Im01,Re10,Im10,Re11,Im11,u1_phase
#     faces.csv:
#       plaq_id,edges   # edges column is space-separated "edge_id:sign" tokens: e12:+1 e23:+1 e31:+1
#
# Outputs:
#   plaquettes.csv with columns:
#     plaq_id,su2_tr,su2_theta,u1_phase,length,edges
#     - su2_tr     : real trace Tr[U_p]
#     - su2_theta  : θ_p in [0, π], from Tr U_p = 2 cos θ_p
#     - u1_phase   : φ_p ∈ (-π, π]
#     - length     : number of edges in the plaquette
#     - edges      : canonicalized loop specification "e12:+1 e23:+1 e31:+1"
#
# Cycle discovery (when faces are not provided):
#   - Builds a directed version of the (undirected) graph with both orientations.
#   - Uses a Johnson-style simple cycle search limited by --max-len (default 6) to avoid explosion.
#   - Deduplicates cycles up to rotation & reversal & edge-rename to get elementary plaquettes.
#
# Notes:
#   - The script gently "unitarizes" edge SU(2) matrices via polar decomposition and enforces det=+1.
#   - Orientation handling: if the loop traverses an edge opposite to its stored orientation,
#     the SU(2) factor is U_e† and the U(1) phase contributes with a minus sign.
#
# Usage examples:
#   JSON model with provided faces:
#     python extract_plaquettes.py --model model.json --out plaquettes.csv
#
#   JSON model WITHOUT faces (auto-discover cycles up to length 6):
#     python extract_plaquettes.py --model model.json --out plaquettes.csv --discover --max-len 6
#
#   CSV inputs with faces:
#     python extract_plaquettes.py --edges edges.csv --faces faces.csv --out plaquettes.csv
#
#   CSV edges only (auto-discover):
#     python extract_plaquettes.py --edges edges.csv --out plaquettes.csv --discover --max-len 6
#
import argparse, csv, json, math, sys
from typing import Dict, List, Tuple, Optional
import numpy as np

# -------- IO helpers --------

def load_json_model(path: str):
   with open(path, "r", encoding="utf-8") as f:
       data = json.load(f)
   nodes = {n["id"]: n for n in data.get("nodes", [])}
   edges = {}
   for e in data["edges"]:
       eid = e["id"]
       U = su2_from_list(e["su2"])
       # store endpoints + orientation
       edges[eid] = {
           "u": e["u"], "v": e["v"],
           "dir": int(e.get("dir", 1)),
           "U": su2_unitarize(U),
           "alpha": float(e["u1_phase"]),
       }
   faces = data.get("faces", None)
   if faces is not None:
       faces = [{"id": f["id"], "edges": [(eid, int(s)) for eid, s in f["edges"]]} for f in faces]
   return nodes, edges, faces

def load_edges_csv(path: str):
   edges = {}
   with open(path, newline="", encoding="utf-8") as f:
       for r in csv.DictReader(f):
           eid = r["edge_id"].strip()
           U = np.array([
               [float(r["Re00"])+1j*float(r["Im00"]), float(r["Re01"])+1j*float(r["Im01"])],
               [float(r["Re10"])+1j*float(r["Im10"]), float(r["Re11"])+1j*float(r["Im11"])],
           ], dtype=complex)
           edges[eid] = {
               "u": r["u"].strip(),
               "v": r["v"].strip(),
               "dir": int(r.get("dir","1")),
               "U": su2_unitarize(U),
               "alpha": float(r["u1_phase"]),
           }
   return edges

def load_faces_csv(path: str):
   faces = []
   with open(path, newline="", encoding="utf-8") as f:
       for r in csv.DictReader(f):
           pid = r["plaq_id"].strip()
           tokens = parse_edge_tokens(r["edges"])
           faces.append({"id": pid, "edges": tokens})
   return faces

def su2_from_list(block: List[List[float]]) -> np.ndarray:
   # block is [[Re00,Im00, Re01,Im01],[Re10,Im10, Re11,Im11]]
   return np.array([
       [block[0][0]+1j*block[0][1], block[0][2]+1j*block[0][3]],
       [block[1][0]+1j*block[1][1], block[1][2]+1j*block[1][3]],
   ], dtype=complex)

# -------- SU(2) sanitation --------

def su2_unitarize(U: np.ndarray) -> np.ndarray:
   # Polar decomposition: U ← U (U†U)^(-1/2); then enforce det=+1 by removing global phase.
   try:
       H = U.conj().T @ U
       w, V = np.linalg.eigh(H)
       Hm12 = V @ np.diag(w**-0.5) @ V.conj().T
       Uc = U @ Hm12
       detU = np.linalg.det(Uc)
       Uc = Uc / detU**0.5
       return Uc
   except Exception:
       return U

# -------- Parsing helpers --------

def parse_edge_tokens(s: str) -> List[Tuple[str,int]]:
   toks = s.strip().split()
   out = []
   for t in toks:
       name, sign = t.split(":")
       out.append((name.strip(), int(sign)))
   return out

def canonical_edge_string(loop: List[Tuple[str,int]]) -> str:
   # produce a canonical rotation/reversal-invariant string for deduplication
   seq = [f"{e}:{s:+d}" for e,s in loop]
   # all rotations and their reversals, pick lexicographically min
   rots = []
   n = len(seq)
   for k in range(n):
       rots.append(seq[k:]+seq[:k])
   rev = [f"{e}:{-s:+d}" for e,s in loop[::-1]]  # reverse and flip signs
   for k in range(n):
       rots.append(rev[k:]+rev[:k])
   return min(" ".join(r) for r in rots)

# -------- Graph and cycle discovery --------

class Graph:
   def __init__(self):
       self.adj: Dict[str, List[Tuple[str,str,int]]] = {}  # node -> list of (edge_id, nbr, rel_dir)

   def add_edge(self, eid: str, u: str, v: str, stored_dir: int):
       # store both directions; rel_dir=+1 when traversing with stored orientation, -1 otherwise
       self.adj.setdefault(u, []).append((eid, v, +1 if stored_dir==+1 else -1))
       self.adj.setdefault(v, []).append((eid, u, +1 if stored_dir==-1 else -1))

def discover_cycles_small(graph: Graph, max_len: int = 6, max_cycles: int = 50000) -> List[List[Tuple[str,int]]]:
   """
   Find simple cycles up to length max_len.
   Returns cycles as list of (edge_id, sign) following traversal direction (sign = +1 with stored dir, -1 against).
   """
   seen_can = set()
   cycles: List[List[Tuple[str,int]]] = []

   # DFS paths (node, incoming_edge_id or None, incoming_rel_dir)
   def dfs(start: str):
       stack = [(start, [], set([start]))]  # (node, path list of (eid, next_node, sign), visited_nodes)
       while stack:
           node, path, visited = stack.pop()
           for (eid, nxt, rel) in graph.adj.get(node, []):
               if path and eid == path[-1][0]:
                   continue  # don't go back via the same edge immediately
               if nxt == start and len(path) >= 2:
                   # closed cycle found: build edge list with signs as traversed
                   edges_with_signs = []
                   last_node = start
                   for (peid, pnext, prel) in path + [(eid, nxt, rel)]:
                       # prel is the relative orientation used when traversed
                       edges_with_signs.append((peid, prel))
                       last_node = pnext
                   if 3 <= len(edges_with_signs) <= max_len:
                       can = canonical_edge_string(edges_with_signs)
                       if can not in seen_can:
                           seen_can.add(can)
                           cycles.append(edges_with_signs)
                           if len(cycles) >= max_cycles:
                               return
                   continue
               if nxt in visited:
                   continue
               if len(path) + 1 > max_len:
                   continue
               stack.append((nxt, path + [(eid, nxt, rel)], visited | {nxt}))
   for start in list(graph.adj.keys()):
       dfs(start)
       if len(cycles) >= max_cycles:
           break
   return cycles

# -------- Plaquette holonomy computation --------

def wrap_pi(phi: float) -> float:
   return ((phi + math.pi) % (2*math.pi)) - math.pi

def compute_plaquette(edges: Dict[str,dict], loop: List[Tuple[str,int]]) -> Tuple[float,float]:
   """
   loop: list of (edge_id, sign) following traversal order.
   Returns (tr_su2_real, theta, u1_phase_wrapped)
   """
   Up = np.eye(2, dtype=complex)
   phi = 0.0
   for eid, sign in loop:
       E = edges[eid]
       rel = sign  # +1 with stored dir, -1 against
       # choose matrix according to traversal relative to stored orientation
       Ue = E["U"] if rel == +1 else E["U"].conj().T
       Up = Up @ Ue
       alpha = E["alpha"] if rel == +1 else -E["alpha"]
       phi += alpha
   tr = float(np.real(np.trace(Up)))
   x = max(-2.0, min(2.0, tr))
   theta = math.acos(x / 2.0)  # in [0, π]
   return tr, theta, wrap_pi(phi)

# -------- Main --------

def main():
   ap = argparse.ArgumentParser(description="Extract plaquette SU(2)/U(1) holonomies from an LQG–SM model.")
   ap.add_argument("--model", help="JSON model file with edges [+ optional faces].")
   ap.add_argument("--edges", help="edges.csv if not using JSON.")
   ap.add_argument("--faces", help="faces.csv (optional; if omitted, cycles will be discovered if --discover).")
   ap.add_argument("--discover", action="store_true", help="Discover small cycles as plaquettes when faces not provided.")
   ap.add_argument("--max-len", type=int, default=6, help="Max cycle length to discover (default 6).")
   ap.add_argument("--max-cycles", type=int, default=50000, help="Safety cap on number of cycles.")
   ap.add_argument("--out", default="plaquettes.csv", help="Output CSV (plaquettes).")
   args = ap.parse_args()

   # Load inputs
   if args.model:
       nodes, edges, faces = load_json_model(args.model)
   else:
       if not args.edges:
           print("Provide --model JSON or --edges CSV.", file=sys.stderr); sys.exit(1)
       edges = load_edges_csv(args.edges)
       faces = load_faces_csv(args.faces) if args.faces else None

   # If no faces and not discovering, abort
   if faces is None and not args.discover:
       print("No faces provided and --discover not set. Nothing to do.", file=sys.stderr)
       sys.exit(2)

   # Discover cycles if requested
   if faces is None and args.discover:
       # Build undirected graph
       G = Graph()
       for eid, E in edges.items():
           G.add_edge(eid, E["u"], E["v"], E["dir"])
       print(f"[info] discovering cycles up to length {args.max_len} ...")
       cycles = discover_cycles_small(G, max_len=args.max_len, max_cycles=args.max_cycles)
       # build face list
       faces = [{"id": f"p{idx+1:06d}", "edges": cyc} for idx, cyc in enumerate(cycles)]
       print(f"[ok] discovered {len(faces)} candidate plaquettes")

   # Compute plaquette holonomies
   out_rows = []
   for f in faces:
       loop = f["edges"]  # list[(eid,sign)]
       tr, theta, phi = compute_plaquette(edges, loop)
       out_rows.append({
           "plaq_id": f["id"],
           "su2_tr": f"{tr:.12g}",
           "su2_theta": f"{theta:.12g}",
           "u1_phase": f"{phi:.12g}",
           "length": str(len(loop)),
           "edges": canonical_edge_string(loop),
       })

   # Write output
   with open(args.out, "w", newline="", encoding="utf-8") as f:
       w = csv.DictWriter(f, fieldnames=["plaq_id","su2_tr","su2_theta","u1_phase","length","edges"])
       w.writeheader(); w.writerows(out_rows)
   print(f"[ok] wrote {args.out}  (plaquettes={len(out_rows)})")

if __name__ == "__main__":
   main()