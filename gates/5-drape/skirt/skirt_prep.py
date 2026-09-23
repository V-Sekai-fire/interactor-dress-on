"""Prepare the fitted-skirt drape fixture from Gate 8's integration evidence and
print its geometry statistics (the inputs the drape's kernels see)."""
import json, math, sys, os

SRC = sys.argv[1]  # integration dir
OUT = sys.argv[2]  # fixture dir
os.makedirs(OUT, exist_ok=True)

V, F = [], []
for ln in open(os.path.join(SRC, "fitdrape3.fitted.obj")):
    p = ln.split()
    if not p:
        continue
    if p[0] == "v":
        V.append(tuple(float(x) for x in p[1:4]))
    elif p[0] == "f":
        F.append(tuple(int(x.split("/")[0]) - 1 for x in p[1:4]))
print("nV", len(V), "nF", len(F))

# boundary loops, as project/util/mesh_topo.gd (ido-8) builds them
count = {}
for t in F:
    for k in range(3):
        a, b = t[k], t[(k + 1) % 3]
        key = (min(a, b), max(a, b))
        count[key] = count.get(key, 0) + 1
nxt = {}
for key, c in count.items():
    if c == 1:
        for a, b in ((key[0], key[1]), (key[1], key[0])):
            nxt.setdefault(a, []).append(b)
used = set()
loops = []
for start in nxt:
    if start in used:
        continue
    loop = [start]
    used.add(start)
    prev, cur = -1, start
    while True:
        step = -1
        for c in nxt[cur]:
            if c != prev and c not in used:
                step = c
                break
        if step < 0:
            break
        loop.append(step)
        used.add(step)
        prev, cur = cur, step
    loops.append(loop)
my = lambda L: sum(V[i][1] for i in L) / max(1, len(L))
best = max(range(len(loops)), key=lambda i: my(loops[i]))
pins = loops[best]
print("loops", [len(l) for l in loops], "pins", len(pins), "mean_y", my(pins))
nonman = sum(1 for c in count.values() if c > 2)
print("non-manifold edges", nonman)

def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(dot(a, a))

for s in (5.0, 10.0):
    P = [(x*s, y*s, z*s) for x, y, z in V]
    areas, invmax, minedge, heights = [], [], [], []
    for t in F:
        p0, p1, p2 = P[t[0]], P[t[1]], P[t[2]]
        e0, e1 = sub(p1, p0), sub(p2, p0)
        a = 0.5 * norm(cross(e0, e1))
        areas.append(a)
        l = [norm(e0), norm(e1), norm(sub(p2, p1))]
        minedge.append(min(l))
        heights.append(2 * a / max(l) if max(l) > 0 else 0)
        n0 = norm(e0)
        P0 = tuple(c / n0 for c in e0)
        q = sub(e1, tuple(c * dot(e1, P0) for c in P0))
        nq = norm(q)
        P1 = tuple(c / nq for c in q) if nq > 0 else (0, 0, 0)
        d00, d01, d10, d11 = dot(P0, e0), dot(P0, e1), dot(P1, e0), dot(P1, e1)
        det = d00 * d11 - d10 * d01
        inv = [d11/det, -d01/det, -d10/det, d00/det] if det != 0 else [float('inf')]*4
        invmax.append(max(abs(x) for x in inv))
    order = sorted(range(len(F)), key=lambda i: areas[i])
    print(f"scale {s}: area min {areas[order[0]]:.3e} (tri {order[0]}) median {sorted(areas)[len(areas)//2]:.3e}; "
          f"min edge {min(minedge):.3e}; min height {min(heights):.3e}; max |invUV| {max(invmax):.3e}")
    print("  5 smallest-area tris:", [(i, f"{areas[i]:.2e}", f"h={heights[i]:.2e}", f"inv={invmax[i]:.2e}") for i in order[:5]])
    # radii: min connected edge / 2 - 0.01 (updateCollisionRadii)
    vt = [[] for _ in V]
    for ti, t in enumerate(F):
        for r in range(3):
            vt[t[r]].append(ti)
    radii = []
    for i in range(len(V)):
        me = 100.0
        for ti in vt[i]:
            t = F[ti]
            p2, p3 = t[0], t[1]
            if p2 == i: p2 = t[2]
            if p3 == i: p3 = t[2]
            me = min(me, norm(sub(P[p2], P[i])), norm(sub(P[p3], P[i])))
        radii.append(me / 2 - 0.01)
    rs = sorted(radii)
    print(f"  radii min {rs[0]:.4f} median {rs[len(rs)//2]:.4f} max {rs[-1]:.4f}; unreferenced vertices {sum(1 for x in vt if not x)}")

with open(os.path.join(OUT, "fitted_skirt.obj"), "w", newline="\n") as f:
    f.write("# Gate 8 fitdrape3: fit.elf's fitted LCL_Skirt_DressEvening_003 on FoxGirl (body metres)\n")
    for v in V:
        f.write("v %.9f %.9f %.9f\n" % v)
    for t in F:
        f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))
with open(os.path.join(OUT, "fitted_skirt_pins.txt"), "w", newline="\n") as f:
    f.write(" ".join(str(i) for i in pins) + "\n")
caps = json.load(open(os.path.join(SRC, "fitdrape3.json")))["capsules"]
def v3(s): return [float(x) for x in s.strip("()").split(",")]
with open(os.path.join(OUT, "fitted_skirt_capsules.txt"), "w", newline="\n") as f:
    f.write("# bottom(3) axis(3) radius length, body metres; radius = median skin - 0.1/10 (Gate 8 at drape scale 10)\n")
    for c in caps:
        b, a = v3(c["bottom"]), v3(c["axis"])
        f.write("%.6f %.6f %.6f %.6f %.6f %.6f %.9f %.9f\n" % (*b, *a, c["radius"], c["length"]))
print("capsules", len(caps))
