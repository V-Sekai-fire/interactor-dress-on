import re, statistics, sys
p = sys.argv[1]
ccd, stat, col, nnz, ns, solvers = [], [], [], [], set(), {}
for line in open(p, errors="replace"):
    m = re.search(r"\[amdahl\] candidates (ccd|static) vv (\d+) ev (\d+) ee (\d+) fv (\d+)", line)
    if m:
        tot = sum(int(x) for x in m.groups()[1:])
        (ccd if m.group(1) == "ccd" else stat).append(tot)
        continue
    m = re.search(r"\[amdahl\] collisions vv (\d+) ev (\d+) ee (\d+) fv (\d+)", line)
    if m:
        col.append(sum(int(x) for x in m.groups()))
        continue
    m = re.search(r"\[amdahl\] linsolve (\S+) n (\d+) nnz (\d+)", line)
    if m:
        ns.add(int(m.group(2)))
        nnz.append(int(m.group(3)))
        solvers[m.group(1)] = solvers.get(m.group(1), 0) + 1
def st(a):
    if not a:
        return "none"
    s = sorted(a)
    return f"n={len(a)} min={s[0]} median={statistics.median(s)} max={s[-1]} mean={sum(s)/len(s):.0f} total={sum(s)}"
print("ccd candidates per build:", st(ccd))
if ccd:
    s = sorted(ccd, reverse=True)
    print("  top5 share:", round(sum(s[:5]) / sum(s), 3), "top5:", s[:5])
    print("  8B/pair readback max MB:", round(s[0] * 8 / 1e6, 2), "mean MB:", round(sum(s) / len(s) * 8 / 1e6, 3))
print("static candidates:", st(stat))
print("collisions per build:", st(col))
print("n:", sorted(ns), "nnz:", st(nnz), "solvers:", solvers)
