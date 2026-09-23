# Aggregate raw census TSVs (patched ggml-cpu, one line per graph node) into
#   census_<app>.csv : op, sub, dst_type, src_types, contiguity, shape_class, count, graphs, example_ne
#   census_union.csv : op/sub x app counts + whether the op is in the K1-K8 family set
# usage: aggregate.py <outdir> <app>=<raw.tsv>[,<raw.tsv>...] ...
import csv, sys, collections, os

# K1..K8 of plan Cut 3 (ggml op names; UNARY ops as UNARY:<name>)
K = {
    "K1": ["UNARY:SILU", "UNARY:GELU", "UNARY:GELU_ERF", "UNARY:SIGMOID", "UNARY:NEG", "SCALE", "DIAG_MASK_INF"],
    "K2": ["ADD", "MUL", "CPY", "CONT", "DUP", "GET_ROWS", "CONCAT", "REPEAT"],
    "K3": ["NORM", "RMS_NORM", "MEAN"],
    "K4": ["SOFT_MAX"],
    "K5": ["ROPE"],
    "K6": ["MUL_MAT"],
    "K7": ["IM2COL", "CONV_3D"],
    "K8": ["FLASH_ATTN_EXT"],
}
FAMILY = {op: fam for fam, ops in K.items() for op in ops}
NOOP = {"NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
BINARY = {"ADD", "MUL", "SUB", "DIV", "ADD1", "ADD_ID"}


def parse_t(cols):
    typ, con, ne, nb = cols
    if typ == "-":
        return None
    return {"type": typ, "c": con, "ne": [int(x) for x in ne.split(",")], "nb": [int(x) for x in nb.split(",")]}


def bucket(k):
    for b in (1, 4, 16, 64, 128, 256, 1024, 4096, 16384):
        if k <= b:
            return f"<={b}"
    return ">16384"


def ndim(t):
    n = 4
    while n > 1 and t["ne"][n - 1] == 1:
        n -= 1
    return n


def shape_class(op, sub, d, s):
    s0, s1 = s[0], s[1]
    if op in NOOP:
        return "view"
    if op == "MUL_MAT":
        K_, M = s0["ne"][0], s0["ne"][1]
        N = s1["ne"][1]
        batch = s1["ne"][2] * s1["ne"][3]
        b0 = s0["ne"][2] * s0["ne"][3]
        kind = "vec" if N == 1 else ("small-N" if N < 32 else "mat")
        bc = "" if batch == 1 else (f" batch{'-bcast' if b0 != batch else ''}")
        return f"{kind} K{bucket(K_)} M{bucket(M)}{bc}"
    if op == "FLASH_ATTN_EXT":
        q, k = s0, s1
        return f"D={q['ne'][0]} Lq{bucket(q['ne'][1])} Lk{bucket(k['ne'][1])} H={q['ne'][2]}{' mask' if s[3] else ''}"
    if op in BINARY and s1 is not None:
        if s1["ne"] == s0["ne"]:
            return f"same nd{ndim(s0)}"
        mask = "".join("b" if (s1["ne"][i] == 1 and s0["ne"][i] != 1) else "." for i in range(4))
        return f"bcast[{mask}] nd{ndim(s0)}"
    if op in ("SOFT_MAX",):
        return f"row{bucket(s0['ne'][0])}{' mask' if s1 else ''}"
    if op in ("NORM", "RMS_NORM", "MEAN", "SUM_ROWS", "ARGMAX"):
        return f"row{bucket(s0['ne'][0])}"
    if op == "ROPE":
        return f"hd={s0['ne'][0]}"
    if op in ("GET_ROWS", "SET_ROWS"):
        return f"row{bucket(s0['ne'][0])} idx{bucket(s1['ne'][0])}"
    if op in ("CPY", "DUP", "CONT"):
        return f"nd{ndim(d)} {s0['type']}->{d['type']}"
    return f"nd{ndim(d)}"


def load(paths):
    rows = collections.Counter()
    graphs = collections.defaultdict(set)
    example = {}
    nodes = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 25:
                    continue
                gid, idx, op, sub = c[0], c[1], c[2], c[3]
                d = parse_t(c[4:8])
                srcs = [parse_t(c[8 + 4 * k: 12 + 4 * k]) for k in range(4)]
                params = c[24] if len(c) > 24 else "-"
                nodes += 1
                src_types = "|".join(x["type"] for x in srcs if x) or "-"
                contig = d["c"] + ":" + "".join(x["c"] for x in srcs if x)
                key = (op, sub, d["type"], src_types, contig, shape_class(op, sub, d, srcs))
                rows[key] += 1
                graphs[key].add(p + gid)
                if key not in example:
                    example[key] = "dst" + ",".join(map(str, d["ne"])) + " " + " ".join(
                        f"s{k}:" + ",".join(map(str, x["ne"])) for k, x in enumerate(srcs) if x) + f" p={params}"
    return rows, graphs, example, nodes


def main():
    out = sys.argv[1]
    per_app = {}
    for arg in sys.argv[2:]:
        app, paths = arg.split("=", 1)
        paths = [p for p in paths.split(",") if os.path.exists(p)]
        rows, graphs, example, nodes = load(paths)
        per_app[app] = rows
        with open(os.path.join(out, f"census_{app}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "sub", "dst_type", "src_types", "contiguity(dst:srcs)", "shape_class", "count", "graphs", "family", "example"])
            for key, n in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[1])):
                op, sub = key[0], key[1]
                name = f"UNARY:{sub}" if op == "UNARY" else (f"GLU:{sub}" if op == "GLU" else op)
                fam = "noop" if op in NOOP else FAMILY.get(name, "NEW")
                w.writerow(list(key) + [n, len(graphs[key]), fam, example[key]])
        print(f"{app}: {nodes} nodes, {len(rows)} (op,type,contig,shape) classes")
    ops = collections.defaultdict(lambda: collections.Counter())
    for app, rows in per_app.items():
        for key, n in rows.items():
            op, sub = key[0], key[1]
            name = f"UNARY:{sub}" if op == "UNARY" else (f"GLU:{sub}" if op == "GLU" else op)
            ops[name][app] += n
    apps = list(per_app)
    with open(os.path.join(out, "census_union.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family"] + apps)
        for name in sorted(ops):
            base = name.split(":")[0]
            fam = "noop" if base in NOOP else FAMILY.get(name, "NEW")
            w.writerow([name, fam] + [ops[name].get(a, 0) for a in apps])
    for name in sorted(ops):
        base = name.split(":")[0]
        fam = "noop" if base in NOOP else FAMILY.get(name, "NEW")
        print(f"  {name:28s} {fam:5s} " + " ".join(f"{a}={ops[name].get(a, 0)}" for a in apps))


main()
