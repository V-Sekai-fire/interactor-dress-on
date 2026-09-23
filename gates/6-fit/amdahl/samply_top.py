#!/usr/bin/env python3
"""List the heaviest functions of a samply profile by inclusive and self time.

samply record --save-only --unstable-presymbolicate writes <out>.json.gz and
<out>.json.syms.json; this resolves frames through the sidecar and prints the
top functions of the busiest thread, so the bucket rules in reduce_samply.py
can be written against real names.

  python samply_top.py profile.json.gz [N] [filter-substring]
"""
import collections
import gzip
import json
import sys


def load(path):
    p = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    base = path[:-3] if path.endswith(".gz") else path
    syms = json.load(open(base + ".syms.json"))
    return p, syms


def symbolicator(p, syms):
    st = syms["string_table"]
    by_id = {}
    for e in syms["data"]:
        key = e["debug_id"].replace("-", "").upper()
        table = e["symbol_table"]
        known = {a: table[i] for a, i in e["known_addresses"]}
        by_id[key] = known
    libs = p["libs"]

    def resolve(thread):
        ft, fn, rt = thread["frameTable"], thread["funcTable"], thread["resourceTable"]
        sa = thread["stringArray"]
        names = []
        for fi in range(ft["length"]):
            func = ft["func"][fi]
            res = fn["resource"][func]
            addr = ft["address"][fi]
            name = None
            libname = "?"
            if res is not None and res >= 0:
                lib = libs[rt["lib"][res]]
                libname = lib["name"]
                known = by_id.get(lib["breakpadId"].upper())
                if known is not None and addr is not None and addr >= 0:
                    s = known.get(addr)
                    if s is not None:
                        name = st[s["symbol"]]
            if name is None:
                name = sa[fn["name"][func]]
            names.append((libname, name))
        return names

    return resolve


def main_thread(p):
    return max(p["threads"], key=lambda t: t["samples"]["length"])


def stacks(thread):
    """Yield (weight, [frame indices root..leaf]) per sample."""
    stt = thread["stackTable"]
    cache = {}

    def chain(si):
        if si in cache:
            return cache[si]
        out = []
        s = si
        while s is not None:
            out.append(stt["frame"][s])
            s = stt["prefix"][s]
        out.reverse()
        cache[si] = out
        return out

    sm = thread["samples"]
    w = sm.get("weight") or [1] * sm["length"]
    for i in range(sm["length"]):
        si = sm["stack"][i]
        if si is None:
            continue
        yield w[i], chain(si)


if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    flt = sys.argv[3] if len(sys.argv) > 3 else None
    p, syms = load(path)
    t = main_thread(p)
    names = symbolicator(p, syms)(t)
    incl = collections.Counter()
    selfc = collections.Counter()
    total = 0
    for w, ch in stacks(t):
        total += w
        seen = set()
        for f in ch:
            nm = names[f][1]
            if nm not in seen:
                incl[nm] += w
                seen.add(nm)
        selfc[names[ch[-1]][1]] += w
    print(f"thread {t['name']} samples {total} interval {p['meta']['interval']} ms")
    print("== inclusive")
    for nm, c in incl.most_common():
        if flt and flt not in nm:
            continue
        print(f"{c:7d} {100.0*c/total:6.2f}%  {nm[:220]}")
        n -= 1
        if n <= 0:
            break
    print("== self")
    for nm, c in selfc.most_common(40):
        print(f"{c:7d} {100.0*c/total:6.2f}%  {nm[:220]}")
