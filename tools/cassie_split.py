"""cassie_split -- train / validation / test splits of CASSIE's curves corpus, as a .usda.

  python tools/cassie_split.py <datasource-cassie>/data/curves <out.usda> [--rev=SHA]

Split by group, never by file: a participant's six sketches (NN-a-b.curves) are one
group, and so are a named object's variants (architecture, architecture-2), so no
person's or subject's strokes sit on both sides. Groups are shuffled with a fixed seed
and cut 60/20/20. The sketches this work already opened (SEEN) stay in train: a set
that has been looked at cannot be held out after the fact. Test is withheld: it is
listed with each file's BLAKE3 so a later change to it is detectable, and
tools/curves_usd.py refuses to convert a test file without --unblind.
"""
import argparse
import os
import random
import re
import sys

from blake3 import blake3
from pxr import Sdf, Usd, UsdGeom, Vt

SEED = 20260925
SEEN = {"dress", "hat", "flower", "vintage_car"}
CUT = (0.6, 0.2)


def group_of(name):
    m = re.match(r"^(\d{2})-\d-\d$", name)
    return "participant-" + m.group(1) if m else re.sub(r"-\d+$", "", name)


def split(files):
    groups = {}
    for f in sorted(files):
        groups.setdefault(group_of(f[:-len(".curves")]), []).append(f)
    seen = sorted(g for g in groups if g in SEEN)
    rest = sorted(g for g in groups if g not in SEEN)
    random.Random(SEED).shuffle(rest)
    n_train = round(len(groups) * CUT[0]) - len(seen)
    n_val = round(len(groups) * CUT[1])
    parts = {"train": seen + rest[:n_train], "validation": rest[n_train:n_train + n_val],
             "test": rest[n_train + n_val:]}
    return {k: sorted(f for g in v for f in groups[g]) for k, v in parts.items()}, groups


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("curves_dir")
    ap.add_argument("out")
    ap.add_argument("--rev", default="")
    o = ap.parse_args(argv)
    files = [f for f in os.listdir(o.curves_dir) if f.endswith(".curves")]
    parts, groups = split(files)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/Splits", "Scope")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().customLayerData = {
        "dataset": "V-Sekai/datasource-cassie data/curves", "dataset_rev": o.rev, "seed": SEED,
        "rule": "group split (participant, or named subject), seeded shuffle, 60/20/20; seen sketches stay in train",
        "seen": Vt.StringArray(sorted(SEEN)), "withheld": "test"}
    for k, fs in parts.items():
        p = stage.DefinePrim("/Splits/" + k, "Scope")
        p.CreateAttribute("files", Sdf.ValueTypeNames.StringArray, custom=True).Set(Vt.StringArray(fs))
        sums = [blake3(open(os.path.join(o.curves_dir, f), "rb").read()).hexdigest()[:12] for f in fs]
        p.CreateAttribute("blake3", Sdf.ValueTypeNames.StringArray, custom=True).Set(Vt.StringArray(sums))
        p.CreateAttribute("groups", Sdf.ValueTypeNames.Int, custom=True).Set(
            len({group_of(f[:-len(".curves")]) for f in fs}))
    open(o.out, "w").write(stage.GetRootLayer().ExportToString())
    print("ok %d files in %d groups: %s -> %s" % (len(files), len(groups),
          ", ".join("%s %d" % (k, len(v)) for k, v in parts.items()), o.out))
    return 0


def test_split_of(splits_path, name):
    """The split a .curves file belongs to, per a cassie_split .usda ("" when unlisted)."""
    stage = Usd.Stage.Open(splits_path)
    for k in ("train", "validation", "test"):
        p = stage.GetPrimAtPath("/Splits/" + k)
        if p and name in list(p.GetAttribute("files").Get() or []):
            return k
    return ""


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
