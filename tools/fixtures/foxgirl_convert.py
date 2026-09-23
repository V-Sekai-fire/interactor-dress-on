#!/usr/bin/env python3
"""FoxGirl fixture: cloth-fit's garment-data (Blender frame) -> Godot frame.

    python tools/fixtures/foxgirl_convert.py <garment-data dir> [out dir]

<garment-data dir> is vendor/cloth-fit/garment-data (Cut 6's subtree). The
output (default project/fixtures/foxgirl/) holds the files Gate 8 uses as the
infer and rig fixtures and, with --allow-fixture=curvenet, the garment:

    avatar.obj            FoxGirl body          (assets/avatars/FoxGirl/avatar.obj)
    skeleton.obj          FoxGirl 15 joints     (assets/avatars/FoxGirl/skeleton.obj)
    garment.obj           LCL skirt             (assets/garments/LCL_Skirt_DressEvening_003/garment.obj)
    garment_skeleton.obj  the skirt's skeleton  (.../LCL_Skirt_DressEvening_003/skeleton.obj)
    no-fit.txt            the skirt's no-fit ids (copied as is)

One map for every file: Blender (x, y, z), +Z up, -Y forward, becomes Godot
(x, z, -y), +Y up, +Z forward; then scaled by s = 1.7 / (the avatar's height)
and shifted so the avatar's lowest vertex sits on y = 0. The skirt and its
skeleton are in their own source frame (cloth-fit retargets them by the
skeleton pair); the same map keeps them consistent with each other. Only v, f
and l records are written (no vt/vn/o/usemtl); f keeps only the position
index; winding is unchanged (a rotation keeps it).
"""
import os
import sys

TARGET_HEIGHT = 1.7


def read_obj(path):
    v, rest = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                v.append((float(p[1]), float(p[2]), float(p[3])))
            elif p[0] == "f":
                rest.append(("f", [t.split("/")[0] for t in p[1:]]))
            elif p[0] == "l":
                rest.append(("l", p[1:]))
    return v, rest


def to_godot(p, s, dy):
    x, y, z = p
    return (s * x, s * z + dy, -s * y)


def write_obj(path, header, v, rest, s, dy):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for h in header:
            f.write("# " + h + "\n")
        for p in v:
            q = to_godot(p, s, dy)
            f.write("v %.7g %.7g %.7g\n" % q)
        for kind, ids in rest:
            f.write(kind + " " + " ".join(ids) + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "..", "project", "fixtures", "foxgirl")
    os.makedirs(out, exist_ok=True)
    av = os.path.join(src, "assets", "avatars", "FoxGirl")
    ga = os.path.join(src, "assets", "garments", "LCL_Skirt_DressEvening_003")
    body_v, body_rest = read_obj(os.path.join(av, "avatar.obj"))
    zmin = min(p[2] for p in body_v)
    zmax = max(p[2] for p in body_v)
    s = TARGET_HEIGHT / (zmax - zmin)
    dy = -s * zmin
    header = [
        "FoxGirl fixture, cloth-fit garment-data, Blender (x,y,z) -> Godot (x,z,-y)",
        "scale %.9g (avatar height %.6f -> %.3f m), then y += %.9g (feet on y=0)" % (s, zmax - zmin, TARGET_HEIGHT, dy),
        "written by tools/fixtures/foxgirl_convert.py; see CITATION.cff",
    ]
    jobs = [
        (os.path.join(av, "avatar.obj"), "avatar.obj"),
        (os.path.join(av, "skeleton.obj"), "skeleton.obj"),
        (os.path.join(ga, "garment.obj"), "garment.obj"),
        (os.path.join(ga, "skeleton.obj"), "garment_skeleton.obj"),
    ]
    for src_path, name in jobs:
        v, rest = read_obj(src_path)
        write_obj(os.path.join(out, name), header, v, rest, s, dy)
        print("%-22s %6d v %6d f/l" % (name, len(v), len(rest)))
    with open(os.path.join(ga, "no-fit.txt"), encoding="utf-8") as f:
        ids = f.read()
    with open(os.path.join(out, "no-fit.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(ids)
    print("scale %.9g dy %.9g" % (s, dy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
