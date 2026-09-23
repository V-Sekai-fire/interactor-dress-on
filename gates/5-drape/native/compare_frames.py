"""Compare two sphere-demo runs frame by frame (OBJ vertex positions).

usage: python compare_frames.py <iter0 dir A> <iter0 dir B> [frames...]

Prints, per frame, the vertex count of each, whether the face lists are
identical, and max |dx| over all vertex coordinates. Frames default to
0 1 10 50 100 350 (the frames kept in iter0/).
"""
import os
import sys


def load(path):
    v, f = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(tuple(float(t) for t in line.split()[1:4]))
            elif line.startswith("f "):
                f.append(tuple(t.split("/")[0] for t in line.split()[1:]))
    return v, f


def main():
    a, b = sys.argv[1], sys.argv[2]
    frames = sys.argv[3:] or ["0", "1", "10", "50", "100", "350"]
    worst = 0.0
    for fr in frames:
        pa, pb = os.path.join(a, fr + ".obj"), os.path.join(b, fr + ".obj")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"frame {fr:>4}: missing ({os.path.exists(pa)}, {os.path.exists(pb)})")
            continue
        va, fa = load(pa)
        vb, fb = load(pb)
        if len(va) != len(vb):
            print(f"frame {fr:>4}: vertex count {len(va)} vs {len(vb)}")
            continue
        d = max((abs(x - y) for p, q in zip(va, vb) for x, y in zip(p, q)), default=0.0)
        worst = max(worst, d)
        print(f"frame {fr:>4}: nv={len(va)} nf={len(fa)} faces_equal={fa == fb} max|dx|={d:.9g}")
    print(f"worst max|dx| over listed frames = {worst:.9g}")


if __name__ == "__main__":
    main()
