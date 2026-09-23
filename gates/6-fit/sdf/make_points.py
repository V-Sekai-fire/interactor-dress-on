"""Gate 6a sample points: FitForm<4>'s fit samples on a garment.

For every garment face not in not_fit_fids (all three vertices listed in
no-fit.txt, optimize.cpp:972-987), the 15 points P * M of
upsample_standard<4>() (FitForm.cpp): barycentrics (i, j, 4-i-j)/4 over
i = 0..4, j = 0..4-i, in that order. Written one "x y z" per line with 17
significant digits, so the points round-trip exactly into openvdb_dump and
the gate test.

usage: make_points.py <garment.obj> <no-fit.txt> <out.txt> [--all]
  --all  every garment face (no-fit ones too): a second, larger population
"""
import sys


def read_obj(path):
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                v.append([float(t) for t in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    f.append((idx[0], idx[k], idx[k + 1]))
    return v, f


def main():
    garment, nofit, out = sys.argv[1:4]
    v, f = read_obj(garment)
    vids = set(int(t) for t in open(nofit).read().split())
    fit = [fi for fi, (a, b, c) in enumerate(f) if not (a in vids and b in vids and c in vids)]
    nofit_faces = len(f) - len(fit)
    if '--all' in sys.argv[4:]:
        fit = list(range(len(f)))
    N = 4
    P = [(i / N, j / N, (N - i - j) / N) for i in range(N + 1) for j in range(N + 1 - i)]
    n = 0
    with open(out, 'w') as fh:
        for fi in fit:
            M = [v[k] for k in f[fi]]
            for (a, b, c) in P:
                p = [a * M[0][d] + b * M[1][d] + c * M[2][d] for d in range(3)]
                fh.write('%.17g %.17g %.17g\n' % tuple(p))
                n += 1
    print('garment verts %d faces %d, no-fit faces %d, faces sampled %d, samples %d'
          % (len(v), len(f), nofit_faces, len(fit), n))


if __name__ == '__main__':
    main()
