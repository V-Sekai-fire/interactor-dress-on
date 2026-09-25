"""Gate S host control: the host's OpenUSD on both sides of the guest.

  pixi run python gates/S-strokes/host_oracle.py make [--curves=<datasource-cassie>/data/curves]
      -> inputs/dress.usda (tools/curves_usd.py) and host-make.log: one line a stroke,
         "dress.usda <i> <name> <blake3_12 of the f32 xyz bytes>", as usd-core reads it
  pixi run python gates/S-strokes/host_oracle.py check
      -> host-check.log: every guest-written out/*.usda parses, validates and
         round-trips (check_usd_valid.py's checks), and usd-core's per-stroke sums
         and boundary marks equal the guest's out/guest.log
"""
import os
import subprocess
import sys

import numpy as np
from blake3 import blake3

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "tools")
sys.path.insert(0, TOOLS)
import curves_usd as cu  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CURVES = os.path.join(WORKSPACE, "6-datasource", "cassie", "data", "curves")
VALIDATOR = os.path.join(WORKSPACE, "2-contract", "manuals-weftspun", "scripts", "check_usd_valid.py")


def sums(text):
    strokes, names, boundary, _ = cu.read_usda(text)
    marks = set(boundary)
    return [(i, names[i], blake3(np.ascontiguousarray(s, dtype=np.float32).tobytes()).hexdigest()[:12], int(i in marks))
            for i, s in enumerate(strokes)]


def make(curves_dir):
    src = os.path.join(curves_dir, "dress.curves")
    if not os.path.exists(src):
        print("FAIL: no %s (repo sync 6-datasource/cassie, or pass --curves=)" % src)
        return 1
    os.makedirs(os.path.join(HERE, "inputs"), exist_ok=True)
    dst = os.path.join(HERE, "inputs", "dress.usda")
    rev = subprocess.run(["git", "-C", curves_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rc = cu.main(["to-usd", src, dst, "--source-rev=" + rev])
    if rc:
        return rc
    with open(os.path.join(HERE, "host-make.log"), "w") as log:
        log.write("# usd-core reading inputs/dress.usda, written by tools/curves_usd.py from datasource-cassie data/curves/dress.curves @ %s\n" % rev[:7])
        for i, name, b3, _ in sums(open(dst).read()):
            log.write("dress.usda %d %s %s\n" % (i, name, b3))
    print("ok host-make.log")
    return 0


def check():
    out = os.path.join(HERE, "out")
    guest = {}
    for line in open(os.path.join(out, "guest.log")):
        f, i, name, b3, mark = line.split()
        guest[(f, int(i))] = (name, b3, int(mark))
    files = sorted({f for f, _ in guest})
    bad = 0
    lines = []
    for f in files:
        path = os.path.join(out, f)
        v = subprocess.run([sys.executable, VALIDATOR, path], capture_output=True, text=True)
        ok_valid = v.returncode == 0
        host = sums(open(path).read())
        diff = [i for i, name, b3, mark in host if guest.get((f, i)) != (name, b3, mark)]
        n_guest = sum(1 for g in guest if g[0] == f)
        ok = ok_valid and not diff and n_guest == len(host)
        bad += not ok
        lines.append("%s %s valid=%s strokes host=%d guest=%d differing=%s" % (
            "PASS" if ok else "FAIL", f, ok_valid, len(host), n_guest, diff or "none"))
    lines.append("RESULT: %s (%d guest-written layers)" % ("PASS" if bad == 0 and files else "FAIL", len(files)))
    open(os.path.join(HERE, "host-check.log"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad or not files else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--curves=")), CURVES)
    sys.exit(make(arg) if cmd == "make" else check() if cmd == "check" else 2)
