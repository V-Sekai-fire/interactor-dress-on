"""First difference between two polyfem debug traces (guest vs native).

    python gates/6-fit/trace_diff.py GUEST.log NATIVE.log [--context N]

GUEST.log is a gate_fit.gd arm's Godot log (runs/<tag>.godot.log), NATIVE.log
fit_native's stdout at debug level. Only the [polyfem] lines are compared;
their timestamps, timing lines and the stopping-criteria print (it shows an
uninitialised double) are removed. Prints how many lines agree, where the
first difference is (which phase, which Newton iteration), and the lines
around it. Exit 0 if the traces are equal, 1 if they differ.
"""
import re
import sys

STAMP = re.compile(r"^\[[0-9-]+ [0-9:.]+\] ")
CRIT = re.compile(r" \(stopping criteria:.*\)$")


def lines(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            s = raw.rstrip("\n")
            if "[polyfem]" not in s:
                continue
            s = STAMP.sub("", s)
            s = CRIT.sub("", s)
            if re.search(r"timing|\btook\b|\btime\b|[0-9.]+ ?s\)$|\bseconds\b", s):
                continue
            out.append(s)
    return out


def where(ls, k):
    """Phase (count of 'Start substep' / 'Solving' headers) and Newton iteration at line k."""
    solve = -1
    it = None
    for s in ls[: k + 1]:
        if "Starting SparseNewton" in s:
            solve += 1
            it = None
        m = re.search(r"iters=(\d+)", s)
        if m and "[SparseNewton]" in s:
            it = int(m.group(1))
    return solve, it


ASCII = {"Δ": "D", "‖": "|", "∇": "grad ", "⋅": ".", "₀": "0"}


def show(s):
    """polyfem's symbols as ASCII: Godot's OS.execute does not decode UTF-8."""
    for k, v in ASCII.items():
        s = s.replace(k, v)
    return s


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = lines(sys.argv[1]), lines(sys.argv[2])
    ctx = int(sys.argv[sys.argv.index("--context") + 1]) if "--context" in sys.argv else 2
    n = min(len(a), len(b))
    k = next((i for i in range(n) if a[i] != b[i]), None)
    if k is None and len(a) == len(b):
        print(f"traces equal: {len(a)} polyfem lines")
        return 0
    if k is None:
        k = n
    solve, it = where(b, k)
    after = "before any Newton iteration" if it is None else f"after Newton iteration {it}"
    print(f"{k} of {len(a)} guest / {len(b)} native polyfem lines agree; first difference at line {k + 1}: "
          f"Newton solve {solve} (0 = phase 0), {after}")
    for i in range(max(0, k - ctx), min(n, k + ctx + 1)):
        mark = "  " if i < k else "! "
        print(f"{mark}guest : {show(a[i])}")
        if a[i] != b[i]:
            print(f"{mark}native: {show(b[i])}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
