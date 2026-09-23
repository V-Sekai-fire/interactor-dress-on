#!/usr/bin/env python3
"""Copy polysolve-guest and ipc-toolkit to a scratch dir and add counters.

The counters only print (stderr, "[amdahl] ..." lines); they read sizes and
the linear solver's info after the fact and change no numbers. The run built
from these copies must reproduce the uninstrumented fit_native bitwise
(Newton count, 17-digit energy); measure.md records that check.

  python make_counters_src.py <forks-dir> <out-dir>

Hooks:
  ipc-toolkit Candidates::build (both overloads): candidate counts per build,
    tagged "static" (x only: the collision set) or "ccd" (x0 -> x1: CCD).
  ipc-toolkit Collisions::build(candidates, ...): active constraints.
  polysolve Newton::solve_sparse_linear_system: n, nnz(H), the linear
    solver's get_info (iterations and error for Eigen iterative solvers).
"""
import os
import shutil
import sys


def patch(path, anchor, insert, after=True, count=1):
    s = open(path, encoding="utf-8").read()
    n = s.count(anchor)
    if n < count:
        raise SystemExit(f"{path}: anchor found {n} times, want >= {count}: {anchor!r}")
    i = s.index(anchor)
    j = i + len(anchor) if after else i
    s = s[:j] + insert + s[j:]
    open(path, "w", encoding="utf-8", newline="").write(s)


def main():
    forks, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ps = os.path.join(out, "polysolve")
    ipc = os.path.join(out, "ipc-toolkit")
    for src, dst in ((os.path.join(forks, "polysolve-guest"), ps), (os.path.join(forks, "ipc-toolkit"), ipc)):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))

    cand = os.path.join(ipc, "src", "ipc", "candidates", "candidates.cpp")
    patch(cand, '#include "candidates.hpp"\n', "#include <cstdio>\n")
    printer = (
        "\n    struct AmdahlCandPrint {{ const Candidates* c; ~AmdahlCandPrint() {{"
        " std::fprintf(stderr, \"[amdahl] candidates {tag} vv %zu ev %zu ee %zu fv %zu\\n\","
        " c->vv_candidates.size(), c->ev_candidates.size(), c->ee_candidates.size(), c->fv_candidates.size()); }} }}"
        " amdahl_cand_print{{this}};\n"
    )
    patch(cand, "const int dim = vertices.cols();\n", printer.format(tag="static"))
    patch(cand, "const int dim = vertices_t0.cols();\n", printer.format(tag="ccd"))

    col = os.path.join(ipc, "src", "ipc", "collisions", "collisions.cpp")
    patch(col, '#include "collisions.hpp"\n', "#include <cstdio>\n")
    patch(
        col,
        "    clear();\n\n    // Cull the candidates",
        "    struct AmdahlColPrint { const Collisions* c; ~AmdahlColPrint() {"
        " std::fprintf(stderr, \"[amdahl] collisions vv %zu ev %zu ee %zu fv %zu pv %zu\\n\","
        " c->vv_collisions.size(), c->ev_collisions.size(), c->ee_collisions.size(), c->fv_collisions.size(),"
        " c->pv_collisions.size()); } } amdahl_col_print{this};\n",
        after=False,
    )

    newton = os.path.join(ps, "src", "polysolve", "nonlinear", "descent_strategies", "Newton.cpp")
    patch(newton, '#include "Newton.hpp"\n', "#include <cstdio>\n")
    patch(
        newton,
        "        internal_solver_info.push_back(info);\n",
        "        std::fprintf(stderr, \"[amdahl] linsolve %s n %lld nnz %lld residual %.6e info %s\\n\","
        " name().c_str(), (long long)hessian.rows(), (long long)hessian.nonZeros(), residual, info.dump().c_str());\n",
    )
    print("ok", out)


if __name__ == "__main__":
    main()
