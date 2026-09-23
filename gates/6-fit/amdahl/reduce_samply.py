#!/usr/bin/env python3
"""Reduce a samply profile of fit_native to the Amdahl buckets of measure.md.

  python reduce_samply.py <profile.json.gz> [--newton N] [--label NAME] [--md]

Needs the sidecar <profile>.json.syms.json (samply record
--unstable-presymbolicate). Frames resolve through fit_native.exe's COFF
symbol table (no PDB): inlined code is charged to its nearest non-inlined
caller.

Each sample of the busiest thread goes to exactly one bucket: the bucket of
the DEEPEST frame on its stack that matches a rule below (so the partition
sums to 100%). The window is the samples under fit::FitDriver::step (the
phases); the rest of the process (inputs, FitDriver::begin, exit) is
reported on its own line. "alloc" is an overlay, not a bucket: the share of
each bucket's samples whose stack holds an allocator frame
(malloc/free/new/delete/Rtl*Heap/Nt*VirtualMemory).
"""
import argparse
import collections
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from samply_top import load, symbolicator, main_thread, stacks  # noqa: E402

# (regex, bucket). Order does not matter for the deepest-match rule except
# among rules matching the SAME frame: the first listed wins there.
RULES = [
    (r"fit::FitDriver::step\(", "other (phase driver, AL outer loop)"),
    (r"ALSolver<.*>::solve_al", "other (phase driver, AL outer loop)"),
    (r"GarmentNLProblem::init\(", "other (phase driver, AL outer loop)"),
    (r"polysolve::nonlinear::Solver::minimize", "other (Newton loop bookkeeping)"),
    (r"polysolve::nonlinear::Solver::verify_gradient", "other (Newton loop bookkeeping)"),
    (r"::post_step\(|::line_search_end\(", "other (Newton loop bookkeeping)"),
    # gradient
    (r"::gradient\(|::first_derivative", "gradient"),
    # Hessian
    (r"Newton::compute_hessian|GarmentNLProblem::hessian\(|FullNLProblem::hessian\(",
     "hessian: sum of forms, P^T H P, copies"),
    (r"polyfem::solver::(\w+(?:<\d+>)?)::second_derivative", "hessian: {1}"),
    # linear solve
    (r"Newton::solve_sparse_linear_system", "linear: other (H*dx residual)"),
    (r"polysolve::linear::\w+<.*>::analyze_pattern|analyzePattern", "linear: analyze"),
    (r"polysolve::linear::\w+<.*>::factorize|::factorize<", "linear: factorize"),
    (r"polysolve::linear::\w+<.*>::solve\(", "linear: solve"),
    # line search
    (r"LineSearch::line_search\(|compute_nan_free_step_size|compute_descent_step_size", "line search: other"),
    (r"::value\(|::value_unweighted", "energy"),
    (r"::is_step_valid\(", "line search: other"),
    (r"line_search_begin\(", "line search: other"),
    (r"ipc::Candidates::build|ipc::BroadPhase::|ipc::BVH::|SimpleBVH::", "broad phase (candidates)"),
    (r"compute_max_step_size|::max_step_size\(|compute_collision_free_stepsize|is_step_collision_free|::ccd\(|_ccd\(",
     "CCD / max step size"),
    (r"::solution_changed\(|update_collision_set|ipc::Collisions::build", "constraint-set update"),
]
RULES = [(re.compile(p), b) for p, b in RULES]
ALLOC = re.compile(
    r"^(malloc|_malloc_base|free|free_base|_free_base|realloc|_realloc_base|calloc|_calloc_base|"
    r"operator new|operator delete|RtlAllocateHeap|RtlFreeHeap|RtlpAllocateHeap|RtlpFreeHeap|"
    r"RtlReAllocateHeap|NtFreeVirtualMemory|NtAllocateVirtualMemory)\b")
LS = re.compile(r"LineSearch::line_search\(")
CS = re.compile(r"update_collision_set|::solution_changed\(")


def bucket_of(frame_names):
    """Deepest matching rule; context splits for energy and broad phase."""
    best = None
    in_ls = False
    in_cs = False
    for nm in frame_names:
        if LS.search(nm):
            in_ls = True
        if CS.search(nm):
            in_cs = True
        for rx, b in RULES:
            m = rx.search(nm)
            if m:
                best = (b, m, in_ls, in_cs)
                break
    if best is None:
        return None
    b, m, in_ls, in_cs = best
    if "{1}" in b:
        b = b.replace("{1}", m.group(1))
    if b == "energy":
        b = "line search: energy evals" if in_ls else "energy (Newton loop)"
    if b == "broad phase (candidates)":
        b = "broad phase: constraint set (dhat)" if in_cs else "broad phase: CCD candidates (x0->x1)"
    return b


ORDER = [
    "gradient",
    "hessian",
    "linear",
    "broad phase",
    "CCD / max step size",
    "line search: energy evals",
    "line search",
    "constraint-set update",
    "energy (Newton loop)",
    "other",
]


def order_key(b):
    for i, p in enumerate(ORDER):
        if b.startswith(p):
            return (i, b)
    return (len(ORDER), b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--newton", type=int, default=0, help="Newton iterations in the window (per-iteration column)")
    ap.add_argument("--label", default="")
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args()
    p, syms = load(a.profile)
    t = main_thread(p)
    names = [n for _, n in symbolicator(p, syms)(t)]
    interval_s = p["meta"]["interval"] / 1000.0
    win = collections.Counter()
    win_alloc = collections.Counter()
    outside = 0
    total = 0
    for w, ch in stacks(t):
        total += w
        fn = [names[f] for f in ch]
        if not any("fit::FitDriver::step(" in n for n in fn):
            outside += w
            continue
        b = bucket_of(fn) or "other (unmatched)"
        win[b] += w
        if any(ALLOC.search(n) for n in fn):
            win_alloc[b] += w
    wt = sum(win.values())
    at = sum(win_alloc.values())
    rows = sorted(win.items(), key=lambda kv: order_key(kv[0]))
    nn = a.newton
    if a.md:
        print(f"| bucket | s | fraction | ms / Newton ({nn}) | of which alloc/free |")
        print("|---|---|---|---|---|")
        for b, c in rows:
            s = c * interval_s
            per = f"{1000.0 * s / nn:.1f}" if nn else "-"
            print(f"| {b} | {s:.2f} | {100.0 * c / wt:.1f}% | {per} | {100.0 * win_alloc[b] / c:.0f}% |")
        s = wt * interval_s
        print(f"| **phase window (FitDriver::step)** | **{s:.2f}** | 100% | {1000.0 * s / nn:.0f} | {100.0 * at / wt:.0f}% |" if nn else "")
        print(f"| outside the window (inputs, begin, exit) | {outside * interval_s:.2f} | - | - | - |")
        # grouped view
        groups = collections.Counter()
        for b, c in rows:
            g = b.split(":")[0] if ":" in b else b
            groups[g] += c
        print()
        print("| group | s | fraction |")
        print("|---|---|---|")
        for g, c in sorted(groups.items(), key=lambda kv: order_key(kv[0])):
            print(f"| {g} | {c * interval_s:.2f} | {100.0 * c / wt:.1f}% |")
        print(f"| allocation overlay (stacks with an allocator frame, across buckets) | {at * interval_s:.2f} | {100.0 * at / wt:.1f}% |")
    else:
        print(f"{a.label} samples {total} (interval {interval_s * 1000:.3f} ms); window {wt}; outside {outside}")
        for b, c in rows:
            s = c * interval_s
            per = f"{1000.0 * s / nn:8.1f} ms/it" if nn else ""
            print(f"{b:45s} {s:8.2f} s {100.0 * c / wt:6.1f}%  {per}  alloc {100.0 * win_alloc[b] / c:3.0f}%")
        print(f"{'alloc overlay':45s} {at * interval_s:8.2f} s {100.0 * at / wt:6.1f}%")


if __name__ == "__main__":
    main()
