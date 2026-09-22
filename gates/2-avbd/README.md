# Stage 2 — AVBD in the guest, one source, two targets

**Result: PASS** (`results.txt`, `run.log`). The 24 AVBD kernels are emitted
from `cloth-dynamics/lean` (`kernels/avbd/gen.sh`) and compiled twice by
slangc: `-target cpp` for `AvbdCpu`, the in-guest CPU path vendored from the
org's `guest-avbd`, and `-target spirv` for `AvbdRd`, the GPU path over
`rd_compute`. Same Slang, same driver (`avbd_sim.h`, templated over the
backend), same oracle.

## The oracle, both backends

The org's `guest-avbd/tests/oracle.cpp` inside the guest (`guest/avbd_fixture.cpp`):
the closed-form 4-vertex step from cloth-dynamics' `test_avbd_solver`
(spring + attachment + triangle + bending; expected positions are exact
rationals, tolerance 1e-5), the same with a perturbed rest length (must
differ), and a pinned two-vertex spring.

```
fixture cpu host_us=     482  PASS cpu: four_vertex_all_constraints max_abs_diff=0 ...
fixture rd  host_us=  403559  PASS rd:  four_vertex_all_constraints max_abs_diff=2.38419e-07 ...
```

The rd fixture's 400 ms is 13 shader compiles and pipeline creations on a
fresh device; every later construction hits the driver's pipeline cache.

## The bench: a pinned panel dropped under gravity

5 substeps of 10 outer iterations, host-timed around the whole vmcall
(construction, coloring, uploads, the solve, readbacks):

| panel | verts | colours | cpu (interpreter) | rd, one submit per iteration | rd-batched, one submit per substep |
|---|---|---|---|---|---|
| 8×8 | 64 | 4 | 6.0 ms/substep | 7.1 | 5.3 |
| 16×16 | 256 | 4 | 17.0 | 6.4 | 5.5 |
| 32×32 | 1024 | 4 | 72.2 | 8.3 | 7.1 |
| 64×64 | 4096 | 4 | — | 16.0 | 14.7 |

All runs finite; `ymin` matches ½·g·t² for 25 ms of fall.

What it settles for the batch/CPU/GPU rule:
- **The crossover is about 100 vertices.** Below it the interpreter wins
  because the GPU path pays ~40 host calls and a submit per iteration
  regardless of size; above it the GPU wins and keeps winning (10× at 1024,
  and 4096 vertices cost only 2× what 1024 did).
- **Batching iterations into one compute list gains 10–25 %**, less than the
  per-submit cost would suggest, because the per-iteration cost is dominated
  by the dispatch calls (4 colours × ~10 dispatches × ~7 µs) rather than the
  submit. The next lever is fewer, fatter dispatches, not fewer submits.
- **Barrier placement holds**: the four force kernels of a colour share a
  segment (disjoint outputs); gathers into the shared scratch are serialised;
  the exact fixture confirms the ordering is right.

## What this cut covers, and what it does not

Forward step, the three dual updates, greedy colouring, gamma scale, penalty
ramp params, `run(iters, duals)` batching. Not yet wired on `AvbdRd`: the
self-collision scan and the ten backward kernels (they are emitted, embedded
and in the table; their buffers and the grad readbacks are the next cut).
`AvbdCpu` never had them; they come to both backends together.

## Files

- `results.txt`, `run.log` — the gate; `import.log` — the ELF import
- `project/gate_avbd.gd` — the runner; `project/main.gd` — `avbd_fixture`,
  `avbd_bench` for MCP
- `kernels/avbd/` — `gen.sh`, `kernels.txt`, the committed `slang/` and
  `cpp/`, the regenerated `AvbdKernelTable.inc`
- `guest/avbd/` — `avbd_cpu`, `avbd_rd`, `avbd_topology`, `avbd_sim.h`,
  `cloth_grid`, `slang-rt`, `CITATION.cff`
