# Gate 3, families K3 and K4: NORM, RMS_NORM, MEAN, SOFT_MAX

**Result: PASS.** `results.txt` ends `RESULT: PASS`. ggml's own
`test-backend-ops -o NORM,RMS_NORM,MEAN,SOFT_MAX -b RD0` runs in the guest
(`ggml_test.elf`) against the in-guest ggml-cpu and reports **79/79 passed,
0 FAIL, `Backend RD0: OK`**. By op: NORM 30, RMS_NORM 21, MEAN 7, SOFT_MAX 21.
The same 79 pass with a barrier after every dispatch. 191 SOFT_MAX cases
report "not supported", and every one of them has a mask or sinks (84 mask,
19 sinks, 88 both). No census-required case is unsupported: the census has
f32 rows only, and no mask, sinks or max_bias.

Setup: Godot 4.7.2, RTX 4090 (Godot's device #0), Ryzen 7 3800X. The
branch is `cut-3-k3k4`, on `cut-3` (40079d0).

```
godot --path project --script gate_ggml_rd.gd --rendering-driver vulkan --xr-mode off -- \
    --ops=NORM,RMS_NORM,MEAN,SOFT_MAX --fault=NORM,RMS_NORM,SOFT_MAX \
    --out=gates/3-ggml-rd/ops-k3k4 --probe=rows_perf:sweep --probe=rows_perf
```

## Verdicts

| level | verdict | numbers |
|---|---|---|
| L0 | **PASS** | `lake build Ggml`. Every pin holds: the full `norm_f32` text and its Serial sibling's, the pieces `Rows` builds from, and the stated differences of each `_t64` kernel from its 256-thread kernel. `gen.sh` check mode finds `slang/` equal to Lean's emission (`../kernels/lean-build.log`). The new control (1b) changes `norm_f32`'s first tree step from 128 to 127, and `native_decide` rejects it (`../kernels/lean-negative-control.log`). |
| L1 | **PASS** | spirv-val and the fixed-layout check pass for all 16 kernels. Every `_serial` cpp emit compiles for riscv64. The 7 group-shared kernels have no cpp target and are listed as SKIP, with the sibling named (`../kernels/l1.log`). |
| L2 | **PASS** | 104/104 cases (63 of them K3/K4) against ggml-cpu, NMSE at most 4.9e-14 against test-backend-ops' 1e-7 (1e-6 for SOFT_MAX); `../kernels/l2.log`. Forcing the variant, the cases pass with the 64-thread kernels (`l2-rows64.log`) and with the 256-thread kernels (`l2-rows256.log`). The swapped-stride control catches 95 cases, misses 0, and finds 9 no-ops (`l2-control.log`). |
| L3 G3.ops | **PASS** | 79 OK, 0 FAIL, in 80 frames and 19.6 s. Barrier-all gives the same 79. |
| L3 fault | **PASS** | `GGML_RD_FAULT=1` (a source read one element off) fails 72 of 72 NORM, RMS_NORM and SOFT_MAX cases. MEAN is left out of the verdict: in a first run with MEAN included, 6 of its 7 cases failed, and the 7th, `MEAN(ne=[32769,1,1,1])`, passed. One mean over 32769 elements moves by only (x[n] - x[0]) / n, which is below NMSE 1e-7 (`run-ops_fault-with-mean.log`). |
| L3 perf | **PASS** | probe `rows_perf`: all 11 census shapes checked (`check=ok`) and timed. The table is below. |
| rule 4 | **PASS** | `rule4_same_frame_syncs=0`; `close: permanent_slots=0`. |
| regression | **PASS** | ADD,MUL on the same ELF: see `../ops-k3k4-regress-add-mul/results.txt`. dress_on, drape, curvenet and probes.elf rebuild byte-identical. |

## The kernels (lean/Ggml/SlangCodegen/{Rows,Norm,SoftMax}.lean)

- **One work group per row**, grid-strided. Each reduction is tree-reduced in
  `groupshared float sh[tg]`. NORM uses two passes (the mean, then the centred
  variance), RMS_NORM one, SOFT_MAX two (max, then Σexp). The last pass is
  the map. SOFT_MAX recomputes `exp` in the map instead of storing it and
  reading it back. Thread `t` reads and writes only elements j ≡ t (mod tg),
  so in-place is safe with no barrier on the data. Word 55 holds the row
  count (`ops/rows.h`, `W_ROWS`).
- **Two sizes.** Each op has a 256-thread kernel and a 64-thread `_t64` kernel
  (MEAN has only the 256-thread one; the census has no MEAN). The packer
  (`ops/rows.h`, `pick_threads`) sends rows of at most `kShortRow = 1024`
  elements to `_t64`, and `GGML_RD_ROW_THREADS=64|256` forces one size. The
  1024 comes from the measured sweep below. It is not a guess.
- **Serial siblings.** slangc's cpp target rejects group-shared memory and
  barriers (E36107), so each kernel `k` has a sibling `k_serial`. It takes the
  same params and grid (one group per row) but has one thread, a local
  `sh[tg]`, and a loop over `t` for each phase. It adds the same partials in
  the same order. `gen.sh` gives a group-shared kernel no cpp emit and
  requires the sibling. `gen_host_kernels.py` runs the sibling in the
  kernel's place, so L2 checks the GPU kernels' summation order.
- **Supported exactly as used.** NORM, RMS_NORM, MEAN and SOFT_MAX need f32
  in and out, strides in the params words, and no src1 or src2. For
  SOFT_MAX, max_bias must be 0. Any stride pattern works, including views
  with strided rows, permuted rows and in place.

## GPU time on the census's hottest shapes (probe rows_perf)

Each shape is one tensor in an RD buffer, run as an in-place chain of N ops
(N - 1 barriers, like consecutive model ops). The time is measured on the
host clock from after the submit to after the next frame's sync:
per op = (t(N) - t(1)) / (N - 1). The GB/s column counts every pass (NORM
3R+1W, RMS_NORM 2R+1W, SOFT_MAX 3R+1W). Tensors of 72 MB or less can stay in
the 4090's L2 between ops, which is why some rows exceed DRAM bandwidth.
This machine is shared with other agents' GPU runs, so a single number moves
by up to 2x between runs. The table is from the committed `results.txt`.

| op, shape | census | kernel | per op | GB/s |
|---|---|---|---|---|
| RMS_NORM [896,512] | Skin-Tokens x21947 | t64 | 5.8 us | 953 |
| RMS_NORM [128,8,514] | Skin-Tokens q/k x21560 | t64 | 8.2 us | 771 |
| SOFT_MAX [512,512,8] | Skin-Tokens attention x11542 | t64 | 15.8 us | 2118 |
| NORM [512,54000] | Skin-Tokens x1964 | t64 | 483 us | 915 |
| SOFT_MAX [54000,512,8] (884 MB) | Skin-Tokens x4 | 256 | 6.69 ms | 529 |
| RMS_NORM [128,12,4096] | Pixal3D ss_flow x240 | t64 | 70 us | 1075 |
| NORM [1536,4096] | Pixal3D ss_flow x182 | 256 | 49 us | 2050 |
| NORM [1024,1029] | Pixal3D DINO x80 | t64 | 13 us | 1265 |
| SOFT_MAX [1029,1029,16] | Pixal3D DINO x24 | 256 | 197 us | 1380 |
| NORM [32,64,64,64] | Pixal3D ss_dec x12 | t64 | 570 us | 236 |
| NORM [128,32,32,32] | Pixal3D ss_dec x14 | t64 | 72 us | 934 |

**The sweep** (`rows_perf:sweep`) runs 2^23 elements per shape at row
lengths 32 to 65536, with each thread count forced. The 64-thread kernels
win at every length up to 1024: 1.6 to 3.8x at 32 to 512, and 1.2 to 2.6x at
1024. The 256-thread kernels win from 2048 up: 1.3 to 1.5x at 2048, and 3.4
to 3.8x at 65536. All three ops agree. Hence `kShortRow = 1024`.

Where time could still go (not done here): rows of 32 use only half of a
64-thread group (NORM [32,64,64,64] runs at 236 GB/s), and a group that
takes several short rows would fill it. Rows larger than L2 are re-read
from DRAM on each pass. Caching the row in group-shared memory would save
two of NORM's passes, and an online max/Σexp would save one of
SOFT_MAX's.

## Files

- `results.txt` is the gate run: probes, ops_main, ops_barrier_all,
  ops_fault, then rows_perf:sweep and rows_perf. `run-<name>.log` holds each
  job's output, and `run.log` is Godot's.
- `run-ops_fault-with-mean.log` is the fault control with MEAN included,
  6 of 7 failing (see above).
- `../ops-k3k4-regress-add-mul/`: ADD,MUL on this ELF.
- `../kernels/l2-rows64.log` and `l2-rows256.log` are L2 with each thread
  count forced.
