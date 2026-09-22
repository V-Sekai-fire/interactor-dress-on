# Stage 1 — `rd_compute`, the one GPU layer in the guest

**Result: PASS.** `guest/rd_compute.{h,cpp}` (+ `rd_enums.h`) is Gate 0A's call
sequence factored into a class and built as a static library every ELF links.
`guest/main.cpp` is `dress_on.elf`, the program that exercises it.

Flat run (`project/gate_rd_compute.gd`, `run3.log`):

```
open:  ok
probe: PASS: GPU compute reached from the guest (device held from an earlier vmcall)
bench nd=  64 ns=  1 barrier=true   OK value=64  expected=64
bench nd= 256 ns=  1 barrier=true   OK value=256 expected=256
bench nd=   1 ns= 64 barrier=true   OK value=64  expected=64
bench nd=  16 ns= 16 barrier=true   OK value=256 expected=256
no-barrier arm: 4 of 7 counts wrong -> the render graph does NOT order same-buffer dispatches; barrier() is mandatory
RESULT: PASS
```

MCP drive (`mcp_run.sh`: play the project, poll 8789, `call_method` on
`/root/Main` for `rd_open`, `rd_probe`, `rd_bench`, `rd_close`):
`MCP RESULT: PASS`, responses in `mcp_resp_*.json`.

## What it proved

- **A guest static can hold the RenderingDevice across vmcalls.** In the
  sandbox's default unrestricted mode a plain-`Object` handle is the engine
  instance id, resolved live; the local device is a plain Object. RefCounted
  helpers (`RDUniform`, `RDShaderSPIRV`) are held only for the call, so the
  layer never retains them — it keeps RIDs, which are integers.
- **Barriers are mandatory and there is no cheaper implicit form.** Godot's
  render graph orders only *between* compute-list commands.
  `compute_list_add_barrier` is `compute_list_end(); compute_list_begin();`
  plus a rebind (engine `rendering_device.cpp:7197`); that split *is* the
  implicit barrier. Without it the accumulate kernel counted 7 of 16, 23 of
  64, 76 of 256 — deterministically, in every run, in GDScript too
  (`control.log`). Dispatches over disjoint buffers can share a segment.

## What it costs (host-clocked)

`project/probe_rd_calls.gd` times one call kind per vmcall from the host,
1000× for the cheap kinds, 32× for the create/free kinds (`calls.log`,
`calls2.log`, three rounds each):

| call from the guest | µs each |
|---|---|
| `Time.get_ticks_usec()` (trivial singleton call) | 0.1 |
| `rd.limit_get()` (trivial RD call) | 0.3–1.7 |
| `compute_list_bind_compute_pipeline` | 7–11 |
| `compute_list_dispatch` | 6–12 |
| `compute_list_add_barrier` | 7–13 |
| storage buffer create + free | 40 |
| shader from SPIR-V create + free | 60–115 |
| compute pipeline create + free | 45 |
| uniform set create + free | 20–30 (`calls3.log`, with `references_max` raised; the 190 in `calls2.log` was the failing run) |
| 4-byte `buffer_get_data` | 130 |
| `RDUniform` instantiate | 20 |
| **`submit()` + `sync()`, empty list** | **60–85 in `calls.log`, 2340–2390 in `calls2.log`** |

GDScript on the same device (`control_rd_compute.gd`, `control.log`) does a
whole barrier'd dispatch — bind, bind, dispatch, barrier — in 4–5 µs and a
submit+sync in ~70 µs. So the boundary is a 2–3× tax per call, not an order
of magnitude. Batching pays per *dispatch* (each is four calls), not per call.

## Two things that are not settled, kept honest

- **`submit()+sync()` from the guest is bimodal per process.** Some runs
  ~70 µs, some ~2.4 ms, for an *empty* compute list, same binary. In
  `probe_rd_mix.gd` (`mix.log`) the guest sits at ~9–10 ms per vmcall for
  every shape while GDScript in the *same process, same device* stays at
  0.4–1.3 ms, so it is not GPU state. Whatever it is lives on the sandbox →
  RenderingDevice path and behaves like a ~2 ms sleep. Unexplained. The
  design rule "state machines and queues, not waits" means no stage syncs in
  the frame it submits, which makes this a latency, not a stall — but it
  stays on the risk list until it is understood.
- **The guest clock is not a clock.** `std::chrono::steady_clock::now()`
  inside the guest jumps between two time bases ~1000 s apart within one
  call (`setup_us=-1047969145`). Every number above is from the host's
  `Time.get_ticks_usec()` around the vmcall. Do not time anything in the
  guest.

## One cap to plan around

Building 32 uniform sets in one vmcall failed with `Maximum number of scoped
variants reached` (`calls2.log`): every `Array`, `RDUniform` and returned
`Variant` is scoped to the call. `AvbdSolverRD::buildColoring` needs
kernels × colours of them, so the host raises the Sandbox's `max_refs` and the
guest builds sets in chunks across calls.

## Files

- `run.log`, `run2.log`, `run3.log` — the flat gate, three runs
- `control.log` — the GDScript control, no sandbox
- `calls.log`, `calls2.log` — per-call-kind costs, a fast and a slow process
- `mix.log` — GDScript and guest interleaved in one process
- `mcp_run.sh`, `mcp_resp_*.json`, `mcp_game.log` — the MCP drive
- `import.log` — the `--headless --import` that registers the ELF
