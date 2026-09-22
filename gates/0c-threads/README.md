# Gate 0C — do guest threads run?

**Result: PASS-SEQUENTIAL.**

```
PASS-SEQUENTIAL spawn=ok hw=2 started=4 finished=4 max_live=1
```

`std::thread` links (`libpthread.a` is in the sysroot), spawns, and completes
in the guest — and **at no point do two threads overlap**. libriscv serializes
them onto one hart. `hardware_concurrency()` reporting 2 is cosmetic.

This settles two questions at once:

- **Parallel Delaunay (PDEL).** SCsub's comment says it "needs OpenMP"; it
  does not — `parallel_delaunay_3d.cpp` uses `GEO::Thread` over pthreads. It
  would build and run in the guest and gain nothing, because its whole design
  is concurrent zone insertion. Serialized, that is the sequential algorithm
  plus partitioning and sync overhead. Sequential `Delaunay3d` stays.
- **Sub-island parallel drape inside the guest** has nowhere to run. The only
  real parallelism in this architecture is the GPU through `RenderingDevice`.

Note that `mujoco-sandbox-demo/guest-avbd/tests/parallel_islands.cpp` is
host-native evidence (`build.sh` builds it under `== native tests ==` with
`-pthread` and runs the `.exe`), not evidence about the guest.

The probe busy-waits 20 ms per thread so overlap would be observable;
`max_live=1` with four threads is therefore a robust result, not a timing
accident.
