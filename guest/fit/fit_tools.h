// fit_tools: the guest-only side of fit.elf that needs the solver's headers
// (Eigen, polysolve, nlohmann::json, the SDF grid), compiled with fit_driver
// under the guest numerics (cmake/fit.cmake) so guest/fit/main.cpp stays a thin
// Godot-facing layer. Also owns the file-open counters behind the
// -Wl,--wrap=open,... link options.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fit {

// --- I/O counter: every open/openat/fopen (and 64-bit variants) in the ELF --
// is counted and refused with EACCES; the guest has no filesystem.
int io_attempts();
// The first few paths tried, "fn path" each.
std::vector<std::string> io_paths();

// --- guest heap, from the native heap's meminfo syscall (libriscv) ----------
struct HeapInfo {
	bool ok = false;
	uint64_t bytes_free = 0;
	uint64_t bytes_used = 0;
	uint64_t chunks_used = 0;
};
HeapInfo heap_info();

// The setup JSON with the keys that name files removed (*_path, root_path):
// the guest takes meshes as arrays. Throws on invalid JSON.
std::string strip_path_keys(const std::string &setup_json);

// The SDF grid of the last FitForm (SdfGrid::current()) sampled at world
// points (solve frame, xyz triples) through the Lean kernel: 10 doubles per
// point (x, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz), kernels/fit's
// contract: x in solve units, g and h per index step (FitForm divides by h
// and h^2). Throws if no grid exists yet (before the first
// phase that builds the fit form).
std::vector<double> sdf_sample(const std::vector<double> &points, double *voxel_size);

// Probes, each returning "PASS ..." or "FAIL ...".
std::string probe_io();         // fopen, ifstream, open: all refused, all counted
std::string probe_ldlt();       // polysolve Eigen::SimplicialLDLT on a 1D Laplacian
std::string probe_exceptions(); // a throw through std::function, caught by type

} // namespace fit
