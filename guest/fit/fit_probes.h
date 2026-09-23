// fit_probes: Gate 6.0's probes, compiled into fit.elf (fit_core) and into
// fit_native (tests/native/fit) from the same source, so each answer can be
// held to its native twin bit for bit. None of them touches a file or a clock:
// the guest's clock is not a clock (AGENTS.md), so the host times the vmcall
// and fit_native times the call.
#pragma once

#include <string>

namespace fit {

// Each returns "PASS ..." or "FAIL ...".
std::string probe_ldlt();       // polysolve Eigen::SimplicialLDLT on a 1D Laplacian, n = 200
std::string probe_exceptions(); // a throw through std::function, caught by type

// The 8k-DOF solve: polysolve's Eigen::SimplicialLDLT (through the embedded
// spec) on a 7-point 3D Laplacian, 20^3 = 8000 unknowns (the foxgirl garment
// has 2682 x 3 = 8046). analyze + factorize + solve, once per call. The line
// carries the residual and an FNV-1a hash of the solution's bits, so guest
// and native can be compared exactly.
std::string probe_ldlt8k();

// libm, bitwise: 20000 deterministic inputs through each of the functions the
// solve can reach (exp, log, pow, sqrt, sin, cos, tan, asin, acos, atan,
// atan2, cbrt, hypot, log1p, expm1, log2, log10, exp2, tanh, fmod). One
// "name:hash" per function (FNV-1a over the result bits); two builds agree on
// a function iff their hashes match (up to hash collisions).
std::string probe_libm();

// The C++ library, bitwise: the order std::sort, std::nth_element and
// std::partial_sort leave tied keys in (20000 elements, keys 0..99), and a sum
// taken in the sorted order. libstdc++ (guest) and libc++ (fit_native) are
// free to differ here; a solver that sums over a sorted candidate list follows
// whichever order it gets.
std::string probe_stl();

} // namespace fit
