// fit_tools: see fit_tools.h.
#include "fit_tools.h"

#include <polyfem/solver/forms/garment_forms/SdfGrid.hpp>
#include <polyfem/solver/forms/garment_forms/SdfSpline.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polysolve/linear/Solver.hpp>

#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <stdexcept>

// ---------------------------------------------------------------------------
// File-open counters. cmake/fit.cmake links fit.elf with
// -Wl,--wrap=open,--wrap=open64,--wrap=openat,--wrap=openat64,--wrap=fopen,--wrap=fopen64,
// so every reference to those in the ELF (the driver, PolyFEM, libstdc++'s
// filebuf, glibc's own callers outside libc's internal aliases) lands here.
// None is passed on: the guest cannot open a file anyway (Gate 0F probe 3,
// EBADF), and refusing with EACCES makes the attempt visible and uniform.
namespace {
int g_io_attempts = 0;
std::vector<std::string> *g_io_paths = nullptr;

void io_note(const char *fn, const char *path) {
	++g_io_attempts;
	if (!g_io_paths)
		g_io_paths = new std::vector<std::string>();
	if (g_io_paths->size() < 32)
		g_io_paths->push_back(std::string(fn) + " " + (path ? path : "(null)"));
}
} // namespace

extern "C" {
int __wrap_open(const char *path, int, ...) {
	io_note("open", path);
	errno = EACCES;
	return -1;
}
int __wrap_open64(const char *path, int, ...) {
	io_note("open64", path);
	errno = EACCES;
	return -1;
}
int __wrap_openat(int, const char *path, int, ...) {
	io_note("openat", path);
	errno = EACCES;
	return -1;
}
int __wrap_openat64(int, const char *path, int, ...) {
	io_note("openat64", path);
	errno = EACCES;
	return -1;
}
FILE *__wrap_fopen(const char *path, const char *) {
	io_note("fopen", path);
	errno = EACCES;
	return nullptr;
}
FILE *__wrap_fopen64(const char *path, const char *) {
	io_note("fopen64", path);
	errno = EACCES;
	return nullptr;
}
}

namespace fit {

int io_attempts() {
	return g_io_attempts;
}

std::vector<std::string> io_paths() {
	return g_io_paths ? *g_io_paths : std::vector<std::string>();
}

// ---------------------------------------------------------------------------
HeapInfo heap_info() {
	HeapInfo h;
#if defined(__riscv) && __riscv_xlen == 64
	// libriscv's native heap: syscall base 480 (sandbox-api native.cpp), +4 is
	// meminfo, filling {bytes_free, bytes_used, chunks_used}; returns 0.
	struct {
		uint64_t bf, bu, cu;
	} r{};
	register long a0 asm("a0") = reinterpret_cast<long>(&r);
	register long a7 asm("a7") = 484;
	asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
	if (a0 == 0) {
		h.ok = true;
		h.bytes_free = r.bf;
		h.bytes_used = r.bu;
		h.chunks_used = r.cu;
	}
#endif
	return h;
}

// ---------------------------------------------------------------------------
std::string strip_path_keys(const std::string &setup_json) {
	nlohmann::json j = nlohmann::json::parse(setup_json);
	if (!j.is_object())
		throw std::runtime_error("setup JSON is not an object");
	for (const char *k : {"avatar_mesh_path", "garment_mesh_path", "source_skeleton_path", "target_skeleton_path",
			 "avatar_skin_weights_path", "no_fit_spec_path", "root_path"})
		j.erase(k);
	return j.dump();
}

// ---------------------------------------------------------------------------
std::vector<double> sdf_sample(const std::vector<double> &points, double *voxel_size) {
	using namespace polyfem::solver;
	const auto grid = SdfGrid::current();
	if (!grid)
		throw std::runtime_error("no SDF grid yet (it is built by the first phase with the fit form)");
	if (points.size() % 3)
		throw std::runtime_error("points must be xyz triples");
	const size_t n = points.size() / 3;
	std::vector<double> st(n * sdf_spline::kStencil), uvw(n * 3), out(n * sdf_spline::kOut);
	for (size_t i = 0; i < n; i++)
		grid->sample_point(&points[3 * i], &st[i * sdf_spline::kStencil], &uvw[3 * i]);
	sdf_spline::hessian_batch(st.data(), uvw.data(), out.data(), n);
	if (voxel_size)
		*voxel_size = grid->voxel_size();
	return out;
}

// ---------------------------------------------------------------------------
std::string probe_io() {
	const int before = g_io_attempts;
	errno = 0;
	FILE *f = std::fopen("/etc/hostname", "r");
	const int e_fopen = errno;
	std::ifstream in("res://project.godot");
	const bool ifs_open = in.is_open();
	errno = 0;
	const int fd = ::open("guest_probe.txt", O_RDONLY);
	const int e_open = errno;
	const int counted = g_io_attempts - before;
	char b[256];
	std::snprintf(b, sizeof b, "%s io: fopen %s (errno %d), ifstream %s, open %d (errno %d); counted %d of 3 (total %d)",
			(!f && !ifs_open && fd < 0 && counted >= 3 && e_fopen == EACCES && e_open == EACCES) ? "PASS" : "FAIL",
			f ? "OPENED" : "refused", e_fopen, ifs_open ? "OPENED" : "refused", fd, e_open, counted, g_io_attempts);
	if (f)
		std::fclose(f);
	return b;
}

std::string probe_ldlt() {
	// 1D Laplacian (2 on the diagonal, -1 off it), b = A * x_true with
	// x_true_i = sin(i): the solver must recover x_true.
	const int n = 200;
	polysolve::StiffnessMatrix A(n, n);
	std::vector<Eigen::Triplet<double>> t;
	for (int i = 0; i < n; i++) {
		t.emplace_back(i, i, 2.0);
		if (i + 1 < n) {
			t.emplace_back(i, i + 1, -1.0);
			t.emplace_back(i + 1, i, -1.0);
		}
	}
	A.setFromTriplets(t.begin(), t.end());
	Eigen::VectorXd x_true(n);
	for (int i = 0; i < n; i++)
		x_true[i] = std::sin(double(i));
	const Eigen::VectorXd b = A * x_true;
	// Through the embedded linear-solver spec (polysolve's rules are not on disk).
	auto solver = polysolve::linear::Solver::create(nlohmann::json{{"solver", "Eigen::SimplicialLDLT"}}, polyfem::logger());
	solver->analyze_pattern(A, n);
	solver->factorize(A);
	Eigen::VectorXd x(n);
	x.setZero();
	solver->solve(b, x);
	const double res = (A * x - b).norm() / b.norm();
	const double err = (x - x_true).cwiseAbs().maxCoeff();
	char buf[256];
	std::snprintf(buf, sizeof buf, "%s ldlt: %s n=%d, relative residual %.3e, max |x - x_true| %.3e",
			(res < 1e-12 && err < 1e-9) ? "PASS" : "FAIL", solver->name().c_str(), n, res, err);
	return buf;
}

namespace {
struct ProbeError : std::runtime_error {
	int code;
	ProbeError(int c) : std::runtime_error("probe"), code(c) {}
};
} // namespace

std::string probe_exceptions() {
	std::function<int(int)> inner = [](int v) -> int {
		if (v > 10)
			throw ProbeError(v);
		return v + 1;
	};
	std::function<int(int)> outer = [&](int v) { return inner(v) * 2; };
	int caught = -1, normal = -1;
	try {
		outer(41);
	} catch (const ProbeError &e) {
		caught = e.code;
	}
	try {
		normal = outer(4);
	} catch (...) {
		normal = -2;
	}
	char b[160];
	std::snprintf(b, sizeof b, "%s exceptions: typed catch through two std::function got %d (want 41); control %d (want 10)",
			(caught == 41 && normal == 10) ? "PASS" : "FAIL", caught, normal);
	return b;
}

} // namespace fit
