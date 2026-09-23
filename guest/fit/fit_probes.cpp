// fit_probes: see fit_probes.h.
#include "fit_probes.h"

#include <polyfem/utils/Logger.hpp>
#include <polysolve/linear/Solver.hpp>

#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fit {

namespace {

uint64_t fnv1a(const double *x, size_t n, uint64_t h = 1469598103934665603ull) {
	for (size_t i = 0; i < n; i++) {
		uint64_t b;
		std::memcpy(&b, &x[i], sizeof b);
		for (int k = 0; k < 8; k++) {
			h ^= (b >> (8 * k)) & 0xffu;
			h *= 1099511628211ull;
		}
	}
	return h;
}

// xorshift64*: the same sequence on every target, no <random> distribution
// (whose algorithms differ between libstdc++ and libc++).
struct Rng {
	uint64_t s;
	explicit Rng(uint64_t seed) : s(seed) {}
	uint64_t next() {
		s ^= s >> 12;
		s ^= s << 25;
		s ^= s >> 27;
		return s * 2685821657736338717ull;
	}
	// uniform in [lo, hi), from 53 bits
	double uni(double lo, double hi) { return lo + (hi - lo) * (double(next() >> 11) * 0x1.0p-53); }
};

std::unique_ptr<polysolve::linear::Solver> ldlt_solver() {
	// Through the embedded linear-solver spec (polysolve's rules are not on disk).
	return polysolve::linear::Solver::create(nlohmann::json{{"solver", "Eigen::SimplicialLDLT"}}, polyfem::logger());
}

} // namespace

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
	auto solver = ldlt_solver();
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

std::string probe_ldlt8k() {
	const int m = 20, n = m * m * m;
	auto id = [m](int i, int j, int k) { return (k * m + j) * m + i; };
	std::vector<Eigen::Triplet<double>> t;
	t.reserve(size_t(n) * 7);
	for (int k = 0; k < m; k++)
		for (int j = 0; j < m; j++)
			for (int i = 0; i < m; i++) {
				const int r = id(i, j, k);
				// Dirichlet 7-point Laplacian plus a small mass term (SPD, as
				// the garment Hessian after PSD projection).
				t.emplace_back(r, r, 6.01);
				if (i + 1 < m) t.emplace_back(r, id(i + 1, j, k), -1.0), t.emplace_back(id(i + 1, j, k), r, -1.0);
				if (j + 1 < m) t.emplace_back(r, id(i, j + 1, k), -1.0), t.emplace_back(id(i, j + 1, k), r, -1.0);
				if (k + 1 < m) t.emplace_back(r, id(i, j, k + 1), -1.0), t.emplace_back(id(i, j, k + 1), r, -1.0);
			}
	polysolve::StiffnessMatrix A(n, n);
	A.setFromTriplets(t.begin(), t.end());
	// No libm in the data (sin would mix a libm difference into the hash).
	Eigen::VectorXd x_true(n);
	for (int i = 0; i < n; i++)
		x_true[i] = double((i * 7919) % 1000) / 1000.0 - 0.5;
	const Eigen::VectorXd b = A * x_true;
	auto solver = ldlt_solver();
	solver->analyze_pattern(A, n);
	solver->factorize(A);
	Eigen::VectorXd x(n);
	x.setZero();
	solver->solve(b, x);
	const double res = (A * x - b).norm() / b.norm();
	const double err = (x - x_true).cwiseAbs().maxCoeff();
	char buf[320];
	std::snprintf(buf, sizeof buf, "%s ldlt8k: %s n=%d nnz=%lld, relative residual %.3e, max |x - x_true| %.3e, x fnv1a %016llx",
			(res < 1e-12 && err < 1e-9) ? "PASS" : "FAIL", solver->name().c_str(), n, (long long)A.nonZeros(), res, err,
			(unsigned long long)fnv1a(x.data(), size_t(n)));
	return buf;
}

std::string probe_libm() {
	struct F {
		const char *name;
		double lo, hi;
		std::function<double(double, double)> f; // second argument drawn from the same range
	};
	const std::vector<F> fs = {
			{"exp", -50, 50, [](double a, double) { return std::exp(a); }},
			{"log", 1e-12, 1e3, [](double a, double) { return std::log(a); }},
			{"pow", 1e-6, 1e2, [](double a, double b) { return std::pow(a, b * 0.05 - 2.0); }},
			{"sqrt", 0, 1e6, [](double a, double) { return std::sqrt(a); }},
			{"sin", -20, 20, [](double a, double) { return std::sin(a); }},
			{"cos", -20, 20, [](double a, double) { return std::cos(a); }},
			{"tan", -1.5, 1.5, [](double a, double) { return std::tan(a); }},
			{"asin", -1, 1, [](double a, double) { return std::asin(a); }},
			{"acos", -1, 1, [](double a, double) { return std::acos(a); }},
			{"atan", -1e3, 1e3, [](double a, double) { return std::atan(a); }},
			{"atan2", -10, 10, [](double a, double b) { return std::atan2(a, b); }},
			{"cbrt", -1e6, 1e6, [](double a, double) { return std::cbrt(a); }},
			{"hypot", -1e3, 1e3, [](double a, double b) { return std::hypot(a, b); }},
			{"log1p", -0.999, 1e3, [](double a, double) { return std::log1p(a); }},
			{"expm1", -30, 30, [](double a, double) { return std::expm1(a); }},
			{"log2", 1e-12, 1e6, [](double a, double) { return std::log2(a); }},
			{"log10", 1e-12, 1e6, [](double a, double) { return std::log10(a); }},
			{"exp2", -60, 60, [](double a, double) { return std::exp2(a); }},
			{"tanh", -20, 20, [](double a, double) { return std::tanh(a); }},
			{"fmod", -1e3, 1e3, [](double a, double b) { return std::fmod(a, b * 0.01 + 10.5); }},
	};
	const int N = 20000;
	std::string out = "libm";
	std::vector<double> y(N);
	for (size_t k = 0; k < fs.size(); k++) {
		Rng r(0x9e3779b97f4a7c15ull + k);
		for (int i = 0; i < N; i++) {
			const double a = r.uni(fs[k].lo, fs[k].hi);
			const double b = r.uni(fs[k].lo, fs[k].hi);
			y[i] = fs[k].f(a, b);
		}
		char b[64];
		std::snprintf(b, sizeof b, " %s:%016llx", fs[k].name, (unsigned long long)fnv1a(y.data(), N));
		out += b;
	}
	return out;
}

std::string probe_stl() {
	// Keys with many ties (0..99 over 20000 elements), each element carrying
	// its original index: the order the algorithms leave the ties in is the
	// library's choice, and a summation over the result follows it.
	const int N = 20000;
	Rng r(12345);
	std::vector<std::pair<int, int>> base(N);
	for (int i = 0; i < N; i++)
		base[i] = {int(r.next() % 100), i};
	auto key_less = [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.first < b.first; };
	auto hash_order = [](const std::vector<std::pair<int, int>> &v) {
		std::vector<double> d(v.size());
		for (size_t i = 0; i < v.size(); i++)
			d[i] = double(v[i].second);
		return fnv1a(d.data(), d.size());
	};
	std::vector<std::pair<int, int>> a = base;
	std::sort(a.begin(), a.end(), key_less);
	const uint64_t h_sort = hash_order(a);
	a = base;
	std::nth_element(a.begin(), a.begin() + N / 2, a.end(), key_less);
	const uint64_t h_nth = hash_order(a);
	a = base;
	std::partial_sort(a.begin(), a.begin() + N / 10, a.end(), key_less);
	const uint64_t h_psort = hash_order(a);
	// The same values summed in the sorted order: what a tie order changes.
	std::vector<double> vals(N);
	for (int i = 0; i < N; i++)
		vals[i] = 1.0 / double(base[i].second + 1) * (base[i].first % 7 == 0 ? 1e8 : 1.0);
	a = base;
	std::sort(a.begin(), a.end(), key_less);
	double s = 0;
	for (const auto &p : a)
		s += vals[size_t(p.second)];
	char b[256];
	std::snprintf(b, sizeof b, "stl sort:%016llx nth_element:%016llx partial_sort:%016llx sum_in_sorted_order:%.17g",
			(unsigned long long)h_sort, (unsigned long long)h_nth, (unsigned long long)h_psort, s);
	return b;
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
