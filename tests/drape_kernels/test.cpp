// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Level-2 test of the Lean-emitted L-BFGS-B kernels (kernels/drape/cpp)
// against LBFGSpp's component fixtures (gates/5-drape/oracle/components).
//
// Each fixture is replayed through the kernels exactly as the drape driver
// will dispatch them: reset, then per history pair lb_multi_dot_serial x5 ->
// lb_compact -> lb_ring_store; then lb_apply_m, lb_pg_inf_serial, lb_cauchy,
// saxpby + dot_reduce_serial + lb_step_max_serial (drt0, step_max0),
// lb_subspace, dot_reduce_serial (dg) and lb_step_max_serial (step_max).
// Everything is float32 (the fixtures' inputs are float32-exact); the
// fixtures' outputs are LBFGSpp's double results.
//
//   test.exe <components dir> [--corrupt-theta]
//
// Pass: every quantity within rel 1e-4 (drt within 5e-4), the state
// (ncorr, ptr) equal, and the sets (newact, free, brk0, subspace L/U/P)
// identical as sets. --corrupt-theta scales theta by 1.25 after the history
// is loaded and refactors T with it; every fixture must then fail on what
// theta feeds (M, the Cauchy point, the direction), theta's own check off.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "slang-cpp-prelude.h"

// Each emit defines main_0 / GlobalParams_0 at file scope under extern "C";
// emptying the EXTERN_C macros lets every emit live in its own namespace in
// one TU (the guest/avbd/avbd_cpu.cpp pattern).
#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END

namespace k_md {
#include "lb_multi_dot_serial_emit.cpp"
}
namespace k_cmp {
#include "lb_compact_emit.cpp"
}
namespace k_rs {
#include "lb_ring_store_emit.cpp"
}
namespace k_am {
#include "lb_apply_m_emit.cpp"
}
namespace k_pg {
#include "lb_pg_inf_serial_emit.cpp"
}
namespace k_sm {
#include "lb_step_max_serial_emit.cpp"
}
namespace k_cau {
#include "lb_cauchy_emit.cpp"
}
namespace k_sub {
#include "lb_subspace_emit.cpp"
}
namespace k_bp {
#include "lb_box_project_emit.cpp"
}
namespace k_sx {
#include "saxpby_emit.cpp"
}
namespace k_dr {
#include "dot_reduce_serial_emit.cpp"
}

namespace {

using ThreadFn = void (*)(ComputeThreadVaryingInput *, void *, void *);
using FVec = std::vector<float>;
using UVec = std::vector<uint32_t>;

void dispatch(uint32_t count, ThreadFn fn, void *gp) {
	for (uint32_t lane = 0; lane < count; ++lane) {
		ComputeThreadVaryingInput t{};
		t.groupID = uint3(0u, 0u, 0u);
		t.groupThreadID = uint3(lane, 0u, 0u);
		fn(&t, nullptr, gp);
	}
}

template <class B, class T>
void bind(B &b, std::vector<T> &v) {
	b.data = v.data();
	b.count = v.size();
}

// ---------------------------------------------------------------- fixture
struct Fixture {
	std::map<std::string, std::vector<double>> vec; // key -> values (sets too)
	std::vector<std::vector<double>> s, y;          // pairs in insertion order
	std::vector<double> M;                          // K x K row-major
	int K = 0;
	std::vector<std::vector<double>> mvIn, mvOut;
	double scalar(const char *k) const { return vec.at(k).at(0); }
};

bool isScalar(const std::string &k) {
	static const char *const keys[] = { "seed", "n", "m", "npairs", "ncorr", "ptr", "theta", "pg_inf",
		"step_max0", "dg", "step_max" };
	for (const char *s : keys) {
		if (k == s) {
			return true;
		}
	}
	return false;
}

bool load(const std::string &path, Fixture &f) {
	std::ifstream in(path);
	if (!in) {
		return false;
	}
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::istringstream ss(line);
		std::vector<std::string> tok;
		std::string t;
		while (ss >> t) {
			tok.push_back(t);
		}
		auto num = [](const std::string &x) { return std::strtod(x.c_str(), nullptr); };
		const std::string &key = tok[0];
		if (key == "s" || key == "y") {
			// s P slot J len v...
			std::vector<double> v;
			for (size_t i = 5; i < tok.size(); ++i) {
				v.push_back(num(tok[i]));
			}
			(key == "s" ? f.s : f.y).push_back(v);
		} else if (key == "M") {
			f.K = std::atoi(tok[1].c_str());
			for (size_t i = 3; i < tok.size(); ++i) {
				f.M.push_back(num(tok[i]));
			}
		} else if (key == "mv_in" || key == "mv_out") {
			std::vector<double> v;
			for (size_t i = 3; i < tok.size(); ++i) {
				v.push_back(num(tok[i]));
			}
			(key == "mv_in" ? f.mvIn : f.mvOut).push_back(v);
		} else if (isScalar(key)) {
			f.vec[key] = { num(tok[1]) };
		} else {
			// a vector or an index set: key <count> items...
			std::vector<double> v;
			for (size_t i = 2; i < tok.size(); ++i) {
				v.push_back(num(tok[i]));
			}
			f.vec[key] = v;
		}
	}
	return true;
}

// ---------------------------------------------------------------- checks
struct Report {
	int fails = 0;
	std::string text;
	void add(bool ok, const char *what, double err, double tol) {
		char b[160];
		std::snprintf(b, sizeof b, " %s=%.2e%s", what, err, ok ? "" : "(FAIL)");
		text += b;
		(void)tol;
		if (!ok) {
			++fails;
		}
	}
	void flag(bool ok, const char *what) {
		if (!ok) {
			text += std::string(" ") + what + "(FAIL)";
			++fails;
		}
	}
};

constexpr double kBig = 1e30; // at or beyond: infinite

// max|got - ref| / max(max|ref|, tiny); infinite refs must match as infinite.
double relErr(const std::vector<float> &got, const std::vector<double> &ref, size_t n) {
	double num = 0.0, den = 0.0;
	for (size_t i = 0; i < n; ++i) {
		const bool ri = std::fabs(ref[i]) >= kBig, gi = std::fabs(got[i]) >= kBig;
		if (ri || gi) {
			if (ri != gi || (ref[i] > 0) != (got[i] > 0)) {
				return INFINITY;
			}
			continue;
		}
		num = std::fmax(num, std::fabs(got[i] - ref[i]));
		den = std::fmax(den, std::fabs(ref[i]));
	}
	return den > 0.0 ? num / den : num;
}
double relScalar(double got, double ref) {
	if (std::fabs(ref) >= kBig || std::fabs(got) >= kBig) {
		return (std::fabs(ref) >= kBig && std::fabs(got) >= kBig && (ref > 0) == (got > 0)) ? 0.0 : INFINITY;
	}
	const double den = std::fabs(ref);
	return den > 0.0 ? std::fabs(got - ref) / den : std::fabs(got - ref);
}
std::set<int> asSet(const std::vector<double> &v) {
	std::set<int> s;
	for (double i : v) {
		s.insert((int)i);
	}
	return s;
}

// A fixture's inf bound becomes FLT_MAX: the kernels treat |b| >= 1e30 as absent.
float toF(double v) {
	if (std::isinf(v)) {
		return v > 0 ? FLT_MAX : -FLT_MAX;
	}
	return (float)v;
}
FVec toFV(const std::vector<double> &v) {
	FVec o;
	for (double d : v) {
		o.push_back(toF(d));
	}
	return o;
}

// ---------------------------------------------------------------- kernels
struct Lb {
	uint32_t n, mc;
	FVec S, Y, bf, dots;
	UVec st;
	Lb(uint32_t n_, uint32_t mc_) :
			n(n_), mc(mc_), S(n_ * mc_, 0.0f), Y(n_ * mc_, 0.0f), bf(4 + 4 * mc_ * mc_ + mc_, 0.0f),
			dots(4 * mc_ + 6, 0.0f), st(5, 0u) {}

	void multiDot(FVec &a, uint32_t ncols, FVec &v, uint32_t dstOff) {
		k_md::LbMultiDotParams_0 p{ n, ncols, 0u, n, 0u, dstOff };
		k_md::GlobalParams_0 gp{};
		gp.params_0 = &p;
		bind(gp.a_0, a);
		bind(gp.vv_0, v);
		bind(gp.dst_0, dots);
		dispatch(ncols, &k_md::main_0_Thread, &gp);
	}
	void compact(uint32_t mode) {
		k_cmp::LbCompactParams_0 p{ mc, mode };
		k_cmp::GlobalParams_0 gp{};
		gp.params_0 = &p;
		bind(gp.dots_0, dots);
		bind(gp.st_0, st);
		bind(gp.bf_0, bf);
		dispatch(1, &k_cmp::main_0_Thread, &gp);
	}
	void ringStore(FVec &s, FVec &y) {
		k_rs::LbRingStoreParams_0 p{ n };
		k_rs::GlobalParams_0 gp{};
		gp.params_0 = &p;
		bind(gp.st_0, st);
		bind(gp.s_0, s);
		bind(gp.y_0, y);
		bind(gp.S_0, S);
		bind(gp.Y_0, Y);
		dispatch(n, &k_rs::main_0_Thread, &gp);
	}
	// The add_correction sequence.
	void add(FVec &s, FVec &y) {
		multiDot(S, mc, s, 0);
		multiDot(Y, mc, s, 2 * mc);
		multiDot(y, 1, s, 4 * mc);
		multiDot(y, 1, y, 4 * mc + 2);
		multiDot(s, 1, s, 4 * mc + 4);
		compact(0);
		ringStore(s, y);
	}
	void applyM(FVec &vin, FVec &vout) {
		k_am::LbApplyMParams_0 p{ mc, 0u, 0u };
		k_am::GlobalParams_0 gp{};
		gp.params_0 = &p;
		bind(gp.st_0, st);
		bind(gp.bf_0, bf);
		bind(gp.vin_0, vin);
		bind(gp.vout_0, vout);
		dispatch(1, &k_am::main_0_Thread, &gp);
	}
};

float pgInf(FVec &x, FVec &g, FVec &lb, FVec &ub) {
	FVec out(1, 0.0f);
	k_pg::LbPgInfParams_0 p{ (uint32_t)x.size(), 0u };
	k_pg::GlobalParams_0 gp{};
	gp.params_0 = &p;
	bind(gp.x_0, x);
	bind(gp.g_0, g);
	bind(gp.lb_0, lb);
	bind(gp.ub_0, ub);
	bind(gp.dst_0, out);
	dispatch(1, &k_pg::main_0_Thread, &gp);
	return out[0];
}
float stepMax(FVec &x, FVec &d, FVec &lb, FVec &ub) {
	FVec out(1, 0.0f);
	k_sm::LbStepMaxParams_0 p{ (uint32_t)x.size(), 0u };
	k_sm::GlobalParams_0 gp{};
	gp.params_0 = &p;
	bind(gp.x_0, x);
	bind(gp.d_0, d);
	bind(gp.lb_0, lb);
	bind(gp.ub_0, ub);
	bind(gp.dst_0, out);
	dispatch(1, &k_sm::main_0_Thread, &gp);
	return out[0];
}
void saxpby(FVec &dst, FVec &x, FVec &y, float a, float b) {
	k_sx::SaxpbyParams_0 p{ (uint32_t)dst.size(), a, b };
	k_sx::GlobalParams_0 gp{};
	gp.params_0 = &p;
	bind(gp.x_0, x);
	bind(gp.y_0, y);
	bind(gp.dst_0, dst);
	dispatch((uint32_t)dst.size(), &k_sx::main_0_Thread, &gp);
}
float dot(FVec &a, FVec &b) {
	FVec out(2, 0.0f);
	k_dr::DotReduceSerialParams_0 p{ (uint32_t)a.size() };
	k_dr::GlobalParams_0 gp{};
	gp.params_0 = &p;
	bind(gp.a_0, a);
	bind(gp.b_0, b);
	bind(gp.dst_0, out);
	dispatch(1, &k_dr::main_0_Thread, &gp);
	return out[0] + out[1];
}
void boxProject(FVec &x, FVec &lb, FVec &ub) {
	k_bp::LbBoxProjectParams_0 p{ (uint32_t)x.size() };
	k_bp::GlobalParams_0 gp{};
	gp.params_0 = &p;
	bind(gp.x_0, x);
	bind(gp.lb_0, lb);
	bind(gp.ub_0, ub);
	dispatch((uint32_t)x.size(), &k_bp::main_0_Thread, &gp);
}

// ---------------------------------------------------------------- one fixture
int runFixture(const std::string &path, bool corrupt, std::string &line) {
	Fixture f;
	if (!load(path, f)) {
		line = "cannot read " + path;
		return 1;
	}
	const uint32_t n = (uint32_t)f.scalar("n"), m = (uint32_t)f.scalar("m");
	const uint32_t ncorr = (uint32_t)f.scalar("ncorr");
	Report r;
	FVec lb = toFV(f.vec["lb"]), ub = toFV(f.vec["ub"]), x = toFV(f.vec["x"]), g = toFV(f.vec["g"]);

	// x already lies in [lb, ub]: the projection must leave it bit-identical.
	{
		FVec xp = x;
		boxProject(xp, lb, ub);
		r.flag(xp == x, "box_project");
	}

	Lb lbs(n, m);
	lbs.compact(1);
	for (size_t p = 0; p < f.s.size(); ++p) {
		FVec s = toFV(f.s[p]), y = toFV(f.y[p]);
		lbs.add(s, y);
		r.flag(lbs.st[3] == 1u, "curvature");
	}
	if (corrupt) {
		// A wrong theta, refactored into T: zero products make lb_compact's
		// curvature test fail, so it only rebuilds L, T and the factor.
		lbs.bf[0] *= 1.25f;
		std::fill(lbs.dots.begin(), lbs.dots.end(), 0.0f);
		lbs.compact(0);
	}
	r.flag(lbs.st[0] == ncorr && lbs.st[1] == (uint32_t)f.scalar("ptr"), "state");
	r.flag(lbs.st[4] == 1u, "chol");
	if (!corrupt) { // the control is judged on what theta feeds, not on theta itself
		const double eTheta = relScalar(lbs.bf[0], f.scalar("theta"));
		r.add(eTheta <= 1e-4, "theta", eTheta, 1e-4);
	}

	// M by columns, then the three random products.
	const uint32_t K = 2 * ncorr;
	if (K > 0) {
		FVec Mg(K * K), e(K), col(K);
		for (uint32_t j = 0; j < K; ++j) {
			std::fill(e.begin(), e.end(), 0.0f);
			e[j] = 1.0f;
			lbs.applyM(e, col);
			for (uint32_t i = 0; i < K; ++i) {
				Mg[i * K + j] = col[i];
			}
		}
		const double eM = relErr(Mg, f.M, K * K);
		r.add(eM <= 1e-4, "M", eM, 1e-4);
		double eMv = 0.0;
		for (size_t t = 0; t < f.mvIn.size(); ++t) {
			FVec vi = toFV(f.mvIn[t]), vo(K);
			lbs.applyM(vi, vo);
			eMv = std::fmax(eMv, relErr(vo, f.mvOut[t], K));
		}
		r.add(eMv <= 1e-4, "Mv", eMv, 1e-4);
	}

	const double ePg = relScalar(pgInf(x, g, lb, ub), f.scalar("pg_inf"));
	r.add(ePg <= 1e-4, "pg_inf", ePg, 1e-4);

	// Cauchy point.
	FVec xcp(n), vecc(2 * m, 0.0f), wk(10 * n, 0.0f);
	UVec iset(8 + 3 * n, 0u);
	{
		k_cau::LbCauchyParams_0 p{ n, m };
		k_cau::GlobalParams_0 gp{};
		gp.params_0 = &p;
		bind(gp.x_0, x);
		bind(gp.g_0, g);
		bind(gp.lb_0, lb);
		bind(gp.ub_0, ub);
		bind(gp.S_0, lbs.S);
		bind(gp.Y_0, lbs.Y);
		bind(gp.st_0, lbs.st);
		bind(gp.bf_0, lbs.bf);
		bind(gp.xcp_0, xcp);
		bind(gp.vecc_0, vecc);
		bind(gp.wk_0, wk);
		bind(gp.iset_0, iset);
		dispatch(1, &k_cau::main_0_Thread, &gp);
	}
	const double eXcp = relErr(xcp, f.vec["xcp"], n);
	r.add(eXcp <= 1e-4, "xcp", eXcp, 1e-4);
	if (K > 0) {
		const double eC = relErr(vecc, f.vec["vecc"], K);
		r.add(eC <= 1e-4, "vecc", eC, 1e-4);
	}
	std::set<int> nact, fv, brk0;
	for (uint32_t k = 0; k < iset[0]; ++k) {
		nact.insert((int)iset[8 + k]);
	}
	for (uint32_t k = 0; k < iset[1]; ++k) {
		fv.insert((int)iset[8 + n + k]);
	}
	for (uint32_t i = 0; i < n; ++i) {
		if (!nact.count((int)i) && !fv.count((int)i)) {
			brk0.insert((int)i);
		}
	}
	r.flag(nact == asSet(f.vec["newact_sorted"]), "newact");
	r.flag(fv == asSet(f.vec["fv_sorted"]), "fv");
	r.flag(brk0 == asSet(f.vec["brk0"]), "brk0");

	// First-iteration direction: normalize(xcp - x), and its step_max.
	{
		FVec d0(n);
		saxpby(d0, xcp, x, 1.0f, -1.0f);
		const float nn = dot(d0, d0);
		if (nn > 0.0f) {
			saxpby(d0, d0, d0, 1.0f / std::sqrt(nn), 0.0f);
		}
		const double eD0 = relErr(d0, f.vec["drt0"], n);
		r.add(eD0 <= 1e-4, "drt0", eD0, 1e-4);
		const double eS0 = relScalar(stepMax(x, d0, lb, ub), f.scalar("step_max0"));
		r.add(eS0 <= 1e-4, "step_max0", eS0, 1e-4);
	}

	// Subspace minimization, max_submin = 10.
	FVec drt(n, 0.0f);
	{
		k_sub::LbSubspaceParams_0 p{ n, m, 10u };
		k_sub::GlobalParams_0 gp{};
		gp.params_0 = &p;
		bind(gp.x_0, x);
		bind(gp.g_0, g);
		bind(gp.lb_0, lb);
		bind(gp.ub_0, ub);
		bind(gp.S_0, lbs.S);
		bind(gp.Y_0, lbs.Y);
		bind(gp.st_0, lbs.st);
		bind(gp.bf_0, lbs.bf);
		bind(gp.xcp_0, xcp);
		bind(gp.vecc_0, vecc);
		bind(gp.iset_0, iset);
		bind(gp.drt_0, drt);
		bind(gp.wk_0, wk);
		dispatch(1, &k_sub::main_0_Thread, &gp);
	}
	const double eDrt = relErr(drt, f.vec["drt"], n);
	r.add(eDrt <= 5e-4, "drt", eDrt, 5e-4);
	std::set<int> sL, sU, sP;
	for (int i : fv) {
		if (drt[i] == lb[i] - x[i]) {
			sL.insert(i);
		} else if (drt[i] == ub[i] - x[i]) {
			sU.insert(i);
		} else {
			sP.insert(i);
		}
	}
	r.flag(sL == asSet(f.vec["sub_L"]) && sU == asSet(f.vec["sub_U"]) && sP == asSet(f.vec["sub_P"]), "subLUP");
	const double eDg = relScalar(dot(g, drt), f.scalar("dg"));
	r.add(eDg <= 5e-4, "dg", eDg, 5e-4);
	const double eSm = relScalar(stepMax(x, drt, lb, ub), f.scalar("step_max"));
	r.add(eSm <= 5e-4, "step_max", eSm, 5e-4);

	char head[200];
	std::snprintf(head, sizeof head, "%s n=%u m=%u ncorr=%u |newact|=%u |fv|=%u sub_rounds=%u sub_path=%u:",
			r.fails ? "FAIL" : "PASS", n, m, ncorr, iset[0], iset[1], iset[4], iset[5]);
	line = head + r.text;
	return r.fails ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <components dir> [--corrupt-theta]\n", argv[0]);
		return 2;
	}
	const std::string dir = argv[1];
	const bool corrupt = argc > 2 && std::strcmp(argv[2], "--corrupt-theta") == 0;
	int failed = 0;
	for (int id = 0; id < 20; ++id) {
		char name[32];
		std::snprintf(name, sizeof name, "comp_%02d.txt", id);
		std::string line;
		const int bad = runFixture(dir + "/" + name, corrupt, line);
		failed += bad;
		std::printf("%s %s\n", name, line.c_str());
	}
	std::printf("%s: %d/20 fixtures pass, %d fail\n", corrupt ? "control (theta x1.25)" : "fixtures", 20 - failed,
			failed);
	if (corrupt) {
		return failed == 20 ? 0 : 1; // the control must fail every fixture
	}
	return failed == 0 ? 0 : 1;
}
