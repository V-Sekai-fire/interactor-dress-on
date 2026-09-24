// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The L2 test of the Lean-emitted ANNY kernels (kernels/anny): the forward
// chain (blend -> joint regressor -> 6D forward kinematics -> skinning) and
// its vector-Jacobian products, run from their slangc cpp emits, against a
// double-precision restatement of the same forward and its central finite
// differences.
//
//   anny_kernels_test            both cases; exit 1 on any failure
//   anny_kernels_test --control  the backward fed dverts x 1.25: every
//                                gradient group must FAIL (the flat control,
//                                AGENTS.md rule 7)
//
// Cases: "small" (P 40, 5 blendshapes, 6 joints, every parameter
// differenced) and "anny-size" (P 13,718 and J 104 as ANNY's mesh and rig,
// 20 blendshapes, 64 parameters differenced). Inputs are made in double,
// rounded to float, and both sides read the rounded values. The random
// draws use the engine's raw output only (AGENTS.md: libc++ and libstdc++
// distributions differ from the same seed).
//
// The loss is L = 1/2 |verts - target|^2 (AnnyInverter's per-vertex error)
// over x = [coeffs | rot6 | trans]; the bind rotations rb are inputs, held
// fixed on both sides (the kernels treat them as a stop-gradient).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "slang-cpp-prelude.h"

#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END

namespace k_blend {
#include "anny_blend_emit.cpp"
}
namespace k_blend_bw {
#include "anny_blend_backward_emit.cpp"
}
namespace k_csr {
#include "anny_csr_gemv3_emit.cpp"
}
namespace k_fk {
#include "anny_fk_emit.cpp"
}
namespace k_fk_bw {
#include "anny_fk_backward_emit.cpp"
}
namespace k_lbs {
#include "anny_lbs_emit.cpp"
}
namespace k_lbs_bone {
#include "anny_lbs_backward_bone_emit.cpp"
}
namespace k_lbs_bind {
#include "anny_lbs_backward_bind_emit.cpp"
}
namespace k_res {
#include "anny_vert_residual_emit.cpp"
}

namespace {

using std::vector;
using KernelFn = void (*)(ComputeVaryingInput *, void *, void *);

void dispatch(KernelFn fn, void *gp, uint32_t threads, uint32_t group) {
	ComputeVaryingInput vi{};
	vi.startGroupID = uint3(0u, 0u, 0u);
	vi.endGroupID = uint3((threads + group - 1) / group, 1u, 1u);
	fn(&vi, nullptr, gp);
}

template <class B, class T>
void bind(B &b, vector<T> &v) {
	b.data = reinterpret_cast<decltype(b.data)>(v.data());
	b.count = v.size();
}

// ---- inputs ----------------------------------------------------------------

struct Rng {
	std::mt19937 e;
	explicit Rng(uint32_t seed) : e(seed) {}
	double u01() { return (e() >> 8) * (1.0 / 16777216.0); }
	double sym() { return 2.0 * u01() - 1.0; }
	uint32_t below(uint32_t n) { return e() % n; }
};

double fr(double x) { return (double)(float)x; }

struct Model {
	uint32_t P = 0, NB = 0, J = 0;
	vector<double> templ, blend;         // P*3, NB*P*3
	vector<uint32_t> rowptr, col;        // the joint regressor M (J x P), CSR
	vector<double> val;
	vector<uint32_t> trow, tcol;         // M^T (P x J), CSR
	vector<double> tval;
	vector<uint32_t> parents;            // J, parents[j] < j
	vector<double> rb;                   // J*9
	vector<double> weights;              // P*J
	vector<double> target;               // P*3
	size_t nx() const { return NB + J * 6 + 3; }
};

// ---- the double reference ---------------------------------------------------

void rot6_ref(const double *a, double R[9]) {
	const double n1 = std::max(std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), 1e-20);
	double r1[3] = { a[0] / n1, a[1] / n1, a[2] / n1 };
	const double d = r1[0] * a[3] + r1[1] * a[4] + r1[2] * a[5];
	double u[3] = { a[3] - d * r1[0], a[4] - d * r1[1], a[5] - d * r1[2] };
	const double nu = std::max(std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), 1e-20);
	double r2[3] = { u[0] / nu, u[1] / nu, u[2] / nu };
	double r3[3] = { r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0] };
	for (int k = 0; k < 3; ++k) {
		R[k] = r1[k];
		R[3 + k] = r2[k];
		R[6 + k] = r3[k];
	}
}

void mm(const double *A, const double *B, double *C) { // C = A B
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}

void mtm(const double *A, const double *B, double *C) { // C = A^T B
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			C[3 * i + j] = A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j];
}

// verts(x); returns L.
double forward_ref(const Model &M, const vector<double> &x, vector<double> *verts_out = nullptr) {
	const double *coeffs = x.data();
	const double *rot6 = x.data() + M.NB;
	const double *trans = x.data() + M.NB + M.J * 6;
	vector<double> vanny(M.templ);
	for (uint32_t c = 0; c < M.NB; ++c)
		for (size_t i = 0; i < (size_t)M.P * 3; ++i)
			vanny[i] += coeffs[c] * M.blend[(size_t)c * M.P * 3 + i];
	vector<double> jpos((size_t)M.J * 3, 0.0);
	for (uint32_t i = 0; i < M.J; ++i)
		for (uint32_t k = M.rowptr[i]; k < M.rowptr[i + 1]; ++k)
			for (int a = 0; a < 3; ++a)
				jpos[i * 3 + a] += M.val[k] * vanny[M.col[k] * 3 + a];
	vector<double> P((size_t)M.J * 9), q((size_t)M.J * 3), bone((size_t)M.J * 12);
	const double I[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	for (uint32_t j = 0; j < M.J; ++j) {
		const bool root = j == 0;
		const uint32_t p = root ? 0 : M.parents[j];
		const double *Rbp = root ? I : &M.rb[p * 9];
		double jp[3], Pp[9], qp[3];
		for (int a = 0; a < 3; ++a) {
			jp[a] = root ? -trans[a] : jpos[p * 3 + a];
			qp[a] = root ? 0.0 : q[p * 3 + a];
		}
		for (int k = 0; k < 9; ++k)
			Pp[k] = root ? I[k] : P[p * 9 + k];
		double R[9], G[9], A[9];
		rot6_ref(rot6 + j * 6, R);
		mtm(Rbp, &M.rb[j * 9], G);
		mm(G, R, A);
		double e[3], c[3];
		for (int a = 0; a < 3; ++a)
			e[a] = jpos[j * 3 + a] - jp[a];
		for (int a = 0; a < 3; ++a)
			c[a] = Rbp[a] * e[0] + Rbp[3 + a] * e[1] + Rbp[6 + a] * e[2];
		mm(Pp, A, &P[j * 9]);
		for (int a = 0; a < 3; ++a)
			q[j * 3 + a] = Pp[3 * a] * c[0] + Pp[3 * a + 1] * c[1] + Pp[3 * a + 2] * c[2] + qp[a];
		// B = P Rb_j^T, b = q - B jpos_j
		const double *Pj = &P[j * 9];
		const double *Rbj = &M.rb[j * 9];
		for (int a = 0; a < 3; ++a) {
			double bj = q[j * 3 + a];
			for (int b = 0; b < 3; ++b) {
				const double B = Pj[3 * a] * Rbj[3 * b] + Pj[3 * a + 1] * Rbj[3 * b + 1] + Pj[3 * a + 2] * Rbj[3 * b + 2];
				bone[j * 12 + 4 * a + b] = B;
				bj -= B * jpos[j * 3 + b];
			}
			bone[j * 12 + 4 * a + 3] = bj;
		}
	}
	vector<double> verts((size_t)M.P * 3, 0.0);
	double L = 0.0;
	for (uint32_t v = 0; v < M.P; ++v) {
		for (uint32_t j = 0; j < M.J; ++j) {
			const double w = M.weights[(size_t)v * M.J + j];
			if (w == 0.0)
				continue;
			for (int a = 0; a < 3; ++a) {
				const double *r = &bone[j * 12 + 4 * a];
				verts[v * 3 + a] += w * (r[0] * vanny[v * 3] + r[1] * vanny[v * 3 + 1] + r[2] * vanny[v * 3 + 2] + r[3]);
			}
		}
		for (int a = 0; a < 3; ++a) {
			const double d = verts[v * 3 + a] - M.target[v * 3 + a];
			L += 0.5 * d * d;
		}
	}
	if (verts_out)
		*verts_out = verts;
	return L;
}

// ---- the kernels ------------------------------------------------------------

vector<float> f32(const vector<double> &v) { return vector<float>(v.begin(), v.end()); }

struct KernelRun {
	vector<float> verts, dcoeffs, drot6, dtrans;
	double loss = 0.0;
};

// Forward, then backward with dverts scaled by `scale` (1 for the test,
// 1.25 for the control).
KernelRun run_kernels(const Model &M, const vector<double> &x, float scale) {
	vector<float> templ = f32(M.templ), blend = f32(M.blend), val = f32(M.val), tval = f32(M.tval);
	vector<float> rb = f32(M.rb), weights = f32(M.weights), target = f32(M.target);
	vector<uint32_t> rowptr = M.rowptr, col = M.col, trow = M.trow, tcol = M.tcol, parents = M.parents;
	vector<float> coeffs(x.begin(), x.begin() + M.NB);
	vector<float> rot6(x.begin() + M.NB, x.begin() + M.NB + M.J * 6);
	vector<float> trans(x.begin() + M.NB + M.J * 6, x.end());
	const size_t P3 = (size_t)M.P * 3;
	vector<float> vanny(P3), jpos((size_t)M.J * 3), world((size_t)M.J * 12), bone((size_t)M.J * 12);
	vector<float> verts(P3), dverts(P3), dbone((size_t)M.J * 12), dvanny(P3), dworld((size_t)M.J * 12);
	vector<float> drot6((size_t)M.J * 6), dtrans(3), djpos((size_t)M.J * 3), dcoeffs(M.NB);

	{
		k_blend::AnnyBlendParams_0 pr{ M.P, M.NB };
		k_blend::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.templ_0, templ);
		bind(gp.blend_0, blend);
		bind(gp.coeffs_0, coeffs);
		bind(gp.vanny_0, vanny);
		dispatch(&k_blend::main_0, &gp, M.P, 64);
	}
	{
		k_csr::AnnyCsrParams_0 pr{ M.J, 0u };
		k_csr::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.rowptr_0, rowptr);
		bind(gp.col_0, col);
		bind(gp.val_0, val);
		bind(gp.x_0, vanny);
		bind(gp.y_0, jpos);
		dispatch(&k_csr::main_0, &gp, M.J, 64);
	}
	{
		k_fk::AnnyFkParams_0 pr{ M.J };
		k_fk::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.parents_0, parents);
		bind(gp.rb_0, rb);
		bind(gp.jpos_0, jpos);
		bind(gp.rot6_0, rot6);
		bind(gp.trans_0, trans);
		bind(gp.world_0, world);
		bind(gp.bone_0, bone);
		dispatch(&k_fk::main_0, &gp, 1, 1);
	}
	{
		k_lbs::AnnyLbsParams_0 pr{ M.P, M.J };
		k_lbs::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.bind_0, vanny);
		bind(gp.weights_0, weights);
		bind(gp.bone_0, bone);
		bind(gp.verts_0, verts);
		dispatch(&k_lbs::main_0, &gp, M.P, 64);
	}
	{
		k_res::AnnyResidualParams_0 pr{ (uint32_t)P3 };
		k_res::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.verts_0, verts);
		bind(gp.target_0, target);
		bind(gp.dverts_0, dverts);
		dispatch(&k_res::main_0, &gp, (uint32_t)P3, 256);
	}
	KernelRun out;
	for (float d : dverts)
		out.loss += 0.5 * (double)d * (double)d;
	for (float &d : dverts)
		d *= scale;
	{
		k_lbs_bone::AnnyLbsParams_0 pr{ M.P, M.J };
		k_lbs_bone::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.bind_0, vanny);
		bind(gp.weights_0, weights);
		bind(gp.dverts_0, dverts);
		bind(gp.dbone_0, dbone);
		dispatch(&k_lbs_bone::main_0, &gp, M.J * 3, 64);
	}
	{
		k_lbs_bind::AnnyLbsParams_0 pr{ M.P, M.J };
		k_lbs_bind::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.weights_0, weights);
		bind(gp.bone_0, bone);
		bind(gp.dverts_0, dverts);
		bind(gp.dbind_0, dvanny);
		dispatch(&k_lbs_bind::main_0, &gp, M.P, 64);
	}
	{
		k_fk_bw::AnnyFkParams_0 pr{ M.J };
		k_fk_bw::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.parents_0, parents);
		bind(gp.rb_0, rb);
		bind(gp.jpos_0, jpos);
		bind(gp.rot6_0, rot6);
		bind(gp.trans_0, trans);
		bind(gp.world_0, world);
		bind(gp.dbone_0, dbone);
		bind(gp.dworld_0, dworld);
		bind(gp.drot6_0, drot6);
		bind(gp.dtrans_0, dtrans);
		bind(gp.djpos_0, djpos);
		dispatch(&k_fk_bw::main_0, &gp, 1, 1);
	}
	{
		k_csr::AnnyCsrParams_0 pr{ M.P, 1u };
		k_csr::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.rowptr_0, trow);
		bind(gp.col_0, tcol);
		bind(gp.val_0, tval);
		bind(gp.x_0, djpos);
		bind(gp.y_0, dvanny);
		dispatch(&k_csr::main_0, &gp, M.P, 64);
	}
	{
		k_blend_bw::AnnyBlendParams_0 pr{ M.P, M.NB };
		k_blend_bw::GlobalParams_0 gp{};
		gp.params_0 = &pr;
		bind(gp.blend_0, blend);
		bind(gp.dvanny_0, dvanny);
		bind(gp.dcoeffs_0, dcoeffs);
		dispatch(&k_blend_bw::main_0, &gp, M.NB, 64);
	}
	out.verts = verts;
	out.dcoeffs = dcoeffs;
	out.drot6 = drot6;
	out.dtrans = dtrans;
	return out;
}

// ---- cases ------------------------------------------------------------------

void random_rotation(Rng &r, double R[9]) {
	double a[6];
	for (double &t : a)
		t = r.sym();
	rot6_ref(a, R);
}

// x near the identity pose: rot6 = (1 0 0 0 1 0) + noise.
vector<double> random_x(const Model &M, Rng &r, double pose_noise) {
	vector<double> x(M.nx());
	for (uint32_t c = 0; c < M.NB; ++c)
		x[c] = fr(0.5 * r.sym());
	for (uint32_t j = 0; j < M.J; ++j)
		for (int k = 0; k < 6; ++k)
			x[M.NB + j * 6 + k] = fr((k == 0 || k == 4 ? 1.0 : 0.0) + pose_noise * r.sym());
	for (int a = 0; a < 3; ++a)
		x[M.NB + M.J * 6 + a] = fr(0.2 * r.sym());
	return x;
}

Model make_model(uint32_t P, uint32_t NB, uint32_t J, uint32_t per_vertex, uint32_t seed, vector<double> *x0) {
	Rng r(seed);
	Model M;
	M.P = P;
	M.NB = NB;
	M.J = J;
	M.templ.resize((size_t)P * 3);
	for (double &t : M.templ)
		t = fr(r.sym());
	M.blend.resize((size_t)NB * P * 3);
	for (double &t : M.blend)
		t = fr(0.1 * r.sym());
	// Regressor: each joint from 4 vertices, weights summing to ~1.
	M.rowptr.push_back(0);
	for (uint32_t i = 0; i < J; ++i) {
		double w[4], s = 0.0;
		for (double &t : w)
			s += (t = 0.1 + r.u01());
		for (int k = 0; k < 4; ++k) {
			M.col.push_back(r.below(P));
			M.val.push_back(fr(w[k] / s));
		}
		M.rowptr.push_back((uint32_t)M.col.size());
	}
	// Its transpose, rows in vertex order and, within a row, in joint order.
	vector<vector<std::pair<uint32_t, double>>> byv(P);
	for (uint32_t i = 0; i < J; ++i)
		for (uint32_t k = M.rowptr[i]; k < M.rowptr[i + 1]; ++k)
			byv[M.col[k]].push_back({ i, M.val[k] });
	M.trow.push_back(0);
	for (uint32_t v = 0; v < P; ++v) {
		for (auto &e : byv[v]) {
			M.tcol.push_back(e.first);
			M.tval.push_back(e.second);
		}
		M.trow.push_back((uint32_t)M.tcol.size());
	}
	M.parents.assign(J, 0);
	for (uint32_t j = 1; j < J; ++j)
		M.parents[j] = j - 1 - r.below(std::min<uint32_t>(j, 5));
	M.rb.resize((size_t)J * 9);
	for (uint32_t j = 0; j < J; ++j) {
		double R[9];
		random_rotation(r, R);
		for (int k = 0; k < 9; ++k)
			M.rb[j * 9 + k] = fr(R[k]);
	}
	M.weights.assign((size_t)P * J, 0.0);
	for (uint32_t v = 0; v < P; ++v) {
		double s = 0.0;
		vector<uint32_t> js;
		for (uint32_t k = 0; k < per_vertex; ++k)
			js.push_back(r.below(J));
		vector<double> w(per_vertex);
		for (double &t : w)
			s += (t = 0.1 + r.u01());
		for (uint32_t k = 0; k < per_vertex; ++k)
			M.weights[(size_t)v * J + js[k]] += w[k] / s;
		for (uint32_t j = 0; j < J; ++j)
			M.weights[(size_t)v * J + j] = fr(M.weights[(size_t)v * J + j]);
	}
	// Target: the model at another pose and shape, plus noise.
	M.target.assign((size_t)P * 3, 0.0);
	vector<double> xt = random_x(M, r, 0.3), vt;
	forward_ref(M, xt, &vt);
	for (size_t i = 0; i < vt.size(); ++i)
		M.target[i] = fr(vt[i] + 0.01 * r.sym());
	*x0 = random_x(M, r, 0.3);
	return M;
}

struct Group {
	const char *name;
	size_t lo, hi; // range in x
};

bool run_case(const char *name, uint32_t P, uint32_t NB, uint32_t J, uint32_t per_vertex, size_t fd_max,
		uint32_t seed, bool control) {
	vector<double> x;
	const Model M = make_model(P, NB, J, per_vertex, seed, &x);
	vector<double> vref;
	const double Lref = forward_ref(M, x, &vref);
	const KernelRun K = run_kernels(M, x, control ? 1.25f : 1.0f);

	double fwd = 0.0, scale = 0.0;
	for (size_t i = 0; i < vref.size(); ++i) {
		fwd = std::max(fwd, std::fabs((double)K.verts[i] - vref[i]));
		scale = std::max(scale, std::fabs(vref[i]));
	}
	const double fwd_rel = fwd / scale;
	const double loss_rel = std::fabs(K.loss - Lref) / Lref;
	printf("case %s: P %u NB %u J %u, %u joints per vertex; L = %.6e\n", name, P, NB, J, per_vertex, Lref);
	printf("  forward: max|verts - ref| / max|ref| = %.2e, |L - ref| / ref = %.2e  %s\n", fwd_rel, loss_rel,
			fwd_rel < 1e-5 ? "OK" : "FAIL");
	bool ok = fwd_rel < 1e-5;

	vector<double> g(x.size());
	for (size_t i = 0; i < M.NB; ++i)
		g[i] = K.dcoeffs[i];
	for (size_t i = 0; i < (size_t)M.J * 6; ++i)
		g[M.NB + i] = K.drot6[i];
	for (int a = 0; a < 3; ++a)
		g[M.NB + M.J * 6 + a] = K.dtrans[a];

	// Which coordinates to difference: all of them, or fd_max spread over the groups.
	const Group groups[3] = { { "coeffs", 0, M.NB }, { "rot6", M.NB, M.NB + (size_t)M.J * 6 },
		{ "trans", M.NB + (size_t)M.J * 6, x.size() } };
	vector<size_t> pick;
	if (x.size() <= fd_max) {
		for (size_t i = 0; i < x.size(); ++i)
			pick.push_back(i);
	} else {
		Rng r(seed ^ 0x9e3779b9u);
		for (const Group &gr : groups) {
			const size_t n = gr.hi - gr.lo;
			const size_t take = std::min(n, gr.hi == x.size() ? n : (gr.lo == 0 ? size_t(10) : fd_max - 13));
			for (size_t k = 0; k < take; ++k)
				pick.push_back(take == n ? gr.lo + k : gr.lo + r.below((uint32_t)n));
		}
	}
	vector<double> fd(x.size(), 0.0);
	vector<char> have(x.size(), 0);
	const double h = 1e-5;
	for (size_t i : pick) {
		vector<double> xp = x, xm = x;
		xp[i] += h;
		xm[i] -= h;
		fd[i] = (forward_ref(M, xp) - forward_ref(M, xm)) / (2.0 * h);
		have[i] = 1;
	}
	int gpass = 0;
	for (const Group &gr : groups) {
		double num = 0.0, den = 0.0, worst = 0.0;
		size_t n = 0;
		for (size_t i = gr.lo; i < gr.hi; ++i) {
			if (!have[i])
				continue;
			num += (g[i] - fd[i]) * (g[i] - fd[i]);
			den += fd[i] * fd[i];
			worst = std::max(worst, std::fabs(g[i] - fd[i]));
			++n;
		}
		const double rel = std::sqrt(num / std::max(den, 1e-300));
		const bool pass = rel < 1e-4;
		printf("  d%-7s %3zu of %3zu differenced: |g - fd| / |fd| = %.2e (worst |g - fd| %.2e, |fd| %.2e)  %s\n",
				gr.name, n, gr.hi - gr.lo, rel, worst, std::sqrt(den), pass ? "OK" : "FAIL");
		ok = ok && pass;
		gpass += pass ? 1 : 0;
	}
	// The control holds only if no gradient group passes on the scaled cotangent.
	return control ? gpass == 0 : ok;
}

} // namespace

int main(int argc, char **argv) {
	const bool control = argc > 1 && std::strcmp(argv[1], "--control") == 0;
	int failed = 0, n = 0;
	const bool a = run_case("small", 40, 5, 6, 3, 1000, 1u, control);
	const bool b = run_case("anny-size", 13718, 20, 104, 4, 64, 2u, control);
	for (bool r : { a, b }) {
		++n;
		failed += r ? 0 : 1;
	}
	if (control) {
		printf("RESULT: control %s (%d of %d cases with every gradient group failing)\n",
				failed == 0 ? "OK" : "FAIL", n - failed, n);
		return failed == 0 ? 0 : 1;
	}
	printf("RESULT: %s (%d of %d cases passed)\n", failed == 0 ? "PASS" : "FAIL", n - failed, n);
	return failed == 0 ? 0 : 1;
}
