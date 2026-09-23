// fit_sim_hessian -- the similarity Hessian's session data and the kernels'
// cpp twin. See fit_sim_hessian.h.
#include "fit_sim_hessian.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

// The emits carry their own copy of the Slang C++ prelude behind
// SLANG_CPP_PRELUDE_H; including it first at file scope and emptying the
// EXTERN_C macros lets each emit sit in a namespace of its own as ordinary
// C++ (the SdfSpline.cpp / avbd_cpu.cpp arrangement).
#include "slang-cpp-prelude.h"
#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END
// slangc's cpp target passes the Slang `precise` qualifier through as text
// (SPIR-V gets NoContraction from it). C++ has no such keyword; this build
// compiles with -ffp-contract=off (cmake/fit.cmake), which is the same promise.
#define precise

namespace k_simhess {
#include "similarity_hessian_block_emit.cpp"
} // namespace k_simhess
namespace k_psd {
#include "project_psd12_emit.cpp"
} // namespace k_psd
namespace k_gather {
#include "csr_gather_df32_emit.cpp"
} // namespace k_gather

namespace fit_gpu {

static_assert(sizeof(Df) == sizeof(k_simhess::df_0), "df layout");
static_assert(sizeof(Df) == 8, "df is two floats");

// --- SimProblem ---------------------------------------------------------------------

void SimProblem::build(const polyfem::solver::SimilarityForm &form) {
	const Eigen::MatrixXd &V = form.rest_vertices();
	const Eigen::MatrixXi &F = form.faces();
	const Eigen::MatrixXi &TT = form.adjacency();
	const Eigen::MatrixXi &TTi = form.adjacency_index();
	const Eigen::VectorXd &areas = form.rest_areas();
	const Eigen::MatrixXd &coeffs = form.rest_coeffs();
	n = int(V.rows()) * 3;

	// The CPU path's hinge walk (GarmentForm.cpp): le = {{0,1},{1,2},{2,0}}, lv = {2,0,1}.
	const int le[3][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 } };
	const int lv[3] = { 2, 0, 1 };
	std::vector<int> slot_of(size_t(V.rows()), -1);
	slot_vertex.clear();
	hinge_v.clear();
	coef.clear();
	weight.clear();
	std::vector<int> hinge_vertex; // 4 per hinge, collision vertex ids (for the pattern)
	for (int i = 0, k = 0; i < TT.rows(); i++) {
		for (int j = 0; j < TT.cols(); j++, k++) {
			if (TT(i, j) < 0)
				continue;
			const int idx[4] = { F(i, le[j][0]), F(i, le[j][1]), F(i, lv[j]), F(TT(i, j), lv[TTi(i, j)]) };
			for (int a = 0; a < 4; a++) {
				if (slot_of[size_t(idx[a])] < 0) {
					slot_of[size_t(idx[a])] = int(slot_vertex.size());
					slot_vertex.push_back(idx[a]);
				}
				hinge_v.push_back(uint32_t(slot_of[size_t(idx[a])]));
				hinge_vertex.push_back(idx[a]);
			}
			for (int c = 0; c < 6; c++)
				coef.push_back(to_df(coeffs(k, c)));
			weight.push_back(to_df(0.5 * (areas(i) + areas(TT(i, j)))));
		}
	}
	n_hinges = int(weight.size());
	n_slots = int(slot_vertex.size());
	rest.resize(size_t(n_slots) * 3);
	for (int s = 0; s < n_slots; s++)
		for (int d = 0; d < 3; d++)
			rest[size_t(s) * 3 + size_t(d)] = V(slot_vertex[size_t(s)], d);

	// The pattern: the CPU path's triplets with value 0, through the same
	// setFromTriplets, so rows, columns and the index arrays are its own.
	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(size_t(n_hinges) * kBlock);
	for (int h = 0; h < n_hinges; h++) {
		const int *idx = &hinge_vertex[size_t(h) * 4];
		for (int lx = 0; lx < 4; lx++)
			for (int ly = 0; ly < 4; ly++)
				for (int dx = 0; dx < 3; dx++)
					for (int dy = 0; dy < 3; dy++)
						triplets.emplace_back(idx[lx] * 3 + dx, idx[ly] * 3 + dy, 0.0);
	}
	pattern.setZero();
	pattern.resize(n, n);
	pattern.setFromTriplets(triplets.begin(), triplets.end());
	pattern.makeCompressed();
	triplets.clear();
	triplets.shrink_to_fit();

	// Every (hinge, entry) to its nonzero: column-major, rows sorted within
	// a column, so a binary search over the column's inner indices.
	const auto *outer = pattern.outerIndexPtr();
	const auto *inner = pattern.innerIndexPtr();
	std::vector<std::pair<uint32_t, uint32_t>> contrib; // (nonzero, hinge * 144 + entry)
	contrib.reserve(size_t(n_hinges) * kBlock);
	for (int h = 0; h < n_hinges; h++) {
		const int *idx = &hinge_vertex[size_t(h) * 4];
		for (int lx = 0; lx < 4; lx++)
			for (int dx = 0; dx < 3; dx++)
				for (int ly = 0; ly < 4; ly++)
					for (int dy = 0; dy < 3; dy++) {
						const int row = idx[lx] * 3 + dx, col = idx[ly] * 3 + dy;
						const auto *b = inner + outer[col], *e = inner + outer[col + 1];
						const auto *it = std::lower_bound(b, e, row);
						const uint32_t nz = uint32_t(it - inner);
						const int r = lx * 3 + dx, c = ly * 3 + dy; // row-major H(r, c)
						contrib.emplace_back(nz, uint32_t(h * kBlock + r * 12 + c));
					}
	}
	std::sort(contrib.begin(), contrib.end());
	const size_t nnz_ = size_t(pattern.nonZeros());
	ptr.assign(nnz_ + 1, 0);
	src.resize(contrib.size());
	for (size_t i = 0; i < contrib.size(); i++) {
		ptr[contrib[i].first + 1]++;
		src[i] = contrib[i].second;
	}
	for (size_t i = 0; i < nnz_; i++)
		ptr[i + 1] += ptr[i];
}

void SimProblem::positions(const Eigen::VectorXd &x, std::vector<Df> &pos) const {
	pos.resize(size_t(n_slots) * 3);
	for (int s = 0; s < n_slots; s++) {
		const int vtx = slot_vertex[size_t(s)];
		for (int d = 0; d < 3; d++)
			pos[size_t(s) * 3 + size_t(d)] = to_df(x(vtx * 3 + d) + rest[size_t(s) * 3 + size_t(d)]);
	}
}

void SimProblem::assemble(const std::vector<Df> &values, polyfem::StiffnessMatrix &out) const {
	out = pattern;
	double *v = out.valuePtr();
	const size_t m = std::min(values.size(), size_t(out.nonZeros()));
	for (size_t i = 0; i < m; i++)
		v[i] = from_df(values[i]);
}

std::string SimProblem::same_pattern(const polyfem::StiffnessMatrix &m) const {
	if (m.rows() != pattern.rows() || m.cols() != pattern.cols())
		return "size " + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + " vs " +
				std::to_string(pattern.rows()) + "x" + std::to_string(pattern.cols());
	if (!m.isCompressed())
		return "not compressed";
	if (m.nonZeros() != pattern.nonZeros())
		return "nnz " + std::to_string(m.nonZeros()) + " vs " + std::to_string(pattern.nonZeros());
	for (Eigen::Index c = 0; c <= m.cols(); c++)
		if (m.outerIndexPtr()[c] != pattern.outerIndexPtr()[c])
			return "outer[" + std::to_string(c) + "] differs";
	for (Eigen::Index i = 0; i < m.nonZeros(); i++)
		if (m.innerIndexPtr()[i] != pattern.innerIndexPtr()[i])
			return "inner[" + std::to_string(i) + "] differs";
	return "";
}

size_t SimProblem::bytes() const {
	return slot_vertex.size() * sizeof(int) + hinge_v.size() * 4 + coef.size() * sizeof(Df) + weight.size() * sizeof(Df) +
			ptr.size() * 4 + src.size() * 4 + size_t(pattern.nonZeros()) * (sizeof(double) + sizeof(int)) +
			size_t(pattern.cols() + 1) * sizeof(int);
}

// --- the cpp twin --------------------------------------------------------------------

namespace {

template <class Buf, class T>
void bind(Buf &b, const std::vector<T> &v) {
	using Elem = typename std::remove_reference<decltype(*b.data)>::type;
	b.data = reinterpret_cast<Elem *>(const_cast<T *>(v.data()));
	b.count = v.size();
}

} // namespace

void SimTwin::blocks(const SimProblem &p, const std::vector<Df> &pos, float sign, std::vector<Df> &blocks) {
	blocks.assign(size_t(p.n_hinges) * kBlock, Df{});
	k_simhess::SimHessParams_0 prm{};
	prm.count_0 = uint32_t(p.n_hinges);
	prm.sign_0 = sign;
	k_simhess::GlobalParams_0 gp{};
	bind(gp.pos_0, pos);
	bind(gp.hinge_v_0, p.hinge_v);
	bind(gp.coef_0, p.coef);
	bind(gp.blocks_0, blocks);
	gp.params_0 = &prm;
	for (int lane = 0; lane < p.n_hinges; lane++) {
		ComputeThreadVaryingInput t{};
		t.groupID = uint3(0u, 0u, 0u);
		t.groupThreadID = uint3(uint32_t(lane), 0u, 0u);
		k_simhess::main_0_Thread(&t, nullptr, &gp);
	}
}

void SimTwin::project_psd(const SimProblem &p, bool enabled, std::vector<Df> &blocks) {
	k_psd::PsdParams_0 prm{};
	prm.count_0 = uint32_t(p.n_hinges);
	prm.enabled_0 = enabled ? 1u : 0u;
	k_psd::GlobalParams_0 gp{};
	bind(gp.blocks_0, blocks);
	gp.params_0 = &prm;
	for (int lane = 0; lane < p.n_hinges; lane++) {
		ComputeThreadVaryingInput t{};
		t.groupID = uint3(0u, 0u, 0u);
		t.groupThreadID = uint3(uint32_t(lane), 0u, 0u);
		k_psd::main_0_Thread(&t, nullptr, &gp);
	}
}

void SimTwin::gather(const SimProblem &p, const std::vector<Df> &blocks, std::vector<Df> &values) {
	values.assign(p.nnz(), Df{});
	k_gather::GatherParams_0 prm{};
	prm.nnz_0 = uint32_t(p.nnz());
	k_gather::GlobalParams_0 gp{};
	bind(gp.ptr_0, p.ptr);
	bind(gp.src_0, p.src);
	bind(gp.blocks_0, blocks);
	bind(gp.weight_0, p.weight);
	bind(gp.values_0, values);
	gp.params_0 = &prm;
	for (size_t lane = 0; lane < p.nnz(); lane++) {
		ComputeThreadVaryingInput t{};
		t.groupID = uint3(0u, 0u, 0u);
		t.groupThreadID = uint3(uint32_t(lane), 0u, 0u);
		k_gather::main_0_Thread(&t, nullptr, &gp);
	}
}

bool SimTwin::assemble(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, polyfem::StiffnessMatrix &out) {
	if (p.n_hinges == 0 || x.size() != p.n)
		return false;
	std::vector<Df> pos, blk, val;
	p.positions(x, pos);
	blocks(p, pos, sign, blk);
	project_psd(p, psd, blk);
	gather(p, blk, val);
	p.assemble(val, out);
	return true;
}

// --- comparisons ---------------------------------------------------------------------

namespace {

template <class Ref>
BlockCompare compare_impl(const Ref &ref, const std::vector<Df> &got, int n_hinges, double (*rf)(const Ref &, size_t)) {
	BlockCompare r;
	r.n = n_hinges;
	double max_ref = 0;
	std::vector<double> num(size_t(n_hinges), 0), den(size_t(n_hinges), 0);
	for (int h = 0; h < n_hinges; h++) {
		for (int e = 0; e < kBlock; e++) {
			const size_t i = size_t(h) * kBlock + size_t(e);
			const double a = rf(ref, i), b = from_df(got[i]);
			num[size_t(h)] += (b - a) * (b - a);
			den[size_t(h)] += a * a;
		}
		max_ref = std::max(max_ref, std::sqrt(den[size_t(h)]));
	}
	double sum = 0;
	for (int h = 0; h < n_hinges; h++) {
		double rel;
		if (den[size_t(h)] > 0)
			rel = std::sqrt(num[size_t(h)] / den[size_t(h)]);
		else {
			r.zero_ref++;
			rel = max_ref > 0 ? std::sqrt(num[size_t(h)]) / max_ref : (num[size_t(h)] > 0 ? HUGE_VAL : 0.0);
		}
		sum += rel;
		if (rel > r.max_rel || h == 0) {
			r.max_rel = rel;
			r.worst = h;
		}
	}
	r.mean_rel = n_hinges > 0 ? sum / n_hinges : 0;
	return r;
}

double ref_d(const std::vector<double> &v, size_t i) { return v[i]; }
double ref_df(const std::vector<Df> &v, size_t i) { return from_df(v[i]); }

} // namespace

BlockCompare compare_blocks(const std::vector<double> &ref, const std::vector<Df> &got, int n_hinges) {
	if (ref.size() < size_t(n_hinges) * kBlock || got.size() < size_t(n_hinges) * kBlock)
		return BlockCompare{ HUGE_VAL, HUGE_VAL, -1, n_hinges, 0 };
	return compare_impl<std::vector<double>>(ref, got, n_hinges, &ref_d);
}

BlockCompare compare_blocks(const std::vector<Df> &ref, const std::vector<Df> &got, int n_hinges) {
	if (ref.size() < size_t(n_hinges) * kBlock || got.size() < size_t(n_hinges) * kBlock)
		return BlockCompare{ HUGE_VAL, HUGE_VAL, -1, n_hinges, 0 };
	return compare_impl<std::vector<Df>>(ref, got, n_hinges, &ref_df);
}

std::string BlockCompare::text() const {
	char b[160];
	std::snprintf(b, sizeof b, "max_rel %.3e (block %d) mean_rel %.3e over %d blocks, %d with a zero reference",
			max_rel, worst, mean_rel, n, zero_ref);
	return b;
}

double values_max_rel(const polyfem::StiffnessMatrix &a, const polyfem::StiffnessMatrix &b) {
	if (a.nonZeros() != b.nonZeros())
		return HUGE_VAL;
	double amax = 0, dmax = 0;
	for (Eigen::Index i = 0; i < a.nonZeros(); i++) {
		amax = std::max(amax, std::fabs(a.valuePtr()[i]));
		dmax = std::max(dmax, std::fabs(a.valuePtr()[i] - b.valuePtr()[i]));
	}
	return amax > 0 ? dmax / amax : (dmax > 0 ? HUGE_VAL : 0.0);
}

} // namespace fit_gpu
