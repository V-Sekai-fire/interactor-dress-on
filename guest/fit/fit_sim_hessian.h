// fit_sim_hessian -- SimilarityForm's Hessian assembly as the three Lean
// kernels see it (cut 6g-C, stage 1), and the kernels' cpp twin.
//
// The CPU path (vendor/cloth-fit GarmentForm.cpp, SimilarityForm::
// second_derivative_unweighted) walks the hinges, runs the sympy autogen
// similarity_hessian per hinge, projects each 12x12 to PSD, weights it and
// lets Eigen sum 144 triplets per hinge. The GPU path splits that into what
// changes per Newton iteration (the positions) and what does not (the
// hinges, their coefficients and weights, and the sparse pattern the sum
// produces): SimProblem holds the latter, built once per session, and the
// kernels (kernels/fit: similarity_hessian_block, project_psd12,
// csr_gather_df32) do the former in df32.
//
// SimTwin runs the same kernels as slangc's cpp emit, one lane at a time,
// on the guest CPU or natively: the second target of the one source
// (AGENTS.md rule 2), which the C1 gate holds to the GPU within FMA noise
// and to the CPU double path at 1e-9. This file is part of fit_core
// (cmake/fit.cmake; tests/native/fit for fit_native), so it has no
// sandbox or RenderingDevice dependency; guest/fit/fit_gpu.cpp is the
// rd_compute side.
#pragma once

#include <polyfem/solver/forms/garment_forms/GarmentForm.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

namespace fit_gpu {

// A df32 pair as the kernels store it: struct df { float hi; float lo; }.
struct Df {
	float hi = 0;
	float lo = 0;
};
inline Df to_df(double v) {
	const float h = float(v);
	return { h, float(v - double(h)) };
}
inline double from_df(Df d) {
	return double(d.hi) + double(d.lo);
}

constexpr int kBlock = 144; // 12 x 12 entries per hinge

struct SimProblem {
	int n = 0;         // the complete space: x.size() = 3 x collision vertices
	int n_hinges = 0;  // hinges with a neighbour (TT >= 0), in the CPU path's order
	int n_slots = 0;   // compact vertex slots the hinges reference
	std::vector<int> slot_vertex;    // slot -> collision vertex
	std::vector<double> rest;        // 3 per slot: the rest position (x is a displacement)
	std::vector<uint32_t> hinge_v;   // 4 per hinge: ve0, ve1, vf0, vf1 as slots
	std::vector<Df> coef;            // 6 per hinge
	std::vector<Df> weight;          // per hinge: 0.5 (area_i + area_j)
	// The pattern setFromTriplets produces for these hinges (compressed,
	// column-major, values 0), and for every nonzero the contributions
	// hinge * 144 + entry that land on it, ascending.
	polyfem::StiffnessMatrix pattern;
	std::vector<uint32_t> ptr; // nnz + 1
	std::vector<uint32_t> src; // ptr[nnz] entries
	size_t nnz() const { return ptr.empty() ? 0 : ptr.size() - 1; }

	void build(const polyfem::solver::SimilarityForm &form);
	// The slots' positions at x (x + rest), 3 pairs per slot.
	void positions(const Eigen::VectorXd &x, std::vector<Df> &pos) const;
	// out = pattern with the nonzeros' values from the gather's pairs.
	void assemble(const std::vector<Df> &values, polyfem::StiffnessMatrix &out) const;
	// "" when `m` has the pattern's rows, columns and index arrays exactly,
	// else the first difference.
	std::string same_pattern(const polyfem::StiffnessMatrix &m) const;
	size_t bytes() const;
};

// The kernels' cpp twin (slangc -target cpp of the committed Slang).
struct SimTwin {
	static void blocks(const SimProblem &p, const std::vector<Df> &pos, float sign, std::vector<Df> &blocks);
	static void project_psd(const SimProblem &p, bool enabled, std::vector<Df> &blocks);
	static void gather(const SimProblem &p, const std::vector<Df> &blocks, std::vector<Df> &values);
	// The whole assembly at x, into out (= the pattern with values).
	static bool assemble(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, polyfem::StiffnessMatrix &out);
};

// Per-block comparison: max and mean over hinges of ||got - ref||_F / ||ref||_F
// (a block with ||ref||_F = 0 compares by ||got||_F against the largest ref
// norm instead, and is counted in zero_ref).
struct BlockCompare {
	double max_rel = 0;
	double mean_rel = 0;
	int worst = -1;
	int n = 0;
	int zero_ref = 0;
	std::string text() const;
};
BlockCompare compare_blocks(const std::vector<double> &ref, const std::vector<Df> &got, int n_hinges);
BlockCompare compare_blocks(const std::vector<Df> &ref, const std::vector<Df> &got, int n_hinges);
// Value arrays of two matrices with the same pattern: max |a - b| / max |a|.
double values_max_rel(const polyfem::StiffnessMatrix &a, const polyfem::StiffnessMatrix &b);

} // namespace fit_gpu
