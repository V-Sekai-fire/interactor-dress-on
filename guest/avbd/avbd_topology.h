// CPU-side topology shared by both AVBD backends: the vertex->constraint CSR
// each gather kernel walks, and the greedy vertex colouring that turns one
// Jacobi sweep into Gauss-Seidel colour passes. Faithful to AvbdSolver.mm /
// AvbdSolverVk.cpp; lifted out of the org's avbd_cpu.cpp so AvbdRd cannot
// drift from AvbdCpu.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <vector>

namespace avbd {

// offsets[nV+1], flat idx and role arrays. K entries per constraint, role r
// in 0..K-1.
void build_csr(uint32_t nV, uint32_t nCon, uint32_t K, const uint32_t *conIdx,
		std::vector<uint32_t> &offsets, std::vector<uint32_t> &idxArr,
		std::vector<uint32_t> &roleArr);

// Greedy first-fit colouring over the spring 2-cliques, triangle 3-cliques and
// bending 4-cliques. vertPerm lists vertices colour by colour; colorOffsets
// (numColors+1) bounds each colour's range in vertPerm.
void greedy_coloring(uint32_t nV, uint32_t nSprings, const uint32_t *p1, const uint32_t *p2,
		uint32_t nTri, const uint32_t *triIdx, uint32_t nBend, const uint32_t *bendIdx,
		std::vector<uint32_t> &vertPerm, std::vector<uint32_t> &colorOffsets);

} // namespace avbd
