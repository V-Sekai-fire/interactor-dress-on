// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "avbd_topology.h"

namespace avbd {

void build_csr(uint32_t nV, uint32_t nCon, uint32_t K, const uint32_t *conIdx,
		std::vector<uint32_t> &offsets, std::vector<uint32_t> &idxArr,
		std::vector<uint32_t> &roleArr) {
	std::vector<uint32_t> counts(nV, 0u);
	for (uint32_t c = 0; c < nCon; ++c) {
		for (uint32_t r = 0; r < K; ++r) {
			counts[conIdx[K * c + r]]++;
		}
	}
	offsets.assign(nV + 1, 0u);
	for (uint32_t v = 0; v < nV; ++v) {
		offsets[v + 1] = offsets[v] + counts[v];
	}
	const uint32_t total = offsets[nV];
	idxArr.assign(total, 0u);
	roleArr.assign(total, 0u);
	std::vector<uint32_t> cursor(nV, 0u);
	for (uint32_t c = 0; c < nCon; ++c) {
		for (uint32_t r = 0; r < K; ++r) {
			const uint32_t v = conIdx[K * c + r];
			const uint32_t p = offsets[v] + cursor[v];
			idxArr[p] = c;
			roleArr[p] = r;
			cursor[v]++;
		}
	}
}

void greedy_coloring(uint32_t nV, uint32_t nSprings, const uint32_t *p1, const uint32_t *p2,
		uint32_t nTri, const uint32_t *triIdx, uint32_t nBend, const uint32_t *bendIdx,
		std::vector<uint32_t> &vertPerm, std::vector<uint32_t> &colorOffsets) {
	std::vector<std::vector<uint32_t>> adj(nV);
	auto edge = [&](uint32_t a, uint32_t b) {
		if (a == b) {
			return;
		}
		adj[a].push_back(b);
		adj[b].push_back(a);
	};
	for (uint32_t c = 0; c < nSprings; ++c) {
		edge(p1[c], p2[c]);
	}
	for (uint32_t c = 0; c < nTri; ++c) {
		const uint32_t i0 = triIdx[3 * c], i1 = triIdx[3 * c + 1], i2 = triIdx[3 * c + 2];
		edge(i0, i1);
		edge(i0, i2);
		edge(i1, i2);
	}
	for (uint32_t c = 0; c < nBend; ++c) {
		const uint32_t i0 = bendIdx[4 * c], i1 = bendIdx[4 * c + 1], i2 = bendIdx[4 * c + 2],
					   i3 = bendIdx[4 * c + 3];
		edge(i0, i1);
		edge(i0, i2);
		edge(i0, i3);
		edge(i1, i2);
		edge(i1, i3);
		edge(i2, i3);
	}

	std::vector<int32_t> color(nV, -1);
	std::vector<bool> used;
	for (uint32_t v = 0; v < nV; ++v) {
		used.assign(adj[v].size() + 1, false);
		for (uint32_t n : adj[v]) {
			const int32_t cn = color[n];
			if (cn >= 0 && uint32_t(cn) < used.size()) {
				used[cn] = true;
			}
		}
		uint32_t c = 0;
		while (c < used.size() && used[c]) {
			++c;
		}
		color[v] = int32_t(c);
	}
	uint32_t nc = 0;
	for (int32_t c : color) {
		if (uint32_t(c + 1) > nc) {
			nc = uint32_t(c + 1);
		}
	}
	if (nc == 0) {
		nc = 1;
	}
	std::vector<uint32_t> counts(nc, 0);
	for (int32_t c : color) {
		counts[uint32_t(c)]++;
	}
	std::vector<uint32_t> offsets(nc + 1, 0);
	for (uint32_t c = 0; c < nc; ++c) {
		offsets[c + 1] = offsets[c] + counts[c];
	}
	std::vector<uint32_t> cursor(nc, 0);
	vertPerm.assign(nV, 0);
	for (uint32_t v = 0; v < nV; ++v) {
		const uint32_t c = uint32_t(color[v]);
		vertPerm[offsets[c] + cursor[c]] = v;
		cursor[c]++;
	}
	colorOffsets = std::move(offsets);
}

} // namespace avbd
