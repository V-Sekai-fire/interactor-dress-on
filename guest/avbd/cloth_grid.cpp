// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "cloth_grid.h"

#include <array>
#include <cmath>
#include <map>
#include <utility>

namespace {

using V3 = std::array<double, 3>;

inline V3 sub(const V3 &a, const V3 &b) { return { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
inline double dot(const V3 &a, const V3 &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double norm(const V3 &a) { return std::sqrt(dot(a, a)); }
inline V3 scale(const V3 &a, double s) { return { a[0] * s, a[1] * s, a[2] * s }; }

// DiffCloth Triangle::inv_deltaUV: Gram-Schmidt the two rest edges into the
// triangle's own 2D frame, form the 2x2 material matrix, invert it.
void triangle_inv_uv(const V3 &p0, const V3 &p1, const V3 &p2, float out[4]) {
	const V3 e0 = sub(p1, p0);
	const V3 e1 = sub(p2, p0);
	const double n0 = norm(e0);
	const V3 P0 = scale(e0, 1.0 / n0);
	V3 t = sub(e1, scale(P0, dot(e1, P0)));
	const V3 P1 = scale(t, 1.0 / norm(t));
	// deltaUV = P^T * [e0 e1]  (upper-triangular: P1.e0 == 0)
	const double d00 = dot(P0, e0), d01 = dot(P0, e1);
	const double d10 = dot(P1, e0), d11 = dot(P1, e1);
	const double det = d00 * d11 - d01 * d10;
	// inv row-major [m00, m01, m10, m11].
	out[0] = float(d11 / det);
	out[1] = float(-d01 / det);
	out[2] = float(-d10 / det);
	out[3] = float(d00 / det);
}

double heron(double a, double b, double c) {
	const double r = 0.5 * (a + b + c);
	const double s = r * (r - a) * (r - b) * (r - c);
	return s > 0.0 ? std::sqrt(s) : 0.0;
}

} // namespace

ClothMesh build_cloth_mesh(uint32_t nx, uint32_t ny, float width, float height,
		float panelTop, float clothMass) {
	ClothMesh m;
	m.nx = nx;
	m.ny = ny;
	m.width = width;
	m.height = height;
	const uint32_t nV = nx * ny;
	const double sx = width / double(nx - 1);
	const double sz = height / double(ny - 1);

	auto vid = [ny](uint32_t ix, uint32_t iy) { return ix * ny + iy; };
	std::vector<V3> rest(nV);
	m.positions.assign(3 * nV, 0.0f);
	for (uint32_t ix = 0; ix < nx; ++ix) {
		for (uint32_t iy = 0; iy < ny; ++iy) {
			const uint32_t v = vid(ix, iy);
			const V3 p = { double(ix) * sx - width * 0.5, double(panelTop), double(iy) * sz };
			rest[v] = p;
			m.positions[3 * v + 0] = float(p[0]);
			m.positions[3 * v + 1] = float(p[1]);
			m.positions[3 * v + 2] = float(p[2]);
		}
	}
	m.mass.assign(nV, clothMass / float(nV));

	// Two triangles per cell, consistent winding.
	auto addTri = [&](uint32_t a, uint32_t b, uint32_t c) {
		m.triIdx.push_back(a);
		m.triIdx.push_back(b);
		m.triIdx.push_back(c);
		float uv[4];
		triangle_inv_uv(rest[a], rest[b], rest[c], uv);
		for (int j = 0; j < 4; ++j) {
			m.triInvUV.push_back(uv[j]);
		}
	};
	for (uint32_t ix = 0; ix + 1 < nx; ++ix) {
		for (uint32_t iy = 0; iy + 1 < ny; ++iy) {
			const uint32_t v00 = vid(ix, iy), v10 = vid(ix + 1, iy);
			const uint32_t v01 = vid(ix, iy + 1), v11 = vid(ix + 1, iy + 1);
			addTri(v00, v10, v11);
			addTri(v00, v11, v01);
		}
	}
	m.faces = m.triIdx;

	// Dihedral bending: every edge shared by two triangles gives a 4-vertex
	// stencil (edge endpoints + the two opposite corners), cotangent weights.
	std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edgeMap;
	const uint32_t nT = m.nTri();
	for (uint32_t ti = 0; ti < nT; ++ti) {
		const uint32_t a = m.triIdx[3 * ti], b = m.triIdx[3 * ti + 1], c = m.triIdx[3 * ti + 2];
		const uint32_t tri[3] = { a, b, c };
		for (int i = 0; i < 3; ++i) {
			for (int j = i + 1; j < 3; ++j) {
				const uint32_t lo = std::min(tri[i], tri[j]), hi = std::max(tri[i], tri[j]);
				const uint32_t other = a + b + c - tri[i] - tri[j];
				edgeMap[{ lo, hi }].push_back(other);
			}
		}
	}
	for (const auto &kv : edgeMap) {
		if (kv.second.size() != 2) {
			continue;
		}
		const uint32_t p0 = kv.first.first, p1 = kv.first.second;
		const uint32_t p2 = kv.second[0], p3 = kv.second[1];
		const V3 &x0 = rest[p0], &x1 = rest[p1], &x2 = rest[p2], &x3 = rest[p3];
		const double l01 = norm(sub(x1, x0)), l02 = norm(sub(x2, x0)), l03 = norm(sub(x3, x0));
		const double l12 = norm(sub(x1, x2)), l13 = norm(sub(x1, x3));
		const double A0 = heron(l01, l02, l12), A1 = heron(l01, l03, l13);
		if (A0 <= 0.0 || A1 <= 0.0) {
			continue;
		}
		const double cot02 = (l01 * l01 - l02 * l02 + l12 * l12) / (4.0 * A0);
		const double cot12 = (l01 * l01 + l02 * l02 - l12 * l12) / (4.0 * A0);
		const double cot03 = (l01 * l01 - l03 * l03 + l13 * l13) / (4.0 * A1);
		const double cot13 = (l01 * l01 + l03 * l03 - l13 * l13) / (4.0 * A1);
		const double w[4] = { cot02 + cot03, cot12 + cot13, -(cot02 + cot12), -(cot03 + cot13) };
		V3 s = { 0, 0, 0 };
		const V3 *xs[4] = { &x0, &x1, &x2, &x3 };
		for (int i = 0; i < 4; ++i) {
			s = { s[0] + w[i] * (*xs[i])[0], s[1] + w[i] * (*xs[i])[1], s[2] + w[i] * (*xs[i])[2] };
		}
		m.bendIdx.push_back(p0);
		m.bendIdx.push_back(p1);
		m.bendIdx.push_back(p2);
		m.bendIdx.push_back(p3);
		for (int i = 0; i < 4; ++i) {
			m.bendWeight.push_back(float(w[i]));
		}
		m.bendNTarget.push_back(float(norm(s)));
	}

	// Pin the whole far edge (iy = ny-1) so the panel hangs from it like a
	// curtain and swings down under gravity.
	for (uint32_t ix = 0; ix < nx; ++ix) {
		const uint32_t v = vid(ix, ny - 1);
		m.attachVert.push_back(v);
		m.attachFixed.push_back(m.positions[3 * v + 0]);
		m.attachFixed.push_back(m.positions[3 * v + 1]);
		m.attachFixed.push_back(m.positions[3 * v + 2]);
	}
	return m;
}

ClothMesh combine_meshes(const std::vector<ClothMesh> &parts) {
	ClothMesh m;
	uint32_t voff = 0;
	for (const ClothMesh &p : parts) {
		m.positions.insert(m.positions.end(), p.positions.begin(), p.positions.end());
		m.mass.insert(m.mass.end(), p.mass.begin(), p.mass.end());
		for (uint32_t v : p.triIdx) {
			m.triIdx.push_back(v + voff);
		}
		m.triInvUV.insert(m.triInvUV.end(), p.triInvUV.begin(), p.triInvUV.end());
		for (uint32_t v : p.bendIdx) {
			m.bendIdx.push_back(v + voff);
		}
		m.bendWeight.insert(m.bendWeight.end(), p.bendWeight.begin(), p.bendWeight.end());
		m.bendNTarget.insert(m.bendNTarget.end(), p.bendNTarget.begin(), p.bendNTarget.end());
		for (uint32_t v : p.attachVert) {
			m.attachVert.push_back(v + voff);
		}
		m.attachFixed.insert(m.attachFixed.end(), p.attachFixed.begin(), p.attachFixed.end());
		for (uint32_t v : p.faces) {
			m.faces.push_back(v + voff);
		}
		voff += p.nVerts();
	}
	return m;
}

namespace {
uint32_t uf_find(std::vector<uint32_t> &parent, uint32_t x) {
	while (parent[x] != x) {
		parent[x] = parent[parent[x]];
		x = parent[x];
	}
	return x;
}
void uf_union(std::vector<uint32_t> &parent, uint32_t a, uint32_t b) {
	parent[uf_find(parent, a)] = uf_find(parent, b);
}
} // namespace

std::vector<ClothMesh> partition_islands(const ClothMesh &mesh) {
	const uint32_t nV = mesh.nVerts();
	std::vector<uint32_t> parent(nV);
	for (uint32_t v = 0; v < nV; ++v) {
		parent[v] = v;
	}
	for (uint32_t t = 0; t < mesh.nTri(); ++t) {
		uf_union(parent, mesh.triIdx[3*t], mesh.triIdx[3*t+1]);
		uf_union(parent, mesh.triIdx[3*t], mesh.triIdx[3*t+2]);
	}
	for (uint32_t b = 0; b < mesh.nBend(); ++b) {
		for (int r = 1; r < 4; ++r) {
			uf_union(parent, mesh.bendIdx[4*b], mesh.bendIdx[4*b+r]);
		}
	}

	// Assign each root a dense island id, then bucket vertices.
	std::vector<uint32_t> islandOf(nV), localOf(nV);
	std::map<uint32_t, uint32_t> rootToIsland;
	uint32_t nIsl = 0;
	for (uint32_t v = 0; v < nV; ++v) {
		const uint32_t r = uf_find(parent, v);
		auto it = rootToIsland.find(r);
		if (it == rootToIsland.end()) { rootToIsland[r] = nIsl; islandOf[v] = nIsl; ++nIsl; }
		else {
			islandOf[v] = it->second;
		}
	}
	std::vector<ClothMesh> out(nIsl);
	for (uint32_t v = 0; v < nV; ++v) {
		ClothMesh &im = out[islandOf[v]];
		localOf[v] = im.nVerts();
		im.positions.push_back(mesh.positions[3*v+0]);
		im.positions.push_back(mesh.positions[3*v+1]);
		im.positions.push_back(mesh.positions[3*v+2]);
		im.mass.push_back(mesh.mass[v]);
	}
	for (uint32_t t = 0; t < mesh.nTri(); ++t) {
		ClothMesh &im = out[islandOf[mesh.triIdx[3*t]]];
		for (int r = 0; r < 3; ++r) {
			im.triIdx.push_back(localOf[mesh.triIdx[3*t+r]]);
		}
		for (int j = 0; j < 4; ++j) {
			im.triInvUV.push_back(mesh.triInvUV[4*t+j]);
		}
	}
	for (uint32_t b = 0; b < mesh.nBend(); ++b) {
		ClothMesh &im = out[islandOf[mesh.bendIdx[4*b]]];
		for (int r = 0; r < 4; ++r) {
			im.bendIdx.push_back(localOf[mesh.bendIdx[4*b+r]]);
		}
		for (int r = 0; r < 4; ++r) {
			im.bendWeight.push_back(mesh.bendWeight[4*b+r]);
		}
		im.bendNTarget.push_back(mesh.bendNTarget[b]);
	}
	for (uint32_t a = 0; a < mesh.nAttach(); ++a) {
		const uint32_t v = mesh.attachVert[a];
		ClothMesh &im = out[islandOf[v]];
		im.attachVert.push_back(localOf[v]);
		for (int j = 0; j < 3; ++j) {
			im.attachFixed.push_back(mesh.attachFixed[3*a+j]);
		}
	}
	for (ClothMesh &im : out) {
		im.faces = im.triIdx;
	}
	return out;
}

std::vector<PanelTile> split_connected_panel(const ClothMesh &mesh, uint32_t nTiles) {
	const uint32_t nV = mesh.nVerts();
	float xmin = mesh.positions[0];
	float xmax = mesh.positions[0];
	for (uint32_t v = 0; v < nV; ++v) {
		const float x = mesh.positions[3 * v + 0];
		if (x < xmin) {
			xmin = x;
		}
		if (x > xmax) {
			xmax = x;
		}
	}
	const float span = (xmax > xmin) ? (xmax - xmin) : 1.0f;

	std::vector<uint32_t> tileOf(nV, 0);
	for (uint32_t v = 0; v < nV; ++v) {
		float f = (mesh.positions[3 * v + 0] - xmin) / span * float(nTiles);
		uint32_t t = uint32_t(f);
		if (t >= nTiles) {
			t = nTiles - 1;
		}
		tileOf[v] = t;
	}

	// Which vertices each tile touches: owned first, then ghosts (verts a
	// constraint pulls in from a neighbour band).
	std::vector<std::vector<uint32_t>> owned(nTiles), ghost(nTiles);
	std::vector<std::vector<char>> seen(nTiles, std::vector<char>(nV, 0));
	auto touch = [&](uint32_t t, uint32_t v) {
		if (seen[t][v]) {
			return;
		}
		seen[t][v] = 1;
		if (tileOf[v] == t) {
			owned[t].push_back(v);
		} else {
			ghost[t].push_back(v);
		}
	};
	for (uint32_t v = 0; v < nV; ++v) {
		touch(tileOf[v], v);
	}
	auto touchConstraint = [&](const uint32_t *ids, uint32_t k) {
		uint32_t bands[8];
		uint32_t nb = 0;
		for (uint32_t r = 0; r < k; ++r) {
			const uint32_t t = tileOf[ids[r]];
			bool have = false;
			for (uint32_t j = 0; j < nb; ++j) {
				if (bands[j] == t) {
					have = true;
				}
			}
			if (!have) {
				bands[nb++] = t;
			}
		}
		for (uint32_t j = 0; j < nb; ++j) {
			for (uint32_t r = 0; r < k; ++r) {
				touch(bands[j], ids[r]);
			}
		}
	};
	for (uint32_t t = 0; t < mesh.nTri(); ++t) {
		touchConstraint(&mesh.triIdx[3 * t], 3);
	}
	for (uint32_t b = 0; b < mesh.nBend(); ++b) {
		touchConstraint(&mesh.bendIdx[4 * b], 4);
	}

	std::vector<PanelTile> tiles(nTiles);
	for (uint32_t t = 0; t < nTiles; ++t) {
		PanelTile &pt = tiles[t];
		pt.nOwned = uint32_t(owned[t].size());
		std::vector<int32_t> localOf(nV, -1);
		auto add = [&](uint32_t g) {
			localOf[g] = int32_t(pt.globalId.size());
			pt.globalId.push_back(g);
			pt.mesh.positions.push_back(mesh.positions[3 * g + 0]);
			pt.mesh.positions.push_back(mesh.positions[3 * g + 1]);
			pt.mesh.positions.push_back(mesh.positions[3 * g + 2]);
			pt.mesh.mass.push_back(mesh.mass[g]);
		};
		for (uint32_t g : owned[t]) {
			add(g);
		}
		for (uint32_t g : ghost[t]) {
			add(g);
		}
		auto inTile = [&](const uint32_t *ids, uint32_t k) {
			for (uint32_t r = 0; r < k; ++r) {
				if (tileOf[ids[r]] == t) {
					return true;
				}
			}
			return false;
		};
		for (uint32_t ti = 0; ti < mesh.nTri(); ++ti) {
			if (!inTile(&mesh.triIdx[3 * ti], 3)) {
				continue;
			}
			for (int r = 0; r < 3; ++r) {
				pt.mesh.triIdx.push_back(uint32_t(localOf[mesh.triIdx[3 * ti + r]]));
			}
			for (int j = 0; j < 4; ++j) {
				pt.mesh.triInvUV.push_back(mesh.triInvUV[4 * ti + j]);
			}
		}
		for (uint32_t b = 0; b < mesh.nBend(); ++b) {
			if (!inTile(&mesh.bendIdx[4 * b], 4)) {
				continue;
			}
			for (int r = 0; r < 4; ++r) {
				pt.mesh.bendIdx.push_back(uint32_t(localOf[mesh.bendIdx[4 * b + r]]));
			}
			for (int r = 0; r < 4; ++r) {
				pt.mesh.bendWeight.push_back(mesh.bendWeight[4 * b + r]);
			}
			pt.mesh.bendNTarget.push_back(mesh.bendNTarget[b]);
		}
		for (uint32_t a = 0; a < mesh.nAttach(); ++a) {
			const uint32_t g = mesh.attachVert[a];
			if (tileOf[g] != t) {
				continue;
			}
			pt.mesh.attachVert.push_back(uint32_t(localOf[g]));
			for (int j = 0; j < 3; ++j) {
				pt.mesh.attachFixed.push_back(mesh.attachFixed[3 * a + j]);
			}
		}
		pt.mesh.faces = pt.mesh.triIdx;
	}
	return tiles;
}
