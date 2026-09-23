// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "body_mesh.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace {

double box_dist2(const v3d &lo, const v3d &hi, const v3d &p) {
	double d2 = 0.0;
	for (int k = 0; k < 3; ++k) {
		const double e = p[k] < lo[k] ? lo[k] - p[k] : (p[k] > hi[k] ? p[k] - hi[k] : 0.0);
		d2 += e * e;
	}
	return d2;
}

double angle_at(const v3d &o, const v3d &a, const v3d &b) {
	const v3d u = (a - o).normalized(), w = (b - o).normalized();
	return std::acos(std::max(-1.0, std::min(1.0, u.dot(w))));
}

} // namespace

bool BodyMesh::build(const std::vector<float> &pos, const std::vector<int32_t> &tris, double scale, std::string &err) {
	if (pos.size() < 9 || pos.size() % 3 != 0 || tris.size() < 3 || tris.size() % 3 != 0) {
		err = "body mesh: 3 floats per vertex and 3 indices per triangle, at least one triangle";
		return false;
	}
	const uint32_t nV = uint32_t(pos.size() / 3);
	v.resize(nV);
	for (uint32_t i = 0; i < nV; ++i) {
		v[i] = v3d(double(pos[3 * i]) * scale, double(pos[3 * i + 1]) * scale, double(pos[3 * i + 2]) * scale);
	}
	t.clear();
	for (int32_t x : tris) {
		if (x < 0 || uint32_t(x) >= nV) {
			err = "body mesh: triangle index out of range";
			return false;
		}
		t.push_back(uint32_t(x));
	}
	const uint32_t nT = nTri();
	faceN_.assign(nT, v3d());
	vertN_.assign(nV, v3d());
	edgeN_.assign(size_t(3) * nT, v3d());
	std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edges;
	for (uint32_t f = 0; f < nT; ++f) {
		const v3d &a = v[t[3 * f]], &b = v[t[3 * f + 1]], &c = v[t[3 * f + 2]];
		const v3d n = (b - a).cross(c - a).normalized();
		faceN_[f] = n;
		vertN_[t[3 * f]] += n * angle_at(a, b, c);
		vertN_[t[3 * f + 1]] += n * angle_at(b, c, a);
		vertN_[t[3 * f + 2]] += n * angle_at(c, a, b);
		for (int e = 0; e < 3; ++e) {
			const uint32_t i = t[3 * f + e], j = t[3 * f + (e + 1) % 3];
			edges[{ std::min(i, j), std::max(i, j) }].push_back(f);
		}
	}
	for (v3d &n : vertN_) {
		n = n.normalized();
	}
	for (uint32_t f = 0; f < nT; ++f) {
		for (int e = 0; e < 3; ++e) {
			const uint32_t i = t[3 * f + e], j = t[3 * f + (e + 1) % 3];
			v3d s;
			for (uint32_t g : edges[{ std::min(i, j), std::max(i, j) }]) {
				s += faceN_[g];
			}
			edgeN_[3 * f + e] = s.normalized();
		}
	}
	std::vector<v3d> cent(nT);
	order_.resize(nT);
	for (uint32_t f = 0; f < nT; ++f) {
		cent[f] = (v[t[3 * f]] + v[t[3 * f + 1]] + v[t[3 * f + 2]]) / 3.0;
		order_[f] = f;
	}
	nodes_.clear();
	nodes_.reserve(size_t(2) * nT / 2 + 1);
	buildNode(0, nT, cent);
	return true;
}

uint32_t BodyMesh::buildNode(uint32_t start, uint32_t count, const std::vector<v3d> &cent) {
	const uint32_t id = uint32_t(nodes_.size());
	nodes_.push_back(Node());
	v3d lo(1e300, 1e300, 1e300), hi(-1e300, -1e300, -1e300), clo = lo, chi = hi;
	for (uint32_t k = start; k < start + count; ++k) {
		const uint32_t f = order_[k];
		for (int r = 0; r < 3; ++r) {
			const v3d &p = v[t[3 * f + r]];
			for (int a = 0; a < 3; ++a) {
				lo[a] = std::min(lo[a], p[a]);
				hi[a] = std::max(hi[a], p[a]);
			}
		}
		for (int a = 0; a < 3; ++a) {
			clo[a] = std::min(clo[a], cent[f][a]);
			chi[a] = std::max(chi[a], cent[f][a]);
		}
	}
	nodes_[id].lo = lo;
	nodes_[id].hi = hi;
	if (count <= 4) {
		nodes_[id].start = start;
		nodes_[id].count = count;
		return id;
	}
	int axis = 0;
	for (int a = 1; a < 3; ++a) {
		if (chi[a] - clo[a] > chi[axis] - clo[axis]) {
			axis = a;
		}
	}
	const uint32_t mid = start + count / 2;
	std::nth_element(order_.begin() + start, order_.begin() + mid, order_.begin() + start + count,
			[&](uint32_t x, uint32_t y) { return cent[x][axis] < cent[y][axis]; });
	const uint32_t l = buildNode(start, mid - start, cent);
	const uint32_t r = buildNode(mid, start + count - mid, cent);
	nodes_[id].left = l;
	nodes_[id].right = r;
	return id;
}

void BodyMesh::closestInTri(uint32_t f, const v3d &p, Hit &h) const {
	const v3d &a = v[t[3 * f]], &b = v[t[3 * f + 1]], &c = v[t[3 * f + 2]];
	const v3d ab = b - a, ac = c - a, ap = p - a;
	Hit o;
	o.tri = f;
	const double d1 = ab.dot(ap), d2 = ac.dot(ap);
	const v3d bp = p - b;
	const double d3 = ab.dot(bp), d4 = ac.dot(bp);
	const v3d cp = p - c;
	const double d5 = ab.dot(cp), d6 = ac.dot(cp);
	const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
	if (d1 <= 0 && d2 <= 0) {
		o.point = a;
		o.region = Region::Vertex;
		o.sub = 0;
	} else if (d3 >= 0 && d4 <= d3) {
		o.point = b;
		o.region = Region::Vertex;
		o.sub = 1;
	} else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
		o.point = a + ab * (d1 / (d1 - d3));
		o.region = Region::Edge;
		o.sub = 0;
		o.edgeDir = ab.normalized();
	} else if (d6 >= 0 && d5 <= d6) {
		o.point = c;
		o.region = Region::Vertex;
		o.sub = 2;
	} else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
		o.point = a + ac * (d2 / (d2 - d6));
		o.region = Region::Edge;
		o.sub = 2;
		o.edgeDir = ac.normalized();
	} else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
		o.point = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
		o.region = Region::Edge;
		o.sub = 1;
		o.edgeDir = (c - b).normalized();
	} else {
		const double den = 1.0 / (va + vb + vc);
		o.point = a + ab * (vb * den) + ac * (vc * den);
		o.region = Region::Face;
	}
	o.dist2 = (p - o.point).squaredNorm();
	if (!(o.dist2 < h.dist2)) {
		return;
	}
	switch (o.region) {
		case Region::Face:
			o.pseudo = faceN_[f];
			break;
		case Region::Edge:
			o.pseudo = edgeN_[3 * f + o.sub];
			break;
		case Region::Vertex:
			o.pseudo = vertN_[t[3 * f + o.sub]];
			break;
	}
	h = o;
}

bool BodyMesh::closest(const v3d &p, double maxDist2, Hit &h) const {
	h = Hit();
	h.dist2 = maxDist2;
	if (nodes_.empty()) {
		return false;
	}
	bool found = false;
	uint32_t stack[64];
	int sp = 0;
	stack[sp++] = 0;
	while (sp > 0) {
		const Node &n = nodes_[stack[--sp]];
		if (!(box_dist2(n.lo, n.hi, p) < h.dist2)) {
			continue;
		}
		if (n.left == 0) {
			for (uint32_t k = n.start; k < n.start + n.count; ++k) {
				const double before = h.dist2;
				closestInTri(order_[k], p, h);
				found = found || h.dist2 < before;
			}
			continue;
		}
		// Nearer child last, so it is popped first.
		const double dl = box_dist2(nodes_[n.left].lo, nodes_[n.left].hi, p);
		const double dr = box_dist2(nodes_[n.right].lo, nodes_[n.right].hi, p);
		if (sp + 2 > 64) {
			break;
		}
		if (dl < dr) {
			stack[sp++] = n.right;
			stack[sp++] = n.left;
		} else {
			stack[sp++] = n.left;
			stack[sp++] = n.right;
		}
	}
	return found;
}
