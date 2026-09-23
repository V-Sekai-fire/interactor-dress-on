// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "drape_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>

namespace {

v3d restd(const DrapeScene &s, uint32_t i) {
	return { double(s.rest[3 * i + 0]), double(s.rest[3 * i + 1]), double(s.rest[3 * i + 2]) };
}

// axisToRotation(finalDir, initialDir) (UtilityFunctions.h:62-73) as Eigen's
// AngleAxis::toRotationMatrix builds it; row-major 3x3.
void axis_to_rotation(v3d finalDir, v3d initialDir, double R[9]) {
	for (int i = 0; i < 9; ++i) {
		R[i] = (i % 4 == 0) ? 1.0 : 0.0;
	}
	finalDir = finalDir.normalized();
	initialDir = initialDir.normalized();
	if ((finalDir - initialDir).norm() > 1e-5) {
		const v3d a = initialDir.cross(finalDir).normalized();
		const double angle = std::acos(finalDir.dot(initialDir));
		const v3d sa = a * std::sin(angle);
		const double c = std::cos(angle);
		const v3d ca = a * (1.0 - c);
		double tmp = ca.x * a.y;
		R[1] = tmp - sa.z;
		R[3] = tmp + sa.z;
		tmp = ca.x * a.z;
		R[2] = tmp + sa.y;
		R[6] = tmp - sa.y;
		tmp = ca.y * a.z;
		R[5] = tmp - sa.x;
		R[7] = tmp + sa.x;
		R[0] = ca.x * a.x + c;
		R[4] = ca.y * a.y + c;
		R[8] = ca.z * a.z + c;
	}
}

} // namespace

void DrapeScene::applyMaterial(const DrapeConfig &cfg) {
	// updateMassMatrix: m = area * density.
	massD.resize(nV);
	mass.resize(nV);
	for (uint32_t i = 0; i < nV; ++i) {
		massD[i] = vertArea[i] * cfg.density;
		mass[i] = float(massD[i]);
	}
	// constrainWeightSqrt^2, as addAvbdConstraint uploads it.
	triK.resize(nTri());
	for (uint32_t t = 0; t < nTri(); ++t) {
		const double c = std::sqrt(triArea[t] * cfg.kTri);
		triK[t] = cfg.rawStiffness ? float(cfg.kTri) : float(c * c);
	}
	bendK.resize(nBend());
	for (uint32_t b = 0; b < nBend(); ++b) {
		const double c = std::sqrt(cfg.kBend * 3.0 / bendAreaSum[b]);
		bendK[b] = cfg.rawStiffness ? float(cfg.kBend) : float(c * c);
	}
	// Pins at kAttach (AttachmentSpring::k_stiff); the fit set at fitK times
	// the vertex's lumped area (updateAreaMatrix's third of each incident
	// triangle), so the pull is area-weighted like cloth-fit's fit term
	// (FitForm::value_unweighted: fit_weight * area(f) * sum_i w_i sdf(p_i)^2)
	// and a mesh's refinement does not change the balance against the
	// membrane, whose stiffness is kTri * area too.
	attachK.assign(nAttach(), float(cfg.kAttach));
	for (uint32_t a = nPin(); a < nPin() + nFit; ++a) {
		attachK[a] = float(fitK * vertArea[attachVert[a]]);
	}
	for (uint32_t a = nPin() + nFit; a < nAttach(); ++a) {
		attachK[a] = float(anchorK);
	}
}

std::string DrapeScene::describe() const {
	char b[256];
	std::snprintf(b, sizeof b, "[drape-scene] %s nV=%u nTri=%u nBend=%u nAttach=%u prims=%zu", name.c_str(), nV,
			nTri(), nBend(), nAttach(), prims.size());
	std::string out = b;
	for (size_t i = 0; i < prims.size(); ++i) {
		out += " | prim" + std::to_string(i) + " " + prims[i].describe();
	}
	return out;
}

void scene_finish(DrapeScene &s) {
	const uint32_t nT = uint32_t(s.tri.size() / 3);
	s.triInvUV.assign(size_t(4) * nT, 0.0f);
	s.triArea.assign(nT, 0.0);
	// Triangle::Triangle (Triangle.cpp:498-511).
	for (uint32_t t = 0; t < nT; ++t) {
		const v3d p0 = restd(s, s.tri[3 * t + 0]);
		const v3d p1 = restd(s, s.tri[3 * t + 1]);
		const v3d p2 = restd(s, s.tri[3 * t + 2]);
		const v3d e0 = p1 - p0, e1 = p2 - p0;
		const v3d P0 = e0.normalized();
		const v3d P1 = (e1 - P0 * e1.dot(P0)).normalized();
		const double d00 = P0.dot(e0), d01 = P0.dot(e1), d10 = P1.dot(e0), d11 = P1.dot(e1);
		// Eigen's 2x2 inverse: det, its reciprocal, then products.
		const double det = d00 * d11 - d10 * d01;
		const double inv = 1.0 / det;
		s.triInvUV[4 * t + 0] = float(d11 * inv);
		s.triInvUV[4 * t + 1] = float(-d01 * inv);
		s.triInvUV[4 * t + 2] = float(-d10 * inv);
		s.triInvUV[4 * t + 3] = float(d00 * inv);
		s.triArea[t] = std::abs(det * 0.5);
	}
	// updateAreaMatrix.
	s.vertArea.assign(s.nV, 0.0);
	for (uint32_t t = 0; t < nT; ++t) {
		const double a = s.triArea[t] / 3.0;
		for (int r = 0; r < 3; ++r) {
			s.vertArea[s.tri[3 * t + r]] += a;
		}
	}
	// createBendingConstraints: edge (min, max) -> the opposite corners, in
	// triangle order; std::map iteration order is the bending order.
	std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edges;
	for (uint32_t t = 0; t < nT; ++t) {
		const uint32_t id[3] = { s.tri[3 * t], s.tri[3 * t + 1], s.tri[3 * t + 2] };
		for (int a = 0; a < 3; ++a) {
			for (int b = a + 1; b < 3; ++b) {
				const uint32_t lo = std::min(id[a], id[b]), hi = std::max(id[a], id[b]);
				edges[{ lo, hi }].push_back(id[3 - (a + b)]);
			}
		}
	}
	s.bendIdx.clear();
	s.bendW.clear();
	s.bendN.clear();
	s.bendAreaSum.clear();
	for (const auto &kv : edges) {
		if (kv.second.size() < 2) {
			continue;
		}
		// Upstream exits on a non-manifold edge; the first two are used here.
		const uint32_t idx[4] = { kv.first.first, kv.first.second, kv.second[0], kv.second[1] };
		v3d pos[4];
		for (int i = 0; i < 4; ++i) {
			pos[i] = restd(s, idx[i]);
		}
		const double l01 = (pos[1] - pos[0]).norm();
		const double l02 = (pos[2] - pos[0]).norm();
		const double l03 = (pos[3] - pos[0]).norm();
		const double l12 = (pos[1] - pos[2]).norm();
		const double l13 = (pos[1] - pos[3]).norm();
		const double r0 = 0.5 * (l01 + l02 + l12);
		const double A0 = std::sqrt(r0 * (r0 - l01) * (r0 - l02) * (r0 - l12));
		const double r1 = 0.5 * (l01 + l13 + l03);
		const double A1 = std::sqrt(r1 * (r1 - l01) * (r1 - l03) * (r1 - l13));
		const double cot02 = ((l01 * l01) - (l02 * l02) + (l12 * l12)) / (4.0 * A0);
		const double cot12 = ((l01 * l01) + (l02 * l02) - (l12 * l12)) / (4.0 * A0);
		const double cot03 = ((l01 * l01) - (l03 * l03) + (l13 * l13)) / (4.0 * A1);
		const double cot13 = ((l01 * l01) + (l03 * l03) - (l13 * l13)) / (4.0 * A1);
		const float w[4] = { float(cot02 + cot03), float(cot12 + cot13), float(-(cot02 + cot12)),
			float(-(cot03 + cot13)) };
		v3d sum;
		for (int i = 0; i < 4; ++i) {
			sum += pos[i] * double(w[i]);
		}
		for (int i = 0; i < 4; ++i) {
			s.bendIdx.push_back(idx[i]);
			s.bendW.push_back(w[i]);
		}
		s.bendN.push_back(float(sum.norm()));
		s.bendAreaSum.push_back(A0 + A1);
	}
	// updateCollisionRadii (Simulation.cpp:3015-3040), its neighbour pick
	// verbatim: p2 = p0 unless that is the vertex (then p2), p3 = p1 likewise.
	std::vector<std::vector<uint32_t>> vt(s.nV);
	for (uint32_t t = 0; t < nT; ++t) {
		for (int r = 0; r < 3; ++r) {
			vt[s.tri[3 * t + r]].push_back(t);
		}
	}
	s.radii.assign(s.nV, 0.0);
	for (uint32_t i = 0; i < s.nV; ++i) {
		const v3d p = restd(s, i);
		double minEdge = 100;
		for (uint32_t t : vt[i]) {
			uint32_t p2 = s.tri[3 * t + 0], p3 = s.tri[3 * t + 1];
			if (p2 == i) {
				p2 = s.tri[3 * t + 2];
			}
			if (p3 == i) {
				p3 = s.tri[3 * t + 2];
			}
			minEdge = std::min(minEdge, (restd(s, p2) - p).norm());
			minEdge = std::min(minEdge, (restd(s, p3) - p).norm());
		}
		s.radii[i] = minEdge / 2.0 - 0.01;
	}
}

void scene_grid(DrapeScene &s, int gridNumX, int gridNumY, double clothDimX, double clothDimY, GridOrientation o,
		double restMin[3], double restMax[3]) {
	s = DrapeScene();
	const double gridSizeX = clothDimX / (gridNumX - 1);
	const double gridSizeY = clothDimY / (gridNumY - 1);
	// getInitParticlePos(i, j).
	std::vector<v3d> pin;
	for (int i = 0; i < gridNumY; ++i) {
		for (int j = 0; j < gridNumX; ++j) {
			const v3d origin(-(gridNumY - 1) / 4.0 * gridSizeY, 15, 0);
			pin.push_back(v3d(j * gridSizeY, -i * gridSizeX, 0) + origin);
		}
	}
	// rotatePointsAccordingToConfig -> rotatePointsAroundCenter.
	if (o == GridOrientation::Down) {
		double R[9];
		axis_to_rotation(v3d(0, 1, 0), v3d(0, 0, 1), R);
		v3d mn = pin[0];
		for (const v3d &p : pin) {
			for (int k = 0; k < 3; ++k) {
				mn[k] = std::min(mn[k], p[k]);
			}
		}
		for (v3d &p : pin) {
			const v3d q = p - mn;
			p = v3d(R[0] * q.x + R[1] * q.y + R[2] * q.z, R[3] * q.x + R[4] * q.y + R[5] * q.z,
					R[6] * q.x + R[7] * q.y + R[8] * q.z);
		}
	}
	v3d mn = pin[0], mx = pin[0];
	for (const v3d &p : pin) {
		for (int k = 0; k < 3; ++k) {
			mn[k] = std::min(mn[k], p[k]);
			mx[k] = std::max(mx[k], p[k]);
		}
	}
	const v3d dim = mx - mn;
	for (v3d &p : pin) {
		p -= mn;
		p -= dim / 2;
	}
	for (int k = 0; k < 3; ++k) {
		restMax[k] = dim[k] - dim[k] / 2.0;
		restMin[k] = 0.0 - dim[k] / 2.0;
	}
	s.nV = uint32_t(pin.size());
	s.rest.resize(3 * s.nV);
	for (uint32_t i = 0; i < s.nV; ++i) {
		for (int k = 0; k < 3; ++k) {
			s.rest[3 * i + k] = float(pin[i][k]);
		}
	}
	s.x0 = s.rest;
	s.v0.assign(3 * s.nV, 0.0f);
	auto gid = [&](int a, int b) -> int {
		if (a < 0 || b < 0 || a >= gridNumY || b >= gridNumX) {
			return -1;
		}
		return a * gridNumX + b;
	};
	auto createTriangle = [&](int a, int b, int c) {
		if (a < 0 || b < 0 || c < 0) {
			return;
		}
		s.tri.push_back(uint32_t(c));
		s.tri.push_back(uint32_t(b));
		s.tri.push_back(uint32_t(a));
	};
	for (int i = 0; i < gridNumY; ++i) {
		for (int j = 0; j < gridNumX; ++j) {
			const int thisIdx = gid(i, j), leftIdx = gid(i, j - 1), upIdx = gid(i - 1, j),
					  upRightIdx = gid(i - 1, j + 1);
			createTriangle(thisIdx, upIdx, upRightIdx);
			createTriangle(upIdx, thisIdx, leftIdx);
		}
	}
	scene_finish(s);
}

void scene_sphere_demo(DrapeScene &s, const DrapeConfig &cfg) {
	double restMin[3], restMax[3];
	scene_grid(s, 25, 25, 4.5, 4.5, GridOrientation::Down, restMin, restMax);
	s.name = "sphere_demo";
	// initScene PLANE_AND_SPHERE: centerLowPoint, plane1, sphere2 (radius 2).
	v3d centerLowPoint((restMin[0] + restMax[0]) * 0.5, (restMin[1] + restMax[1]) * 0.5,
			(restMin[2] + restMax[2]) * 0.5);
	centerLowPoint[1] = restMin[1];
	const double radius = 2.0;
	const v3d plane1 = centerLowPoint - v3d(0, radius * 2 + 0.1, 0);
	const v3d center = plane1 + v3d(radius * 0.3, radius, radius * 0.1);
	s.prims.clear();
	s.prims.push_back(make_sphere(center, radius, cfg.mu));
	s.applyMaterial(cfg);
}

bool scene_mesh(DrapeScene &s, const std::vector<float> &pos, const std::vector<int32_t> &tris,
		const std::vector<int32_t> &pins, std::string &err) {
	if (pos.size() < 9 || pos.size() % 3 != 0) {
		err = "positions must be 3 floats per vertex, at least 3 vertices";
		return false;
	}
	if (tris.empty() || tris.size() % 3 != 0) {
		err = "triangles must be 3 indices each";
		return false;
	}
	const uint32_t nV = uint32_t(pos.size() / 3);
	for (int32_t t : tris) {
		if (t < 0 || uint32_t(t) >= nV) {
			err = "triangle index out of range";
			return false;
		}
	}
	for (int32_t p : pins) {
		if (p < 0 || uint32_t(p) >= nV) {
			err = "pin index out of range";
			return false;
		}
	}
	for (float f : pos) {
		if (!std::isfinite(f)) {
			err = "non-finite position";
			return false;
		}
	}
	std::vector<Primitive> keep = s.prims;
	s = DrapeScene();
	s.name = "mesh";
	s.prims = keep;
	s.nV = nV;
	s.rest = pos;
	s.x0 = pos;
	s.v0.assign(pos.size(), 0.0f);
	for (int32_t t : tris) {
		s.tri.push_back(uint32_t(t));
	}
	scene_finish(s);
	for (int32_t p : pins) {
		s.attachVert.push_back(uint32_t(p));
		for (int k = 0; k < 3; ++k) {
			s.attachFixed.push_back(s.rest[3 * p + k]);
		}
	}
	return true;
}

bool scene_fit_set(DrapeScene &s, const std::vector<int32_t> &verts, double k, double gap, int refresh, bool similarity,
		const std::vector<int32_t> &anchor, int restEvery, int settle, double anchorK, std::string &err) {
	if (s.nV == 0) {
		err = "no scene";
		return false;
	}
	if (refresh < 1 || restEvery < 1 || settle < 0) {
		err = "refresh and restEvery must be >= 1, settle >= 0";
		return false;
	}
	for (int32_t v : verts) {
		if (v < 0 || uint32_t(v) >= s.nV) {
			err = "fit vertex index out of range";
			return false;
		}
	}
	for (int32_t v : anchor) {
		if (v < 0 || uint32_t(v) >= s.nV) {
			err = "anchor vertex index out of range";
			return false;
		}
	}
	s.fitAnchor.assign(anchor.begin(), anchor.end());
	const uint32_t nPin = s.nPin();
	s.attachVert.resize(nPin);
	s.attachFixed.resize(size_t(3) * nPin);
	for (int32_t v : verts) {
		s.attachVert.push_back(uint32_t(v));
		for (int c = 0; c < 3; ++c) {
			s.attachFixed.push_back(s.rest[3 * v + c]);
		}
	}
	s.nFit = uint32_t(verts.size());
	for (int32_t v : anchor) {
		s.attachVert.push_back(uint32_t(v));
		for (int c = 0; c < 3; ++c) {
			s.attachFixed.push_back(s.rest[3 * v + c]);
		}
	}
	s.nAnchorAtt = uint32_t(anchor.size());
	s.anchorK = anchorK;
	s.fitK = k;
	s.fitGap = gap;
	s.fitRefresh = refresh;
	s.fitSimilarity = similarity;
	s.fitRestEvery = restEvery;
	s.fitSettle = settle;
	return true;
}
