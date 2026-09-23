// body_mesh -- a triangle-mesh collider for the drape: the body itself in
// place of capsules along its bones.
//
// Not in upstream DiffCloth (whose primitives are analytic); driver code like
// primitives.cpp, on the CPU in double. A static triangle soup with a
// bounding-volume hierarchy (median split on the longest centroid axis, four
// triangles a leaf) answers the closest surface point to a query point
// (Ericson, Real-Time Collision Detection 5.1.5, with the Voronoi region it
// falls in). The sign of the distance comes from the angle-weighted
// pseudo-normal of that region -- the face normal, the edge's two faces, or
// the vertex's incident faces weighted by their angles (Baerentzen & Aanaes
// 2005) -- which is exact for a closed, consistently wound mesh; the body
// from infer / the FoxGirl fixture is one.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "primitives.h"

struct BodyMesh {
	// Region of the closest point: the face, an edge (0 ab, 1 bc, 2 ca), or a
	// corner (0 a, 1 b, 2 c).
	enum class Region : uint8_t { Face, Edge, Vertex };
	struct Hit {
		v3d point;    // closest surface point
		v3d pseudo;   // the region's pseudo-normal (unit, outward)
		double dist2 = 0.0;
		uint32_t tri = 0;
		Region region = Region::Face;
		uint8_t sub = 0;
		v3d edgeDir;  // unit edge direction for Region::Edge
	};

	std::vector<v3d> v;
	std::vector<uint32_t> t; // 3 per triangle
	// false with err on bad input (out-of-range index, too few triangles).
	bool build(const std::vector<float> &pos, const std::vector<int32_t> &tris, double scale, std::string &err);
	// The closest surface point to p within sqrt(maxDist2); false if none.
	bool closest(const v3d &p, double maxDist2, Hit &h) const;
	uint32_t nTri() const { return uint32_t(t.size() / 3); }
	size_t nodes() const { return nodes_.size(); }

private:
	struct Node {
		v3d lo, hi;
		uint32_t left = 0, right = 0; // children, or 0 for a leaf
		uint32_t start = 0, count = 0; // leaf range into order_
	};
	std::vector<Node> nodes_;
	std::vector<uint32_t> order_;
	std::vector<v3d> faceN_;
	std::vector<v3d> vertN_;  // angle-weighted
	std::vector<v3d> edgeN_;  // 3 per triangle: the edge's two face normals summed
	uint32_t buildNode(uint32_t start, uint32_t count, const std::vector<v3d> &cent);
	void closestInTri(uint32_t tri, const v3d &p, Hit &h) const;
};
