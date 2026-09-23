// primitives -- the collision primitives of the drape: Sphere, Plane, Capsule,
// and Mesh (a triangle-mesh body, body_mesh.h; not in upstream).
//
// Ports of DiffCloth's Primitive.cpp (cloth-dynamics-standalone e361584):
// Sphere::isInContact (241-275, COLLISION_EPSILON 0.1), Plane::isInContact
// (95-156, epsilon 0.4, point-in-two-triangles, the near-edge fallback) and
// Capsule::isInContact (576-608, delta 0.1, the +0.1 skin on the body and top
// cap but not the bottom cap, as upstream). No Eigen: a three-double vector.
// The geometry stays in double, as upstream's Vec3d: the per-vertex state it
// is applied to is float (Particle::pos is Vec3f), and the drape rounds the
// same way upstream does (drape_sim.h).
//
// Beyond upstream, each primitive can give the Jacobian of its position
// projection p' = p - dist * normal, which the recompute backward modes chain
// through (drape_backward.h). The normal's own derivative is not carried into
// the velocity response or the friction blend (exact for the plane, whose
// normal is constant).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

struct BodyMesh;

struct v3d {
	double x = 0.0, y = 0.0, z = 0.0;
	v3d() = default;
	v3d(double a, double b, double c) : x(a), y(b), z(c) {}
	double &operator[](int i) { return (&x)[i]; }
	double operator[](int i) const { return (&x)[i]; }
	v3d operator+(const v3d &o) const { return { x + o.x, y + o.y, z + o.z }; }
	v3d operator-(const v3d &o) const { return { x - o.x, y - o.y, z - o.z }; }
	v3d operator-() const { return { -x, -y, -z }; }
	v3d operator*(double s) const { return { x * s, y * s, z * s }; }
	v3d operator/(double s) const { return { x / s, y / s, z / s }; }
	v3d &operator+=(const v3d &o) { x += o.x; y += o.y; z += o.z; return *this; }
	v3d &operator-=(const v3d &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	double dot(const v3d &o) const { return x * o.x + y * o.y + z * o.z; }
	v3d cross(const v3d &o) const { return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x }; }
	double norm() const { return std::sqrt(x * x + y * y + z * z); }
	double squaredNorm() const { return x * x + y * y + z * z; }
	// Eigen's normalized(): a zero vector stays zero.
	v3d normalized() const {
		const double n2 = squaredNorm();
		if (n2 > 0.0) {
			const double n = std::sqrt(n2);
			return { x / n, y / n, z / n };
		}
		return *this;
	}
};
inline v3d operator*(double s, const v3d &v) { return v * s; }

// A 3x3 matrix, row-major, for the projection Jacobians.
struct m3d {
	double m[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	static m3d identity() {
		m3d r;
		r.m[0] = r.m[4] = r.m[8] = 1.0;
		return r;
	}
	static m3d outer(const v3d &a, const v3d &b) {
		m3d r;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				r.m[3 * i + j] = a[i] * b[j];
			}
		}
		return r;
	}
	m3d operator+(const m3d &o) const {
		m3d r;
		for (int i = 0; i < 9; ++i) r.m[i] = m[i] + o.m[i];
		return r;
	}
	m3d operator-(const m3d &o) const {
		m3d r;
		for (int i = 0; i < 9; ++i) r.m[i] = m[i] - o.m[i];
		return r;
	}
	m3d operator*(double s) const {
		m3d r;
		for (int i = 0; i < 9; ++i) r.m[i] = m[i] * s;
		return r;
	}
	v3d mul(const v3d &v) const {
		return { m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
			m[6] * v.x + m[7] * v.y + m[8] * v.z };
	}
	v3d mulT(const v3d &v) const {
		return { m[0] * v.x + m[3] * v.y + m[6] * v.z, m[1] * v.x + m[4] * v.y + m[7] * v.z,
			m[2] * v.x + m[5] * v.y + m[8] * v.z };
	}
};

enum class PrimKind { Sphere, Plane, Capsule, Mesh };

struct Primitive {
	PrimKind kind = PrimKind::Sphere;
	v3d center;          // sphere centre, plane centre, capsule bottom-cap centre
	v3d velocity;        // translation velocity (0 for the static primitives upstream uses)
	double mu = 0.0;     // friction coefficient (clamped to [0,1] where applied)
	bool isStatic = true;
	bool rotates = false; // Sphere: v_out += (0,1,0) x normal * 8
	// Sphere / Capsule.
	double radius = 1.0;
	// Capsule: unit axis from the bottom cap to the top cap, and its length.
	v3d axis{ 0, 1, 0 };
	double length = 1.0;
	// Plane: corners relative to the centre (Plane::Plane), the normal
	// upperRight x upperLeft, the half-thickness below it, the bounding radius.
	v3d upperLeft, upperRight, lowerLeft, lowerRight, planeNormal;
	double planeNormalNorm = 1.0, thickness = 5.0, boundaryRadius = 0.0;
	// Mesh: the body (shared, immutable), the skin its contact surface sits
	// out along the normal (the capsule's +0.1), the contact band (the
	// capsule's delta 0.1), and how deep inside a point is still pushed out.
	std::shared_ptr<const BodyMesh> body;
	double skin = 0.1, band = 0.1, depth = 1.0;

	// Upstream's isInContact(center, pos, velocity, normal, dist, v_out):
	// the signed distance (negative inside), the outward normal, and the
	// primitive's surface velocity at the contact.
	bool isInContact(const v3d &pos, const v3d &vel, v3d &normal, double &dist, v3d &v_out) const;
	// d(pos - dist * normal)/d pos at `pos` (called only where dist < 0).
	m3d projectionJacobian(const v3d &pos) const;
	// Primitive::step for a moving primitive (static ones do not move).
	void step(double h) {
		if (!isStatic) {
			center += velocity * h;
		}
	}
	std::string describe() const;
};

Primitive make_sphere(const v3d &center, double radius, double mu);
// A rectangle given by its centre and two absolute corners, as Plane::Plane.
Primitive make_plane(const v3d &center, const v3d &upperLeft, const v3d &upperRight, double mu);
// A capsule from its bottom-cap centre along `axis` (normalised here).
Primitive make_capsule(const v3d &bottom, const v3d &axis, double radius, double length, double mu);
// A triangle-mesh body (body_mesh.h): contact where the signed distance to
// the surface, less `skin`, is below `band`; points deeper than `depth`
// inside are not seen.
Primitive make_mesh_collider(std::shared_ptr<const BodyMesh> body, double skin, double band, double depth, double mu);
