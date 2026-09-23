// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "primitives.h"

#include <algorithm>
#include <cstdio>

namespace {

// Upstream stores the plane's corner points as Particle::pos, a Vec3f.
v3d f32(const v3d &v) { return { double(float(v.x)), double(float(v.y)), double(float(v.z)) }; }

// Primitive::pointInsideTriangle (Primitive.h:190-208).
bool point_inside_triangle(const v3d &p0, const v3d &p1, const v3d &p2, const v3d &p) {
	const v3d AB = p1 - p0, AC = p2 - p0;
	const v3d n = AB.cross(AC);
	const double n2 = n.squaredNorm();
	const v3d AP = p - p0;
	const double alpha = AB.cross(AP).dot(n) / n2;
	const double beta = AP.cross(AC).dot(n) / n2;
	const double gamma = 1 - alpha - beta;
	return alpha >= 0 && beta >= 0 && gamma >= 0 && gamma <= 1 && alpha <= 1 && beta <= 1;
}

// Primitive::projectionOnLine (Primitive.h:215-228): the foot and a signed t.
v3d projection_on_line(const v3d &a, const v3d &b, const v3d &p, double &t) {
	const v3d AP = p - a, AB = b - a;
	const v3d proj = AB * (AP.dot(AB) / AB.dot(AB));
	const v3d P = a + proj;
	const double AB_l = (b - a).norm();
	const double AP_l = (P - a).norm();
	const double PB_l = (P - b).norm();
	t = AP_l / AB_l;
	if (PB_l > AB_l) {
		t *= -1;
	}
	return P;
}

// d/dp of c + R * (p - c)/|p - c|: (R/|p - c|)(I - n n^T).
m3d radial_jacobian(const v3d &d, double R) {
	const double l = d.norm();
	if (!(l > 0.0)) {
		return m3d::identity();
	}
	const v3d n = d / l;
	return (m3d::identity() - m3d::outer(n, n)) * (R / l);
}

} // namespace

Primitive make_sphere(const v3d &center, double radius, double mu) {
	Primitive p;
	p.kind = PrimKind::Sphere;
	p.center = center;
	p.radius = radius;
	p.mu = mu;
	return p;
}

Primitive make_plane(const v3d &center, const v3d &ul, const v3d &ur, double mu) {
	// Plane::Plane (Primitive.cpp:39-73).
	Primitive p;
	p.kind = PrimKind::Plane;
	p.center = center;
	p.mu = mu;
	p.upperLeft = ul - center;
	p.upperRight = ur - center;
	p.lowerRight = -p.upperLeft;
	p.lowerLeft = -p.upperRight;
	p.boundaryRadius = std::max(p.upperLeft.norm(), p.upperRight.norm());
	p.planeNormal = p.upperRight.cross(p.upperLeft).normalized();
	p.planeNormalNorm = std::sqrt(p.planeNormal.squaredNorm());
	p.thickness = 5.0;
	return p;
}

Primitive make_capsule(const v3d &bottom, const v3d &axis, double radius, double length, double mu) {
	Primitive p;
	p.kind = PrimKind::Capsule;
	p.center = bottom;
	p.axis = axis.normalized();
	p.radius = radius;
	p.length = length;
	p.mu = mu;
	return p;
}

bool Primitive::isInContact(const v3d &pos, const v3d &vel, v3d &normal, double &dist, v3d &v_out) const {
	(void)vel;
	switch (kind) {
		case PrimKind::Sphere: {
			// Sphere::isInContact (Primitive.cpp:241-275), not discretized.
			const double COLLISION_EPSILON = 0.1;
			dist = (pos - center).norm() - radius;
			normal = (pos - center).normalized();
			const bool collides = dist < COLLISION_EPSILON;
			v_out = velocity;
			if (rotates) {
				v_out += v3d(0, 1, 0).cross(normal) * 8.0;
			}
			return collides;
		}
		case PrimKind::Plane: {
			// Plane::isInContact (Primitive.cpp:95-156), d = 0.
			const v3d posShifted = pos - center;
			const double COLLISION_EPSILON = 0.4;
			const double distToCenter = posShifted.norm();
			if (distToCenter > boundaryRadius + COLLISION_EPSILON) {
				return false;
			}
			const double distToPlane = planeNormal.dot(posShifted) / planeNormalNorm;
			dist = distToPlane;
			if (std::abs(distToPlane) > COLLISION_EPSILON) {
				return false;
			}
			if (distToPlane < 0 && distToPlane * -1 > COLLISION_EPSILON + thickness) {
				return false;
			}
			const v3d p_prime = posShifted - planeNormal * planeNormal.dot(posShifted);
			const v3d ul = f32(upperLeft), ur = f32(upperRight), ll = f32(lowerLeft), lr = f32(lowerRight);
			if (point_inside_triangle(ul, ur, ll, p_prime) || point_inside_triangle(ll, ur, lr, p_prime)) {
				normal = planeNormal * (distToPlane < -COLLISION_EPSILON ? -1.0 : 1.0);
				v_out = velocity;
				return true;
			}
			const v3d edges[4][2] = { { upperLeft, upperRight }, { upperRight, lowerRight }, { lowerLeft, lowerRight },
				{ upperLeft, lowerLeft } };
			const double edgeTol = 0.0005;
			for (const auto &e : edges) {
				double t = 0.0;
				const v3d foot = projection_on_line(e[0], e[1], posShifted, t);
				if ((posShifted - foot).norm() < edgeTol && t > -edgeTol && t < 1 + edgeTol) {
					if (t < 0) {
						normal = (posShifted - e[0]).normalized();
					} else if (t > 1) {
						normal = (posShifted - e[1]).normalized();
					} else {
						normal = (posShifted - foot).normalized();
					}
					v_out = velocity;
					return true;
				}
			}
			return false;
		}
		case PrimKind::Capsule: {
			// Capsule::isInContact (Primitive.cpp:576-608).
			const double delta = 0.1;
			const v3d posLocal = pos - center;
			v_out = velocity;
			const v3d bottom(0, 0, 0);
			const v3d top = axis * length;
			double t = 0.0;
			const v3d foot = projection_on_line(bottom, top, posLocal, t);
			if (t < 0 - radius / length || t > 1 + radius / length) {
				return false;
			}
			if (t < 0) {
				dist = posLocal.norm() - radius;
				normal = posLocal.normalized();
			} else if (t > 1) {
				dist = (posLocal - top).norm() - (radius + 0.1);
				normal = (posLocal - top).normalized();
			} else {
				dist = (posLocal - foot).norm() - (radius + 0.1);
				normal = (posLocal - foot).normalized();
			}
			return dist < delta;
		}
	}
	return false;
}

m3d Primitive::projectionJacobian(const v3d &pos) const {
	switch (kind) {
		case PrimKind::Sphere:
			// p' = c + r n.
			return radial_jacobian(pos - center, radius);
		case PrimKind::Plane: {
			// p' = p - (N.p) N on the face; the near-edge branch is a point or
			// line contact, projected like a sphere of radius 0.
			const v3d N = planeNormal / planeNormalNorm;
			return m3d::identity() - m3d::outer(N, N);
		}
		case PrimKind::Capsule: {
			const v3d posLocal = pos - center;
			const v3d top = axis * length;
			double t = 0.0;
			const v3d foot = projection_on_line(v3d(0, 0, 0), top, posLocal, t);
			if (t < 0) {
				return radial_jacobian(posLocal, radius);
			}
			if (t > 1) {
				return radial_jacobian(posLocal - top, radius + 0.1);
			}
			// p' = c + a a^T p_l + R d/|d|, d = (I - a a^T) p_l.
			const v3d d = posLocal - foot;
			const double l = d.norm();
			const m3d aa = m3d::outer(axis, axis);
			if (!(l > 0.0)) {
				return m3d::identity();
			}
			const v3d n = d / l;
			return aa + (m3d::identity() - aa - m3d::outer(n, n)) * ((radius + 0.1) / l);
		}
	}
	return m3d::identity();
}

std::string Primitive::describe() const {
	char b[256];
	switch (kind) {
		case PrimKind::Sphere:
			std::snprintf(b, sizeof b, "sphere c=(%g,%g,%g) r=%g mu=%.6f", center.x, center.y, center.z, radius, mu);
			break;
		case PrimKind::Plane:
			std::snprintf(b, sizeof b, "plane c=(%g,%g,%g) n=(%g,%g,%g) R=%g mu=%.6f", center.x, center.y, center.z,
					planeNormal.x, planeNormal.y, planeNormal.z, boundaryRadius, mu);
			break;
		case PrimKind::Capsule:
			std::snprintf(b, sizeof b, "capsule c=(%g,%g,%g) a=(%g,%g,%g) r=%g len=%g mu=%.6f", center.x, center.y,
					center.z, axis.x, axis.y, axis.z, radius, length, mu);
			break;
	}
	return b;
}
