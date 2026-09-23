// similarity -- the best similarity transform between two point sets
// (Umeyama, "Least-squares estimation of transformation parameters between
// two point patterns", IEEE TPAMI 13(4), 1991): T(p) = s R (p - cs) + ct
// minimising sum_i |T(src_i) - dst_i|^2 over rotation R, scale s and the
// translation. Eigen-free (AGENTS.md rule 3): the 3x3 SVD is a cyclic Jacobi
// eigen-decomposition of Sigma^T Sigma. Cut 6d's fit phase uses it to keep
// the garment's rest shape a uniformly scaled copy of the authored one, the
// one-transform form of cloth-fit's SimilarityForm (rotation + uniform scale
// of each element).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "primitives.h"

struct Similarity {
	double s = 1.0;
	double R[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // row-major
	v3d cs, ct;
	bool rotated = false; // R is not the identity (a degenerate fit keeps I)

	v3d apply(const v3d &p) const {
		const v3d q = p - cs;
		return v3d(R[0] * q.x + R[1] * q.y + R[2] * q.z, R[3] * q.x + R[4] * q.y + R[5] * q.z,
					   R[6] * q.x + R[7] * q.y + R[8] * q.z) *
						s +
				ct;
	}
	// The rotation angle in radians (0 for the identity).
	double angle() const {
		const double c = std::max(-1.0, std::min(1.0, (R[0] + R[4] + R[8] - 1.0) * 0.5));
		return std::acos(c);
	}
};

namespace similarity_detail {

// Eigen-decomposition of a symmetric 3x3 A (row-major) by cyclic Jacobi:
// A = V diag(w) V^T, V's columns the eigenvectors.
inline void jacobi3(double A[9], double w[3], double V[9]) {
	for (int i = 0; i < 9; ++i) {
		V[i] = (i % 4 == 0) ? 1.0 : 0.0;
	}
	for (int sweep = 0; sweep < 50; ++sweep) {
		double off = 0.0;
		for (int p = 0; p < 3; ++p) {
			for (int q = p + 1; q < 3; ++q) {
				off += A[3 * p + q] * A[3 * p + q];
			}
		}
		if (off < 1e-30) {
			break;
		}
		for (int p = 0; p < 3; ++p) {
			for (int q = p + 1; q < 3; ++q) {
				const double apq = A[3 * p + q];
				if (std::abs(apq) < 1e-300) {
					continue;
				}
				const double theta = (A[3 * q + q] - A[3 * p + p]) / (2.0 * apq);
				const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
				const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
				for (int k = 0; k < 3; ++k) {
					const double akp = A[3 * k + p], akq = A[3 * k + q];
					A[3 * k + p] = c * akp - sn * akq;
					A[3 * k + q] = sn * akp + c * akq;
				}
				for (int k = 0; k < 3; ++k) {
					const double apk = A[3 * p + k], aqk = A[3 * q + k];
					A[3 * p + k] = c * apk - sn * aqk;
					A[3 * q + k] = sn * apk + c * aqk;
				}
				for (int k = 0; k < 3; ++k) {
					const double vkp = V[3 * k + p], vkq = V[3 * k + q];
					V[3 * k + p] = c * vkp - sn * vkq;
					V[3 * k + q] = sn * vkp + c * vkq;
				}
			}
		}
	}
	for (int i = 0; i < 3; ++i) {
		w[i] = A[3 * i + i];
	}
}

inline double det3(const double M[9]) {
	return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

} // namespace similarity_detail

// src, dst: 3 floats per point, n points. False (and the identity) for n < 3
// or a degenerate source. A near-singular cross-covariance (a flat or
// collinear set) keeps R = I and fits the scale and translation alone. With
// `pivotSrc` / `pivotDst` the transform maps that source point to that
// destination point (no translation fitted: the scale and rotation are
// fitted about them), as cloth-fit's curve_center_target holds a garment's
// boundary curve centre on its bone.
inline bool similarity_fit(const float *src, const float *dst, uint32_t n, Similarity &out, const v3d *pivotSrc = nullptr,
		const v3d *pivotDst = nullptr) {
	out = Similarity();
	if (n < 3) {
		return false;
	}
	v3d cs, ct;
	if (pivotSrc && pivotDst) {
		cs = *pivotSrc;
		ct = *pivotDst;
	} else {
		for (uint32_t i = 0; i < n; ++i) {
			cs += v3d(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
			ct += v3d(dst[3 * i], dst[3 * i + 1], dst[3 * i + 2]);
		}
		cs = cs / double(n);
		ct = ct / double(n);
	}
	double S[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // sum (dst - ct)(src - cs)^T
	double var = 0.0;
	for (uint32_t i = 0; i < n; ++i) {
		const v3d a = v3d(src[3 * i], src[3 * i + 1], src[3 * i + 2]) - cs;
		const v3d b = v3d(dst[3 * i], dst[3 * i + 1], dst[3 * i + 2]) - ct;
		var += a.squaredNorm();
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				S[3 * r + c] += b[r] * a[c];
			}
		}
	}
	out.cs = cs;
	out.ct = ct;
	if (!(var > 0.0)) {
		return false;
	}
	// Sigma = U D V^T from the eigen-decomposition of Sigma^T Sigma = V D^2 V^T.
	double StS[9];
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			double acc = 0.0;
			for (int k = 0; k < 3; ++k) {
				acc += S[3 * k + r] * S[3 * k + c];
			}
			StS[3 * r + c] = acc;
		}
	}
	double w[3], V[9];
	similarity_detail::jacobi3(StS, w, V);
	double d[3], dmax = 0.0;
	for (int i = 0; i < 3; ++i) {
		d[i] = std::sqrt(std::max(0.0, w[i]));
		dmax = std::max(dmax, d[i]);
	}
	bool full = dmax > 0.0;
	for (int i = 0; i < 3; ++i) {
		full = full && d[i] > 1e-9 * dmax;
	}
	if (!full) {
		// Scale and translation only.
		double num = 0.0;
		for (int i = 0; i < 3; ++i) {
			num += S[3 * i + i];
		}
		out.s = num / var;
		return true;
	}
	// U = Sigma V D^-1; R = U diag(1, 1, sign det Sigma) V^T.
	double U[9];
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			double acc = 0.0;
			for (int k = 0; k < 3; ++k) {
				acc += S[3 * r + k] * V[3 * k + c];
			}
			U[3 * r + c] = acc / d[c];
		}
	}
	const double sign = similarity_detail::det3(S) < 0.0 ? -1.0 : 1.0;
	// The reflection, if any, goes on the smallest singular value.
	int imin = 0;
	for (int i = 1; i < 3; ++i) {
		if (d[i] < d[imin]) {
			imin = i;
		}
	}
	double diag[3] = { 1.0, 1.0, 1.0 };
	diag[imin] = sign;
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			double acc = 0.0;
			for (int k = 0; k < 3; ++k) {
				acc += U[3 * r + k] * diag[k] * V[3 * c + k];
			}
			out.R[3 * r + c] = acc;
		}
	}
	double tr = 0.0;
	for (int i = 0; i < 3; ++i) {
		tr += d[i] * diag[i];
	}
	out.s = tr / var;
	out.rotated = true;
	return true;
}
