// similarity -- the best similarity transform between two point sets
// (Umeyama, "Least-squares estimation of transformation parameters between
// two point patterns", IEEE TPAMI 13(4), 1991): T(p) = s R (p - cs) + ct
// minimising sum_i |T(src_i) - dst_i|^2 over rotation R, scale s and the
// translation. The rotation comes from the org's fitter, sinew-mocap/solve's
// Align.lean as its C port (vendor/sinew-align: rodrigues for one pair, a
// Newton-Schulz-orthonormalised covariance for two or more, the Jacobi-SVD
// kabsch fallback when valid9 rejects the fast path), fed the centred
// cross-covariance; the scale is Umeyama's trace(R^T H) / sum |src - cs|^2.
// Eigen-free (AGENTS.md rule 3); the rotation is a row-major 3x3 (rule 11).
// Cut 6d's fit phase uses it to keep the garment's rest shape a uniformly
// scaled copy of the authored one, the one-transform form of cloth-fit's
// SimilarityForm (rotation + uniform scale of each element).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "primitives.h"
// The vendored C port declares no extern "C" itself (it is built as C in
// interactor-gyre); the guest and the host harness compile this header as C++.
extern "C" {
#include "sinew_align.h"
}

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
	// The rotation angle in radians from the trace (a readable summary only).
	double angle() const {
		const double c = std::max(-1.0, std::min(1.0, (R[0] + R[4] + R[8] - 1.0) * 0.5));
		return std::acos(c);
	}
};

// src, dst: 3 floats per point, n points. False (and the identity) for n < 3
// or a degenerate source. A rotation the fitter cannot vouch for (valid9
// false: a flat or collinear set) keeps R = I and fits the scale and
// translation alone. With `pivotSrc` / `pivotDst` the transform maps that
// source point to that destination point (no translation fitted: the scale
// and rotation are fitted about them), as cloth-fit's curve_center_target
// holds a garment's boundary curve centre on its bone.
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
	// H = sum (dst - ct) (src - cs)^T: sinew_align's own accumulation
	// (targets[i] ~= R sources[i], h[i][j] += a[i] b[j]).
	double H[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	double var = 0.0;
	double a0[3] = { 0, 0, 0 }, a1[3] = { 0, 0, 0 }, b0[3] = { 0, 0, 0 }, b1[3] = { 0, 0, 0 };
	for (uint32_t i = 0; i < n; ++i) {
		const v3d a = v3d(src[3 * i], src[3 * i + 1], src[3 * i + 2]) - cs;
		const v3d b = v3d(dst[3 * i], dst[3 * i + 1], dst[3 * i + 2]) - ct;
		var += a.squaredNorm();
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				H[3 * r + c] += b[r] * a[c];
			}
		}
		if (i < 2) {
			double *ta = i == 0 ? a0 : a1;
			double *tb = i == 0 ? b0 : b1;
			for (int k = 0; k < 3; ++k) {
				ta[k] = b[k];
				tb[k] = a[k];
			}
		}
	}
	out.cs = cs;
	out.ct = ct;
	if (!(var > 0.0)) {
		return false;
	}
	double R[9];
	sinew_finish_align(H, int(n), a0, a1, b0, b1, R);
	if (!sinew_valid9(R)) {
		// Scale and translation only.
		double num = 0.0;
		for (int i = 0; i < 3; ++i) {
			num += H[3 * i + i];
		}
		out.s = num / var;
		return true;
	}
	for (int i = 0; i < 9; ++i) {
		out.R[i] = R[i];
	}
	// trace(R^T H) = sum_i d_i for R = U V^T (Umeyama's scale numerator).
	double tr = 0.0;
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			tr += R[3 * r + c] * H[3 * r + c];
		}
	}
	out.s = tr / var;
	out.rotated = true;
	return true;
}
