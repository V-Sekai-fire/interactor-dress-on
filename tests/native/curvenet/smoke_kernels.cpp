// curvenet kernel smoke: the five Lean-emitted Cassie kernels, called through
// the vendored dispatchers (vendor/cassie/src/solver/slang_dispatch), run and
// give the answers their math says. Built only with
// CURVENET_KERNELS_PENDING=OFF. Each check has a control that must come out
// differently, so "passes" cannot mean "nothing ran".
//
// Prelude-free (no godot_lite): the dispatch headers are raw-pointer only.
#include "curve_casteljau_dispatch.h"
#include "curve_generate_bezier_dispatch.h"
#include "curve_newton_dispatch.h"
#include "curve_rdp_dispatch.h"
#include "spmv_dispatch.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace sd = cassie_slang_dispatch;

static int g_fail = 0;

static void check(bool p_ok, const char *p_what) {
	std::printf("%s %s\n", p_ok ? "PASS" : "FAIL", p_what);
	if (!p_ok) {
		++g_fail;
	}
}

// The cubic every curve check uses: P0 (0,0,0), P1 (1,2,0), P2 (3,2,0), P3 (4,0,0).
static const float P0[3] = { 0, 0, 0 }, P1[3] = { 1, 2, 0 }, P2[3] = { 3, 2, 0 }, P3[3] = { 4, 0, 0 };

static void bezier(float u, float r[3]) {
	const float o = 1.0f - u;
	const float w0 = o * o * o, w1 = 3 * o * o * u, w2 = 3 * o * u * u, w3 = u * u * u;
	for (int k = 0; k < 3; ++k) {
		r[k] = w0 * P0[k] + w1 * P1[k] + w2 * P2[k] + w3 * P3[k];
	}
}

static float dist3(const float a[3], const float b[3]) {
	return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

int main() {
	// 1. De Casteljau at u = 0.5: the cut point is B(0.5) = (2, 1.5, 0),
	//    exact in float; la = P0, rd = P3. Control: u = 0.25 cuts elsewhere.
	{
		float la[3], lb[3], lc[3], ld[3], ra[3], rb[3], rc[3], rd[3];
		sd::curve_casteljau(P0, P1, P2, P3, 0.5f, la, lb, lc, ld, ra, rb, rc, rd);
		std::printf("casteljau u=0.5: split (%g, %g, %g)\n", ld[0], ld[1], ld[2]);
		check(ld[0] == 2.0f && ld[1] == 1.5f && ld[2] == 0.0f && ra[0] == ld[0] && ra[1] == ld[1],
				"curve_casteljau: split at u=0.5 is (2, 1.5, 0), ld == ra");
		check(dist3(la, P0) == 0.0f && dist3(rd, P3) == 0.0f, "curve_casteljau: la == P0, rd == P3");
		sd::curve_casteljau(P0, P1, P2, P3, 0.25f, la, lb, lc, ld, ra, rb, rc, rd);
		float b[3];
		bezier(0.25f, b);
		check(dist3(ld, b) < 1e-6f && ld[0] != 2.0f, "control: curve_casteljau at u=0.25 cuts at B(0.25), not (2, 1.5, 0)");
	}
	// 2. RDP, tolerance 0.5: five colinear points keep only the endpoints (2).
	//    Control: lift the middle one by 1 and it is kept too (3); points 1
	//    and 3 then sit 1/sqrt(5) = 0.447 off their sub-chords and still go.
	{
		float pts[15] = { 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0 };
		uint32_t keep[5];
		const uint32_t n0 = sd::curve_rdp_reduce(pts, 5, 0.5f, keep);
		pts[7] = 1.0f;
		const uint32_t n1 = sd::curve_rdp_reduce(pts, 5, 0.5f, keep);
		std::printf("rdp: colinear keeps %u, bumped keeps %u (keep[2] = %u)\n", n0, n1, keep[2]);
		check(n0 == 2, "curve_rdp: 5 colinear points keep 2");
		check(n1 == 3 && keep[2] == 1, "control: curve_rdp keeps the lifted middle point (3)");
	}
	// 3. Newton: points sampled at exact u leave u where it is; a u pushed
	//    by +0.05 comes back toward the true one.
	{
		const int n = 5;
		float pts[3 * n], u[n], u_off[n], out[n], out_off[n];
		for (int i = 0; i < n; ++i) {
			u[i] = 0.1f + 0.2f * float(i);
			u_off[i] = u[i] + 0.05f;
			bezier(u[i], pts + 3 * i);
		}
		sd::curve_newton_reparameterize(P0, P1, P2, P3, n, pts, u, out);
		sd::curve_newton_reparameterize(P0, P1, P2, P3, n, pts, u_off, out_off);
		float moved = 0, err_before = 0, err_after = 0;
		for (int i = 0; i < n; ++i) {
			moved = std::fmax(moved, std::fabs(out[i] - u[i]));
			err_before = std::fmax(err_before, std::fabs(u_off[i] - u[i]));
			err_after = std::fmax(err_after, std::fabs(out_off[i] - u[i]));
		}
		std::printf("newton: exact u moved by %g; offset u error %g -> %g\n", moved, err_before, err_after);
		check(moved < 1e-5f, "curve_newton: exact parameters are a fixed point");
		check(err_after < 0.1f * err_before, "control: curve_newton pulls u+0.05 back by >10x");
	}
	// 4. Schneider LSQ: 9 exact samples of the cubic with its unit end
	//    tangents recover P1 and P2. Control: 3 samples of the chord (a
	//    straight line) give a different P1.
	{
		const int n = 9;
		float pts[3 * n], u[n], ctrl[12];
		for (int i = 0; i < n; ++i) {
			u[i] = float(i) / float(n - 1);
			bezier(u[i], pts + 3 * i);
		}
		const float s = 1.0f / std::sqrt(5.0f);
		const float ta[3] = { 1 * s, 2 * s, 0 }, tb[3] = { -1 * s, 2 * s, 0 };
		sd::curve_generate_bezier(ta, tb, n, pts, u, ctrl);
		const float e1 = dist3(ctrl + 3, P1), e2 = dist3(ctrl + 6, P2);
		std::printf("generate_bezier: |P1 - fit| %g, |P2 - fit| %g\n", e1, e2);
		check(e1 < 1e-4f && e2 < 1e-4f && dist3(ctrl, P0) == 0.0f && dist3(ctrl + 9, P3) == 0.0f,
				"curve_generate_bezier: recovers P1, P2 from 9 exact samples");
		float line[9] = { 0, 0, 0, 2, 0, 0, 4, 0, 0 };
		float ul[3] = { 0, 0.5f, 1 };
		sd::curve_generate_bezier(ta, tb, 3, line, ul, ctrl);
		check(dist3(ctrl + 3, P1) > 0.1f, "control: curve_generate_bezier on a straight chord does not give P1");
	}
	// 5. SpMV (df32): [[2,0,1],[0,3,0],[1,0,4]] (1,2,3) = (5,6,13). Control:
	//    drop the (2,0) entry and row 2 gives 12.
	{
		const int32_t rp[4] = { 0, 2, 3, 5 }, ci[5] = { 0, 2, 1, 0, 2 };
		const float v[5] = { 2, 1, 3, 1, 4 }, x[3] = { 1, 2, 3 };
		float y[3];
		sd::spmv(3, rp, ci, 5, v, x, 3, y);
		std::printf("spmv: (%g, %g, %g)\n", y[0], y[1], y[2]);
		check(y[0] == 5 && y[1] == 6 && y[2] == 13, "spmv: A x = (5, 6, 13)");
		const float v0[5] = { 2, 1, 3, 0, 4 };
		sd::spmv(3, rp, ci, 5, v0, x, 3, y);
		check(y[2] == 12, "control: spmv without A[2][0] gives y[2] = 12");
	}
	std::printf("curvenet kernel smoke: %s (%d failed)\n", g_fail ? "FAILED" : "OK", g_fail);
	return g_fail ? 1 : 0;
}
