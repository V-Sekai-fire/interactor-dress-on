// SPDX-License-Identifier: Apache-2.0 OR MIT

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../avbd_cpu.h"

static int g_fails = 0;

static void check(const char *tag, const std::vector<float> &got,
		const std::vector<float> &ref, float tol) {
	float maxd = 0.0f;
	for (size_t i = 0; i < ref.size(); ++i) {
		const float d = std::fabs(got[i] - ref[i]);
		if (d > maxd) {
			maxd = d;
		}
	}
	const bool ok = maxd <= tol;
	std::printf("%s  %s  (max_abs_diff=%g)\n", ok ? "PASS" : "FAIL", tag, maxd);
	if (!ok) {
		for (size_t i = 0; i < ref.size(); ++i) {
			std::printf("    [%zu] got %g  ref %g\n", i, got[i], ref[i]);
		}
		++g_fails;
	}
}

static void run_four_vertex(bool perturb, const char *tag) {
	AvbdCpu s;
	const float pos[12] = { 0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0 };
	const float pred[12] = { 1, 2, 3, 3, 0, 0, 0, 4, 0, 3, 4, 0 };
	const float mass[4] = { 2, 1, 1, 1 };
	s.setupMesh(4, pos, pred, mass, 2.0f);

	const uint32_t sp1[1] = { 0 }, sp2[1] = { 1 };
	const float srest[1] = { perturb ? 5.0f : 2.0f }, sk[1] = { 1 };
	s.uploadSprings(1, sp1, sp2, srest, sk);

	const uint32_t av[1] = { 2 };
	const float afix[3] = { 0, 4, 10 }, ak[1] = { 4 };
	s.uploadAttachments(1, av, afix, ak);

	const uint32_t ti[3] = { 0, 1, 2 };
	const float tuv[4] = { 1, 0, 0, 1 }, tk[1] = { 1 };
	s.uploadTriangles(1, ti, tuv, tk);

	const uint32_t bi[4] = { 0, 1, 2, 3 };
	const float bw[4] = { 1, 1, -1, -1 }, bnt[1] = { 4 }, bk[1] = { 1 };
	s.uploadBendings(1, bi, bw, bnt, bk);

	s.step();
	std::vector<float> out;
	s.readPositions(out);

	const std::vector<float> expected = {
		7.0f / 8.0f, 15.0f / 7.0f, 12.0f / 7.0f,
		12.0f / 5.0f, 1.0f, 0.0f,
		0.0f, 25.0f / 8.0f, 5.0f,
		3.0f, 8.0f / 3.0f, 0.0f
	};
	if (!perturb) {
		check(tag, out, expected, 1e-5f);
	} else {
		float maxd = 0.0f;
		for (size_t i = 0; i < expected.size(); ++i) {
			maxd = std::fmax(maxd, std::fabs(out[i] - expected[i]));
		}
		const bool moved = maxd > 1e-4f;
		std::printf("%s  %s  (max_abs_diff=%g, expected to differ)\n",
				moved ? "PASS" : "FAIL", tag, maxd);
		if (!moved) {
			++g_fails;
		}
	}
}

static void run_two_vertex() {
	AvbdCpu s;
	const float pos[6] = { 0, 0, 0, 2, 0, 0 };
	const float pred[6] = { 0, 0, 0, 1.9f, 0, 0 };
	const float mass[2] = { 1, 1 };
	s.setupMesh(2, pos, pred, mass, 100.0f);

	const uint32_t sp1[1] = { 1 }, sp2[1] = { 0 };
	const float srest[1] = { 1 }, sk[1] = { 50 };
	s.uploadSprings(1, sp1, sp2, srest, sk);

	const uint32_t av[1] = { 0 };
	const float afix[3] = { 0, 0, 0 }, ak[1] = { 1000 };
	s.uploadAttachments(1, av, afix, ak);

	s.step();
	std::vector<float> out;
	s.readPositions(out);

	const bool v0_pinned = std::fabs(out[0]) < 0.1f;
	const bool v1_moved = out[3] < 2.0f && out[3] > 1.0f;
	const bool ok = v0_pinned && v1_moved;
	std::printf("%s  two_vertex_spring  (v0.x=%g pinned, v1.x=%g in (1,2))\n",
			ok ? "PASS" : "FAIL", out[0], out[3]);
	if (!ok) {
		++g_fails;
	}
}

int main() {
	run_four_vertex(false, "four_vertex_all_constraints");
	run_four_vertex(true, "four_vertex_perturbed_control");
	run_two_vertex();
	std::printf("%s\n", g_fails == 0 ? "all oracle checks OK" : "ORACLE FAILED");
	return g_fails == 0 ? 0 : 1;
}
