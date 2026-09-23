// The AVBD oracle from the org's guest-avbd (tests/oracle.cpp), run inside
// the guest against either backend. Three checks:
//   four_vertex_all_constraints  the closed-form fixture from cloth-dynamics'
//                                test_avbd_solver: one spring, one attachment,
//                                one triangle, one bending stencil, one step;
//                                the expected positions are exact rationals,
//                                tolerance 1e-5
//   four_vertex_perturbed        the same with the spring rest length changed;
//                                the result must move (a solver that ignores
//                                its inputs would pass the first check)
//   two_vertex_spring            a pinned vertex stays, its spring partner
//                                moves into (1, 2)
// A backend that passes all three has every forward kernel, the gather, the
// 3x3 solve and its buffer plumbing right.
#include "avbd_fixture.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "avbd/avbd_cpu.h"

namespace {

template <class Solver>
struct Report {
	std::string text;
	int fails = 0;
	void line(const char *tag, bool ok, const std::string &detail) {
		text += std::string(ok ? "PASS " : "FAIL ") + tag + " (" + detail + "); ";
		if (!ok) {
			++fails;
		}
	}
};

template <class Solver>
void four_vertex(Solver &s, bool perturb, Report<Solver> &r) {
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

	if (s.step() != 0) {
		r.line(perturb ? "four_vertex_perturbed" : "four_vertex_all_constraints", false, "step() failed");
		return;
	}
	std::vector<float> out;
	s.readPositions(out);
	if (out.size() != 12) {
		r.line(perturb ? "four_vertex_perturbed" : "four_vertex_all_constraints", false,
				"readPositions returned " + std::to_string(out.size()) + " floats");
		return;
	}
	const float expected[12] = {
		7.0f / 8.0f, 15.0f / 7.0f, 12.0f / 7.0f,
		12.0f / 5.0f, 1.0f, 0.0f,
		0.0f, 25.0f / 8.0f, 5.0f,
		3.0f, 8.0f / 3.0f, 0.0f,
	};
	float maxd = 0.0f;
	for (int i = 0; i < 12; ++i) {
		maxd = std::fmax(maxd, std::fabs(out[i] - expected[i]));
	}
	char b[160];
	std::snprintf(b, sizeof b, "max_abs_diff=%g v0=(%g,%g,%g)", maxd, out[0], out[1], out[2]);
	if (!perturb) {
		r.line("four_vertex_all_constraints", maxd <= 1e-5f, b);
	} else {
		r.line("four_vertex_perturbed", maxd > 1e-4f, std::string(b) + ", expected to differ");
	}
}

template <class Solver>
void two_vertex(Solver &s, Report<Solver> &r) {
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
	// No triangles or bendings this time: the backend must cope with empty
	// families.
	s.uploadTriangles(0, nullptr, nullptr, nullptr);
	s.uploadBendings(0, nullptr, nullptr, nullptr, nullptr);
	if (s.step() != 0) {
		r.line("two_vertex_spring", false, "step() failed");
		return;
	}
	std::vector<float> out;
	s.readPositions(out);
	if (out.size() != 6) {
		r.line("two_vertex_spring", false, "readPositions returned " + std::to_string(out.size()) + " floats");
		return;
	}
	const bool v0_pinned = std::fabs(out[0]) < 0.1f;
	const bool v1_moved = out[3] < 2.0f && out[3] > 1.0f;
	char b[96];
	std::snprintf(b, sizeof b, "v0.x=%g pinned, v1.x=%g in (1,2)", out[0], out[3]);
	r.line("two_vertex_spring", v0_pinned && v1_moved, b);
}

template <class Solver>
std::string run_all(Solver &s, const char *backend) {
	Report<Solver> r;
	four_vertex(s, false, r);
	four_vertex(s, true, r);
	two_vertex(s, r);
	return std::string(r.fails == 0 ? "PASS " : "FAIL ") + backend + ": " + r.text;
}

} // namespace

std::string avbd_fixture_cpu() {
	AvbdCpu s;
	return run_all(s, "cpu");
}
