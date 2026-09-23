// inverse_min -- cloth-dynamics' smallest end-to-end inverse design
// (src/code/slang_solver/test_avbd_inverse_min.cpp, e361584) as a resumable
// objective over AvbdCpu or AvbdRd, for Gate 5 G3.
//
// 4 vertices, 4 chained steps, a synthetic target made by the same solver at
// known parameters, loss 0.5 |x_4 - target|^2. Each step re-uploads the state
// it starts from (setupMesh with predicted = positions, as upstream), so the
// backward walks the chain by re-running step k from states[k] and calling
// stepBackward with the carried cotangent (dL/dx_k = dPos + dPred).
//
// Upstream's two cases share one upload here:
//   case 1 (k_tri):          kTri free,  kBend 1, density 1;
//   case 2 (k_bend, density): kTri 1, kBend free, mass = base mass * density.
// With density 1 the masses are the base masses bit for bit (2*1 = 2), so
// case 1 uploads exactly what upstream's upload() does.
//
// Upstream drives the parameters with backtracking gradient descent; Gate 5
// drives them with the in-guest L-BFGS-B (lbfgsb.h) and compares that with
// LBFGSpp on the same objective compiled for the host
// (tests/inverse_min_oracle). advance() does one stage: on AvbdRd a stage
// that submits ends the caller's tick, and the read happens on a later one
// (AGENTS.md rule 4). Header-only: the host oracle compiles the same code.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace invmin {

constexpr uint32_t NV = 4;
constexpr int STEPS = 4;
inline const float *pos0() {
	static const float p[12] = { 0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0 };
	return p;
}
inline const float *baseMass() {
	static const float m[4] = { 2, 1, 1, 1 };
	return m;
}

struct Params {
	float kTri = 1.0f, kBend = 1.0f, density = 1.0f;
};

template <class S>
void upload(S &s, const float *pos, const Params &p) {
	float m[4];
	for (int i = 0; i < 4; ++i) {
		m[i] = baseMass()[i] * p.density;
	}
	s.setupMesh(NV, pos, pos, m, 2.0f);
	s.uploadSprings(0, nullptr, nullptr, nullptr, nullptr);
	const uint32_t av[1] = { 2 };
	const float af[3] = { 0, 4, 10 }, ak[1] = { 4.0f };
	s.uploadAttachments(1, av, af, ak);
	const uint32_t ti[3] = { 0, 1, 2 };
	const float uv[4] = { 1, 0, 0, 1 }, tk[1] = { p.kTri };
	s.uploadTriangles(1, ti, uv, tk);
	const uint32_t bi[4] = { 0, 1, 2, 3 };
	const float bw[4] = { 1, 1, -1, -1 }, bn[1] = { 4 }, bk[1] = { p.kBend };
	s.uploadBendings(1, bi, bw, bn, bk);
}

// One objective evaluation: the rollout, and with `grad` the chained
// backward. After done(): x (the final state), loss, and dL/d kTri, kBend,
// density.
template <class S>
class Eval {
public:
	Eval(S &s, const Params &p, const std::vector<float> *target, bool grad) :
			s_(s), p_(p), target_(target), grad_(grad) {
		pos_.assign(pos0(), pos0() + 3 * NV);
	}

	// One stage. False once done (or failed).
	bool advance() {
		switch (ph_) {
			case FWD_SUBMIT:
				states_.push_back(pos_);
				upload(s_, pos_.data(), p_);
				if (s_.step() != 0) {
					return fail();
				}
				ph_ = FWD_READ;
				return true;
			case FWD_READ:
				s_.readPositions(pos_);
				if (pos_.size() != 3 * NV) {
					return fail();
				}
				if (++k_ < STEPS) {
					ph_ = FWD_SUBMIT;
					return true;
				}
				x = pos_;
				if (!target_) {
					ph_ = DONE;
					return false;
				}
				loss = 0.0;
				adj_.assign(3 * NV, 0.0f);
				for (size_t i = 0; i < 3 * NV; ++i) {
					const double d = double(x[i]) - double((*target_)[i]);
					loss += 0.5 * d * d;
					adj_[i] = float(d);
				}
				if (!grad_) {
					ph_ = DONE;
					return false;
				}
				k_ = STEPS - 1;
				ph_ = BWD_SUBMIT;
				return true;
			case BWD_SUBMIT:
				upload(s_, states_[k_].data(), p_);
				if (s_.runWithBackward(1, false, adj_.data()) != 0) {
					return fail();
				}
				ph_ = BWD_READ;
				return true;
			case BWD_READ: {
				std::vector<float> a, b, c;
				s_.readTriGrad(a, b, c);
				if (!a.empty()) {
					dkTri += double(a[0]);
				}
				s_.readBendGrad(a, b, c);
				if (!b.empty()) {
					dkBend += double(b[0]);
				}
				s_.readMassGrad(a);
				for (uint32_t v = 0; v < NV && v < a.size(); ++v) {
					ddensity += double(a[v]) * double(baseMass()[v]);
				}
				std::vector<float> gPos, gPred;
				s_.readPositionsGrad(gPos);
				s_.readPredictedGrad(gPred);
				for (size_t i = 0; i < adj_.size(); ++i) {
					const double u = i < gPos.size() ? double(gPos[i]) : 0.0;
					const double w = i < gPred.size() ? double(gPred[i]) : 0.0;
					adj_[i] = float(u + w);
				}
				if (--k_ >= 0) {
					ph_ = BWD_SUBMIT;
					return true;
				}
				ph_ = DONE;
				return false;
			}
			case DONE:
			case FAILED:
				return false;
		}
		return false;
	}
	bool done() const { return ph_ == DONE; }
	bool failed() const { return ph_ == FAILED; }

	std::vector<float> x;
	double loss = NAN, dkTri = 0.0, dkBend = 0.0, ddensity = 0.0;

private:
	enum Phase { FWD_SUBMIT, FWD_READ, BWD_SUBMIT, BWD_READ, DONE, FAILED };
	bool fail() {
		ph_ = FAILED;
		loss = NAN;
		return false;
	}
	S &s_;
	Params p_;
	const std::vector<float> *target_;
	bool grad_;
	Phase ph_ = FWD_SUBMIT;
	int k_ = 0;
	std::vector<float> pos_, adj_;
	std::vector<std::vector<float>> states_;
};

// The two cases: which parameters are free, the truth, the start, and the
// lower bounds (upstream's clamps: k >= 0.05 in case 1, 0.02 in case 2).
struct Case {
	const char *name;
	int n;
	const char *pname[2];
	float truth[2], start[2], lb[2];
};
inline const Case &caseOf(int c) {
	static const Case cs[2] = {
		{ "k_tri", 1, { "k_tri", "" }, { 2.0f, 0.0f }, { 0.5f, 0.0f }, { 0.05f, 0.0f } },
		{ "k_bend_density", 2, { "k_bend", "density" }, { 1.5f, 1.25f }, { 0.4f, 2.5f }, { 0.02f, 0.02f } },
	};
	return cs[c];
}
inline Params paramsOf(int c, const float *x) {
	Params p;
	if (c == 0) {
		p.kTri = x[0];
	} else {
		p.kBend = x[0];
		p.density = x[1];
	}
	return p;
}
template <class E>
void gradOf(int c, const E &e, float *g) {
	if (c == 0) {
		g[0] = float(e.dkTri);
	} else {
		g[0] = float(e.dkBend);
		g[1] = float(e.ddensity);
	}
}

} // namespace invmin
