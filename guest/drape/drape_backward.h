// drape_backward -- losses and the three backward modes over a recorded
// forward (drape_sim.h), staged like the forward.
//
// Losses (on the frames: x0, then each step's predictor s, which is what
// upstream's ForwardInformation::x holds and what the native OBJs export):
//  - MATCH_TRAJECTORY (Simulation.cpp:3865-3903): L = k sum_i |F_i - T_i|^2,
//    k = 1/((N+1) nV); dL/dF_i = 2k (F_i - T_i);
//  - TARGET_POINTS: L = sum_j |F_f[v_j] - p_j|^2 on one frame f (the last by
//    default).
//
// Modes:
//  - native: upstream runBackwardTask's AVBD branch (4495-4795) with
//    stepBackwardAvbd (2107-2382) and AvbdBackwardShim.cpp, for parity. The
//    forward is not re-run: every step's adjoint is the solver's stepBackward
//    on the state the forward left (the last step's last iteration, and the
//    positions/predicted the self-collision pass uploaded). K = bwdTruncateK
//    steps back (then nothing more accumulates), dL/dx_in = carried + this
//    step's loss gradient, clipped to gradientClippingThreshold * nV;
//    dL/dv = 0. dL/dmu = sum over the PREVIOUS record's friction events of
//    (-v_tan) . (h dL/dpredicted); the carry is added twice per step as
//    upstream does (nativeMuDoubleCarry). Stiffness gradients are upstream's
//    per-type sums of per-constraint dL/dk_c (reported as *_raw) and also
//    chained to the scalar parameters (k_c = area k).
//  - step: recompute each step from its recorded start state: updateState(x,
//    s_blend), run(iters) and the solver's backward of the last iteration in
//    one submit; the last iteration's input stands in for the step's start.
//  - unrolled: recompute each step one iteration at a time (a snapshot of the
//    positions before each), then run the backward of every iteration from
//    the last to the first, carrying dL/dx through all of them.
// The recompute modes chain everything the forward did after the solve,
// using the recorded contacts: the self-collision pushes (exact Jacobians, in
// reverse), the primitive projection (projectionJacobian), the velocity
// response (1 - mu on the tangent, the normal inflow clamp), v = (x_avbd -
// x)/h, and the friction predictor blend; so
//   ax_n = gPos + gPred - av'/h,  av_n = h damp P^T gPred   (+ the frame's
//   own gradient through s = x + h v + h^2 M^-1 f),
// with P = I off contact. dL/dmu gathers both the response (-v_tan) and the
// blend (-ds_t). Normal derivatives are not carried (exact for a plane).
// Duals (AL) are not recomputed: the recompute modes refuse cfg.al.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "drape_sim.h"

enum class DrapeLoss { MatchTrajectory, TargetPoints };
enum class DrapeMode { Native, Step, Unrolled };

inline const char *drape_mode_name(DrapeMode m) {
	return m == DrapeMode::Native ? "native" : (m == DrapeMode::Step ? "step" : "unrolled");
}

struct DrapeTarget {
	// MATCH_TRAJECTORY: frames 0..N back to back, frameSize doubles each.
	// (Flat on purpose: a vector of vectors kept alive across stages
	// corrupted the guest heap; see gates/5-drape/README.md.)
	std::vector<double> frames;
	size_t frameSize = 0;
	size_t numFrames() const { return frameSize ? frames.size() / frameSize : 0; }
	const double *frameAt(size_t i) const { return frames.data() + i * frameSize; }
	std::vector<uint32_t> verts;             // TARGET_POINTS
	std::vector<double> pos;                 // 3 per vert
	int frame = -1;                          // -1: the last
};

struct DrapeGrad {
	double loss = 0.0;
	std::vector<double> dmu;   // per primitive
	double dkTri = 0.0, dkBend = 0.0, dkAttach = 0.0, ddensity = 0.0;
	double dkTriRaw = 0.0, dkBendRaw = 0.0, dkAttachRaw = 0.0; // upstream's sums of dL/dk_c
	std::vector<double> muPerStep; // native: this step's dL/dmu (prim 0), from the last step back
	int stepsBack = 0;
	double dx0Norm = 0.0;
	std::string mode;
};

// The loss of a recorded forward against a target (see the header comment).
template <class Solver>
double drape_loss_value(const DrapeSimT<Solver> &sim, const DrapeTarget &tgt, DrapeLoss loss) {
	const size_t N = sim.steps();
	const uint32_t nV = sim.scene.nV;
	double L = 0.0;
	if (loss == DrapeLoss::MatchTrajectory) {
		if (tgt.numFrames() != N + 1 || tgt.frameSize != 3 * size_t(nV)) {
			return NAN;
		}
		const double k = 1.0 / (double(N + 1) * double(nV));
		for (size_t i = 0; i <= N; ++i) {
			const std::vector<double> F = sim.frame(i);
			double sq = 0.0;
			for (size_t b = 0; b < F.size(); ++b) {
				const double d = F[b] - tgt.frameAt(i)[b];
				sq += d * d;
			}
			L += k * sq;
		}
	} else {
		const size_t f = tgt.frame < 0 ? N : size_t(tgt.frame);
		if (f > N) {
			return NAN;
		}
		const std::vector<double> F = sim.frame(f);
		for (size_t j = 0; j < tgt.verts.size(); ++j) {
			for (int k = 0; k < 3; ++k) {
				const double d = F[3 * tgt.verts[j] + k] - tgt.pos[3 * j + k];
				L += d * d;
			}
		}
	}
	return L;
}

template <class Solver>
class DrapeBackwardT {
public:
	DrapeBackwardT(DrapeSimT<Solver> &sim, const DrapeTarget &tgt, DrapeLoss loss, DrapeMode mode)
			: sim_(sim), s_(sim.solver), tgt_(tgt), loss_(loss), mode_(mode) {}

	DrapeGrad g;
	bool failed = false;
	std::string err;

	bool done() const { return st_ == St::Done; }

	// One phase; true once the gradient is complete.
	bool advance() {
		switch (st_) {
			case St::Init: init(); break;
			case St::NativeSubmit: nativeSubmit(); break;
			case St::NativeRead: nativeRead(); break;
			case St::RcBegin: rcBegin(); break;
			case St::RcSnap: rcSnap(); break;
			case St::RcBwdRead: rcBwdRead(); break;
			case St::Done: break;
		}
		return st_ == St::Done;
	}

	std::string summary() const {
		char b[1024];
		std::string mu;
		for (size_t p = 0; p < g.dmu.size(); ++p) {
			char t[64];
			std::snprintf(t, sizeof t, "%sdL/dmu_%zu=%.9g", p ? " " : "", p, g.dmu[p]);
			mu += t;
		}
		std::snprintf(b, sizeof b,
				"mode=%s loss=%.9g %s dL/dkTri=%.9g dL/dkBend=%.9g dL/dkAttach=%.9g dL/ddensity=%.9g "
				"raw(dk_tri=%.9g dk_bend=%.9g dk_attach=%.9g) stepsBack=%d |dL/dx0|=%.6g",
				g.mode.c_str(), g.loss, mu.c_str(), g.dkTri, g.dkBend, g.dkAttach, g.ddensity, g.dkTriRaw, g.dkBendRaw,
				g.dkAttachRaw, g.stepsBack, g.dx0Norm);
		return b;
	}

private:
	enum class St { Init, NativeSubmit, NativeRead, RcBegin, RcSnap, RcBwdRead, Done };
	DrapeSimT<Solver> &sim_;
	Solver &s_;
	const DrapeTarget &tgt_;
	DrapeLoss loss_;
	DrapeMode mode_;
	St st_ = St::Init;
	size_t N_ = 0;
	uint32_t nV_ = 0;
	int idx_ = 0;
	std::vector<double> carry_, ax_, av_, vOut_, axDirect_, gPredSum_;
	std::vector<float> snaps_; // iters x 3 nV, flat
	std::vector<float> v_;
	int it_ = 0;

	void fail(const std::string &e) {
		failed = true;
		err = e;
		st_ = St::Done;
	}

	// dL/dF_i (3 nV), zero when the frame carries no loss.
	std::vector<double> gradFrame(size_t i) const {
		std::vector<double> gF(3 * size_t(nV_), 0.0);
		if (loss_ == DrapeLoss::MatchTrajectory) {
			const double k = 1.0 / (double(N_ + 1) * double(nV_));
			const std::vector<double> F = sim_.frame(i);
			for (size_t b = 0; b < gF.size(); ++b) {
				gF[b] = k * 2.0 * (F[b] - tgt_.frameAt(i)[b]);
			}
		} else {
			const size_t f = tgt_.frame < 0 ? N_ : size_t(tgt_.frame);
			if (i == f) {
				const std::vector<double> F = sim_.frame(i);
				for (size_t j = 0; j < tgt_.verts.size(); ++j) {
					const uint32_t vtx = tgt_.verts[j];
					for (int k = 0; k < 3; ++k) {
						gF[3 * vtx + k] += 2.0 * (F[3 * vtx + k] - tgt_.pos[3 * j + k]);
					}
				}
			}
		}
		return gF;
	}

	double lossValue() const { return drape_loss_value(sim_, tgt_, loss_); }

	void init() {
		N_ = sim_.steps();
		nV_ = sim_.scene.nV;
		g = DrapeGrad();
		g.mode = drape_mode_name(mode_);
		g.dmu.assign(sim_.scene.prims.size(), 0.0);
		if (N_ == 0 || sim_.midStep()) {
			fail("no completed forward to differentiate");
			return;
		}
		if (loss_ == DrapeLoss::MatchTrajectory) {
			if (tgt_.numFrames() != N_ + 1 || tgt_.frameSize != 3 * size_t(nV_)) {
				fail("target has " + std::to_string(tgt_.numFrames()) + " frames, the forward " +
						std::to_string(N_ + 1));
				return;
			}
		} else {
			if (tgt_.verts.empty() || tgt_.pos.size() != 3 * tgt_.verts.size() ||
					(tgt_.frame >= 0 && size_t(tgt_.frame) > N_)) {
				fail("bad target points");
				return;
			}
			for (uint32_t vtx : tgt_.verts) {
				if (vtx >= nV_) {
					fail("target vertex out of range");
					return;
				}
			}
		}
		if (mode_ != DrapeMode::Native && sim_.cfg.al) {
			fail("the recompute modes do not replay AL duals (set al 0)");
			return;
		}
		g.loss = lossValue();
		idx_ = int(N_);
		if (mode_ == DrapeMode::Native) {
			carry_ = gradFrame(N_);
			st_ = St::NativeSubmit;
			nativeSubmit();
		} else {
			ax_.assign(3 * size_t(nV_), 0.0);
			av_.assign(3 * size_t(nV_), 0.0);
			// Frame N is s(x_{N-1}, v_{N-1}): its gradient lands on state N-1,
			// added when step N is finished.
			st_ = St::RcBegin;
			rcBegin();
		}
	}

	// --- native ------------------------------------------------------------------

	void nativeSubmit() {
		const int K = sim_.cfg.bwdTruncateK;
		if (idx_ < 1 || (K > 0 && g.stepsBack >= K)) {
			finishNative();
			return;
		}
		std::vector<double> din = gradFrame(size_t(idx_ - 1));
		for (size_t b = 0; b < din.size(); ++b) {
			din[b] += carry_[b];
		}
		if (sim_.cfg.gradientClipping) {
			const double maxNorm = sim_.cfg.gradientClippingThreshold * double(nV_);
			double n2 = 0.0;
			for (double d : din) {
				n2 += d * d;
			}
			const double n = std::sqrt(n2);
			if (n > maxNorm) {
				for (double &d : din) {
					d *= maxNorm / n;
				}
			}
		}
		std::vector<float> f(din.size());
		for (size_t b = 0; b < din.size(); ++b) {
			f[b] = float(din[b]);
		}
		if (s_.stepBackward(f.data()) != 0) {
			fail("stepBackward failed");
			return;
		}
		st_ = St::NativeRead;
	}

	void nativeRead() {
		std::vector<float> dPos, dPred;
		s_.readPositionsGrad(dPos);
		s_.readPredictedGrad(dPred);
		if (dPos.size() != 3 * size_t(nV_) || dPred.size() != 3 * size_t(nV_)) {
			fail("short gradient readback");
			return;
		}
		accumulateParams();
		const double h = sim_.cfg.h;
		std::vector<double> c(sim_.scene.prims.size(), 0.0);
		if (idx_ - 1 >= 1) {
			for (const DrapeFriction &ev : sim_.recs[size_t(idx_ - 2)].fric) {
				double dot = 0.0;
				for (int k = 0; k < 3; ++k) {
					dot += (-ev.vTan[k]) * (h * double(dPred[3 * ev.vert + k]));
				}
				c[ev.prim] += dot;
			}
		}
		for (size_t p = 0; p < c.size(); ++p) {
			g.dmu[p] = c[p] + g.dmu[p] + (sim_.cfg.nativeMuDoubleCarry ? g.dmu[p] : 0.0);
		}
		g.muPerStep.push_back(c.empty() ? 0.0 : c[0]);
		for (size_t b = 0; b < carry_.size(); ++b) {
			carry_[b] = double(dPos[b]);
		}
		++g.stepsBack;
		--idx_;
		st_ = St::NativeSubmit;
		nativeSubmit();
	}

	void finishNative() {
		double n2 = 0.0;
		for (double d : carry_) {
			n2 += d * d;
		}
		g.dx0Norm = std::sqrt(n2);
		st_ = St::Done;
	}

	// The solver's parameter cotangents of the backward just read, added in.
	void accumulateParams() {
		const DrapeScene &sc = sim_.scene;
		const DrapeConfig &cfg = sim_.cfg;
		std::vector<float> dMass, a, b, c;
		s_.readMassGrad(dMass);
		for (uint32_t i = 0; i < nV_ && i < dMass.size(); ++i) {
			g.ddensity += double(dMass[i]) * sc.vertArea[i];
		}
		if (cfg.membrane && sc.nTri() > 0) {
			s_.readTriGrad(a, b, c);
			for (uint32_t t = 0; t < sc.nTri() && t < a.size(); ++t) {
				g.dkTriRaw += double(a[t]);
				g.dkTri += double(a[t]) * (cfg.rawStiffness ? 1.0 : sc.triArea[t]);
			}
		}
		if (cfg.bending && sc.nBend() > 0) {
			s_.readBendGrad(a, b, c);
			for (uint32_t q = 0; q < sc.nBend() && q < b.size(); ++q) {
				g.dkBendRaw += double(b[q]);
				g.dkBend += double(b[q]) * (cfg.rawStiffness ? 1.0 : 3.0 / sc.bendAreaSum[q]);
			}
		}
		if (sc.nAttach() > 0) {
			s_.readAttachGrad(a, b, c);
			for (uint32_t q = 0; q < sc.nAttach() && q < b.size(); ++q) {
				g.dkAttachRaw += double(b[q]);
				g.dkAttach += double(b[q]);
			}
		}
	}

	// --- recompute (step, unrolled) ---------------------------------------------

	const std::vector<float> &xPrev(size_t n) const { return n >= 2 ? sim_.recs[n - 2].x : sim_.scene.x0; }
	const std::vector<float> &vPrev(size_t n) const { return n >= 2 ? sim_.recs[n - 2].v : sim_.scene.v0; }

	static v3d get(const std::vector<double> &a, uint32_t i) { return { a[3 * i], a[3 * i + 1], a[3 * i + 2] }; }
	static void set(std::vector<double> &a, uint32_t i, const v3d &p) {
		a[3 * i] = p.x;
		a[3 * i + 1] = p.y;
		a[3 * i + 2] = p.z;
	}

	void rcBegin() {
		const int K = sim_.cfg.bwdTruncateK;
		if (idx_ < 1 || (K > 0 && g.stepsBack >= K)) {
			finishRecompute();
			return;
		}
		const size_t n = size_t(idx_);
		const DrapeStepRecord &rec = sim_.recs[n - 1];
		const DrapeScene &sc = sim_.scene;
		const double h = sim_.cfg.h;
		std::vector<double> gx = ax_, gv = av_;
		// Self-collision pushes, last first: a' = a + p(d), b' = b - p(d).
		for (size_t q = rec.pushes.size(); q-- > 0;) {
			const DrapePush &P = rec.pushes[q];
			const v3d Ga = get(gx, P.a), Gb = get(gx, P.b);
			const v3d dG = Ga - Gb;
			const double l = P.dist;
			const double c0 = P.thresh / (2.0 * l) - 0.5;
			const double c1 = P.thresh / (2.0 * l * l * l);
			// J = c0 I - c1 d d^T (symmetric).
			const v3d Jt = dG * c0 - P.diff * (c1 * P.diff.dot(dG));
			set(gx, P.a, Ga + Jt);
			set(gx, P.b, Gb - Jt);
		}
		// Contacts, last first: projection on x, response on v.
		for (size_t q = rec.contacts.size(); q-- > 0;) {
			const DrapeContact &C = rec.contacts[q];
			const Primitive &pr = sc.prims[C.prim];
			v3d av = get(gv, C.vert);
			if (pr.mu > 0.0 && pr.mu < 1.0) {
				g.dmu[C.prim] += av.dot(-C.vTan);
			}
			const m3d nn = m3d::outer(C.normal, C.normal);
			m3d J = (m3d::identity() - nn) * (1.0 - C.muClip);
			if (C.vn > 0.0) {
				J = J + nn;
			}
			set(gv, C.vert, J.mulT(av));
			if (C.penetrating) {
				const m3d P = pr.projectionJacobian(C.posBefore);
				set(gx, C.vert, P.mulT(get(gx, C.vert)));
			}
		}
		// v = (x_avbd - x_prev)/h.
		vOut_.assign(3 * size_t(nV_), 0.0);
		axDirect_.assign(3 * size_t(nV_), 0.0);
		for (size_t b = 0; b < vOut_.size(); ++b) {
			vOut_[b] = gx[b] + gv[b] / h;
			axDirect_[b] = -gv[b] / h;
		}
		gPredSum_.assign(3 * size_t(nV_), 0.0);
		s_.updateState(xPrev(n).data(), rec.sBlend.data());
		if (sc.nAttach() > 0) {
			s_.updateAttachmentFixedPos(sc.attachFixed.data());
		}
		std::vector<float> vo(vOut_.size());
		for (size_t b = 0; b < vo.size(); ++b) {
			vo[b] = float(vOut_[b]);
		}
		const int iters = sim_.cfg.iters;
		if (mode_ == DrapeMode::Step) {
			if (s_.runWithBackward(iters, false, vo.data()) != 0) {
				fail("runWithBackward failed");
				return;
			}
			st_ = St::RcBwdRead;
			return;
		}
		// Unrolled: snapshots of the positions before each iteration.
		snaps_.assign(size_t(iters) * 3 * nV_, 0.0f);
		std::copy(xPrev(n).begin(), xPrev(n).end(), snaps_.begin());
		v_ = vo;
		it_ = 0;
		if (iters > 1) {
			if (s_.run(1, false) != 0) {
				fail("run failed");
				return;
			}
			st_ = St::RcSnap;
			return;
		}
		startUnrolledBackward();
	}

	void rcSnap() {
		std::vector<float> x;
		s_.readPositions(x);
		if (x.size() != 3 * size_t(nV_)) {
			fail("short readback");
			return;
		}
		std::copy(x.begin(), x.end(), snaps_.begin() + size_t(it_ + 1) * 3 * nV_);
		++it_;
		if (it_ + 1 < sim_.cfg.iters) {
			if (s_.run(1, false) != 0) {
				fail("run failed");
			}
			return;
		}
		startUnrolledBackward();
	}

	void startUnrolledBackward() {
		it_ = sim_.cfg.iters - 1;
		submitIteration();
	}

	void submitIteration() {
		const DrapeStepRecord &rec = sim_.recs[size_t(idx_) - 1];
		s_.updateState(snaps_.data() + size_t(it_) * 3 * nV_, rec.sBlend.data());
		if (s_.runWithBackward(1, false, v_.data()) != 0) {
			fail("runWithBackward failed");
			return;
		}
		st_ = St::RcBwdRead;
	}

	void rcBwdRead() {
		std::vector<float> dPos, dPred;
		s_.readPositionsGrad(dPos);
		s_.readPredictedGrad(dPred);
		if (dPos.size() != 3 * size_t(nV_) || dPred.size() != 3 * size_t(nV_)) {
			fail("short gradient readback");
			return;
		}
		accumulateParams();
		for (size_t b = 0; b < gPredSum_.size(); ++b) {
			gPredSum_[b] += double(dPred[b]);
		}
		if (mode_ == DrapeMode::Unrolled) {
			v_ = dPos;
			if (it_ > 0) {
				--it_;
				submitIteration();
				return;
			}
		} else {
			v_ = dPos;
		}
		finishStep();
	}

	void finishStep() {
		const size_t n = size_t(idx_);
		const DrapeStepRecord &rec = sim_.recs[n - 1];
		const DrapeScene &sc = sim_.scene;
		const DrapeConfig &cfg = sim_.cfg;
		const double h = cfg.h;
		// The friction blend: s_blend = x + P_k..P_1 ds, last contact first.
		std::vector<double> gds = gPredSum_;
		for (size_t q = rec.pred.size(); q-- > 0;) {
			const DrapePredContact &C = rec.pred[q];
			const v3d gi = get(gds, C.vert);
			g.dmu[C.prim] += gi.dot(-C.dsT);
			const m3d nn = m3d::outer(C.normal, C.normal);
			const m3d P = nn + (m3d::identity() - nn) * (1.0 - C.muClip);
			set(gds, C.vert, P.mulT(gi));
		}
		const std::vector<double> gF = gradFrame(n);
		const bool wind = cfg.wind[0] != 0.0 || cfg.wind[1] != 0.0 || cfg.wind[2] != 0.0;
		for (uint32_t i = 0; i < nV_; ++i) {
			for (int k = 0; k < 3; ++k) {
				const size_t b = 3 * size_t(i) + k;
				ax_[b] = double(v_[b]) + gPredSum_[b] + axDirect_[b] + gF[b];
				av_[b] = h * double(cfg.damp) * gds[b] + h * gF[b];
				if (wind) {
					// s = ... + h^2 (g + wind/m): d/dm = -h^2 wind/m^2.
					const double m = sc.massD[i];
					g.ddensity += (gds[b] + gF[b]) * (-h * h * cfg.wind[k] / (m * m)) * sc.vertArea[i];
				}
			}
		}
		(void)vPrev(n);
		++g.stepsBack;
		--idx_;
		st_ = St::RcBegin;
		rcBegin();
	}

	void finishRecompute() {
		const std::vector<double> gF0 = gradFrame(0);
		double n2 = 0.0;
		for (size_t b = 0; b < ax_.size(); ++b) {
			const double d = ax_[b] + gF0[b];
			n2 += d * d;
		}
		g.dx0Norm = std::sqrt(n2);
		st_ = St::Done;
	}
};
