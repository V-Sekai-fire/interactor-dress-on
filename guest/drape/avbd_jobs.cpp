// The Stage 2 gate's jobs (see avbd_jobs.h). Every job is a list of stages
// on a jobs::StageQueue; a stage that submits GPU work ends the tick, and the
// stage that reads it back runs on the next one. On the CPU backend nothing
// is ever pending, so the same stages run back to back.
//
// The 4-vertex fixture and the tolerances are the native tests' own
// (cloth-dynamics-standalone/src/code/slang_solver/test_avbd_{solver,
// gradcheck,stategrad}.cpp), so the logs compare line for line with
// gates/2-avbd/native_*.log.
#include "avbd_jobs.h"

#include "../common/blake3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "avbd/avbd_cpu.h"
#include "avbd/avbd_rd.h"
#include "avbd/avbd_sim.h"
#include "avbd/cloth_grid.h"

namespace {

std::string fmt(const char *f, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

// --- the 4-vertex fixture (test_avbd_solver.cpp) -----------------------------

constexpr uint32_t NV = 4;
using V12 = std::array<float, 12>;
const V12 kPos0 = { 0, 0, 0, 3, 0, 0, 0, 4, 0, 3, 4, 0 };
const V12 kPred0 = { 1, 2, 3, 3, 0, 0, 0, 4, 0, 3, 4, 0 };
const float kMass[4] = { 2, 1, 1, 1 };
const float kExpected[12] = {
	7.0f / 8.0f, 15.0f / 7.0f, 12.0f / 7.0f,
	12.0f / 5.0f, 1.0f, 0.0f,
	0.0f, 25.0f / 8.0f, 5.0f,
	3.0f, 8.0f / 3.0f, 0.0f,
};

// The fixture's parameters, and which constraint families are uploaded
// (the stategrad bisect turns them off one family at a time).
struct Fix {
	float springK = 1.0f, attachK = 4.0f, triK = 1.0f, bendK = 1.0f, restLen = 2.0f;
	bool sp = true, at = true, tr = true, be = true;
};

template <class S>
void upload4(S &s, const float *pos, const float *pred, const Fix &p) {
	s.setupMesh(NV, pos, pred, kMass, 2.0f);
	const uint32_t p1[1] = { 0 }, p2[1] = { 1 };
	const float rl[1] = { p.restLen }, sk[1] = { p.springK };
	s.uploadSprings(p.sp ? 1 : 0, p1, p2, rl, sk);
	const uint32_t av[1] = { 2 };
	const float af[3] = { 0, 4, 10 }, ak[1] = { p.attachK };
	s.uploadAttachments(p.at ? 1 : 0, av, af, ak);
	const uint32_t ti[3] = { 0, 1, 2 };
	const float uv[4] = { 1, 0, 0, 1 }, tk[1] = { p.triK };
	s.uploadTriangles(p.tr ? 1 : 0, ti, uv, tk);
	const uint32_t bi[4] = { 0, 1, 2, 3 };
	const float bw[4] = { 1, 1, -1, -1 }, bn[1] = { 4 }, bk[1] = { p.bendK };
	s.uploadBendings(p.be ? 1 : 0, bi, bw, bn, bk);
}

// The oracle's pinned two-vertex spring (guest/avbd_fixture.cpp): a spring and
// an attachment, no triangles or bendings.
template <class S>
void upload2(S &s) {
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
	s.uploadTriangles(0, nullptr, nullptr, nullptr);
	s.uploadBendings(0, nullptr, nullptr, nullptr, nullptr);
}

double half_sq(const std::vector<float> &x) {
	double L = 0.0;
	for (float v : x) {
		L += 0.5 * double(v) * double(v);
	}
	return L;
}

bool all_finite(const std::vector<float> &v) {
	for (float x : v) {
		if (!std::isfinite(x)) {
			return false;
		}
	}
	return true;
}

// test_avbd_stategrad's measure (relative to the finite difference, floor 1).
double rel_state(double a, double n) {
	return std::fabs(a - n) / std::max(1.0, std::fabs(n));
}
// test_avbd_gradcheck's measure (relative to the larger, floor 1).
double rel_param(double a, double n) {
	return std::fabs(a - n) / std::max(1.0, std::max(std::fabs(a), std::fabs(n)));
}

// BLAKE3 (first 12 hex digits) over the pairs as little-endian u32s.
std::string b3_pairs(const std::vector<std::pair<uint32_t, uint32_t>> &p) {
	blake3::Ctx h;
	for (const auto &e : p) {
		const uint32_t w[2] = { e.first, e.second };
		unsigned char le[8];
		for (int i = 0; i < 8; ++i)
			le[i] = (unsigned char)(w[i / 4] >> (8 * (i % 4)));
		h.update(le, 8);
	}
	return h.hex().substr(0, 12);
}

// --- backends ----------------------------------------------------------------

template <class S>
struct Backend;

template <>
struct Backend<AvbdCpu> {
	static const char *name() { return "cpu"; }
	static std::unique_ptr<AvbdCpu> make(rdc::Device &, std::string &) { return std::make_unique<AvbdCpu>(); }
};

template <>
struct Backend<AvbdRd> {
	static const char *name() { return "rd"; }
	static std::unique_ptr<AvbdRd> make(rdc::Device &dev, std::string &err) {
		if (!dev.open()) {
			err = dev.error();
			return nullptr;
		}
		auto s = std::make_unique<AvbdRd>(dev);
		if (!s->ok()) {
			err = "rd: " + s->error();
			return nullptr;
		}
		return s;
	}
};

// Every read* accessor, concatenated, with a label per float: what "cpu ==
// rd" compares.
template <class S>
std::vector<float> collect_all(S &s, std::vector<std::string> *labels) {
	std::vector<float> all;
	auto put = [&](const char *name, const std::vector<float> &v) {
		for (size_t i = 0; i < v.size(); ++i) {
			all.push_back(v[i]);
			if (labels) {
				labels->push_back(fmt("%s[%zu]", name, i));
			}
		}
	};
	std::vector<float> a, b, c;
	s.readPositions(a);
	put("x_out", a);
	s.readPositionsGrad(a);
	put("dPos", a);
	s.readMassGrad(a);
	put("dMass", a);
	s.readPredictedGrad(a);
	put("dPred", a);
	s.readSpringGrad(a, b);
	put("dSpringRest", a);
	put("dSpringK", b);
	s.readAttachGrad(a, b, c);
	put("dAttachFixed", a);
	put("dAttachK", b);
	put("dAttachLambda", c);
	s.readTriGrad(a, b, c);
	put("dTriK", a);
	put("dTriL0", b);
	put("dTriL1", c);
	s.readBendGrad(a, b, c);
	put("dBendN", a);
	put("dBendK", b);
	put("dBendLambda", c);
	return all;
}

// --- the single-backend job base -----------------------------------------------

template <class S>
class SJob : public jobs::Job {
public:
	bool pending() const override { return s_ && s_->pending(); }
	void drain() override {
		if (s_) {
			s_->sync();
		}
	}
	bool init(rdc::Device &dev, std::string &err, const std::string &name) {
		name_ = name + " " + Backend<S>::name();
		s_ = Backend<S>::make(dev, err);
		return s_ != nullptr;
	}
	virtual void build() = 0;

protected:
	std::unique_ptr<S> s_;
	std::string name_;
	std::string detail_;
	int stepFails_ = 0;

	void say(const std::string &line) {
		detail_ += line;
		detail_ += '\n';
	}
	void done(bool pass, const std::string &head) { finish(pass, name_ + ": " + head, detail_); }
	void add(jobs::StageQueue::Fn f, bool insert, bool yield = false) {
		if (insert) {
			q.next(std::move(f), yield);
		} else {
			q.push(std::move(f), yield);
		}
	}
	// One fixture loss, two stages: upload (pos, pred, p) and step -- on rd a
	// submit, which ends the tick -- then, a tick later, read x_out and
	// reduce to 0.5 |x_out|^2 (NaN if the read came back short).
	void loss(const V12 &pos, const V12 &pred, const Fix &p, double *out, bool insert = false) {
		S *s = s_.get();
		add([this, s, pos, pred, p]() {
			upload4(*s, pos.data(), pred.data(), p);
			if (s->step() != 0) {
				++stepFails_;
			}
		},
				insert);
		add([s, out]() {
			std::vector<float> x;
			s->readPositions(x);
			*out = x.size() == size_t(NV) * 3 ? half_sq(x) : NAN;
		},
				insert);
	}
};

// --- fixture: the Stage 2 oracle, staged ---------------------------------------

template <class S>
class FixtureJob : public SJob<S> {
public:
	void build() override {
		S *s = this->s_.get();
		auto *j = this;
		for (int c = 0; c < 2; ++c) {
			const bool perturb = c == 1;
			this->q.push([s, perturb]() {
				Fix p;
				if (perturb) {
					p.restLen = 5.0f;
				}
				upload4(*s, kPos0.data(), kPred0.data(), p);
				s->step();
			});
			this->q.push([s, j, perturb]() {
				std::vector<float> x;
				s->readPositions(x);
				float maxd = x.size() == 12 ? 0.0f : INFINITY;
				for (size_t i = 0; i < x.size() && i < 12; ++i) {
					maxd = std::fmax(maxd, std::fabs(x[i] - kExpected[i]));
				}
				const float v0[3] = { x.size() == 12 ? x[0] : NAN, x.size() == 12 ? x[1] : NAN,
					x.size() == 12 ? x[2] : NAN };
				const bool ok = perturb ? maxd > 1e-4f : maxd <= 1e-5f;
				j->fails_ += ok ? 0 : 1;
				j->say(fmt("%s %s (max_abs_diff=%g v0=(%g,%g,%g)%s)", ok ? "PASS" : "FAIL",
						perturb ? "four_vertex_perturbed" : "four_vertex_all_constraints", maxd, v0[0], v0[1], v0[2],
						perturb ? ", expected to differ" : ""));
				if (!perturb) {
					j->exact_ = maxd;
				}
			});
		}
		this->q.push([s]() {
			upload2(*s);
			s->step();
		});
		this->q.push([s, j]() {
			std::vector<float> x;
			s->readPositions(x);
			const bool ok = x.size() == 6 && std::fabs(x[0]) < 0.1f && x[3] < 2.0f && x[3] > 1.0f;
			j->fails_ += ok ? 0 : 1;
			j->say(fmt("%s two_vertex_spring (v0.x=%g pinned, v1.x=%g in (1,2))", ok ? "PASS" : "FAIL",
					x.size() == 6 ? x[0] : NAN, x.size() == 6 ? x[3] : NAN));
			j->done(j->fails_ == 0, fmt("four_vertex_all_constraints max_abs_diff=%g, %d of 3 checks failed",
											  j->exact_, j->fails_));
		});
	}
	int fails_ = 0;
	float exact_ = NAN;
};

// --- backward_smoke: test_avbd_solver.cpp:148-212 ----------------------------------

template <class S>
class SmokeJob : public SJob<S> {
public:
	void build() override {
		S *s = this->s_.get();
		auto *j = this;
		this->q.push([s]() {
			upload4(*s, kPos0.data(), kPred0.data(), Fix());
			s->step();
		});
		this->q.push([s, j]() {
			std::vector<float> x;
			s->readPositions(x);
			j->fwd_ = x.size() == 12 ? 0.0f : INFINITY;
			for (size_t i = 0; i < x.size() && i < 12; ++i) {
				j->fwd_ = std::fmax(j->fwd_, std::fabs(x[i] - kExpected[i]));
			}
			// Loss = x_0.z: the cotangent e_{v0.z}.
			std::vector<float> v(12, 0.0f);
			v[2] = 1.0f;
			j->bwdRc_ = s->stepBackward(v.data());
		});
		this->q.push([s, j]() {
			std::vector<float> dPos, dSL, dSK, dAF, dAK, dAL, dTK, dTL0, dTL1, dBN, dBK, dBL;
			s->readPositionsGrad(dPos);
			s->readSpringGrad(dSL, dSK);
			s->readAttachGrad(dAF, dAK, dAL);
			s->readTriGrad(dTK, dTL0, dTL1);
			s->readBendGrad(dBN, dBK, dBL);
			const std::vector<float> *all[] = { &dPos, &dSL, &dSK, &dAF, &dAK, &dAL, &dTK, &dTL0, &dTL1, &dBN, &dBK, &dBL };
			const char *names[] = { "dPos", "dSpringL", "dSpringK", "dAttachFixed", "dAttachK", "dAttachLam", "dTriK",
				"dTriLam0", "dTriLam1", "dBendN", "dBendK", "dBendLam" };
			const size_t want[] = { 12, 1, 1, 3, 1, 3, 1, 3, 3, 1, 1, 3 };
			int bad = 0;
			for (int a = 0; a < 12; ++a) {
				const bool ok = all[a]->size() == want[a] && all_finite(*all[a]);
				bad += ok ? 0 : 1;
				std::string vals;
				for (float f : *all[a]) {
					vals += fmt(" %.9g", f);
				}
				j->say(fmt("  %-13s n=%zu finite=%s:%s", names[a], all[a]->size(), ok ? "yes" : "NO", vals.c_str()));
			}
			if (bad == 0) {
				// One scalar per line, keyed, for the gate's comparison with
				// native_solver.log (which prints these at %.3g).
				const float kv[] = { dPos[0], dPos[1], dPos[2], dPos[3], dPos[4], dPos[5], dSL[0], dSK[0], dAF[0], dAF[1],
					dAF[2], dAK[0], dTK[0], dTL0[0], dTL0[1], dTL0[2], dBN[0], dBK[0] };
				const char *kn[] = { "dx_v0x", "dx_v0y", "dx_v0z", "dx_v1x", "dx_v1y", "dx_v1z", "dL_s0", "dk_s0",
					"danchor_x", "danchor_y", "danchor_z", "dk_attach", "dk_tri", "dl0_tri_x", "dl0_tri_y", "dl0_tri_z",
					"dn_bend", "dk_bend" };
				for (int k = 0; k < 18; ++k) {
					j->say(fmt("  smoke %s=%.9g", kn[k], kv[k]));
				}
			}
			const bool pass = j->fwd_ <= 1e-5f && j->bwdRc_ == 0 && bad == 0;
			j->done(pass, fmt("forward max_abs_diff=%g, stepBackward rc=%d, %d of 12 gradient arrays non-finite or short",
								  j->fwd_, j->bwdRc_, bad));
		});
	}
	float fwd_ = NAN;
	int bwdRc_ = -1;
};

// --- gradcheck: test_avbd_gradcheck.cpp -------------------------------------------

template <class S>
class GradcheckJob : public SJob<S> {
public:
	static constexpr int P = 5;
	void build() override {
		S *s = this->s_.get();
		auto *j = this;
		this->q.push([s]() {
			upload4(*s, kPos0.data(), kPred0.data(), Fix());
			s->step();
		});
		this->q.push([s, j]() {
			s->readPositions(j->xOut_);
			j->L_ = j->xOut_.size() == 12 ? half_sq(j->xOut_) : NAN;
			// L = 0.5 |x_out|^2, so dL/dx_out = x_out.
			if (j->xOut_.size() != 12 || s->stepBackward(j->xOut_.data()) != 0) {
				j->bwdFail_ = true;
			}
		});
		this->q.push([s, j]() {
			std::vector<float> dRest, dSK, dF, dAK, dAL, dTK, dT0, dT1, dBN, dBK, dBL;
			s->readSpringGrad(dRest, dSK);
			s->readAttachGrad(dF, dAK, dAL);
			s->readTriGrad(dTK, dT0, dT1);
			s->readBendGrad(dBN, dBK, dBL);
			const std::vector<float> *v[P] = { &dRest, &dSK, &dAK, &dTK, &dBK };
			for (int k = 0; k < P; ++k) {
				j->a_[k] = v[k]->empty() ? 0.0 : double((*v[k])[0]);
			}
		});
		for (int k = 0; k < P; ++k) {
			Fix hi, lo;
			hi.*field(k) = float(double(Fix().*field(k)) + kH);
			lo.*field(k) = float(double(Fix().*field(k)) - kH);
			this->loss(kPos0, kPred0, hi, &hi_[k]);
			this->loss(kPos0, kPred0, lo, &lo_[k]);
		}
		this->q.push([j]() { j->report(); });
	}

private:
	static constexpr double kH = 1e-3;
	static float Fix::*field(int k) {
		static float Fix::*const f[P] = { &Fix::restLen, &Fix::springK, &Fix::attachK, &Fix::triK, &Fix::bendK };
		return f[k];
	}
	static const char *pname(int k) {
		static const char *const n[P] = { "restLen", "k_spring", "k_attach", "k_tri", "k_bend" };
		return n[k];
	}

	void report() {
		this->say(fmt("test_avbd_gradcheck: L = %.9g at the 4-vertex fixture", L_));
		this->say("  single step, dL/dx = x_out, central differences h=1e-3, tol rel 5e-2");
		for (int k = 0; k < P; ++k) {
			const double fd = (hi_[k] - lo_[k]) / (2.0 * kH);
			const double rel = rel_param(a_[k], fd);
			const bool ok = rel <= 5e-2 && std::isfinite(fd) && std::isfinite(a_[k]);
			fails_ += ok ? 0 : 1;
			this->say(fmt("  dL/d %-9s analytic=%.9g  fd=%.9g  rel=%.3g  %s", pname(k), a_[k], fd, rel,
					ok ? "ok" : "MISMATCH"));
		}
		if (fails_ == 0 && !bwdFail_ && this->stepFails_ == 0) {
			this->done(true, fmt("5/5 match finite differences, L=%.9g", L_));
			return;
		}
		// Sweep h before blaming the adjoint: a finite difference that drifts
		// with h is the unreliable side.
		static const double hs[5] = { 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 };
		for (int i = 0; i < 5; ++i) {
			Fix hi, lo;
			hi.springK = float(1.0 + hs[i]);
			lo.springK = float(1.0 - hs[i]);
			this->loss(kPos0, kPred0, hi, &swHi_[i], true);
			this->loss(kPos0, kPred0, lo, &swLo_[i], true);
		}
		auto *j = this;
		this->q.next([j]() {
			j->say("  h-sweep on k_spring (an fd stable across h means the analytic value is the wrong one):");
			for (int i = 0; i < 5; ++i) {
				j->say(fmt("    h=%-9g fd=%.9g", hs[i], (j->swHi_[i] - j->swLo_[i]) / (2.0 * hs[i])));
			}
			j->done(false, fmt("%d of 5 gradients disagree with finite differences%s", j->fails_,
								   j->bwdFail_ ? ", stepBackward failed" : ""));
		});
	}

	std::vector<float> xOut_;
	double L_ = NAN;
	bool bwdFail_ = false;
	double a_[P] = {}, hi_[P] = {}, lo_[P] = {}, swHi_[5] = {}, swLo_[5] = {};
	int fails_ = 0;
};

// --- stategrad: test_avbd_stategrad.cpp ------------------------------------------
//
// 6 forward+backward pairs and 168 finite-difference losses (the full model's
// 24 + 24, each bisect case's 24); the falsifiability arm reuses the full
// model's predicted differences, as the native test's are the same numbers.
// On rd each loss is one tick: the read of one and the upload + submit of the
// next share a frame.

template <class S>
class StategradJob : public SJob<S> {
public:
	static constexpr int NC = 5;
	void build() override {
		S *s = this->s_.get();
		auto *j = this;
		// The full model: both accessors from one backward.
		fwd_bwd(Fix(), [j, s]() {
			s->readPositionsGrad(j->gPos_);
			s->readPredictedGrad(j->gPred_);
		});
		for (int i = 0; i < 12; ++i) {
			V12 hi = kPos0, lo = kPos0;
			hi[i] = float(double(kPos0[i]) + kH);
			lo[i] = float(double(kPos0[i]) - kH);
			this->loss(hi, kPred0, Fix(), &posHi_[i]);
			this->loss(lo, kPred0, Fix(), &posLo_[i]);
		}
		for (int i = 0; i < 12; ++i) {
			V12 hi = kPred0, lo = kPred0;
			hi[i] = float(double(kPred0[i]) + kH);
			lo[i] = float(double(kPred0[i]) - kH);
			this->loss(kPos0, hi, Fix(), &predHi_[i]);
			this->loss(kPos0, lo, Fix(), &predLo_[i]);
		}
		// The bisect of d L / d positions by constraint family.
		for (int c = 0; c < NC; ++c) {
			const Fix f = family(c);
			fwd_bwd(f, [j, s, c]() { s->readPositionsGrad(j->gBis_[c]); });
			for (int i = 0; i < 12; ++i) {
				V12 hi = kPos0, lo = kPos0;
				hi[i] = float(double(kPos0[i]) + kH);
				lo[i] = float(double(kPos0[i]) - kH);
				this->loss(hi, kPred0, f, &bisHi_[c][i]);
				this->loss(lo, kPred0, f, &bisLo_[c][i]);
			}
		}
		this->q.push([j]() { j->report(); });
	}

private:
	static constexpr double kH = 1e-3;
	static Fix family(int c) {
		// inertia only, attachment, spring, triangle, bending
		static const bool m[NC][4] = { { 0, 0, 0, 0 }, { 0, 1, 0, 0 }, { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
		Fix f;
		f.sp = m[c][0];
		f.at = m[c][1];
		f.tr = m[c][2];
		f.be = m[c][3];
		return f;
	}
	static const char *family_name(int c) {
		static const char *const n[NC] = { "inertia only", "attachment", "spring", "triangle", "bending" };
		return n[c];
	}
	// Three stages: upload + step; read x_out and stepBackward(x_out); read.
	template <class Read>
	void fwd_bwd(const Fix &f, Read read) {
		S *s = this->s_.get();
		auto *j = this;
		this->q.push([s, f]() {
			upload4(*s, kPos0.data(), kPred0.data(), f);
			s->step();
		});
		this->q.push([s, j]() {
			std::vector<float> x;
			s->readPositions(x);
			if (x.size() != 12 || s->stepBackward(x.data()) != 0) {
				++j->bwdFails_;
			}
		});
		this->q.push(read);
	}

	int sweep(const char *label, const std::vector<float> &analytic, const double *hi, const double *lo) {
		this->say(fmt("  %s", label));
		this->say(fmt("    %-3s %16s %16s %11s", "i", "analytic", "fd", "rel"));
		int bad = 0;
		for (int i = 0; i < 12; ++i) {
			const double a = size_t(i) < analytic.size() ? double(analytic[i]) : 0.0;
			const double n = (hi[i] - lo[i]) / (2.0 * kH);
			const double rel = rel_state(a, n);
			const bool ok = rel < 0.05;
			bad += ok ? 0 : 1;
			this->say(fmt("    %-3d %16.9g %16.9g %11.3g  %s", i, a, n, rel, ok ? "ok" : "MISMATCH"));
		}
		this->say(fmt("    -> %d of 12 disagree", bad));
		return bad;
	}

	void report() {
		this->say("test_avbd_stategrad: positions and predicted checked SEPARATELY, h=1e-3, rel<0.05 vs max(1,|fd|)");
		// Falsifiability: the doubled predicted gradient must be rejected.
		int rejected = 0;
		for (int i = 0; i < 12; ++i) {
			const double a = size_t(i) < gPred_.size() ? 2.0 * double(gPred_[i]) : 0.0;
			const double n = (predHi_[i] - predLo_[i]) / (2.0 * kH);
			rejected += rel_state(a, n) >= 0.05 ? 1 : 0;
		}
		this->say(fmt("  falsifiability: doubled predictedGrad -> %d of 12 rejected %s", rejected,
				rejected > 0 ? "(good, the check has teeth)" : "(BAD: the check cannot fail)"));
		this->say("  bisect of d L / d positions by constraint family:");
		for (int c = 0; c < NC; ++c) {
			int bad = 0;
			for (int i = 0; i < 12; ++i) {
				const double a = size_t(i) < gBis_[c].size() ? double(gBis_[c][i]) : 0.0;
				bad += rel_state(a, (bisHi_[c][i] - bisLo_[c][i]) / (2.0 * kH)) >= 0.05 ? 1 : 0;
			}
			this->say(fmt("    %-14s %d of 12 disagree", family_name(c), bad));
		}
		const int badPos = sweep("d L / d positions  vs readPositionsGrad", gPos_, posHi_, posLo_);
		const int badPred = sweep("d L / d predicted  vs readPredictedGrad", gPred_, predHi_, predLo_);
		const bool pass = badPos == 0 && badPred == 0 && rejected > 0 && bwdFails_ == 0 && this->stepFails_ == 0;
		this->done(pass, fmt("positions %d/12 ok, predicted %d/12 ok, falsifiability %d/12 rejected%s", 12 - badPos,
								 12 - badPred, rejected, bwdFails_ ? ", a backward failed" : ""));
	}

	std::vector<float> gPos_, gPred_, gBis_[NC];
	double posHi_[12] = {}, posLo_[12] = {}, predHi_[12] = {}, predLo_[12] = {};
	double bisHi_[NC][12] = {}, bisLo_[NC][12] = {};
	int bwdFails_ = 0;
};

// --- gradcheck_duals: the backward sees the duals its step saw ------------------
//
// run(1, true) is a step and then the three dual updates, so after it the live
// duals are not the ones the step's force kernels read. The backward binds
// the copies taken before the step (A2b). To make those duals non-zero a
// warm-up run(1, true) goes first; the state is then reset and the measured
// run(1, true) starts from (kPos0, kPred0) with duals lambda1. The finite
// differences perturb only that second run's start positions, so lambda1 is
// held fixed, which is what the backward assumes. The control binds the live
// (post-dual) duals instead.
//
// Finding (2026-09-22): the control cannot fail. The Lean-emitted backward
// kernels (triangle_membrane_force_al_backward, triangle_bending_force_al_
// backward) declare lambda0/lambda1/lambda but never read them: the AL
// forces are affine in the dual with a coefficient that does not depend on
// the positions (dF/dx is constant for the membrane, the stencil weights for
// the bending), so no adjoint output depends on it. The job therefore
// requires the control to be bit-identical to the snapshot arm and reports it
// as inert; any difference that still matched the finite differences, or any
// mismatch of the snapshot arm, is a FAIL. That the duals are live in the
// forward is shown by the loss: warm duals move x_out off the cold fixture.

template <class S>
class DualsJob : public SJob<S> {
public:
	void build() override {
		S *s = this->s_.get();
		auto *j = this;
		two_runs(kPos0);
		this->q.push([s, j]() {
			s->readPositions(j->xOut_);
			if (j->xOut_.size() != 12 || s->stepBackward(j->xOut_.data()) != 0) {
				++j->bwdFails_;
			}
		});
		this->q.push([s, j]() {
			s->readPositionsGrad(j->gSnap_);
			s->setLambdaSnapshotForTest(false);
			if (j->xOut_.size() != 12 || s->stepBackward(j->xOut_.data()) != 0) {
				++j->bwdFails_;
			}
		});
		this->q.push([s, j]() {
			s->readPositionsGrad(j->gLive_);
			s->setLambdaSnapshotForTest(true);
		});
		for (int i = 0; i < 12; ++i) {
			V12 hi = kPos0, lo = kPos0;
			hi[i] = float(double(kPos0[i]) + kH);
			lo[i] = float(double(kPos0[i]) - kH);
			two_runs(hi);
			read_loss(&hi_[i]);
			two_runs(lo);
			read_loss(&lo_[i]);
		}
		this->q.push([j]() { j->report(); });
	}

private:
	static constexpr double kH = 1e-3;
	// Warm-up run(1, true) from the fixture, then the measured run(1, true)
	// from `pos`: two submits, two ticks on rd.
	void two_runs(const V12 &pos) {
		S *s = this->s_.get();
		auto *j = this;
		this->q.push([s, j]() {
			upload4(*s, kPos0.data(), kPred0.data(), Fix());
			if (s->run(1, true) != 0) {
				++j->stepFails_;
			}
		});
		this->q.push([s, j, pos]() {
			s->updateState(pos.data(), kPred0.data());
			if (s->run(1, true) != 0) {
				++j->stepFails_;
			}
		});
	}
	void read_loss(double *out) {
		S *s = this->s_.get();
		this->q.push([s, out]() {
			std::vector<float> x;
			s->readPositions(x);
			*out = x.size() == 12 ? half_sq(x) : NAN;
		});
	}
	int count_bad(const std::vector<float> &g, bool print) {
		int bad = 0;
		for (int i = 0; i < 12; ++i) {
			const double a = size_t(i) < g.size() ? double(g[i]) : 0.0;
			const double n = (hi_[i] - lo_[i]) / (2.0 * kH);
			const double rel = rel_state(a, n);
			bad += rel < 0.05 ? 0 : 1;
			if (print) {
				const double b = size_t(i) < gLive_.size() ? double(gLive_[i]) : 0.0;
				this->say(fmt("    %-3d %16.9g %16.9g %16.9g %11.3g %11.3g", i, a, b, n, rel, rel_state(b, n)));
			}
		}
		return bad;
	}
	void report() {
		this->say("  d L / d positions after warm-up duals: snapshot arm, live-dual arm, fd (h=1e-3); rel vs max(1,|fd|)");
		this->say(fmt("    %-3s %16s %16s %16s %11s %11s", "i", "snapshot", "live duals", "fd", "rel snap", "rel live"));
		const int badSnap = count_bad(gSnap_, true);
		const int badLive = count_bad(gLive_, false);
		float maxd = 0.0f;
		for (size_t i = 0; i < gSnap_.size() && i < gLive_.size(); ++i) {
			maxd = std::fmax(maxd, std::fabs(gSnap_[i] - gLive_[i]));
		}
		this->say(fmt("  snapshot arm %d of 12 disagree; live-dual arm %d of 12 disagree (max |snap - live| = %g)", badSnap,
				badLive, maxd));
		const double Lwarm = xOut_.size() == 12 ? half_sq(xOut_) : NAN;
		std::vector<float> cold(kExpected, kExpected + 12);
		const double Lcold = half_sq(cold);
		this->say(fmt("  duals live in the forward: L(x_out) with warm duals %.9g vs cold fixture %.9g", Lwarm, Lcold));
		const bool inert = gSnap_.size() == 12 && gLive_.size() == 12 && maxd == 0.0f;
		const bool control = badLive > 0 || inert;
		const bool pass = badSnap == 0 && control && std::fabs(Lwarm - Lcold) > 1e-3 && bwdFails_ == 0 &&
				this->stepFails_ == 0;
		this->done(pass, fmt("snapshot %d/12 ok vs fd; no-snapshot control %s", 12 - badSnap,
								 badLive > 0 ? fmt("%d/12 rejected", badLive).c_str()
								 : inert  ? "INERT, bit-identical (the backward kernels never read the duals: negative result)"
										  : "differs but still matches fd (BAD)"));
	}

	std::vector<float> xOut_, gSnap_, gLive_;
	double hi_[12] = {}, lo_[12] = {};
	int bwdFails_ = 0;
};

// --- self_collision ----------------------------------------------------------------

template <class S>
class SelfCollisionJob : public SJob<S> {
public:
	using Pairs = std::vector<std::pair<uint32_t, uint32_t>>;
	struct Case {
		std::string name;
		std::vector<float> pos, radii;
		uint32_t K = 0;
		Pairs expected;
	};

	void build() override {
		{
			// Two pairs, spacing 0.05 and 0.08 under a 0.1 contact distance,
			// far apart from each other.
			Case c;
			c.name = "four_vertex r=0.05 K=4";
			c.pos = { 0, 0, 0, 0.05f, 0, 0, 1, 0, 0, 1.08f, 0, 0 };
			c.radii.assign(4, 0.05f);
			c.K = 4;
			c.expected = { { 0, 1 }, { 2, 3 } };
			cases_.push_back(c);
		}
		{
			// Six vertices all in contact, K = 3: each records its first
			// three j in ascending order, so the pairs are those among 0..3
			// that a lower vertex recorded.
			Case c;
			c.name = "kcap_cluster 6 verts K=3";
			for (int i = 0; i < 6; ++i) {
				c.pos.insert(c.pos.end(), { 0.01f * float(i), 0.0f, 0.0f });
			}
			c.radii.assign(6, 0.05f);
			c.K = 3;
			c.expected = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 } };
			cases_.push_back(c);
		}
		const ClothMesh m = build_cloth_mesh(8, 8, 1.0f, 1.0f, 1.0f, 0.5f);
		{
			// Spacing 1/7 = 0.1429; contact 0.16 takes the grid edges, not the
			// diagonals (0.202): 2 * 8 * 7 = 112 pairs.
			Case c;
			c.name = "panel 8x8 r=0.08 K=8";
			c.pos = m.positions;
			c.radii.assign(m.nVerts(), 0.08f);
			c.K = 8;
			auto vid = [](uint32_t ix, uint32_t iy) { return ix * 8u + iy; };
			for (uint32_t ix = 0; ix < 8; ++ix) {
				for (uint32_t iy = 0; iy < 8; ++iy) {
					if (ix + 1 < 8) {
						c.expected.emplace_back(vid(ix, iy), vid(ix + 1, iy));
					}
					if (iy + 1 < 8) {
						c.expected.emplace_back(vid(ix, iy), vid(ix, iy + 1));
					}
				}
			}
			cases_.push_back(c);
		}
		{
			// Contact 0.14 < 0.1429: nothing (the negative control).
			Case c;
			c.name = "panel 8x8 r=0.07 K=8";
			c.pos = m.positions;
			c.radii.assign(m.nVerts(), 0.07f);
			c.K = 8;
			cases_.push_back(c);
		}
		for (Case &c : cases_) {
			std::sort(c.expected.begin(), c.expected.end());
		}
		S *s = this->s_.get();
		auto *j = this;
		for (size_t k = 0; k < cases_.size(); ++k) {
			this->q.push([s, j, k]() {
				const Case &c = j->cases_[k];
				const uint32_t n = uint32_t(c.radii.size());
				std::vector<float> mass(n, 1.0f);
				s->setupMesh(n, c.pos.data(), c.pos.data(), mass.data(), 1.0f);
				s->uploadSelfCollisionRadii(c.radii.data(), c.K);
				if (s->submitSelfCollisionScan() != 0) {
					++j->stepFails_;
				}
			});
			this->q.push([s, j, k]() {
				const Case &c = j->cases_[k];
				Pairs got;
				const int rc = s->collectSelfCollisions(got);
				std::sort(got.begin(), got.end());
				const bool ok = rc == 0 && got == c.expected;
				j->fails_ += ok ? 0 : 1;
				std::string first;
				for (size_t i = 0; i < got.size() && i < 6; ++i) {
					first += fmt(" (%u,%u)", got[i].first, got[i].second);
				}
				j->say(fmt("  %s %-24s pairs=%zu expected=%zu blake3=%s first:%s", ok ? "PASS" : "FAIL", c.name.c_str(),
						got.size(), c.expected.size(), b3_pairs(got).c_str(), first.c_str()));
			});
		}
		this->q.push([j]() {
			j->done(j->fails_ == 0 && j->stepFails_ == 0,
					fmt("%d of %zu cases exact", int(j->cases_.size()) - j->fails_, j->cases_.size()));
		});
	}

	std::vector<Case> cases_;
	int fails_ = 0;
};

// --- bench_fwd / bench_bwd ---------------------------------------------------------
//
// The pinned panel under gravity, SUBSTEPS substeps of ITERS iterations per
// size. One substep per tick on both backends: on rd the read of substep k
// and the submit of k+1 share a tick; on cpu each substep stage yields. The
// time is the host's, passed in with every tick: from the start of the tick
// that submits substep 0 to the start of the tick after the last readback,
// SUBSTEPS + 1 frames for SUBSTEPS substeps on either backend, divided by
// SUBSTEPS. bench_bwd runs runWithBackward (forward iterations + the adjoint
// of the last, one submit) and reads the positions gradient too.

template <class S>
class BenchJob : public SJob<S> {
public:
	static constexpr int SUBSTEPS = 10, ITERS = 10;
	explicit BenchJob(bool bwd) :
			bwd_(bwd) {}
	void build() override {
		S *s = this->s_.get();
		auto *j = this;
		const bool cpu = std::strcmp(Backend<S>::name(), "cpu") == 0;
		const uint32_t sizes[] = { 8, 16, 32, 64 };
		for (uint32_t n : sizes) {
			if (cpu && n * n > 1024) {
				continue; // the interpreter is not the point of the CPU path at this size
			}
			this->q.push([s, j, n]() {
				j->sim_.reset(new ClothSimT<S>(*s));
				j->sim_->setup(build_cloth_mesh(n, n, 1.0f, 1.0f, 1.0f, 0.5f));
				s->buildColoring();
				j->sim_->iters = ITERS;
				j->cot_.assign(size_t(3) * j->sim_->mesh.nVerts(), 1.0f);
				j->gradFinite_ = true;
			},
					true);
			for (int k = 0; k < SUBSTEPS; ++k) {
				this->q.push([s, j, k]() {
					if (k == 0) {
						j->t0_ = j->now_us;
					} else {
						j->consume();
					}
					auto &sim = *j->sim_;
					using Sim = ClothSimT<S>;
					const uint32_t nV = sim.mesh.nVerts();
					for (uint32_t i = 0; i < nV; ++i) {
						sim.predicted[3 * i + 0] = sim.pos[3 * i + 0] + Sim::H * sim.vel[3 * i + 0];
						sim.predicted[3 * i + 1] =
								sim.pos[3 * i + 1] + Sim::H * sim.vel[3 * i + 1] + Sim::H * Sim::H * Sim::GRAVITY_Y;
						sim.predicted[3 * i + 2] = sim.pos[3 * i + 2] + Sim::H * sim.vel[3 * i + 2];
					}
					s->updateState(sim.pos.data(), sim.predicted.data());
					const int rc = j->bwd_ ? s->runWithBackward(ITERS, false, j->cot_.data()) : s->run(ITERS, false);
					if (rc != 0) {
						++j->stepFails_;
					}
				},
						true);
			}
			this->q.push([j]() { j->consume(); }, true);
			this->q.push([s, j, n]() {
				const double ms = double(j->now_us - j->t0_) / 1000.0 / SUBSTEPS;
				float ymin = 1e30f;
				for (uint32_t i = 0; i < j->sim_->mesh.nVerts(); ++i) {
					ymin = std::fmin(ymin, j->sim_->pos[3 * i + 1]);
				}
				const bool finite = j->sim_->finite() && j->gradFinite_;
				j->allFinite_ = j->allFinite_ && finite;
				j->say(fmt("  %s %s %2ux%-2u nv=%u colors=%u substeps=%d iters=%d ms/substep=%8.2f finite=%s ymin=%.4f",
						j->bwd_ ? "bench_bwd" : "bench_fwd", Backend<S>::name(), n, n, j->sim_->mesh.nVerts(),
						unsigned(s->numColors()), SUBSTEPS, ITERS, ms, finite ? "yes" : "NO", ymin));
				j->sim_.reset();
			});
		}
		this->q.push([j]() {
			j->done(j->allFinite_ && j->stepFails_ == 0, j->allFinite_ ? "every size finite" : "a size went non-finite");
		});
	}

private:
	// Read substep k's result (on rd: the submit of the previous tick).
	void consume() {
		auto &sim = *sim_;
		std::vector<float> np;
		sim.solver.readPositions(np);
		const uint32_t nV = sim.mesh.nVerts();
		if (np.size() != size_t(3) * nV) {
			++this->stepFails_;
			return;
		}
		for (uint32_t i = 0; i < 3 * nV; ++i) {
			sim.vel[i] = (np[i] - sim.pos[i]) / ClothSimT<S>::H * sim.damp;
		}
		sim.pos = np;
		if (bwd_) {
			std::vector<float> g;
			sim.solver.readPositionsGrad(g);
			gradFinite_ = gradFinite_ && g.size() == size_t(3) * nV && all_finite(g);
		}
	}

	bool bwd_;
	std::unique_ptr<ClothSimT<S>> sim_;
	std::vector<float> cot_;
	int64_t t0_ = 0;
	bool gradFinite_ = true, allFinite_ = true;
};

// --- two_vertex_bwd, two_vertex_bwd_nopad: cpu == rd on the backward -------------
//
// Both backends in one job, two arms: step then stepBackward, and
// runWithBackward(3, duals). The cotangent is all ones, so the pinned
// vertex 0 (the attachment's) carries one. Per float |cpu - rd| <= 1e-5 +
// 1e-4 * max(|cpu|, |rd|). The nopad variant points the rd index padding at
// vertex 0 instead of the dummy vertex (setPadFillForTest(0)): the 63
// ragged-tail lanes of attachment_force_al_backward then write
// v_positions[0] alongside the real lane, the race the padding exists to
// prevent. It must disagree, or be recorded as not reproducing here.

class TwoVertexBwdJob : public jobs::Job {
public:
	explicit TwoVertexBwdJob(bool nopad) :
			nopad_(nopad) {}
	bool pending() const override { return rd_ && rd_->pending(); }
	void drain() override {
		if (rd_) {
			rd_->sync();
		}
	}
	bool init(rdc::Device &dev, std::string &err) {
		rd_ = Backend<AvbdRd>::make(dev, err);
		return rd_ != nullptr;
	}
	void build() {
		auto *j = this;
		AvbdRd *rd = rd_.get();
		const std::vector<float> ones(6, 1.0f);
		for (int arm = 0; arm < 2; ++arm) {
			// The cpu side runs inline in the first stage of the arm.
			q.push([j, rd, arm, ones]() {
				AvbdCpu cpu;
				upload2(cpu);
				if (arm == 0) {
					cpu.step();
					cpu.stepBackward(ones.data());
				} else {
					cpu.runWithBackward(3, true, ones.data());
				}
				j->labels_.clear();
				j->cpu_ = collect_all(cpu, &j->labels_);
				if (j->nopad_) {
					rd->setPadFillForTest(0);
				}
				upload2(*rd);
				if (arm == 0) {
					rd->step();
				} else {
					rd->runWithBackward(3, true, ones.data());
				}
			});
			if (arm == 0) {
				q.push([rd, ones]() { rd->stepBackward(ones.data()); });
			}
			q.push([j, rd, arm]() {
				const std::vector<float> got = collect_all(*rd, nullptr);
				j->compare(arm == 0 ? "step+stepBackward" : "runWithBackward(3,duals)", got);
			});
		}
		q.push([j]() {
			const char *name = j->nopad_ ? "two_vertex_bwd_nopad cpu-vs-rd(pad->0)" : "two_vertex_bwd cpu-vs-rd";
			if (!j->nopad_) {
				j->finish(j->bad_ == 0 && j->sizeBad_ == 0,
						fmt("%s: %d floats disagree over 2 arms (max |cpu-rd| = %g)", name, j->bad_, j->maxd_), j->detail_);
			} else {
				// A negative control: PASS either way, the verdict is in the text.
				j->finish(j->sizeBad_ == 0,
						fmt("%s: %s (%d floats disagree over 2 arms, max |cpu-rd| = %g)", name,
								j->bad_ > 0 ? "race REPRODUCED" : "race did NOT reproduce on this GPU (negative result)",
								j->bad_, j->maxd_),
						j->detail_);
			}
		});
	}

private:
	void compare(const char *arm, const std::vector<float> &rd) {
		if (rd.size() != cpu_.size()) {
			++sizeBad_;
			detail_ += fmt("  %s: size cpu=%zu rd=%zu\n", arm, cpu_.size(), rd.size());
			return;
		}
		int bad = 0;
		float maxd = 0.0f;
		std::string where;
		for (size_t i = 0; i < rd.size(); ++i) {
			const float a = cpu_[i], b = rd[i];
			const float d = std::fabs(a - b);
			const bool ok = std::isfinite(a) && std::isfinite(b) && d <= 1e-5f + 1e-4f * std::fmax(std::fabs(a), std::fabs(b));
			maxd = std::fmax(maxd, d);
			if (!ok) {
				++bad;
				if (bad <= 6) {
					where += fmt(" %s cpu=%.9g rd=%.9g;", labels_[i].c_str(), a, b);
				}
			}
		}
		bad_ += bad;
		maxd_ = std::fmax(maxd_, maxd);
		detail_ += fmt("  %-26s floats=%zu disagree=%d max|cpu-rd|=%g%s\n", arm, rd.size(), bad, maxd, where.c_str());
		// The pinned vertex's positions gradient, the one the race touches.
		detail_ += fmt("    dPos[v0] cpu=(%.9g, %.9g, %.9g) rd=(%.9g, %.9g, %.9g)\n", cpu_[6], cpu_[7], cpu_[8], rd[6], rd[7],
				rd[8]);
	}

	bool nopad_;
	std::unique_ptr<AvbdRd> rd_;
	std::vector<float> cpu_;
	std::vector<std::string> labels_;
	std::string detail_;
	int bad_ = 0, sizeBad_ = 0;
	float maxd_ = 0.0f;
};

// --- the factory -------------------------------------------------------------------

template <class S>
std::unique_ptr<jobs::Job> make_on(const std::string &name, rdc::Device &dev, std::string &err) {
	std::unique_ptr<SJob<S>> j;
	if (name == "fixture") {
		j.reset(new FixtureJob<S>());
	} else if (name == "backward_smoke") {
		j.reset(new SmokeJob<S>());
	} else if (name == "gradcheck") {
		j.reset(new GradcheckJob<S>());
	} else if (name == "stategrad") {
		j.reset(new StategradJob<S>());
	} else if (name == "gradcheck_duals") {
		j.reset(new DualsJob<S>());
	} else if (name == "self_collision") {
		j.reset(new SelfCollisionJob<S>());
	} else if (name == "bench_fwd") {
		j.reset(new BenchJob<S>(false));
	} else if (name == "bench_bwd") {
		j.reset(new BenchJob<S>(true));
	} else {
		err = "unknown job '" + name + "' (jobs: " + avbd_job_names() + ")";
		return nullptr;
	}
	if (!j->init(dev, err, name)) {
		return nullptr;
	}
	j->build();
	return j;
}

} // namespace

const char *avbd_job_names() {
	return "fixture backward_smoke gradcheck stategrad gradcheck_duals self_collision bench_fwd bench_bwd "
		   "two_vertex_bwd two_vertex_bwd_nopad";
}

std::unique_ptr<jobs::Job> make_avbd_job(const std::string &name, const std::string &backend, rdc::Device &dev,
		std::string &err) {
	if (name == "two_vertex_bwd" || name == "two_vertex_bwd_nopad") {
		auto j = std::make_unique<TwoVertexBwdJob>(name == "two_vertex_bwd_nopad");
		if (!j->init(dev, err)) {
			return nullptr;
		}
		j->build();
		return j;
	}
	if (backend == "cpu") {
		return make_on<AvbdCpu>(name, dev, err);
	}
	if (backend == "rd") {
		return make_on<AvbdRd>(name, dev, err);
	}
	err = "backend must be cpu or rd";
	return nullptr;
}
