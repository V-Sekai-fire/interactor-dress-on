// drape_sim -- DiffCloth's AVBD forward step, staged, over AvbdCpu or AvbdRd.
//
// Descended from guest/avbd/avbd_sim.h (ClothSimT); the step itself is a port
// of cloth-dynamics-standalone Simulation::step() (Simulation.cpp:1185-2106,
// e361584), the AVBD path only:
//  1. predictor s = x + h v + h^2 M^-1 f_ext (f_ext = m g + wind, as float,
//     fillForces 245-305); the solver's predictor uses damp on v (1245-1300);
//  2. friction as a tangent-only predictor blend (1302-1392): every vertex a
//     primitive's isInContact accepts at the START of the step (the per-vertex
//     bootstrap: upstream's triangle contact list is never filled on the AVBD
//     path, so the bootstrap runs every step) gets s_t scaled by (1 - mu);
//  3. updateState(x, s_blend), `iters` solver iterations (+ duals if AL);
//  4. v = (x_avbd - x)/h, x = x_avbd (1818-1842);
//  5. primitive projection (x -= dist n where dist < 0) and the velocity
//     response on the relative velocity (normal inflow zeroed, tangent scaled
//     by 1 - mu), recording d v/d mu = -v_tan per contact (frictionSensitivity,
//     1845-1931);
//  6. self-collision: up to selfPasses passes of updateState(x, x) + the
//     solver's scan + a symmetric push to the sum of radii (1932-2020).
// The state is float as upstream's Particle (Vec3f pos/velocity); predictor
// and contact arithmetic are double where upstream's are, and rounded to
// float where upstream stores into a Vec3f.
//
// The record of each step: the predictor s (upstream's ForwardInformation::x:
// what the native OBJ frames and MATCH_TRAJECTORY see), the uploaded
// s_blend, the solver output, the final x and v, the friction sensitivities,
// and what the recompute backward modes need to chain through contact and
// self-collision (the contacts and pushes in the order they were applied).
//
// Staging (AGENTS.md rule 4): advance() runs one phase. A phase that submits
// GPU work returns and the next phase, which reads it back, runs on a later
// tick: solve -> contact (reads x_avbd, submits the scan) -> self pass (reads
// the pairs, maybe submits the next scan) -> done. On AvbdCpu nothing is
// pending and a whole step can run in one tick.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "drape_config.h"
#include "drape_scene.h"
#include "primitives.h"

struct DrapePredContact {
	uint32_t vert, prim;
	v3d normal;
	v3d dsT;       // the tangent part of s - x before the blend
	double muClip; // clamp(mu, 0, 1)
};

struct DrapeContact {
	uint32_t vert, prim;
	v3d normal;
	double dist;
	bool penetrating;
	v3d posBefore; // the position isInContact saw (projection Jacobian)
	double vn;     // relative normal velocity before the response
	double muClip;
	v3d vTan;      // relative tangent velocity before the response
};

struct DrapeFriction {
	uint32_t prim, vert;
	v3d vTan; // d v_new / d mu = -vTan
};

struct DrapePush {
	uint32_t a, b;
	v3d diff; // x_a - x_b before the push
	double dist, thresh;
};

struct DrapeStepRecord {
	std::vector<double> s;     // predictor (the frame)
	std::vector<float> sBlend; // uploaded predictor
	std::vector<float> xAvbd;  // solver output
	std::vector<float> x, v;   // state after the step
	std::vector<DrapePredContact> pred;
	std::vector<DrapeContact> contacts;
	std::vector<DrapeFriction> fric;
	// Every applied push in application order; pass k is [passEnd[k-1], passEnd[k]).
	std::vector<DrapePush> pushes;
	std::vector<uint32_t> passEnd;
	uint32_t selfPairs = 0, projHits = 0;
};

template <class Solver>
class DrapeSimT {
public:
	explicit DrapeSimT(Solver &s) : solver(s) {}

	Solver &solver;
	DrapeConfig cfg;
	DrapeScene scene;
	std::vector<float> x, v;            // current state
	std::vector<DrapeStepRecord> recs;  // recs[k]: step k+1
	std::string err;
	bool failed = false;

	// Upload the scene and material; reset the state and the records.
	bool setup() {
		const uint32_t nV = scene.nV;
		err.clear();
		failed = false;
		if (nV == 0) {
			err = "no scene";
			failed = true;
			return false;
		}
		scene.applyMaterial(cfg);
		const float invHSq = float(1.0 / (cfg.h * cfg.h));
		solver.setupMesh(nV, scene.x0.data(), scene.x0.data(), scene.mass.data(), invHSq);
		solver.uploadSprings(0, nullptr, nullptr, nullptr, nullptr);
		solver.uploadAttachments(scene.nAttach(), scene.attachVert.data(), scene.attachFixed.data(),
				scene.attachK.data());
		if (cfg.membrane && scene.nTri() > 0) {
			solver.uploadTriangles(scene.nTri(), scene.tri.data(), scene.triInvUV.data(), scene.triK.data());
		} else {
			solver.uploadTriangles(0, nullptr, nullptr, nullptr);
		}
		if (cfg.bending) {
			solver.uploadBendings(scene.nBend(), scene.bendIdx.data(), scene.bendW.data(), scene.bendN.data(),
					scene.bendK.data());
		} else {
			solver.uploadBendings(0, nullptr, nullptr, nullptr, nullptr);
		}
		if (cfg.alGammaSet) {
			solver.setGammaScale(cfg.alGamma);
		}
		if (cfg.colors) {
			solver.buildColoring();
		}
		std::vector<float> rad(nV);
		for (uint32_t i = 0; i < nV; ++i) {
			rad[i] = float(scene.radii[i]);
		}
		solver.uploadSelfCollisionRadii(rad.data(), uint32_t(cfg.selfK));
		x = scene.x0;
		v = scene.v0;
		recs.clear();
		phase_ = Phase::Idle;
		return true;
	}

	// Rewind to the initial state keeping the uploads (a new forward with
	// the same parameters).
	void rewind() {
		x = scene.x0;
		v = scene.v0;
		recs.clear();
		phase_ = Phase::Idle;
		failed = false;
		err.clear();
	}

	size_t steps() const { return recs.size(); }
	bool midStep() const { return phase_ != Phase::Idle; }

	// Frame i as upstream exports it: x0, then each step's predictor.
	std::vector<double> frame(size_t i) const {
		if (i == 0) {
			return std::vector<double>(scene.x0.begin(), scene.x0.end());
		}
		return recs[i - 1].s;
	}

	bool finite() const {
		for (float f : x) {
			if (!std::isfinite(f)) {
				return false;
			}
		}
		return true;
	}

	// One phase of the current step (starting one if none is in progress).
	// True when the step completed in this call.
	bool advance() {
		switch (phase_) {
			case Phase::Idle:
				begin();
				return false;
			case Phase::Solved:
				return solved();
			case Phase::Scanned:
				return scanned();
		}
		return false;
	}

private:
	enum class Phase { Idle, Solved, Scanned };
	Phase phase_ = Phase::Idle;
	DrapeStepRecord cur_;
	std::vector<float> pos_, vel_;
	int pass_ = 0;

	static v3d at(const std::vector<float> &a, uint32_t i) {
		return { double(a[3 * i]), double(a[3 * i + 1]), double(a[3 * i + 2]) };
	}
	static void put(std::vector<float> &a, uint32_t i, const v3d &p) {
		a[3 * i] = float(p.x);
		a[3 * i + 1] = float(p.y);
		a[3 * i + 2] = float(p.z);
	}

	void begin() {
		const uint32_t nV = scene.nV;
		const double h = cfg.h, h2 = h * h;
		cur_ = DrapeStepRecord();
		cur_.s.resize(3 * size_t(nV));
		std::vector<double> sb(3 * size_t(nV));
		for (uint32_t i = 0; i < nV; ++i) {
			const double minv = 1.0 / scene.massD[i];
			for (int k = 0; k < 3; ++k) {
				const size_t b = 3 * size_t(i) + k;
				const float fext = float(cfg.gravity[k] * scene.massD[i] + cfg.wind[k]);
				const double xd = double(x[b]), vd = double(v[b]);
				cur_.s[b] = xd + h * vd;
				cur_.s[b] += h2 * minv * double(fext);
				if (cfg.damp == 1.0f) {
					sb[b] = cur_.s[b];
				} else {
					sb[b] = xd + h * double(cfg.damp) * vd;
					sb[b] += h2 * minv * double(fext);
				}
			}
		}
		// The tangent-only friction blend, per-vertex bootstrap.
		if (cfg.contact && cfg.frictionPred) {
			for (uint32_t i = 0; i < nV; ++i) {
				for (uint32_t p = 0; p < scene.prims.size(); ++p) {
					const Primitive &pr = scene.prims[p];
					v3d n, vout;
					double dist = 0.0;
					if (!pr.isInContact(at(x, i), at(v, i), n, dist, vout)) {
						continue;
					}
					if (pr.mu <= 0.0) {
						continue;
					}
					const double muc = std::min(1.0, std::max(0.0, pr.mu));
					const v3d xi = at(x, i);
					const v3d ds(sb[3 * i] - xi.x, sb[3 * i + 1] - xi.y, sb[3 * i + 2] - xi.z);
					const double dsn = ds.dot(n);
					const v3d dsN = n * dsn;
					const v3d dsT = ds - dsN;
					const v3d dsNew = dsN + dsT * (1.0 - muc);
					sb[3 * i] = xi.x + dsNew.x;
					sb[3 * i + 1] = xi.y + dsNew.y;
					sb[3 * i + 2] = xi.z + dsNew.z;
					cur_.pred.push_back({ i, p, n, dsT, muc });
				}
			}
		}
		cur_.sBlend.resize(3 * size_t(nV));
		for (size_t b = 0; b < sb.size(); ++b) {
			cur_.sBlend[b] = float(sb[b]);
		}
		solver.updateState(x.data(), cur_.sBlend.data());
		if (scene.nAttach() > 0) {
			solver.updateAttachmentFixedPos(scene.attachFixed.data());
		}
		if (solver.run(cfg.iters, cfg.al) != 0) {
			failed = true;
			err = "solver run failed";
		}
		phase_ = Phase::Solved;
	}

	bool solved() {
		const uint32_t nV = scene.nV;
		solver.readPositions(cur_.xAvbd);
		if (cur_.xAvbd.size() != 3 * size_t(nV)) {
			failed = true;
			err = "short readback";
			cur_.xAvbd.assign(3 * size_t(nV), NAN);
		}
		const double invH = 1.0 / cfg.h;
		pos_ = cur_.xAvbd;
		vel_.resize(3 * size_t(nV));
		for (uint32_t i = 0; i < nV; ++i) {
			const v3d np = at(cur_.xAvbd, i), sp = at(x, i);
			put(vel_, i, (np - sp) * invH);
		}
		if (cfg.contact) {
			for (uint32_t i = 0; i < nV; ++i) {
				for (uint32_t p = 0; p < scene.prims.size(); ++p) {
					const Primitive &pr = scene.prims[p];
					v3d n, vout;
					double dist = 0.0;
					const v3d pb = at(pos_, i);
					if (!pr.isInContact(pb, at(vel_, i), n, dist, vout)) {
						continue;
					}
					const bool pen = dist < 0.0;
					if (pen) {
						// Vec3f -= Vec3f(dist * normal): rounded, then float.
						const v3d d = n * dist;
						for (int k = 0; k < 3; ++k) {
							pos_[3 * i + k] -= float(d[k]);
						}
						++cur_.projHits;
					}
					// v_rel = velocity - v_out in float (Vec3f::operator-).
					float vr[3];
					for (int k = 0; k < 3; ++k) {
						vr[k] = vel_[3 * i + k] - float(vout[k]);
					}
					const v3d vrel(vr[0], vr[1], vr[2]);
					const double vn = vrel.dot(n);
					const v3d vtan = vrel - n * vn;
					const double clip = std::min(1.0, std::max(0.0, pr.mu));
					const double vnA = vn < 0.0 ? 0.0 : vn;
					const v3d after = vtan * (1.0 - clip) + n * vnA;
					put(vel_, i, after + vout);
					if (pr.mu > 0.0 && pr.mu < 1.0) {
						cur_.fric.push_back({ p, i, vtan });
					}
					cur_.contacts.push_back({ i, p, n, dist, pen, pb, vn, clip, vtan });
				}
			}
		}
		if (cfg.selfCollision) {
			pass_ = 0;
			submitScan();
			phase_ = Phase::Scanned;
			return false;
		}
		finishStep();
		return true;
	}

	void submitScan() {
		solver.updateState(pos_.data(), pos_.data());
		if (solver.submitSelfCollisionScan() != 0) {
			failed = true;
			err = "self-collision scan failed";
		}
	}

	bool scanned() {
		std::vector<std::pair<uint32_t, uint32_t>> pairs;
		solver.collectSelfCollisions(pairs);
		cur_.selfPairs += uint32_t(pairs.size());
		const size_t before = cur_.pushes.size();
		for (const auto &pr : pairs) {
			const uint32_t a = pr.first, b = pr.second;
			float df[3];
			for (int k = 0; k < 3; ++k) {
				df[k] = pos_[3 * a + k] - pos_[3 * b + k];
			}
			const v3d diff(df[0], df[1], df[2]);
			const double dist = diff.norm();
			const double thresh = scene.radii[a] + scene.radii[b];
			if (dist < thresh && dist > 1e-10) {
				const double overlap = thresh - dist;
				const v3d push = (diff / dist) * (overlap * 0.5);
				for (int k = 0; k < 3; ++k) {
					pos_[3 * a + k] += float(push[k]);
					pos_[3 * b + k] -= float(push[k]);
				}
				cur_.pushes.push_back({ a, b, diff, dist, thresh });
			}
		}
		++pass_;
		cur_.passEnd.push_back(uint32_t(cur_.pushes.size()));
		if (cur_.pushes.size() > before && pass_ < cfg.selfPasses) {
			submitScan();
			return false;
		}
		finishStep();
		return true;
	}

	void finishStep() {
		cur_.x = pos_;
		cur_.v = vel_;
		x = pos_;
		v = vel_;
		recs.push_back(std::move(cur_));
		cur_ = DrapeStepRecord();
		phase_ = Phase::Idle;
	}
};
