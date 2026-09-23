// drape_session -- a drape session: one backend's solver, a DrapeSimT over
// it, a target and the last result, driven by stages on a jobs::StageQueue
// (see drape_jobs.h). Header-only apart from the backend factory in
// drape_jobs.cpp, so the host harness (tests/drape_host) runs the same code.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "drape_backward.h"
#include "drape_config.h"
#include "drape_scene.h"
#include "drape_sim.h"
#include "jobs.h"

inline std::string drape_fmt(const char *f, ...) {
	char b[2048];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

class DrapeSession {
public:
	virtual ~DrapeSession() = default;
	virtual const char *backend() const = 0;
	// A GPU submit is in flight (the tick must end; do not destroy this frame).
	virtual bool pending() const = 0;
	virtual void drain() = 0;
	// Upload a scene with a configuration; the state goes back to x0.
	virtual bool load(const DrapeScene &scene, const DrapeConfig &cfg, std::string &err) = 0;
	// Knobs that need no upload (iters, contact flags, mu, truncation, ...).
	virtual void setLive(const DrapeConfig &cfg) = 0;
	virtual void setPrims(const std::vector<Primitive> &prims) = 0;
	virtual void rewind() = 0;
	virtual void enqueueForward(jobs::StageQueue &q, int steps) = 0;
	// The fit phase (Cut 6d): up to maxSteps steps with gravity off and no
	// velocity carried (damp 0, quasi-static), the scene's fit attachments
	// refreshed (DrapeSimT fit mode); stops early once the largest vertex
	// move of a refresh cycle's last step is under tol. Gravity and damp
	// are restored after it. Needs a fit set (scene_fit_set).
	virtual void enqueueFit(jobs::StageQueue &q, int maxSteps, double tol) = 0;
	virtual void enqueueBackward(jobs::StageQueue &q, DrapeLoss loss, DrapeMode mode) = 0;
	virtual DrapeTarget &target() = 0;
	// The recorded frames as a MATCH_TRAJECTORY target.
	virtual void targetFromFrames() = 0;
	virtual size_t steps() const = 0;
	virtual bool midStep() const = 0;
	virtual std::vector<double> frame(size_t i) const = 0;
	virtual const std::vector<float> &positions() const = 0;
	virtual const DrapeScene &scene() const = 0;
	virtual const DrapeConfig &config() const = 0;
	virtual const std::string &result() const = 0;
	virtual const DrapeGrad &grad() const = 0;
	virtual bool ok() const = 0;
	// Timing: the host clock of the tick that started / finished the last op.
	int64_t now_us = 0;
};

template <class S>
class DrapeSessionT : public DrapeSession {
public:
	std::unique_ptr<S> solver;
	std::unique_ptr<DrapeSimT<S>> sim;
	DrapeTarget tgt;
	std::unique_ptr<DrapeBackwardT<S>> bwd;
	DrapeGrad lastGrad;
	std::string res;
	bool okFlag = true;
	const int64_t *clock = nullptr;
	// Called after each completed step (jobs collect statistics).
	std::function<void(const DrapeStepRecord &, const std::vector<float> &xPrev)> onStep;
	int fwdRemaining = 0, fwdAsked = 0;
	int64_t t0 = 0;
	size_t fwdFrom = 0;
	std::vector<float> xPrev;

	int64_t now() const { return clock ? *clock : now_us; }

	const char *backendName = "cpu";
	const char *backend() const override { return backendName; }
	bool pending() const override { return solver && solver->pending(); }
	void drain() override {
		if (solver) {
			solver->sync();
		}
	}
	bool load(const DrapeScene &scene, const DrapeConfig &cfg, std::string &err) override {
		bwd.reset();
		sim = std::make_unique<DrapeSimT<S>>(*solver);
		sim->cfg = cfg;
		sim->scene = scene;
		if (!sim->setup()) {
			err = sim->err;
			okFlag = false;
			return false;
		}
		okFlag = true;
		res = "LOADED " + std::string(backend()) + " " + sim->scene.describe() + "\n" + sim->cfg.dump();
		return true;
	}
	void setLive(const DrapeConfig &c) override {
		if (!sim) {
			return;
		}
		DrapeConfig &d = sim->cfg;
		d.iters = c.iters;
		d.damp = c.damp;
		d.al = c.al;
		d.contact = c.contact;
		d.frictionPred = c.frictionPred;
		d.selfCollision = c.selfCollision;
		d.selfPasses = c.selfPasses;
		d.bwdTruncateK = c.bwdTruncateK;
		d.gradientClipping = c.gradientClipping;
		d.gradientClippingThreshold = c.gradientClippingThreshold;
		d.nativeMuDoubleCarry = c.nativeMuDoubleCarry;
		for (int k = 0; k < 3; ++k) {
			d.gravity[k] = c.gravity[k];
			d.wind[k] = c.wind[k];
		}
		d.mu = c.mu;
	}
	void setPrims(const std::vector<Primitive> &p) override {
		if (sim) {
			sim->scene.prims = p;
		}
	}
	void rewind() override {
		if (sim) {
			sim->rewind();
		}
	}
	DrapeTarget &target() override { return tgt; }
	void targetFromFrames() override {
		tgt.frames.clear();
		tgt.frameSize = sim ? 3 * size_t(sim->scene.nV) : 0;
		for (size_t i = 0; sim && i <= sim->steps(); ++i) {
			const std::vector<double> f = sim->frame(i);
			tgt.frames.insert(tgt.frames.end(), f.begin(), f.end());
		}
	}
	size_t steps() const override { return sim ? sim->steps() : 0; }
	bool midStep() const override { return sim && sim->midStep(); }
	std::vector<double> frame(size_t i) const override {
		if (!sim || i > sim->steps()) {
			return {};
		}
		return sim->frame(i);
	}
	const std::vector<float> &positions() const override {
		static const std::vector<float> none;
		return sim ? sim->x : none;
	}
	const DrapeScene &scene() const override {
		static const DrapeScene none;
		return sim ? sim->scene : none;
	}
	const DrapeConfig &config() const override {
		static const DrapeConfig none;
		return sim ? sim->cfg : none;
	}
	const std::string &result() const override { return res; }
	const DrapeGrad &grad() const override { return lastGrad; }
	bool ok() const override { return okFlag; }

	void enqueueForward(jobs::StageQueue &q, int steps) override {
		jobs::StageQueue *qp = &q; // by pointer: a captured reference parameter dangles across ticks
		q.push([this, qp, steps]() {
			if (!sim) {
				res = "FAIL forward: no scene";
				okFlag = false;
				return;
			}
			fwdRemaining = steps;
			fwdAsked = steps;
			fwdFrom = sim->steps();
			t0 = now();
			xPrev = sim->x;
			forwardStage(*qp);
		});
	}

	void forwardStage(jobs::StageQueue &q) {
		if (fwdRemaining > 0 && !sim->failed) {
			if (sim->advance()) {
				--fwdRemaining;
				if (onStep) {
					onStep(sim->recs.back(), xPrev);
				}
				xPrev = sim->x;
			}
		}
		if (fwdRemaining > 0 && !sim->failed) {
			jobs::StageQueue *qp = &q;
			q.next([this, qp]() { forwardStage(*qp); });
			return;
		}
		finishForward();
	}

	void finishForward() {
		const double wall_ms = double(now() - t0) / 1000.0;
		const size_t done = sim->steps() - fwdFrom;
		size_t fric = 0, proj = 0, pushes = 0;
		for (size_t k = fwdFrom; k < sim->steps(); ++k) {
			fric += sim->recs[k].fric.size();
			proj += sim->recs[k].projHits;
			pushes += sim->recs[k].pushes.size();
		}
		float ymin = INFINITY;
		for (uint32_t i = 0; i < sim->scene.nV; ++i) {
			ymin = std::min(ymin, sim->x[3 * i + 1]);
		}
		const bool fin = sim->finite();
		okFlag = !sim->failed && fin && int(done) == fwdAsked;
		res = drape_fmt("%s forward %s steps=%zu (+%zu of %d) finite=%s ymin=%.6f friction_events=%zu projections=%zu "
				  "self_pushes=%zu wall_ms=%.1f ms/step=%.3f%s%s",
				okFlag ? "DONE" : "FAIL", backend(), sim->steps(), done, fwdAsked, fin ? "yes" : "NO", ymin, fric, proj,
				pushes, wall_ms, done ? wall_ms / double(done) : 0.0, sim->failed ? " error=" : "",
				sim->failed ? sim->err.c_str() : "");
	}

	// --- the fit phase (Cut 6d) ---
	double fitTol = 0.0;
	double fitGravity[3] = { 0.0, 0.0, 0.0 };
	float fitDamp = 1.0f;
	bool fitConverged = false;
	int fitStepsDone = 0, settleDone = 0;

	void enqueueFit(jobs::StageQueue &q, int maxSteps, double tol) override {
		jobs::StageQueue *qp = &q;
		q.push([this, qp, maxSteps, tol]() {
			if (!sim) {
				res = "FAIL fit: no scene";
				okFlag = false;
				return;
			}
			if (sim->scene.nFit == 0) {
				res = "FAIL fit: no fit set (drape_fit_set first)";
				okFlag = false;
				return;
			}
			for (int k = 0; k < 3; ++k) {
				fitGravity[k] = sim->cfg.gravity[k];
				sim->cfg.gravity[k] = 0.0;
			}
			fitDamp = sim->cfg.damp;
			sim->cfg.damp = 0.0f;
			sim->fitActive = true;
			sim->fitSettling = false;
			sim->fitSteps = 0;
			fitStepsDone = 0;
			settleDone = 0;
			sim->fitRefreshed = 0;
			sim->fitMisses = 0;
			sim->fitRest0 = sim->scene.rest;
			sim->fitSim = Similarity();
			sim->fitRestUpdates = 0;
			fitTol = tol;
			fitConverged = false;
			fwdRemaining = maxSteps;
			fwdAsked = maxSteps;
			fwdFrom = sim->steps();
			t0 = now();
			xPrev = sim->x;
			fitStage(*qp);
		});
	}

	void fitStage(jobs::StageQueue &q) {
		if (fwdRemaining > 0 && !sim->failed) {
			if (sim->advance()) {
				--fwdRemaining;
				if (onStep) {
					onStep(sim->recs.back(), xPrev);
				}
				xPrev = sim->x;
				if (sim->fitSettling) {
					++settleDone;
				} else {
					++fitStepsDone;
					// The stop test on the last step of a refresh cycle only: the
					// step after a refresh moves toward new targets.
					const uint32_t r = uint32_t(std::max(1, sim->scene.fitRefresh));
					if (sim->fitSteps % r == 0 && sim->lastMaxDisp < fitTol) {
						fitConverged = true;
					}
				}
			}
		}
		const bool fitOver = sim->failed || fitConverged || fwdRemaining <= 0;
		if (fitOver && !sim->fitSettling && !sim->failed && sim->scene.fitSettle > 0) {
			// The settle: the fit pull off, targets frozen, the pins, the
			// anchor and the collider on.
			sim->fitSettling = true;
			sim->setFitPull(false);
			fwdRemaining = sim->scene.fitSettle;
		}
		if (fwdRemaining > 0 && !sim->failed) {
			jobs::StageQueue *qp = &q;
			q.next([this, qp]() { fitStage(*qp); });
			return;
		}
		if (sim->fitSettling) {
			sim->setFitPull(true);
			sim->fitSettling = false;
		}
		finishFit();
	}

	void finishFit() {
		const double wall_ms = double(now() - t0) / 1000.0;
		const size_t done = sim->steps() - fwdFrom;
		for (int k = 0; k < 3; ++k) {
			sim->cfg.gravity[k] = fitGravity[k];
		}
		sim->cfg.damp = fitDamp;
		sim->fitActive = false;
		size_t proj = 0, pushes = 0;
		for (size_t k = fwdFrom; k < sim->steps(); ++k) {
			proj += sim->recs[k].projHits;
			pushes += sim->recs[k].pushes.size();
		}
		const bool fin = sim->finite();
		okFlag = !sim->failed && fin && done > 0;
		float ymin = INFINITY, ymax = -INFINITY;
		for (uint32_t i = 0; i < sim->scene.nV; ++i) {
			ymin = std::min(ymin, sim->x[3 * i + 1]);
			ymax = std::max(ymax, sim->x[3 * i + 1]);
		}
		res = drape_fmt("%s fit %s steps=%zu (+%zu of %d, settle %d of %d) converged=%s max_disp=%.3g tol=%g refreshed=%u "
				  "misses=%u dist_min=%.4f dist_mean=%.4f gap=%g kFit=%g fit=%u pins=%u iters=%d h=%.6g similarity=%d "
				  "rest_every=%d rest_updates=%u anchor=%zu anchor_k=%g anchor_shift=%.4f sim_s=%.4f sim_rot_deg=%.3f y=%.4f..%.4f "
				  "finite=%s projections=%zu self_pushes=%zu wall_ms=%.1f ms/step=%.3f%s%s",
				okFlag ? "DONE" : "FAIL", backend(), sim->steps(), size_t(fitStepsDone), fwdAsked, settleDone,
				sim->scene.fitSettle, fitConverged ? "yes" : "no",
				sim->lastMaxDisp, fitTol, sim->fitRefreshed, sim->fitMisses, sim->lastFitDistMin, sim->lastFitDistMean,
				sim->scene.fitGap, sim->scene.fitK, sim->scene.nFit, sim->scene.nPin(), sim->cfg.iters, sim->cfg.h,
				int(sim->scene.fitSimilarity), sim->scene.fitRestEvery, sim->fitRestUpdates, sim->scene.fitAnchor.size(),
				sim->scene.anchorK, sim->lastAnchorShift,
				sim->fitSim.s, sim->fitSim.angle() * 57.29577951308232, ymin, ymax, fin ? "yes" : "NO", proj, pushes,
				wall_ms, done ? wall_ms / double(done) : 0.0, sim->failed ? " error=" : "",
				sim->failed ? sim->err.c_str() : "");
	}

	void enqueueBackward(jobs::StageQueue &q, DrapeLoss loss, DrapeMode mode) override {
		jobs::StageQueue *qp = &q;
		q.push([this, qp, loss, mode]() {
			if (!sim || sim->midStep()) {
				res = "FAIL backward: no completed forward";
				okFlag = false;
				return;
			}
			bwd = std::make_unique<DrapeBackwardT<S>>(*sim, tgt, loss, mode);
			t0 = now();
			backwardStage(*qp);
		});
	}

	void backwardStage(jobs::StageQueue &q) {
		if (!bwd->advance()) {
			jobs::StageQueue *qp = &q;
			q.next([this, qp]() { backwardStage(*qp); });
			return;
		}
		lastGrad = bwd->g;
		okFlag = !bwd->failed;
		const double wall_ms = double(now() - t0) / 1000.0;
		res = std::string(bwd->failed ? "FAIL" : "DONE") + " backward " + backend() + " " + bwd->summary() +
				drape_fmt(" wall_ms=%.1f", wall_ms) + (bwd->failed ? " error=" + bwd->err : "");
	}
};

