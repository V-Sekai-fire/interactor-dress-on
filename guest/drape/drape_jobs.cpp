// The drape session and the Gate 5 drape jobs (see drape_jobs.h).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "drape_jobs.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <utility>

#include "avbd/avbd_cpu.h"
#include "avbd/avbd_rd.h"
#include "body_mesh.h"
#include "drape_sim.h"
#include "inverse_jobs.h"
#include "lbfgsb_jobs.h"

namespace {

std::string fmt(const char *f, ...) {
	char b[2048];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

template <class S>
struct DBackend;

template <>
struct DBackend<AvbdCpu> {
	static const char *name() { return "cpu"; }
	static std::unique_ptr<AvbdCpu> make(rdc::Device &, std::string &) { return std::make_unique<AvbdCpu>(); }
};

template <>
struct DBackend<AvbdRd> {
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

// Per-step statistics a forward can collect, computed as the native
// [avbd-step] line computes them (Simulation.cpp:1497-1545, all float):
// |dx| = |x_avbd - x|, pred = |s_blend - x|, drift = |x_avbd - s_blend| per
// component; the mean accumulates in float and divides by 3.0f nV.
struct StepStat {
	size_t step = 0;
	float dxMax = 0.0f, dxMean = 0.0f, predMax = 0.0f, driftMax = 0.0f;
	uint32_t driftVert = 0;
	int driftAxis = 0;
	size_t fric = 0, pred = 0, proj = 0, pushes = 0, pairs = 0;
};

} // namespace

std::unique_ptr<DrapeSession> make_drape_session(const std::string &backend, rdc::Device &dev, std::string &err) {
	if (backend == "cpu") {
		auto s = std::make_unique<DrapeSessionT<AvbdCpu>>();
		s->backendName = "cpu";
		s->solver = DBackend<AvbdCpu>::make(dev, err);
		return s;
	}
	if (backend == "rd") {
		auto s = std::make_unique<DrapeSessionT<AvbdRd>>();
		s->backendName = "rd";
		s->solver = DBackend<AvbdRd>::make(dev, err);
		if (!s->solver) {
			return nullptr;
		}
		return s;
	}
	err = "backend must be cpu, rd or auto";
	return nullptr;
}

std::string drape_pick_backend(const std::string &want, uint32_t nV) {
	if (want == "auto") {
		return nV >= kDrapeAutoRdVerts ? "rd" : "cpu";
	}
	return want;
}

// --- jobs ---------------------------------------------------------------------------

namespace {

struct Args {
	std::map<std::string, std::string> kv;
	explicit Args(const std::string &s) {
		std::istringstream in(s);
		std::string tok;
		while (in >> tok) {
			const size_t e = tok.find('=');
			if (e != std::string::npos) {
				kv[tok.substr(0, e)] = tok.substr(e + 1);
			}
		}
	}
	bool has(const std::string &k) const { return kv.count(k) != 0; }
	double num(const std::string &k, double d) const {
		auto it = kv.find(k);
		return it == kv.end() ? d : std::atof(it->second.c_str());
	}
	std::string str(const std::string &k, const std::string &d) const {
		auto it = kv.find(k);
		return it == kv.end() ? d : it->second;
	}
	std::vector<std::string> list(const std::string &k, const std::string &d) const {
		std::vector<std::string> out;
		std::string v = str(k, d);
		size_t p = 0;
		while (p <= v.size()) {
			const size_t c = v.find(',', p);
			const std::string t = v.substr(p, c == std::string::npos ? std::string::npos : c - p);
			if (!t.empty()) {
				out.push_back(t);
			}
			if (c == std::string::npos) {
				break;
			}
			p = c + 1;
		}
		return out;
	}
	// Every key the DrapeConfig knows goes into cfg; false on a refused value.
	bool applyConfig(DrapeConfig &cfg, const std::set<std::string> &jobKeys, std::string &err) const {
		for (const auto &e : kv) {
			if (jobKeys.count(e.first)) {
				continue;
			}
			if (!cfg.set(e.first, std::atof(e.second.c_str()))) {
				err = "unknown or refused key " + e.first + "=" + e.second;
				return false;
			}
		}
		return true;
	}
};

class DrapeJobBase : public jobs::Job {
public:
	virtual DrapeSession *session() = 0;
};

template <class S>
class DJob : public DrapeJobBase {
public:
	bool pending() const override { return sess_ && sess_->pending(); }
	void drain() override {
		if (sess_) {
			sess_->drain();
		}
	}
	DrapeSession *session() override { return sess_.get(); }
	bool init(rdc::Device &dev, std::string &err, const std::string &name) {
		name_ = name + " " + DBackend<S>::name();
		sess_ = std::make_unique<DrapeSessionT<S>>();
		sess_->backendName = DBackend<S>::name();
		sess_->solver = DBackend<S>::make(dev, err);
		if (!sess_->solver) {
			return false;
		}
		sess_->clock = &now_us;
		// CPU stages are whole solver steps: a few per frame keeps a tick short.
		if (std::string(DBackend<S>::name()) == "cpu") {
			max_stages_per_tick = 4;
		}
		return true;
	}
	virtual bool build(const Args &a, std::string &err) = 0;

protected:
	std::unique_ptr<DrapeSessionT<S>> sess_;
	std::string name_;
	std::string detail_;
	int fails_ = 0;

	void say(const std::string &line) {
		detail_ += line;
		detail_ += '\n';
	}
	void done(bool pass, const std::string &head) { finish(pass, name_ + ": " + head, detail_); }
	// Load a scene, failing the job on error.
	void loadStage(std::function<void(DrapeScene &, DrapeConfig &)> make) {
		q.push([this, make]() {
			DrapeScene sc;
			DrapeConfig cfg;
			make(sc, cfg);
			std::string err;
			if (!sess_->load(sc, cfg, err)) {
				++fails_;
				say("FAIL load: " + err);
			}
		});
	}
	bool fwdOk() {
		if (!sess_->ok()) {
			++fails_;
			say("FAIL " + sess_->result());
			return false;
		}
		return true;
	}
};

// The native reference's per-step lines use these steps.
const std::set<size_t> kStatSteps = { 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 350 };

template <class S>
void collect_stats(DrapeSessionT<S> &s, std::vector<StepStat> &out, bool all = false) {
	DrapeSessionT<S> *sp = &s;
	std::vector<StepStat> *op = &out;
	s.onStep = [sp, op, all](const DrapeStepRecord &r, const std::vector<float> &xPrev) {
		const size_t k = sp->sim->steps();
		if (!all && !kStatSteps.count(k)) {
			return;
		}
		StepStat st;
		st.step = k;
		const size_t nb = std::min(std::min(r.xAvbd.size(), xPrev.size()), r.sBlend.size());
		for (size_t b = 0; b < nb; ++b) {
			const float ax = r.xAvbd[b];
			const float dx = std::fabs(ax - xPrev[b]);
			const float dp = std::fabs(r.sBlend[b] - xPrev[b]);
			const float dd = std::fabs(ax - r.sBlend[b]);
			if (dx > st.dxMax) st.dxMax = dx;
			if (dp > st.predMax) st.predMax = dp;
			if (dd > st.driftMax) {
				st.driftMax = dd;
				st.driftVert = uint32_t(b / 3);
				st.driftAxis = int(b % 3);
			}
			st.dxMean += dx;
		}
		st.dxMean /= (3.0f * float(nb / 3));
		st.fric = r.fric.size();
		st.pred = r.pred.size();
		st.proj = r.projHits;
		st.pairs = r.selfPairs;
		st.pushes = r.pushes.size();
		op->push_back(st);
	};
}

std::string stat_line(const StepStat &st) {
	return fmt("step %4zu |dx|_max=%.9g |dx|_mean=%.9g pred_max=%.9g drift_max=%.9g drift@v%u.%c friction_blends=%zu "
			   "friction_events=%zu projections=%zu self_pairs=%zu self_pushes=%zu",
			st.step, double(st.dxMax), double(st.dxMean), double(st.predMax), double(st.driftMax), st.driftVert,
			"xyz"[st.driftAxis], st.pred, st.fric, st.proj, st.pairs, st.pushes);
}

// --- sphere_forward: the rotating-sphere demo forward ------------------------------
//
// keys: steps (100), mu (kNativeSphereMu0: the native iter0's mu, 0.539770 as
// printed), stats=all (every step's line), any DrapeConfig key.

template <class S>
class SphereForwardJob : public DJob<S> {
public:
	bool build(const Args &a, std::string &err) override {
		steps_ = int(a.num("steps", 100));
		DrapeConfig cfg;
		cfg.mu = kNativeSphereMu0;
		if (!a.applyConfig(cfg, { "steps", "stats" }, err)) {
			return false;
		}
		cfg_ = cfg;
		this->loadStage([this](DrapeScene &sc, DrapeConfig &c) {
			c = cfg_;
			scene_sphere_demo(sc, c);
		});
		collect_stats(*this->sess_, stats_, a.str("stats", "") == "all");
		this->q.push([this]() {
			this->say(this->sess_->config().dump());
			this->say(this->sess_->scene().describe());
		});
		this->sess_->enqueueForward(this->q, steps_);
		this->q.push([this]() {
			this->fwdOk();
			for (const StepStat &st : stats_) {
				this->say(stat_line(st));
			}
			this->say(this->sess_->result());
			const std::string head = fmt("%d steps at mu %.6f: %s", steps_, cfg_.mu, this->sess_->result().c_str());
			this->done(this->fails_ == 0, head.substr(0, head.find(" wall_ms")));
		});
		return true;
	}

private:
	int steps_ = 100;
	DrapeConfig cfg_;
	std::vector<StepStat> stats_;
};

// --- sphere_backward: native dL/dmu against backwardLog ---------------------------
//
// keys: target (0.3: the ground truth), mus (kNativeSphereMu0,0.01: LBFGS iter0 and
// iter1), mode (native), steps (350), any DrapeConfig key.

template <class S>
class SphereBackwardJob : public DJob<S> {
public:
	bool build(const Args &a, std::string &err) override {
		steps_ = int(a.num("steps", 350));
		const double target = a.num("target", 0.3);
		const std::string mode = a.str("mode", "native");
		mode_ = mode == "step" ? DrapeMode::Step : (mode == "unrolled" ? DrapeMode::Unrolled : DrapeMode::Native);
		DrapeConfig cfg;
		if (!a.applyConfig(cfg, { "steps", "target", "mus", "mode" }, err)) {
			return false;
		}
		cfg.mu = target;
		cfg_ = cfg;
		for (const std::string &m : a.list("mus", "0.5397701956236457,0.01")) {
			mus_.push_back(std::atof(m.c_str()));
		}
		auto *s = this->sess_.get();
		this->loadStage([this](DrapeScene &sc, DrapeConfig &c) {
			c = cfg_;
			scene_sphere_demo(sc, c);
		});
		this->q.push([this]() { this->say(this->sess_->config().dump()); });
		s->enqueueForward(this->q, steps_);
		this->q.push([this, s, target]() {
			if (!this->fwdOk()) {
				return;
			}
			s->targetFromFrames();
			this->say(fmt("target: mu %.6f, %zu frames; %s", target, s->target().numFrames(), s->result().c_str()));
		});
		for (double mu : mus_) {
			this->q.push([this, s, mu]() {
				std::vector<Primitive> p = s->scene().prims;
				if (!p.empty()) {
					p[0].mu = mu;
				}
				s->setPrims(p);
				s->rewind();
			});
			s->enqueueForward(this->q, steps_);
			this->q.push([this, s, mu]() {
				this->fwdOk();
				this->say(fmt("mu %.6f %s", mu, s->result().c_str()));
			});
			s->enqueueBackward(this->q, DrapeLoss::MatchTrajectory, mode_);
			this->q.push([this, s, mu]() {
				if (!s->ok()) {
					++this->fails_;
				}
				const DrapeGrad &g = s->grad();
				const double dmu = g.dmu.empty() ? NAN : g.dmu[0];
				std::string per;
				double once = 0.0;
				for (size_t k = 0; k < g.muPerStep.size(); ++k) {
					once += g.muPerStep[k];
					if (k < 6) {
						per += fmt(" %.6g", g.muPerStep[k]);
					}
				}
				this->say(fmt("RESULT mu=%.6f mu_exact=%.16g loss=%.9g dL/dmu=%.9g", mu, mu, g.loss, dmu));
				if (!per.empty()) {
					this->say(fmt("  per-step dL/dmu (step N, N-1, ...):%s; summed once (no double carry) %.9g",
							per.c_str(), once));
				}
				this->say("  " + s->result());
				if (!std::isfinite(g.loss) || !std::isfinite(dmu)) {
					++this->fails_;
				}
			});
		}
		this->q.push([this]() { this->done(this->fails_ == 0, fmt("%zu mu values, mode %s", mus_.size(), drape_mode_name(mode_))); });
		return true;
	}

private:
	int steps_ = 350;
	DrapeMode mode_ = DrapeMode::Native;
	DrapeConfig cfg_;
	std::vector<double> mus_;
};

// --- sim_gradcheck: every backward mode against central finite differences ----------
//
// keys: scene (panel | plane), colors (0), steps (20), modes
// (native,step,unrolled), eps (1e-3, relative), tol (5e-2), any DrapeConfig key.
// panel: an 8x8 1 m panel, horizontal, pinned at two corners. plane: the same
// panel unpinned, moving (0.5, -0.4, 0.2) m/s, over a plane tilted 5.7 deg
// that cuts through one side of it (contact, projection, friction). The loss
// is MATCH_TRAJECTORY against the same scene at different parameters
// (mu 0.25, kTri 110, density 0.36 against mu 0.4, kTri 150, density 0.3).

struct GcParams {
	double mu = 0.4, kTri = 150.0, density = 0.3;
};

void gc_scene(const std::string &kind, const GcParams &p, DrapeConfig &cfg, DrapeScene &sc) {
	double mn[3], mx[3];
	scene_grid(sc, 8, 8, 1.0, 1.0, GridOrientation::Down, mn, mx);
	sc.name = "gradcheck_" + kind;
	cfg.mu = p.mu;
	cfg.kTri = p.kTri;
	cfg.density = p.density;
	if (kind == "panel") {
		for (uint32_t v : { 0u, 7u }) {
			sc.attachVert.push_back(v);
			for (int k = 0; k < 3; ++k) {
				sc.attachFixed.push_back(sc.rest[3 * v + k]);
			}
		}
	} else {
		for (uint32_t i = 0; i < sc.nV; ++i) {
			sc.v0[3 * i + 0] = 0.5f;
			sc.v0[3 * i + 1] = -0.4f;
			sc.v0[3 * i + 2] = 0.2f;
		}
		// Normal (0.1, 1, 0)/|.|: y = -0.1 x - 0.02 lies above the panel for
		// x < -0.2.
		const v3d n = v3d(0.1, 1.0, 0.0).normalized();
		const v3d c(0.0, -0.02, 0.0);
		const v3d t1 = v3d(0, 0, 1).cross(n).normalized(); // in-plane, along -x-ish
		const v3d t2 = n.cross(t1);
		v3d ul = c + (t1 * -3.0) + (t2 * 3.0), ur = c + (t1 * 3.0) + (t2 * 3.0);
		Primitive pl = make_plane(c, ul, ur, p.mu);
		if (pl.planeNormal.dot(n) < 0) {
			pl = make_plane(c, ur, ul, p.mu);
		}
		sc.prims.push_back(pl);
	}
	sc.applyMaterial(cfg);
}

template <class S>
class SimGradcheckJob : public DJob<S> {
public:
	bool build(const Args &a, std::string &err) override {
		kind_ = a.str("scene", "panel");
		if (kind_ != "panel" && kind_ != "plane") {
			err = "scene must be panel or plane";
			return false;
		}
		steps_ = int(a.num("steps", 20));
		eps_ = a.num("eps", 1e-3);
		tol_ = a.num("tol", 5e-2);
		DrapeConfig cfg;
		cfg.colors = false;
		if (!a.applyConfig(cfg, { "scene", "steps", "modes", "eps", "tol" }, err)) {
			return false;
		}
		cfg_ = cfg;
		for (const std::string &m : a.list("modes", "native,step,unrolled")) {
			if (m == "native") modes_.push_back(DrapeMode::Native);
			else if (m == "step") modes_.push_back(DrapeMode::Step);
			else if (m == "unrolled") modes_.push_back(DrapeMode::Unrolled);
			else {
				err = "unknown mode " + m;
				return false;
			}
		}
		// native reads the state the forward left: run it first.
		std::stable_sort(modes_.begin(), modes_.end(), [](DrapeMode x, DrapeMode y) {
			return int(x == DrapeMode::Native) > int(y == DrapeMode::Native);
		});
		auto *s = this->sess_.get();
		GcParams truth;
		truth.mu = 0.25;
		truth.kTri = 110.0;
		truth.density = 0.36;
		forward(truth);
		this->q.push([this, s]() {
			s->targetFromFrames();
			this->say(this->sess_->config().dump());
			this->say(this->sess_->scene().describe());
			this->say("target " + s->result());
		});
		forward(base_);
		for (DrapeMode m : modes_) {
			s->enqueueBackward(this->q, DrapeLoss::MatchTrajectory, m);
			this->q.push([this, s, m]() {
				grads_[int(m)] = s->grad();
				this->say(s->result());
				if (!s->ok()) {
					++this->fails_;
				}
			});
		}
		// Central differences, relative step eps and 10 eps.
		const char *names[3] = { "mu", "kTri", "density" };
		for (int p = 0; p < 3; ++p) {
			for (int e = 0; e < 2; ++e) {
				for (int sg = -1; sg <= 1; sg += 2) {
					GcParams q = base_;
					double *v = p == 0 ? &q.mu : (p == 1 ? &q.kTri : &q.density);
					const double h = (e == 0 ? eps_ : 10 * eps_) * std::fabs(*v);
					*v += sg * h;
					forward(q);
					this->q.push([this, s, p, e, sg, h]() {
						this->fwdOk();
						const double L = drape_loss_value(*s->sim, s->target(), DrapeLoss::MatchTrajectory);
						fdL_[p][e][sg > 0 ? 1 : 0] = L;
						fdH_[p][e] = h;
					});
				}
			}
		}
		this->q.push([this, names]() { report(names); });
		return true;
	}

private:
	std::string kind_;
	int steps_ = 20;
	double eps_ = 1e-3, tol_ = 5e-2;
	DrapeConfig cfg_;
	GcParams base_;
	std::vector<DrapeMode> modes_;
	std::map<int, DrapeGrad> grads_;
	double fdL_[3][2][2] = {};
	double fdH_[3][2] = {};

	void forward(const GcParams &p) {
		this->loadStage([this, p](DrapeScene &sc, DrapeConfig &c) {
			c = cfg_;
			gc_scene(kind_, p, c, sc);
		});
		this->sess_->enqueueForward(this->q, steps_);
	}

	static double analytic(const DrapeGrad &g, int p) {
		if (p == 0) {
			return g.dmu.empty() ? 0.0 : g.dmu[0];
		}
		return p == 1 ? g.dkTri : g.ddensity;
	}

	void report(const char *const names[3]) {
		bool pass = true;
		const bool single = !cfg_.colors;
		this->say(fmt("scene=%s colors=%s steps=%d eps=%g (rel), tol=%g; rel = |a - fd| / max(|a|, |fd|), 0 when "
					  "both are below 1e-12",
				kind_.c_str(), single ? "single" : "multi", steps_, eps_, tol_));
		for (int p = 0; p < 3; ++p) {
			double fd[2];
			for (int e = 0; e < 2; ++e) {
				fd[e] = (fdL_[p][e][1] - fdL_[p][e][0]) / (2.0 * fdH_[p][e]);
			}
			std::string row = fmt("  d/d%-8s fd(eps)=%-15.9g fd(10eps)=%-15.9g", names[p], fd[0], fd[1]);
			for (DrapeMode m : modes_) {
				const double a = analytic(grads_[int(m)], p);
				const double den = std::max(std::fabs(a), std::fabs(fd[0]));
				const double rel = den < 1e-12 ? 0.0 : std::fabs(a - fd[0]) / den;
				row += fmt(" | %s %.9g rel=%.3g", drape_mode_name(m), a, rel);
				if (m == DrapeMode::Unrolled && single && !(rel <= tol_)) {
					pass = false;
				}
				this->say(fmt("GC scene=%s colors=%d mode=%s param=%s analytic=%.9g fd=%.9g rel=%.6g", kind_.c_str(),
						single ? 0 : 1, drape_mode_name(m), names[p], a, fd[0], rel));
			}
			this->say(row);
		}
		if (!single) {
			// The multi-colour table is information (it picks the default mode).
			this->done(this->fails_ == 0, fmt("%s multi-colour table (information)", kind_.c_str()));
			return;
		}
		this->done(this->fails_ == 0 && pass, fmt("%s single colour: unrolled within %g of central FD on mu, kTri, "
												   "density: %s",
													   kind_.c_str(), tol_, pass ? "yes" : "NO"));
	}
};

// --- bench_drape: ms per step by grid size ----------------------------------------
//
// keys: sizes (8,16,24,32,48), steps (10), any DrapeConfig key. Each size is
// the sphere demo's scene at n x n (4.5 m, the same sphere); the time is the
// host clock over the frames the forward took (frame-driven, rule 4).

template <class S>
class BenchDrapeJob : public DJob<S> {
public:
	bool build(const Args &a, std::string &err) override {
		steps_ = int(a.num("steps", 10));
		DrapeConfig cfg;
		if (!a.applyConfig(cfg, { "sizes", "steps" }, err)) {
			return false;
		}
		cfg_ = cfg;
		for (const std::string &t : a.list("sizes", "8,16,24,32,48")) {
			const int n = std::atoi(t.c_str());
			if (n < 2) {
				err = "size must be >= 2";
				return false;
			}
			this->loadStage([this, n](DrapeScene &sc, DrapeConfig &c) {
				c = cfg_;
				double mn[3], mx[3];
				scene_grid(sc, n, n, 4.5, 4.5, GridOrientation::Down, mn, mx);
				sc.name = fmt("bench_%dx%d", n, n);
				v3d low((mn[0] + mx[0]) * 0.5, mn[1], (mn[2] + mx[2]) * 0.5);
				const v3d plane1 = low - v3d(0, 4.1, 0);
				sc.prims.push_back(make_sphere(plane1 + v3d(0.6, 2.0, 0.2), 2.0, c.mu));
				sc.applyMaterial(c);
			});
			this->sess_->enqueueForward(this->q, steps_);
			this->q.push([this, n]() {
				this->fwdOk();
				this->say(fmt("bench_drape %s %2dx%-2d nv=%4d %s", this->sess_->backend(), n, n, n * n,
						this->sess_->result().c_str()));
			});
		}
		this->q.push([this]() {
			this->say(this->sess_->config().dump());
			this->done(this->fails_ == 0, "every size finite");
		});
		return true;
	}

private:
	int steps_ = 10;
	DrapeConfig cfg_;
};

// --- mesh_parity / mesh_bisect: a host mesh on cpu and rd side by side ----------------
//
// The scene comes from drape_job_data (the guest has no filesystem):
//   mesh_obj       Wavefront text: v x y z / f a b c (1-based; a/b/c forms and
//                  polygons fanned from the first corner)
//   mesh_pins      whitespace-separated vertex ids (0-based), each an attachment
//   mesh_capsules  lines "bx by bz ax ay az radius length" (# comments)
// all in body units; `scale` (10) multiplies positions, capsule bottoms,
// radii and lengths, and gravity is -9.8 * scale unless gravityY is given.
// keys: scale, caps (1: the capsules), capmu (0.3), steps, tol, any
// DrapeConfig key.

// Wavefront text: v x y z (as float), f a b c ... (1-based, a/b/c forms,
// polygons fanned from the first corner).
void obj_parse(const std::string &text, std::vector<float> &pos, std::vector<int32_t> &tris) {
	std::istringstream in(text);
	std::string line;
	while (std::getline(in, line)) {
		std::istringstream ls(line);
		std::string tag;
		ls >> tag;
		if (tag == "v") {
			double x = 0, y = 0, z = 0;
			ls >> x >> y >> z;
			pos.push_back(float(x));
			pos.push_back(float(y));
			pos.push_back(float(z));
		} else if (tag == "f") {
			std::vector<int32_t> ids;
			std::string t;
			while (ls >> t) {
				ids.push_back(std::atoi(t.c_str()) - 1);
			}
			for (size_t k = 1; k + 1 < ids.size(); ++k) {
				tris.push_back(ids[0]);
				tris.push_back(ids[k]);
				tris.push_back(ids[k + 1]);
			}
		}
	}
}

bool mesh_scene_from_data(const Args &a, DrapeScene &sc, DrapeConfig &cfg, std::string &err,
		std::shared_ptr<const BodyMesh> *bodyOut = nullptr) {
	auto &d = lbfgsb_job_data();
	auto it = d.find("mesh_obj");
	if (it == d.end()) {
		err = "no data mesh_obj (drape_job_data first)";
		return false;
	}
	const double s = a.num("scale", 10.0);
	std::vector<float> pos;
	std::vector<int32_t> tris, pins;
	obj_parse(it->second, pos, tris);
	for (float &x : pos) {
		// The host's float * scale (pipeline.gd's _scaled), in double then float.
		x = float(double(x) * s);
	}
	it = d.find("mesh_pins");
	if (it != d.end()) {
		std::istringstream in(it->second);
		int32_t p;
		while (in >> p) {
			pins.push_back(p);
		}
	}
	sc = DrapeScene();
	if (a.num("caps", 0.0) != 0.0) {
		it = d.find("mesh_capsules");
		if (it == d.end()) {
			err = "caps=1 but no data mesh_capsules";
			return false;
		}
		std::istringstream in(it->second);
		std::string line;
		const double mu = a.num("capmu", 0.3);
		while (std::getline(in, line)) {
			if (line.empty() || line[0] == '#') {
				continue;
			}
			std::istringstream ls(line);
			double c[8];
			int n = 0;
			while (n < 8 && ls >> c[n]) {
				++n;
			}
			if (n == 8) {
				sc.prims.push_back(make_capsule(v3d(c[0] * s, c[1] * s, c[2] * s), v3d(c[3], c[4], c[5]), c[6] * s,
						c[7] * s, mu));
			}
		}
	}
	if (a.num("body", 0.0) != 0.0 || a.num("bodycheck", 0.0) != 0.0) {
		it = d.find("body_obj");
		if (it == d.end()) {
			err = "body=1 or bodycheck=1 but no data body_obj";
			return false;
		}
		std::vector<float> bv;
		std::vector<int32_t> bt;
		obj_parse(it->second, bv, bt);
		auto body = std::make_shared<BodyMesh>();
		if (!body->build(bv, bt, s, err)) {
			return false;
		}
		if (bodyOut) {
			*bodyOut = body;
		}
		if (a.num("body", 0.0) != 0.0) {
			sc.prims.push_back(make_mesh_collider(body, a.num("bodyskin", 0.1), a.num("bodyband", 0.1),
					a.num("bodydepth", 1.0), a.num("capmu", 0.3)));
		}
	}
	cfg.gravity[1] = -9.8 * s;
	if (!a.applyConfig(cfg, { "scale", "caps", "capmu", "steps", "tol", "body", "bodycheck", "bodyskin", "bodyband", "bodydepth" },
				err)) {
		return false;
	}
	if (!scene_mesh(sc, pos, tris, pins, err)) {
		return false;
	}
	sc.name = fmt("mesh x%g", s);
	sc.applyMaterial(cfg);
	return true;
}

double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b, size_t *at = nullptr) {
	double m = 0.0;
	for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
		if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
			continue;
		}
		const double d = std::fabs(double(a[i]) - double(b[i]));
		if (d > m) {
			m = d;
			if (at) {
				*at = i;
			}
		}
	}
	return m;
}

size_t count_nonfinite(const std::vector<float> &a, long *first = nullptr) {
	size_t n = 0;
	if (first) {
		*first = -1;
	}
	for (size_t i = 0; i < a.size(); ++i) {
		if (!std::isfinite(a[i])) {
			if (first && *first < 0) {
				*first = long(i);
			}
			++n;
		}
	}
	return n;
}

// mesh_parity: the same scene forward on cpu and on rd, compared step by step.
// PASS when every step is finite on both and max |x_cpu - x_rd| <= tol (1e-3
// drape units) at every step.
class MeshParityJob : public DrapeJobBase {
public:
	bool pending() const override { return rd_ && rd_->pending(); }
	void drain() override {
		if (rd_) {
			rd_->drain();
		}
	}
	DrapeSession *session() override { return rd_.get(); }

	bool build(rdc::Device &dev, const Args &a, std::string &err) {
		steps_ = int(a.num("steps", 3));
		tol_ = a.num("tol", 1e-3);
		if (!mesh_scene_from_data(a, scene_, cfg_, err, &body_)) {
			return false;
		}
		cpu_ = std::make_unique<DrapeSessionT<AvbdCpu>>();
		cpu_->backendName = "cpu";
		cpu_->solver = DBackend<AvbdCpu>::make(dev, err);
		rd_ = std::make_unique<DrapeSessionT<AvbdRd>>();
		rd_->backendName = "rd";
		rd_->solver = DBackend<AvbdRd>::make(dev, err);
		if (!rd_->solver) {
			return false;
		}
		cpu_->clock = &now_us;
		rd_->clock = &now_us;
		// A cpu stage is a whole solve here (about a second at 2682 vertices).
		max_stages_per_tick = 2;
		label_ = fmt("mesh_parity %s steps=%d tol=%g", scene_.name.c_str(), steps_, tol_);
		q.push([this]() {
			std::string e;
			if (!cpu_->load(scene_, cfg_, e) || !rd_->load(scene_, cfg_, e)) {
				fail("load: " + e);
			}
			say(cfg_.dump());
			say(rd_->scene().describe());
		});
		auto *cx = &xc_;
		auto *rx = &xr_;
		cpu_->onStep = [this, cx](const DrapeStepRecord &, const std::vector<float> &) { cx->push_back(cpu_->sim->x); };
		rd_->onStep = [this, rx](const DrapeStepRecord &, const std::vector<float> &) { rx->push_back(rd_->sim->x); };
		rd_->enqueueForward(q, steps_);
		q.push([this]() { say(rd_->result()); });
		cpu_->enqueueForward(q, steps_);
		q.push([this]() {
			say(cpu_->result());
			report();
		});
		return true;
	}

private:
	std::unique_ptr<DrapeSessionT<AvbdCpu>> cpu_;
	std::unique_ptr<DrapeSessionT<AvbdRd>> rd_;
	DrapeScene scene_;
	DrapeConfig cfg_;
	int steps_ = 3;
	double tol_ = 1e-3;
	std::vector<std::vector<float>> xc_, xr_;
	std::shared_ptr<const BodyMesh> body_; // body=1 or bodycheck=1: inside counts
	std::string label_, detail_;

	// Vertices strictly inside the body (signed distance to its surface < 0).
	size_t inside(const std::vector<float> &x, double &deepest) const {
		size_t n = 0;
		deepest = 0.0;
		for (size_t i = 0; i + 2 < x.size(); i += 3) {
			const v3d p(x[i], x[i + 1], x[i + 2]);
			BodyMesh::Hit h;
			if (!body_->closest(p, 1e300, h)) {
				continue;
			}
			if ((p - h.point).dot(h.pseudo) < 0.0) {
				++n;
				deepest = std::max(deepest, std::sqrt(h.dist2));
			}
		}
		return n;
	}
	bool failed_ = false;

	void say(const std::string &l) {
		detail_ += l;
		detail_ += '\n';
	}
	void fail(const std::string &why) {
		failed_ = true;
		say("FAIL " + why);
	}
	void report() {
		bool ok = !failed_ && int(xc_.size()) == steps_ && int(xr_.size()) == steps_;
		double worst = 0.0;
		size_t nfC = 0, nfR = 0;
		for (size_t k = 0; k < xc_.size() && k < xr_.size(); ++k) {
			long fc, fr;
			const size_t c = count_nonfinite(xc_[k], &fc), r = count_nonfinite(xr_[k], &fr);
			size_t at = 0;
			const double d = max_abs_diff(xc_[k], xr_[k], &at);
			worst = std::max(worst, d);
			nfC += c;
			nfR += r;
			say(fmt("step %zu: non-finite floats cpu %zu rd %zu (first rd %ld); max|x_cpu - x_rd| = %.3g at v%zu.%c", k + 1,
					c, r, fr, d, at / 3, "xyz"[at % 3]));
		}
		if (body_ && !xr_.empty() && !xc_.empty()) {
			double d0, dc, dr;
			const size_t i0 = inside(scene_.x0, d0), ic = inside(xc_.back(), dc), ir = inside(xr_.back(), dr);
			say(fmt("inside the body (signed distance < 0): start %zu (deepest %.3g), after %zu steps cpu %zu (%.3g) "
					"rd %zu (%.3g); collider %s",
					i0, d0, xr_.size(), ic, dc, ir, dr, scene_.prims.empty() ? "none" : scene_.prims[0].describe().c_str()));
		}
		ok = ok && nfC == 0 && nfR == 0 && worst <= tol_;
		finish(ok, fmt("%s: finite cpu=%s rd=%s, max|x_cpu - x_rd| over %zu steps = %.3g (tol %g)", label_.c_str(),
						   nfC == 0 ? "yes" : "NO", nfR == 0 ? "yes" : "NO", xr_.size(), worst, tol_),
				detail_);
	}
};

// mesh_bisect: where inside the first step rd leaves cpu. The predictor is
// computed once (cpu driver code, as a step would); then
//  1. iterations: run(k) on rd from the predictor for k = 1..iters (one submit
//     per k, read back on the next tick) against k cpu iterations;
//  2. kernels: in the first iteration K where rd is non-finite or further
//     than tol from cpu, every (colour, stage) of K -- stage 0 init + the four
//     force kernels, 1-4 the spring/attachment/triangle/bending gathers, 5 the
//     solve -- on both, each buffer compared, until the first stage whose rd
//     buffers hold a non-finite value;
//  3. the constraint behind the first non-finite row, with its inputs.
class MeshBisectJob : public DrapeJobBase {
public:
	bool pending() const override { return rd_ && rd_->pending(); }
	void drain() override {
		if (rd_) {
			rd_->sync();
		}
	}
	DrapeSession *session() override { return nullptr; }

	bool build(rdc::Device &dev, const Args &a, std::string &err) {
		tol_ = a.num("tol", 1e-3);
		if (!mesh_scene_from_data(a, scene_, cfg_, err)) {
			return false;
		}
		cpu_ = std::make_unique<AvbdCpu>();
		rd_ = DBackend<AvbdRd>::make(dev, err);
		if (!rd_) {
			return false;
		}
		simC_ = std::make_unique<DrapeSimT<AvbdCpu>>(*cpu_);
		simR_ = std::make_unique<DrapeSimT<AvbdRd>>(*rd_);
		max_stages_per_tick = 4;
		q.push([this]() {
			simC_->cfg = cfg_;
			simC_->scene = scene_;
			simR_->cfg = cfg_;
			simR_->scene = scene_;
			if (!simC_->setup() || !simR_->setup()) {
				done(false, "setup failed");
				return;
			}
			simC_->predict(rec_);
			say(cfg_.dump());
			say(scene_.describe() + fmt(" colours cpu %u rd %u", cpu_->numColors(), rd_->numColors()));
			cpu_->updateState(simC_->x.data(), rec_.sBlend.data());
			if (scene_.nAttach()) {
				cpu_->updateAttachmentFixedPos(scene_.attachFixed.data());
			}
			iterStage(1);
		});
		return true;
	}

private:
	std::unique_ptr<AvbdCpu> cpu_;
	std::unique_ptr<AvbdRd> rd_;
	std::unique_ptr<DrapeSimT<AvbdCpu>> simC_;
	std::unique_ptr<DrapeSimT<AvbdRd>> simR_;
	DrapeScene scene_;
	DrapeConfig cfg_;
	DrapeStepRecord rec_;
	double tol_ = 1e-3;
	int K_ = 0;
	std::string detail_;
	std::vector<std::string> lastRdBad_;

	void say(const std::string &l) {
		detail_ += l;
		detail_ += '\n';
	}
	void done(bool pass, const std::string &head) {
		cpu_->setDebugStopForTest(-1, -1);
		rd_->setDebugStopForTest(-1, -1);
		finish(pass, "mesh_bisect " + scene_.name + ": " + head, detail_);
	}
	void rdFromPredictor(int iters) {
		rd_->updateState(simR_->x.data(), rec_.sBlend.data());
		if (scene_.nAttach()) {
			rd_->updateAttachmentFixedPos(scene_.attachFixed.data());
		}
		rd_->run(iters, cfg_.al);
	}
	void cpuFromPredictor(int iters) {
		cpu_->updateState(simC_->x.data(), rec_.sBlend.data());
		if (scene_.nAttach()) {
			cpu_->updateAttachmentFixedPos(scene_.attachFixed.data());
		}
		cpu_->run(iters, cfg_.al);
	}

	// 1. iteration k: cpu one more iteration (no duals unless al), rd from scratch.
	void iterStage(int k) {
		if (cfg_.al) {
			cpuFromPredictor(k);
		} else {
			cpu_->run(1, false);
		}
		rdFromPredictor(k);
		q.next([this, k]() {
			std::vector<float> xc, xr;
			cpu_->readPositions(xc);
			rd_->readPositions(xr);
			long fr;
			const size_t nr = count_nonfinite(xr, &fr), nc = count_nonfinite(xc);
			size_t at = 0;
			const double d = max_abs_diff(xc, xr, &at);
			say(fmt("iteration %2d: non-finite floats cpu %zu rd %zu (first rd v%ld); max|cpu - rd| %.3g at v%zu", k, nc,
					nr, fr < 0 ? -1L : fr / 3, d, at / 3));
			if (nr > 0 || d > tol_) {
				K_ = k;
				kernelStage(0, 0);
				return;
			}
			if (k >= cfg_.iters) {
				done(true, fmt("rd stays finite and within %g of cpu over %d iterations of the first step", tol_, k));
				return;
			}
			iterStage(k + 1);
		});
	}

	// 2. colour c, stage s of iteration K.
	void kernelStage(int c, int s) {
		cpu_->setDebugStopForTest(c, s);
		rd_->setDebugStopForTest(c, s);
		cpuFromPredictor(K_);
		rdFromPredictor(K_);
		q.next([this, c, s]() {
			// Per-constraint buffers first: at stage 0 the vertex scratch rows
			// of later colours still hold the previous run's values.
			static const char *const names[] = { "attachGradV", "attachHess", "triGrad", "triHess", "bendGrad",
				"bendHess", "gScratch", "hScratch", "positions" };
			static const char *const stages[] = { "init+forces", "gather spring", "gather attachment",
				"gather triangle", "gather bending", "solve" };
			std::string line = fmt("iteration %d colour %d after %-17s:", K_, c, stages[s]);
			std::vector<std::string> bad;
			for (const char *n : names) {
				const std::vector<float> vc = cpu_->readDebugForTest(n), vr = rd_->readDebugForTest(n);
				long fr;
				const size_t nr = count_nonfinite(vr, &fr), nc = count_nonfinite(vc);
				const double d = max_abs_diff(vc, vr);
				if (nr || nc) {
					const std::string sn = n;
					const long per = sn == "hScratch" ? 6 : (sn.find("Hess") != std::string::npos ? 1 : 3);
					line += fmt(" %s non-finite cpu %zu rd %zu (first rd row %ld);", n, nc, nr, fr < 0 ? -1L : fr / per);
					bad.push_back(n);
				}
				if (d > 0.0) {
					line += fmt(" %s %.2g;", n, d);
				}
			}
			say(line);
			if (!bad.empty()) {
				explain(bad);
				done(false, fmt("first non-finite rd buffer at iteration %d colour %d after %s: %s", K_, c, stages[s],
								   bad[0].c_str()));
				return;
			}
			int nc2 = c, ns = s + 1;
			if (ns > 5) {
				ns = 0;
				++nc2;
			}
			if (nc2 >= int(rd_->numColors())) {
				done(false, fmt("iteration %d leaves cpu by more than %g without a non-finite rd buffer", K_, tol_));
				return;
			}
			kernelStage(nc2, ns);
		});
	}

	// 3. the constraint behind the first non-finite row of a force buffer.
	void explain(const std::vector<std::string> &bad) {
		for (const std::string &n : bad) {
			const std::vector<float> vr = rd_->readDebugForTest(n.c_str());
			const std::vector<float> xr = rd_->readDebugForTest("positions");
			const std::vector<float> xc = cpu_->readDebugForTest("positions");
			long first;
			count_nonfinite(vr, &first);
			if (n == "bendGrad" || n == "bendHess") {
				const size_t row = size_t(first) / (n == "bendGrad" ? 3 : 1);
				const uint32_t b = uint32_t(row / 4);
				std::string ids;
				double sd[3] = { 0, 0, 0 };
				float sf[3] = { 0, 0, 0 }, sfc[3] = { 0, 0, 0 };
				for (int r = 0; r < 4; ++r) {
					const uint32_t v = scene_.bendIdx[4 * b + r];
					const float w = scene_.bendW[4 * b + r];
					ids += fmt(" v%u w=%.9g rd(%.9g %.9g %.9g) cpu(%.9g %.9g %.9g)", v, double(w), xr[3 * v],
							xr[3 * v + 1], xr[3 * v + 2], xc[3 * v], xc[3 * v + 1], xc[3 * v + 2]);
					for (int k = 0; k < 3; ++k) {
						sd[k] += double(w) * double(xr[3 * v + k]);
						sf[k] += w * xr[3 * v + k];
						sfc[k] += w * xc[3 * v + k];
					}
				}
				const double lenD = std::sqrt(sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]);
				const float lenF = std::sqrt(sf[0] * sf[0] + sf[1] * sf[1] + sf[2] * sf[2]);
				const float lenFc = std::sqrt(sfc[0] * sfc[0] + sfc[1] * sfc[1] + sfc[2] * sfc[2]);
				say(fmt("  %s row %zu = bending %u: nTarget %.9g k %.9g;%s", n.c_str(), row, b, double(scene_.bendN[b]),
						double(scene_.bendK[b]), ids.c_str()));
				say(fmt("  |s| on rd's positions: %.6g (double), %.6g (float, sequential); on cpu's positions %.6g (float)",
						lenD, double(lenF), double(lenFc)));
				size_t below = 0;
				for (uint32_t k = 0; k < scene_.nBend(); ++k) {
					if (scene_.bendN[k] > 1e-6f && scene_.bendN[k] < 1e-4f) {
						++below;
					}
				}
				say(fmt("  bendings with 1e-6 < nTarget < 1e-4: %zu of %u", below, scene_.nBend()));
				return;
			}
			if (n == "triGrad" || n == "triHess") {
				const size_t row = size_t(first) / (n == "triGrad" ? 3 : 1);
				const uint32_t t = uint32_t(row / 3);
				say(fmt("  %s row %zu = triangle %u (%u %u %u) invUV (%.6g %.6g %.6g %.6g) k %.6g", n.c_str(), row, t,
						scene_.tri[3 * t], scene_.tri[3 * t + 1], scene_.tri[3 * t + 2], double(scene_.triInvUV[4 * t]),
						double(scene_.triInvUV[4 * t + 1]), double(scene_.triInvUV[4 * t + 2]),
						double(scene_.triInvUV[4 * t + 3]), double(scene_.triK[t])));
				return;
			}
		}
		say("  first non-finite buffer: " + bad[0] + " (a vertex buffer: see the force buffers of the stage before)");
	}
};

template <class S>
std::unique_ptr<jobs::Job> make_on(const std::string &name, const Args &a, rdc::Device &dev, std::string &err) {
	std::unique_ptr<DJob<S>> j;
	if (name == "sphere_forward") {
		j = std::make_unique<SphereForwardJob<S>>();
	} else if (name == "sphere_backward") {
		j = std::make_unique<SphereBackwardJob<S>>();
	} else if (name == "sim_gradcheck") {
		j = std::make_unique<SimGradcheckJob<S>>();
	} else if (name == "bench_drape") {
		j = std::make_unique<BenchDrapeJob<S>>();
	} else {
		err = "unknown job " + name + " (" + drape_job_names() + ")";
		return nullptr;
	}
	if (!j->init(dev, err, name)) {
		return nullptr;
	}
	if (!j->build(a, err)) {
		return nullptr;
	}
	return j;
}

} // namespace

std::unique_ptr<jobs::Job> make_drape_job(const std::string &name, const std::string &backend,
		const std::string &args, rdc::Device &dev, std::string &err) {
	if (name.rfind("lbfgsb_", 0) == 0) {
		// The L-BFGS-B gates (G1, G2): lbfgsb_jobs.cpp; "auto" is cpu (n <= 1000:
		// see gates/5-drape/lbfgsb for the per-iteration cost on each backend).
		return make_lbfgsb_job(name, backend == "auto" ? "cpu" : backend, args, dev, err);
	}
	if (name == "inverse_min") {
		// G3: 4 vertices, so auto is cpu (rule 5).
		return make_inverse_job(backend == "auto" ? "cpu" : backend, args, dev, err);
	}
	Args a(args);
	if (name == "mesh_parity") {
		// Both backends by construction; `backend` is ignored.
		auto j = std::make_unique<MeshParityJob>();
		if (!j->build(dev, a, err)) {
			return nullptr;
		}
		return j;
	}
	if (name == "mesh_bisect") {
		auto j = std::make_unique<MeshBisectJob>();
		if (!j->build(dev, a, err)) {
			return nullptr;
		}
		return j;
	}
	std::string b = backend;
	if (b == "auto") {
		// The scene sizes the jobs build: the sphere demo is 625 vertices,
		// the gradcheck panels 64; the bench spans the crossover, so it
		// takes an explicit backend.
		if (name == "bench_drape") {
			err = "bench_drape needs cpu or rd";
			return nullptr;
		}
		b = drape_pick_backend("auto", name == "sim_gradcheck" ? 64u : 625u);
	}
	if (b == "cpu") {
		return make_on<AvbdCpu>(name, a, dev, err);
	}
	if (b == "rd") {
		return make_on<AvbdRd>(name, a, dev, err);
	}
	err = "backend must be cpu, rd or auto";
	return nullptr;
}

const char *drape_job_names() {
	return "sphere_forward sphere_backward sim_gradcheck bench_drape mesh_parity mesh_bisect inverse_min "
	       "lbfgsb_components lbfgsb_problems lbfgsb_replay lbfgsb_bench";
}

DrapeSession *drape_job_session(jobs::Job *job) {
	auto *d = dynamic_cast<DrapeJobBase *>(job);
	return d ? d->session() : nullptr;
}
