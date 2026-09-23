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
#include "drape_sim.h"

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

// Per-step statistics a forward can collect (the native [avbd-step] line's
// |dx|_max is max |x_avbd - x| over components).
struct StepStat {
	size_t step = 0;
	double dxMax = 0.0;
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
void collect_stats(DrapeSessionT<S> &s, std::vector<StepStat> &out) {
	DrapeSessionT<S> *sp = &s;
	std::vector<StepStat> *op = &out;
	s.onStep = [sp, op](const DrapeStepRecord &r, const std::vector<float> &xPrev) {
		const size_t k = sp->sim->steps();
		if (!kStatSteps.count(k)) {
			return;
		}
		StepStat st;
		st.step = k;
		for (size_t b = 0; b < r.xAvbd.size() && b < xPrev.size(); ++b) {
			st.dxMax = std::max(st.dxMax, double(std::fabs(r.xAvbd[b] - xPrev[b])));
		}
		st.fric = r.fric.size();
		st.pred = r.pred.size();
		st.proj = r.projHits;
		st.pairs = r.selfPairs;
		st.pushes = r.pushes.size();
		op->push_back(st);
	};
}

std::string stat_line(const StepStat &st) {
	return fmt("step %4zu |dx|_max=%.9g friction_blends=%zu friction_events=%zu projections=%zu self_pairs=%zu "
			   "self_pushes=%zu",
			st.step, st.dxMax, st.pred, st.fric, st.proj, st.pairs, st.pushes);
}

// --- sphere_forward: the rotating-sphere demo forward ------------------------------
//
// keys: steps (100), mu (0.539770: the native iter0's mu), any DrapeConfig key.

template <class S>
class SphereForwardJob : public DJob<S> {
public:
	bool build(const Args &a, std::string &err) override {
		steps_ = int(a.num("steps", 100));
		DrapeConfig cfg;
		cfg.mu = 0.539770;
		if (!a.applyConfig(cfg, { "steps" }, err)) {
			return false;
		}
		cfg_ = cfg;
		this->loadStage([this](DrapeScene &sc, DrapeConfig &c) {
			c = cfg_;
			scene_sphere_demo(sc, c);
		});
		collect_stats(*this->sess_, stats_);
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
// keys: target (0.3: the ground truth), mus (0.539770,0.01: LBFGS iter0 and
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
		for (const std::string &m : a.list("mus", "0.539770,0.01")) {
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
				this->say(fmt("RESULT mu=%.6f loss=%.9g dL/dmu=%.9g", mu, g.loss, dmu));
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
	Args a(args);
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
	return "sphere_forward sphere_backward sim_gradcheck bench_drape";
}

DrapeSession *drape_job_session(jobs::Job *job) {
	auto *d = dynamic_cast<DrapeJobBase *>(job);
	return d ? d->session() : nullptr;
}
