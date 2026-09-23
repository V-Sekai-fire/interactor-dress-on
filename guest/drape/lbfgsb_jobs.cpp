// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "lbfgsb_jobs.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <sstream>
#include <vector>

#include "drape_scene.h"
#include "lbfgsb_gate.h"
#include "vec_cpu.h"
#include "vec_rd.h"

std::map<std::string, std::string> &lbfgsb_job_data() {
	static std::map<std::string, std::string> d;
	return d;
}

std::unique_ptr<lbv::Vec> make_lbfgsb_vec(const std::string &backend, rdc::Device &dev, std::string &err) {
	if (backend == "cpu") {
		return std::make_unique<VecCpu>();
	}
	if (backend == "rd") {
		if (!dev.open()) {
			err = dev.error();
			return nullptr;
		}
		auto v = std::make_unique<VecRd>(dev);
		if (!v->ok()) {
			err = "rd: " + v->error();
			return nullptr;
		}
		return v;
	}
	err = "vec backend must be cpu or rd";
	return nullptr;
}

namespace {

std::string fmt(const char *f, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

std::string argOf(const std::string &args, const std::string &key) {
	std::istringstream ss(args);
	std::string t;
	while (ss >> t) {
		if (t.rfind(key + "=", 0) == 0) {
			return t.substr(key.size() + 1);
		}
	}
	return "";
}

class LbJob : public jobs::Job {
public:
	std::unique_ptr<lbv::Vec> vec;
	bool pending() const override { return vec && vec->pending(); }
	void drain() override {
		if (vec) {
			vec->sync();
		}
	}

protected:
	void say(const std::string &s) { log_ += s + "\n"; }
	std::string log_;
};

// G1: every component fixture, then every fixture again with theta x1.25.
class ComponentsJob : public LbJob {
public:
	bool build(const std::string &args, std::string &err) {
		const std::string only = argOf(args, "only");
		for (int id = 0; id < 20; ++id) {
			char key[16];
			std::snprintf(key, sizeof key, "comp_%02d", id);
			auto it = lbfgsb_job_data().find(key);
			if (it == lbfgsb_job_data().end()) {
				err = std::string("no data ") + key + " (drape_job_data first: gates/5-drape/oracle/components)";
				return false;
			}
			if (!only.empty() && std::string(key).find(only) == std::string::npos) {
				continue;
			}
			names_.push_back(key);
		}
		for (int corrupt = 0; corrupt < 2; ++corrupt) {
			for (size_t i = 0; i < names_.size(); ++i) {
				q.push([this, i, corrupt]() { stepRun(i, corrupt != 0); });
			}
		}
		q.push([this]() { verdict(); });
		return true;
	}

private:
	void stepRun(size_t i, bool corrupt) {
		if (!run_) {
			run_ = std::make_unique<lbg::ComponentRun>(names_[i], lbfgsb_job_data()[names_[i]], corrupt);
			t0_ = now_us;
		}
		if (run_->step(*vec)) {
			q.next([this, i, corrupt]() { stepRun(i, corrupt); });
			return;
		}
		say(std::string(corrupt ? "control " : "") + run_->line() + fmt(" wall_ms=%.1f", (now_us - t0_) / 1000.0));
		if (corrupt) {
			ctlFails_ += run_->pass() ? 0 : 1;
			run_->addWorst(ctlWorst_);
		} else {
			passes_ += run_->pass() ? 1 : 0;
			run_->addWorst(worst_);
		}
		run_.reset();
	}
	void verdict() {
		const int n = int(names_.size());
		const bool ok = passes_ == n && ctlFails_ == n && n > 0;
		say("worst (fixtures):" + worst_.table());
		say("worst (control):" + ctlWorst_.table());
		say(fmt("phases=%lld dispatches=%lld", (long long)vec->phases(), (long long)vec->dispatches()));
		finish(ok,
				fmt("G1 %s: %d/%d fixtures pass, control (theta x1.25) fails %d/%d", vec->name(), passes_, n, ctlFails_,
						n),
				log_);
	}
	std::vector<std::string> names_;
	std::unique_ptr<lbg::ComponentRun> run_;
	int64_t t0_ = 0;
	int passes_ = 0, ctlFails_ = 0;
	lbg::Worst worst_, ctlWorst_;
};

// G2: every oracle trace.
class ProblemsJob : public LbJob {
public:
	bool build(const std::string &args, std::string &err) {
		const std::string only = argOf(args, "only");
		for (const auto &kv : lbfgsb_job_data()) {
			if (kv.first.rfind("trace_", 0) != 0) {
				continue;
			}
			if (!only.empty() && kv.first.find(only) == std::string::npos) {
				continue;
			}
			traces_.emplace_back();
			lbg::Trace &t = traces_.back();
			if (!t.parse(kv.second, err)) {
				err = kv.first + ": " + err;
				return false;
			}
			if (!problems_.count(t.problem)) {
				auto p = lbfgsb_job_data().find("prob_" + t.problem);
				if (p == lbfgsb_job_data().end()) {
					err = "no data prob_" + t.problem;
					return false;
				}
				if (!problems_[t.problem].parse(p->second, err)) {
					return false;
				}
			}
			names_.push_back(kv.first.substr(6));
		}
		if (traces_.empty()) {
			err = "no trace_* data (drape_job_data first: gates/5-drape/oracle/traces)";
			return false;
		}
		// G2's f band per trace comes with its flat control (Cut 5c).
		auto ctl = lbfgsb_job_data().find("f32io_control");
		if (ctl == lbfgsb_job_data().end()) {
			err = "no data f32io_control (drape_job_data first: gates/5-drape/oracle/f32io_control.txt)";
			return false;
		}
		std::vector<lbg::Trace *> tp;
		for (lbg::Trace &t : traces_) {
			tp.push_back(&t);
		}
		if (!lbg::applyF32Control(ctl->second, names_, tp, err)) {
			return false;
		}
		for (size_t i = 0; i < traces_.size(); ++i) {
			q.push([this, i]() { stepRun(i); });
		}
		q.push([this]() { verdict(); });
		return true;
	}

private:
	void stepRun(size_t i) {
		if (!run_) {
			const lbg::Trace &t = traces_[i];
			run_ = std::make_unique<lbg::ProblemRun>(names_[i], problems_[t.problem], t, *vec);
			t0_ = now_us;
			phases0_ = vec->phases();
			stages_ = 0;
		}
		++stages_;
		if (run_->step()) {
			q.next([this, i]() { stepRun(i); });
			return;
		}
		const double ms = (now_us - t0_) / 1000.0;
		const int it = run_->driver().iterations();
		say(run_->line() + fmt(" first_f_divergence=%d wall_ms=%.1f ms/iter=%.3f phases=%lld stages=%d",
								   run_->firstDivergence(), ms, it ? ms / it : 0.0,
								   (long long)(vec->phases() - phases0_), stages_));
		passes_ += run_->pass() ? 1 : 0;
		worstF_ = std::max(worstF_, run_->fRatio);
		worstX_ = std::max(worstX_, run_->xRatio);
		worstIt_ = std::max(worstIt_, run_->itRatio);
		run_.reset();
	}
	void verdict() {
		const int n = int(traces_.size());
		say(fmt("worst measured/limit: f %.3g x %.3g iterations %.3g", worstF_, worstX_, worstIt_));
		finish(passes_ == n, fmt("G2 %s: %d/%d traces pass", vec->name(), passes_, n), log_);
	}
	std::deque<lbg::Trace> traces_;
	std::map<std::string, lbg::Problem> problems_;
	std::vector<std::string> names_;
	std::unique_ptr<lbg::ProblemRun> run_;
	int64_t t0_ = 0, phases0_ = 0;
	int stages_ = 0, passes_ = 0;
	double worstF_ = 0, worstX_ = 0, worstIt_ = 0;
};

// G7: the native sphere demo's L-BFGS-B run replayed through our driver. The
// objective is backwardLog's table: (loss, dL/dmu) at mu 0.539770 (the seed-1
// start, kNativeSphereMu0 exactly; float32 here), at mu 0.01 (the lower bound) and anywhere else (the third point);
// DiffCloth's parameters (m 10, delta 1e-3, max_linesearch 20) and bounds
// [0.01, 0.95]. Native evaluates 0.539770 -> 0.010000 -> 0.375146 and stops
// after one iteration. backwardLog prints dL/dmu at mu 0.539770 to four
// digits (0.01153), and the line search's second trial depends on it, so the
// replay also runs at the ends of that digit's interval (g0 +- 5e-6) and
// passes when that interval of trials reaches native's printed 0.375146
// (+- 5e-7). keys: f0 g0 f1 g1 f2 g2 (the table; defaults backwardLog's),
// tag (a label), strict=0 (report only).
class ReplayJob : public LbJob {
public:
	bool build(const std::string &args, std::string &err) {
		std::istringstream ss(args);
		std::string t;
		while (ss >> t) {
			const size_t e = t.find('=');
			if (e == std::string::npos) {
				err = "lbfgsb_replay wants key=value, got " + t;
				return false;
			}
			const std::string k = t.substr(0, e), v = t.substr(e + 1);
			if (k == "tag") {
				tag_ = v;
				continue;
			}
			if (k == "strict") {
				strict_ = std::atoi(v.c_str()) != 0;
				continue;
			}
			static const char *const keys[6] = { "f0", "g0", "f1", "g1", "f2", "g2" };
			int slot = -1;
			for (int i = 0; i < 6; ++i) {
				if (k == keys[i]) {
					slot = i;
				}
			}
			if (slot < 0) {
				err = "lbfgsb_replay keys: f0 g0 f1 g1 f2 g2 tag strict; got " + k;
				return false;
			}
			tab_[slot] = std::atof(v.c_str());
		}
		for (int arm = 0; arm < 3; ++arm) {
			q.push([this, arm]() { startArm(arm); });
		}
		q.push([this]() { verdict(); });
		return true;
	}

private:
	// f0 g0 (mu 0.539770), f1 g1 (mu 0.01), f2 g2 (elsewhere).
	double tab_[6] = { 0.00132918, 0.01153, 1.65198194, -50.45588, 0.00047714, 0.00781 };
	std::string tag_ = "backwardLog";
	bool strict_ = true;
	std::unique_ptr<Lbfgsb> drv_;
	int arm_ = 0;
	double g0_ = 0.0;
	std::vector<float> xs_[3];
	int iters_[3] = {}, nfev_[3] = {};
	std::string reason_[3];

	void startArm(int arm) {
		arm_ = arm;
		g0_ = tab_[1] + (arm == 1 ? -5e-6 : arm == 2 ? 5e-6 : 0.0);
		LbfgsbParams prm;
		prm.m = 10;
		prm.delta = 1e-3;
		prm.max_linesearch = 20;
		const float x0 = float(kNativeSphereMu0), lb = 0.01f, ub = 0.95f;
		drv_ = std::make_unique<Lbfgsb>(*vec);
		handle(drv_->start(1, &x0, &lb, &ub, prm));
	}
	void handle(Lbfgsb::Status s) {
		for (;;) {
			switch (s) {
				case Lbfgsb::BUSY:
					q.next([this]() { handle(drv_->next()); });
					return;
				case Lbfgsb::NEED_EVAL:
				case Lbfgsb::TRY: {
					std::vector<float> x;
					if (!drv_->readX(x) || x.empty()) {
						reason_[arm_] = "FAIL readX";
						return;
					}
					xs_[arm_].push_back(x[0]);
					double f = tab_[4];
					float g = float(tab_[5]);
					if (std::fabs(double(x[0]) - 0.539770) < 1e-6) {
						f = tab_[0];
						g = float(g0_);
					} else if (std::fabs(double(x[0]) - 0.01) < 1e-7) {
						f = tab_[2];
						g = float(tab_[3]);
					}
					if (!drv_->setGradient(&g)) {
						reason_[arm_] = "FAIL setGradient";
						return;
					}
					s = drv_->next(f);
					break;
				}
				case Lbfgsb::ACCEPT:
					s = drv_->next();
					break;
				case Lbfgsb::CONVERGED:
				case Lbfgsb::FAIL:
					iters_[arm_] = drv_->iterations();
					nfev_[arm_] = drv_->nfev();
					reason_[arm_] = s == Lbfgsb::FAIL ? "FAIL " + drv_->error() : drv_->reason();
					return;
			}
		}
	}
	void verdict() {
		const char *names[3] = { "nominal", "g0-5e-6", "g0+5e-6" };
		double lo = INFINITY, hi = -INFINITY;
		for (int a = 0; a < 3; ++a) {
			std::string seq;
			for (size_t i = 0; i < xs_[a].size(); ++i) {
				seq += fmt("%s%.6f", i ? " -> " : "", double(xs_[a][i]));
			}
			say(fmt("%s %s (g0 %.8g): evaluations %s (third %.9g); iterations %d, evaluations %d, stop %s", tag_.c_str(),
					names[a], a == 0 ? tab_[1] : tab_[1] + (a == 1 ? -5e-6 : 5e-6), seq.c_str(),
					xs_[a].size() > 2 ? double(xs_[a][2]) : NAN, iters_[a], nfev_[a], reason_[a].c_str()));
			if (xs_[a].size() > 2) {
				lo = std::min(lo, double(xs_[a][2]));
				hi = std::max(hi, double(xs_[a][2]));
			}
		}
		const std::vector<float> &x = xs_[0];
		const bool shape = x.size() == 3 && x[0] == float(kNativeSphereMu0) && x[1] == 0.01f && iters_[0] == 1 &&
				reason_[0] == "delta";
		// Native printed 0.375146: its trial lies in [0.3751455, 0.3751465]
		// (float32 x: +- 3e-8).
		const bool third = hi >= 0.3751455 - 3e-8 && lo <= 0.3751465 + 3e-8;
		const std::string x1 = x.size() > 1 ? fmt("%.6f", double(x[1])) : std::string("-");
		const std::string x2 = x.size() > 2 ? fmt("%.6f", double(x[2])) : std::string("-");
		const std::string head = fmt("G7 %s replay (vec %s): 0.539770 -> %s -> %s (the g0 interval gives [%.7f, %.7f]; "
									 "native 0.375146), %d iteration, %d evaluations, stop %s",
				tag_.c_str(), vec->name(), x1.c_str(), x2.c_str(), lo, hi, iters_[0], nfev_[0], reason_[0].c_str());
		finish(!strict_ || (shape && third), head, log_);
	}
};

// G9: L-BFGS-B cost per iteration, cpu against rd, on a tridiagonal box QP
// (0.5 x'Ax - b'x, A diagonally dominant, |x| <= 1) of each size, for `iters`
// iterations (epsilon 0, delta 0: nothing stops it early). One job per size
// and backend; the host times it (a cpu run takes one or two ticks, and the
// guest's clock is the tick's). keys: n (1000), iters (10), m (10).
class BenchJob : public LbJob {
public:
	bool build(const std::string &args, std::string &err) {
		const std::string ns = argOf(args, "n"), is = argOf(args, "iters"), ms = argOf(args, "m");
		n_ = uint32_t(std::atof(ns.empty() ? "1000" : ns.c_str()));
		if (!is.empty()) {
			iters_ = std::atoi(is.c_str());
		}
		if (!ms.empty()) {
			m_ = std::atoi(ms.c_str());
		}
		if (n_ < 2 || n_ > 2000000 || iters_ < 1) {
			err = "lbfgsb_bench: n in [2, 2e6], iters >= 1";
			return false;
		}
		max_stages_per_tick = 8;
		p_.name = "boxqp_bench";
		p_.n = n_;
		p_.lb.assign(n_, -1.0);
		p_.ub.assign(n_, 1.0);
		p_.x0.assign(n_, 0.0);
		p_.aDiag.resize(n_);
		p_.b.resize(n_);
		p_.aOff = -1.0;
		for (uint32_t i = 0; i < n_; ++i) {
			p_.aDiag[i] = 2.5 + 0.5 * double((i * 4u) % 11u);
			p_.b[i] = double(float(8.0 * std::sin(1.7 * double(i) + 0.3)));
		}
		q.push([this]() { stepRun(); });
		return true;
	}

private:
	lbg::Problem p_;
	uint32_t n_ = 1000;
	int iters_ = 10, m_ = 10;
	std::unique_ptr<Lbfgsb> drv_;
	std::vector<float> x_;
	std::vector<double> g_;
	std::vector<float> gf_;
	bool started_ = false;
	int evals_ = 0;
	double f_ = NAN;

	void stepRun() {
		Lbfgsb::Status s;
		if (!started_) {
			started_ = true;
			LbfgsbParams prm;
			prm.m = m_;
			prm.epsilon = 0.0;
			prm.epsilon_rel = 0.0;
			prm.delta = 0.0;
			prm.max_iterations = iters_;
			const std::vector<float> x0(p_.x0.begin(), p_.x0.end()), lb(p_.lb.begin(), p_.lb.end()),
					ub(p_.ub.begin(), p_.ub.end());
			drv_ = std::make_unique<Lbfgsb>(*vec);
			s = drv_->start(n_, x0.data(), lb.data(), ub.data(), prm);
		} else {
			s = drv_->next();
		}
		for (;;) {
			switch (s) {
				case Lbfgsb::BUSY:
					q.next([this]() { stepRun(); });
					return;
				case Lbfgsb::NEED_EVAL:
				case Lbfgsb::TRY: {
					if (!drv_->readX(x_)) {
						finish(false, "lbfgsb_bench readX: " + vec->error(), log_);
						return;
					}
					f_ = p_.fg(x_, g_);
					++evals_;
					gf_.assign(g_.begin(), g_.end());
					if (!drv_->setGradient(gf_.data())) {
						finish(false, "lbfgsb_bench setGradient: " + vec->error(), log_);
						return;
					}
					s = drv_->next(f_);
					break;
				}
				case Lbfgsb::ACCEPT:
					s = drv_->next();
					break;
				case Lbfgsb::CONVERGED:
				case Lbfgsb::FAIL: {
					// A small problem can converge before `iters` (the line
					// search then finds no decrease): a bench, not a verdict on
					// the solver, so any run with an iteration and a finite f counts.
					const bool ok = drv_->iterations() > 0 && std::isfinite(drv_->fx());
					const std::string why = s == Lbfgsb::FAIL ? drv_->error() : drv_->reason();
					finish(ok, fmt("lbfgsb_bench %s n=%u iterations=%d evaluations=%d f=%.9g stop=%s phases=%lld",
									   vec->name(), n_, drv_->iterations(), evals_, drv_->fx(), why.c_str(),
									   (long long)vec->phases()),
							log_);
					return;
				}
			}
		}
	}
};

} // namespace

std::unique_ptr<jobs::Job> make_lbfgsb_job(const std::string &name, const std::string &backend,
		const std::string &args, rdc::Device &dev, std::string &err) {
	if (backend != "cpu" && backend != "rd") {
		err = name + " needs cpu or rd";
		return nullptr;
	}
	std::unique_ptr<LbJob> j;
	if (name == "lbfgsb_components") {
		auto c = std::make_unique<ComponentsJob>();
		c->vec = make_lbfgsb_vec(backend, dev, err);
		if (!c->vec || !c->build(args, err)) {
			return nullptr;
		}
		j = std::move(c);
	} else if (name == "lbfgsb_problems") {
		auto p = std::make_unique<ProblemsJob>();
		p->vec = make_lbfgsb_vec(backend, dev, err);
		if (!p->vec || !p->build(args, err)) {
			return nullptr;
		}
		j = std::move(p);
	} else if (name == "lbfgsb_replay") {
		auto p = std::make_unique<ReplayJob>();
		p->vec = make_lbfgsb_vec(backend, dev, err);
		if (!p->vec || !p->build(args, err)) {
			return nullptr;
		}
		j = std::move(p);
	} else if (name == "lbfgsb_bench") {
		auto p = std::make_unique<BenchJob>();
		p->vec = make_lbfgsb_vec(backend, dev, err);
		if (!p->vec || !p->build(args, err)) {
			return nullptr;
		}
		j = std::move(p);
	} else {
		err = "unknown job " + name;
		return nullptr;
	}
	return j;
}
