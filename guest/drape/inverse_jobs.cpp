// Gate 5 G3: inverse_min (guest/drape/inverse_min.h) driven by the in-guest
// L-BFGS-B, against LBFGSpp on the same objective compiled for the host
// (gates/5-drape/oracle/inverse_min, handed over with drape_job_data as
// invmin_k_tri / invmin_k_bend_density).
//
// Per case (k_tri; k_bend and density):
//  - the target: the solver's own rollout at the truth;
//  - the gradient arm from upstream's start, with upstream's clamps as lower
//    bounds: it must land within 0.05 of the truth (upstream's criterion), in
//    LBFGSpp's iteration count +-2 with the parameters within 1e-3 of
//    LBFGSpp's;
//  - the zero-gradient arm (the falsifiability control, upstream's): the same
//    loop with g forced to 0 must stay more than 0.1 away.
// The solver and the L-BFGS-B vectors run on the job's backend (cpu: AvbdCpu
// and the slangc cpp kernels; rd: AvbdRd and the SPIR-V over rdc::Device);
// `vec=cpu|rd` overrides the vectors. Every solver or vector submit ends the
// tick (AGENTS.md rule 4).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "inverse_jobs.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

#include "avbd/avbd_cpu.h"
#include "avbd/avbd_rd.h"
#include "inverse_min.h"
#include "lbfgsb.h"
#include "lbfgsb_jobs.h"

namespace {

std::string fmt(const char *f, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

// The oracle file: "key rest-of-line" (vectors as "key <n> v...").
struct Oracle {
	std::map<std::string, std::string> line;
	bool parse(const std::string &text) {
		std::istringstream in(text);
		std::string l;
		while (std::getline(in, l)) {
			if (l.empty() || l[0] == '#') {
				continue;
			}
			const size_t sp = l.find(' ');
			line[l.substr(0, sp)] = sp == std::string::npos ? "" : l.substr(sp + 1);
		}
		return line.count("param") && line.count("niter") && line.count("x");
	}
	double num(const std::string &k) const {
		auto it = line.find(k);
		return it == line.end() ? NAN : std::atof(it->second.c_str());
	}
	std::vector<double> vec(const std::string &k) const {
		std::vector<double> v;
		auto it = line.find(k);
		if (it == line.end()) {
			return v;
		}
		std::istringstream in(it->second);
		int n = 0;
		in >> n;
		for (int i = 0; i < n; ++i) {
			double x;
			in >> x;
			v.push_back(x);
		}
		return v;
	}
};

template <class S>
struct SolverOf;
template <>
struct SolverOf<AvbdCpu> {
	static const char *name() { return "cpu"; }
	static std::unique_ptr<AvbdCpu> make(rdc::Device &, std::string &) { return std::make_unique<AvbdCpu>(); }
};
template <>
struct SolverOf<AvbdRd> {
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

template <class S>
class InverseMinJob : public jobs::Job {
public:
	bool pending() const override { return (s_ && s_->pending()) || (vec_ && vec_->pending()); }
	void drain() override {
		if (s_) {
			s_->sync();
		}
		if (vec_) {
			vec_->sync();
		}
	}

	bool build(rdc::Device &dev, const std::string &args, std::string &err) {
		std::string vecName = SolverOf<S>::name();
		std::istringstream in(args);
		std::string tok;
		while (in >> tok) {
			if (tok.rfind("vec=", 0) == 0) {
				vecName = tok.substr(4);
			} else if (tok.rfind("cases=", 0) == 0) {
				cases_.clear();
				for (char ch : tok.substr(6)) {
					if (ch == '0' || ch == '1') {
						cases_.push_back(ch - '0');
					}
				}
			} else {
				err = "inverse_min keys: vec=cpu|rd cases=01; got " + tok;
				return false;
			}
		}
		s_ = SolverOf<S>::make(dev, err);
		if (!s_) {
			return false;
		}
		vec_ = make_lbfgsb_vec(vecName, dev, err);
		if (!vec_) {
			return false;
		}
		for (int c : cases_) {
			const std::string key = std::string("invmin_") + invmin::caseOf(c).name;
			auto it = lbfgsb_job_data().find(key);
			if (it == lbfgsb_job_data().end() || !oracle_[c].parse(it->second)) {
				err = "no oracle " + key + " (drape_job_data: gates/5-drape/oracle/inverse_min/case_*.txt)";
				return false;
			}
		}
		name_ = fmt("inverse_min %s (vec %s)", SolverOf<S>::name(), vec_->name());
		for (int c : cases_) {
			q.push([this, c]() { startCase(c); });
		}
		q.push([this]() { verdict(); });
		return true;
	}

private:
	struct Arm {
		int iters = -1, nfev = 0;
		std::vector<float> x;
		double f = NAN, err = NAN;
		std::string how;
		int64_t us = 0;
	};

	std::unique_ptr<S> s_;
	std::unique_ptr<lbv::Vec> vec_;
	std::vector<int> cases_ = { 0, 1 };
	Oracle oracle_[2];
	std::string name_, log_;
	std::vector<float> target_;
	std::unique_ptr<invmin::Eval<S>> eval_;
	std::unique_ptr<Lbfgsb> drv_;
	int case_ = 0;
	bool zero_ = false;
	Arm arm_;
	int64_t t0_ = 0;
	int fails_ = 0;
	std::string summary_;

	void say(const std::string &s) { log_ += s + "\n"; }

	// Run eval_ one stage per queue slot, then `then`.
	void pump(std::function<void()> then) {
		if (eval_->advance()) {
			q.next([this, then]() { pump(then); });
			return;
		}
		then();
	}

	void startCase(int c) {
		case_ = c;
		const invmin::Case &cs = invmin::caseOf(c);
		eval_ = std::make_unique<invmin::Eval<S>>(*s_, invmin::paramsOf(c, cs.truth), nullptr, false);
		pump([this, c]() {
			if (!eval_->done()) {
				++fails_;
				say(fmt("FAIL %s: the target rollout failed", invmin::caseOf(c).name));
				return;
			}
			target_ = eval_->x;
			// Our target against LBFGSpp's (the host-compiled AvbdCpu).
			const std::vector<double> ot = oracle_[c].vec("target");
			double td = ot.size() == target_.size() ? 0.0 : INFINITY;
			for (size_t i = 0; i < ot.size() && i < target_.size(); ++i) {
				td = std::fmax(td, std::fabs(double(target_[i]) - ot[i]));
			}
			say(fmt("%s: target at the truth, max |ours - host AvbdCpu| = %.3g", invmin::caseOf(c).name, td));
			startArm(false);
		});
	}

	void startArm(bool zero) {
		zero_ = zero;
		arm_ = Arm();
		t0_ = now_us;
		const invmin::Case &cs = invmin::caseOf(case_);
		LbfgsbParams prm;
		std::string err;
		if (!prm.parse(oracle_[case_].line["param"], err)) {
			++fails_;
			say("FAIL param: " + err);
			return;
		}
		std::vector<float> x0(cs.start, cs.start + cs.n), lb(cs.lb, cs.lb + cs.n), ub(cs.n, INFINITY);
		drv_ = std::make_unique<Lbfgsb>(*vec_);
		handle(drv_->start(uint32_t(cs.n), x0.data(), lb.data(), ub.data(), prm));
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
					if (!drv_->readX(x)) {
						armDone("FAIL readX: " + vec_->error());
						return;
					}
					eval_ = std::make_unique<invmin::Eval<S>>(*s_, invmin::paramsOf(case_, x.data()), &target_, true);
					q.next([this]() { pump([this]() { evaluated(); }); });
					return;
				}
				case Lbfgsb::ACCEPT:
					s = drv_->next();
					break;
				case Lbfgsb::CONVERGED:
					armDone(drv_->reason());
					return;
				case Lbfgsb::FAIL:
					armDone("FAIL " + drv_->error());
					return;
			}
		}
	}

	void evaluated() {
		if (!eval_->done()) {
			armDone("FAIL: the objective's rollout failed");
			return;
		}
		const int n = invmin::caseOf(case_).n;
		float g[2] = { 0, 0 };
		invmin::gradOf(case_, *eval_, g);
		if (zero_) {
			g[0] = g[1] = 0.0f;
		}
		if (!zero_) {
			std::vector<float> x;
			drv_->readX(x);
			std::string xs, gs;
			for (int i = 0; i < n; ++i) {
				xs += fmt(" %.9g", double(x[i]));
				gs += fmt(" %.9g", double(g[i]));
			}
			say(fmt("  eval %d x%s f %.9g g%s", drv_->nfev() + 1, xs.c_str(), eval_->loss, gs.c_str()));
		}
		if (!drv_->setGradient(g)) {
			armDone("FAIL setGradient: " + vec_->error());
			return;
		}
		handle(drv_->next(eval_->loss));
	}

	void armDone(const std::string &how) {
		const invmin::Case &cs = invmin::caseOf(case_);
		arm_.how = how;
		arm_.iters = drv_->iterations();
		arm_.nfev = drv_->nfev();
		arm_.f = drv_->fx();
		arm_.us = now_us - t0_;
		drv_->readX(arm_.x);
		arm_.err = 0.0;
		for (int i = 0; i < cs.n && i < int(arm_.x.size()); ++i) {
			arm_.err = std::fmax(arm_.err, std::fabs(double(arm_.x[i]) - double(cs.truth[i])));
		}
		std::string xs;
		for (float v : arm_.x) {
			xs += fmt(" %.9g", double(v));
		}
		const Oracle &o = oracle_[case_];
		if (zero_) {
			const bool ok = arm_.err > 0.1 && how.rfind("FAIL", 0) != 0;
			fails_ += ok ? 0 : 1;
			say(fmt("%s %s zero-gradient arm (control): x%s err %.3g (want > 0.1), %d iterations (%s); LBFGSpp's arm: "
					"err %.3g",
					ok ? "PASS" : "FAIL", cs.name, xs.c_str(), arm_.err, arm_.iters, how.c_str(), o.num("zero_err")));
			summary_ += fmt(" %s control err %.3g;", cs.name, arm_.err);
			return;
		}
		const std::vector<double> ox = o.vec("x");
		double dx = ox.size() == size_t(cs.n) ? 0.0 : INFINITY;
		for (int i = 0; i < cs.n && i < int(ox.size()) && i < int(arm_.x.size()); ++i) {
			dx = std::fmax(dx, std::fabs(double(arm_.x[i]) - ox[i]));
		}
		const int oit = int(o.num("niter"));
		const bool rec = arm_.err <= 0.05;
		const bool its = std::abs(arm_.iters - oit) <= 2;
		const bool par = dx <= 1e-3;
		const bool ok = rec && its && par && how.rfind("FAIL", 0) != 0;
		fails_ += ok ? 0 : 1;
		say(fmt("%s %s: x%s (truth %s) err %.3g (want <= 0.05); iterations %d vs LBFGSpp %d (want +-2), evaluations %d vs "
				"%d; |x - x_LBFGSpp| %.3g (want <= 1e-3); f %.3g; stop %s; host ms %.1f",
				ok ? "PASS" : "FAIL", cs.name, xs.c_str(),
				cs.n == 1 ? fmt("%g", cs.truth[0]).c_str() : fmt("%g,%g", cs.truth[0], cs.truth[1]).c_str(), arm_.err,
				arm_.iters, oit, arm_.nfev, int(o.num("nfev")), dx, arm_.f, how.c_str(), arm_.us / 1000.0));
		summary_ += fmt(" %s err %.3g it %d/%d dx %.2g;", cs.name, arm_.err, arm_.iters, oit, dx);
		// The control next.
		q.next([this]() { startArm(true); });
	}

	void verdict() {
		finish(fails_ == 0,
				fmt("%s: G3 %s:%s", name_.c_str(), fails_ == 0 ? "both cases recover, trace LBFGSpp, controls stay put"
															   : "FAILED",
						summary_.c_str()),
				log_);
	}
};

} // namespace

std::unique_ptr<jobs::Job> make_inverse_job(const std::string &backend, const std::string &args, rdc::Device &dev,
		std::string &err) {
	if (backend == "cpu") {
		auto j = std::make_unique<InverseMinJob<AvbdCpu>>();
		if (!j->build(dev, args, err)) {
			return nullptr;
		}
		return j;
	}
	if (backend == "rd") {
		auto j = std::make_unique<InverseMinJob<AvbdRd>>();
		if (!j->build(dev, args, err)) {
			return nullptr;
		}
		return j;
	}
	err = "inverse_min needs cpu or rd";
	return nullptr;
}
