// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "lbfgsb_jobs.h"

#include <cstdarg>
#include <cstdio>
#include <deque>
#include <sstream>
#include <vector>

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
	} else {
		err = "unknown job " + name;
		return nullptr;
	}
	return j;
}
