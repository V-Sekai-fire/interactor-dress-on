// Gate 5 G3 oracle: LBFGSpp 0.3.0 (the solver DiffCloth's BackwardTaskSolver
// uses) on cloth-dynamics' inverse_min objective, with the objective compiled
// for the host from the same sources drape.elf runs (guest/drape/inverse_min.h
// over guest/avbd/avbd_cpu*.cpp). drape.elf's inverse_min job drives the same
// objective with its own L-BFGS-B and compares with the traces written here:
// iterations within 2, final parameters within 1e-3.
//
//   tests/inverse_min_oracle/build.sh     -> gates/5-drape/oracle/inverse_min/
//
// Per case: the target at the truth, LBFGSpp's run from upstream's start with
// upstream's clamps as lower bounds (every evaluation and iterate logged), the
// zero-gradient arm (LBFGSpp with g forced to 0: it must stay put), and
// upstream's own backtracking gradient descent (information: what
// test_avbd_inverse_min prints, on this objective).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <LBFGSB.h>

#include "avbd_cpu.h"
#include "inverse_min.h"

using invmin::Case;

namespace {

LBFGSpp::LBFGSBParam<double> gateParam() {
	LBFGSpp::LBFGSBParam<double> p; // LBFGSpp's defaults, DiffCloth's m
	p.m = 10;
	p.max_linesearch = 20;
	p.max_iterations = 100;
	return p;
}

void wparam(FILE *f, const LBFGSpp::LBFGSBParam<double> &p) {
	std::fprintf(f,
			"param m %d epsilon %.9g epsilon_rel %.9g past %d delta %.9g max_iterations %d max_submin %d "
			"max_linesearch %d min_step %.9g max_step %.9g ftol %.9g wolfe %.9g\n",
			p.m, p.epsilon, p.epsilon_rel, p.past, p.delta, p.max_iterations, p.max_submin, p.max_linesearch,
			p.min_step, p.max_step, p.ftol, p.wolfe);
}

struct Objective {
	AvbdCpu &s;
	int c;
	const std::vector<float> &target;
	bool zeroGrad;
	FILE *log;
	int nfev = 0;
	double operator()(const Eigen::VectorXd &xd, Eigen::VectorXd &g) {
		const Case &cs = invmin::caseOf(c);
		float x[2] = { 0, 0 };
		for (int i = 0; i < cs.n; ++i) {
			x[i] = float(xd[i]);
		}
		invmin::Eval<AvbdCpu> e(s, invmin::paramsOf(c, x), &target, true);
		while (e.advance()) {
		}
		float gf[2] = { 0, 0 };
		invmin::gradOf(c, e, gf);
		g.resize(cs.n);
		for (int i = 0; i < cs.n; ++i) {
			g[i] = zeroGrad ? 0.0 : double(gf[i]);
		}
		++nfev;
		if (log) {
			std::fprintf(log, "eval %d x", nfev);
			for (int i = 0; i < cs.n; ++i) {
				std::fprintf(log, " %.9g", double(x[i]));
			}
			std::fprintf(log, " f %.9g g", e.loss);
			for (int i = 0; i < cs.n; ++i) {
				std::fprintf(log, " %.9g", g[i]);
			}
			std::fprintf(log, "\n");
		}
		return e.loss;
	}
};

double lossAt(AvbdCpu &s, int c, const float *x, const std::vector<float> &target) {
	invmin::Eval<AvbdCpu> e(s, invmin::paramsOf(c, x), &target, false);
	while (e.advance()) {
	}
	return e.loss;
}

// test_avbd_inverse_min.cpp's descend() / the 2-parameter loop, verbatim in
// its arithmetic (information only).
void upstreamGd(AvbdCpu &s, int c, const std::vector<float> &target, float *out) {
	const Case &cs = invmin::caseOf(c);
	float x[2] = { cs.start[0], cs.start[1] };
	double step = c == 0 ? 0.1 : 0.05;
	const int iters = c == 0 ? 40 : 60, backs = c == 0 ? 30 : 40;
	for (int it = 0; it < iters; ++it) {
		invmin::Eval<AvbdCpu> e(s, invmin::paramsOf(c, x), &target, true);
		while (e.advance()) {
		}
		const double L = e.loss;
		if (!(L == L)) {
			break;
		}
		double g[2] = { c == 0 ? e.dkTri : e.dkBend, e.ddensity };
		const double gn = c == 0 ? std::fabs(g[0]) : std::sqrt(g[0] * g[0] + g[1] * g[1]);
		if (gn < (c == 0 ? 1e-12 : 1e-10)) {
			break;
		}
		double t = step;
		float xn[2] = { x[0], x[1] };
		bool moved = false;
		for (int b = 0; b < backs; ++b) {
			for (int i = 0; i < cs.n; ++i) {
				xn[i] = float(double(x[i]) - t * g[i]);
				if (xn[i] < cs.lb[i]) {
					xn[i] = cs.lb[i];
				}
			}
			const double Ln = lossAt(s, c, xn, target);
			if (Ln == Ln && Ln < L) {
				moved = true;
				break;
			}
			t *= 0.5;
		}
		if (c == 0) {
			if (xn[0] == x[0]) {
				break;
			}
		} else if (!moved) {
			break;
		}
		x[0] = xn[0];
		x[1] = xn[1];
		step = t * 1.5;
	}
	out[0] = x[0];
	out[1] = x[1];
}

} // namespace

int main(int argc, char **argv) {
	const std::string outDir = argc > 1 ? argv[1] : ".";
	int bad = 0;
	for (int c = 0; c < 2; ++c) {
		const Case &cs = invmin::caseOf(c);
		AvbdCpu s;
		invmin::Eval<AvbdCpu> te(s, invmin::paramsOf(c, cs.truth), nullptr, false);
		while (te.advance()) {
		}
		const std::vector<float> target = te.x;
		const std::string path = outDir + "/case_" + cs.name + ".txt";
		FILE *f = std::fopen(path.c_str(), "w");
		if (!f || target.size() != 12) {
			std::printf("cannot write %s or target failed\n", path.c_str());
			return 1;
		}
		std::fprintf(f, "# inverse_min %s: LBFGSpp 0.3.0 LBFGSBSolver<double> (MoreThuente) on the host-compiled AvbdCpu\n",
				cs.name);
		std::fprintf(f, "# objective (guest/drape/inverse_min.h); tests/inverse_min_oracle/oracle.cpp\n");
		std::fprintf(f, "case %s\nn %d\n", cs.name, cs.n);
		const auto prm = gateParam();
		wparam(f, prm);
		std::fprintf(f, "truth %d", cs.n);
		for (int i = 0; i < cs.n; ++i) std::fprintf(f, " %.9g", double(cs.truth[i]));
		std::fprintf(f, "\nstart %d", cs.n);
		for (int i = 0; i < cs.n; ++i) std::fprintf(f, " %.9g", double(cs.start[i]));
		std::fprintf(f, "\nlb %d", cs.n);
		for (int i = 0; i < cs.n; ++i) std::fprintf(f, " %.9g", double(cs.lb[i]));
		std::fprintf(f, "\ntarget 12");
		for (float v : target) std::fprintf(f, " %.9g", double(v));
		std::fprintf(f, "\n");

		for (int zero = 0; zero < 2; ++zero) {
			Objective obj{ s, c, target, zero != 0, zero ? nullptr : f };
			LBFGSpp::LBFGSBSolver<double> solver(prm);
			Eigen::VectorXd x(cs.n), lb(cs.n), ub(cs.n);
			for (int i = 0; i < cs.n; ++i) {
				x[i] = cs.start[i];
				lb[i] = cs.lb[i];
				ub[i] = INFINITY;
			}
			double fx = 0.0;
			int niter = -1;
			std::string status = "converged";
			try {
				niter = solver.minimize(obj, x, fx, lb, ub);
			} catch (const std::exception &e) {
				status = std::string("threw:") + e.what();
				for (char &ch : status) {
					if (ch == ' ') ch = '_';
				}
			}
			const char *pre = zero ? "zero_" : "";
			std::fprintf(f, "%sniter %d\n%snfev %d\n%sstatus %s\n%sf_final %.9g\n%sx %d", pre, niter, pre, obj.nfev, pre,
					status.c_str(), pre, fx, pre, cs.n);
			double err = 0.0;
			for (int i = 0; i < cs.n; ++i) {
				std::fprintf(f, " %.9g", x[i]);
				err = std::fmax(err, std::fabs(x[i] - double(cs.truth[i])));
			}
			std::fprintf(f, "\n%serr %.9g\n", pre, err);
			std::printf("%s %s: LBFGSpp iterations %d evaluations %d x", cs.name, zero ? "zero-gradient" : "gradient", niter,
					obj.nfev);
			for (int i = 0; i < cs.n; ++i) std::printf(" %.9g", x[i]);
			std::printf(" f %.9g err %.3g (%s)\n", fx, err, status.c_str());
			if (!zero && err > 0.05) ++bad;
			if (zero && err <= 0.1) ++bad;
		}
		float gd[2];
		upstreamGd(s, c, target, gd);
		double gerr = 0.0;
		std::fprintf(f, "gd_x %d", cs.n);
		for (int i = 0; i < cs.n; ++i) {
			std::fprintf(f, " %.9g", double(gd[i]));
			gerr = std::fmax(gerr, std::fabs(double(gd[i]) - double(cs.truth[i])));
		}
		std::fprintf(f, "\ngd_err %.9g\n", gerr);
		std::printf("%s upstream gradient descent: x %.9g %.9g err %.3g\n", cs.name, double(gd[0]), double(gd[1]), gerr);
		std::fclose(f);
	}
	std::printf("%s\n", bad ? "ORACLE: LBFGSpp itself misses a G3 criterion" : "ORACLE: LBFGSpp meets both G3 criteria");
	return 0;
}
