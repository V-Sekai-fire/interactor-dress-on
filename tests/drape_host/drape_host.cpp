// Host-native harness for the drape driver on AvbdCpu: the same sources the
// guest compiles (drape_scene, primitives, drape_sim.h, drape_backward.h,
// drape_session.h, jobs.cpp, the Lean-emitted kernels as C++), built for the
// host so a sanitizer can watch them. Not part of any gate verdict.
//
//   SAN=1 tests/drape_host/build.sh && build/drape_host/drape_host sphere 20 [mu]
//   build/drape_host/drape_host session 5      (the StageQueue path, 4 stages a tick)
//   build/drape_host/drape_host gradcheck plane|panel [steps] [colors]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "avbd_cpu.h"
#include "drape_backward.h"
#include "drape_scene.h"
#include "drape_session.h"
#include "drape_sim.h"
#include "jobs.h"

static void tick_all(jobs::StageQueue &q, DrapeSession &s, int cap) {
	int ticks = 0;
	while (!q.empty()) {
		s.now_us += 1000;
		q.tick([&s]() { return s.pending(); }, cap);
		++ticks;
	}
	std::printf("ticks=%d %s\n", ticks, s.result().c_str());
}

int main(int argc, char **argv) {
	const std::string what = argc > 1 ? argv[1] : "sphere";
	if (what == "sphere") {
		const int steps = argc > 2 ? std::atoi(argv[2]) : 10;
		AvbdCpu solver;
		DrapeSimT<AvbdCpu> sim(solver);
		sim.cfg.mu = argc > 3 ? std::atof(argv[3]) : 0.539770;
		scene_sphere_demo(sim.scene, sim.cfg);
		sim.setup();
		for (int s = 1; s <= steps; ++s) {
			std::vector<float> xp = sim.x;
			while (!sim.advance()) {
			}
			const DrapeStepRecord &r = sim.recs.back();
			double dx = 0.0;
			for (size_t b = 0; b < xp.size(); ++b) {
				dx = std::fmax(dx, std::fabs(r.xAvbd[b] - xp[b]));
			}
			const size_t pushes = r.pushes.size();
			std::printf("step %d |dx|_max=%.9g blends=%zu fric=%zu proj=%u pairs=%u pushes=%zu\n", s, dx, r.pred.size(),
					r.fric.size(), r.projHits, r.selfPairs, pushes);
		}
		std::printf("finite=%d\n", int(sim.finite()));
		return 0;
	}
	if (what == "session") {
		const int steps = argc > 2 ? std::atoi(argv[2]) : 3;
		DrapeSessionT<AvbdCpu> s;
		s.solver = std::make_unique<AvbdCpu>();
		DrapeConfig cfg;
		cfg.mu = 0.539770;
		DrapeScene sc;
		scene_sphere_demo(sc, cfg);
		std::string err;
		if (!s.load(sc, cfg, err)) {
			std::printf("load: %s\n", err.c_str());
			return 1;
		}
		jobs::StageQueue q;
		s.enqueueForward(q, steps);
		tick_all(q, s, 4);
		s.targetFromFrames();
		s.rewind();
		s.enqueueForward(q, steps);
		s.enqueueBackward(q, DrapeLoss::MatchTrajectory, DrapeMode::Native);
		tick_all(q, s, 4);
		s.enqueueBackward(q, DrapeLoss::MatchTrajectory, DrapeMode::Step);
		tick_all(q, s, 4);
		s.enqueueBackward(q, DrapeLoss::MatchTrajectory, DrapeMode::Unrolled);
		tick_all(q, s, 4);
		return 0;
	}
	return 1;
}
