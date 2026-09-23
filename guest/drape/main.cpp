// drape.elf -- the drape stage: the AVBD solver on both backends.
//
// One ELF per stage (AGENTS.md rule 6): this one owns the solver, its oracle
// and its bench, with its own RenderingDevice held in a guest static.
// dress_on.elf keeps the Stage 1 GPU-layer probes. Every ADD_API_FUNCTION here
// has a no-argument wrapper in project/main.gd (rule 8).

#include <api.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "rd_compute.h"
#include "avbd_fixture.h"
#include "avbd/avbd_cpu.h"
#include "avbd/avbd_rd.h"
#include "avbd/avbd_sim.h"
#include "avbd/cloth_grid.h"
#include "avbd_jobs.h"
#include "body_mesh.h"
#include "drape_jobs.h"
#include "drape_scene.h"
#include "drape_session.h"
#include "jobs.h"
#include "lbfgsb.h"
#include "similarity.h" // brings sinew_align.h in, as extern "C"
#include "lbfgsb_jobs.h"

// This stage's device, held across vmcalls.
static rdc::Device g_dev;

static Variant text(const std::string &s) {
	return Variant(String(s));
}

// The one-shot MCP entry points are CPU only. On rd a one-shot call would
// submit and sync in one vmcall, a same-frame sync (AGENTS.md rule 4); the rd
// oracle and bench are the jobs "fixture" and "bench_fwd"/"bench_bwd" through
// avbd_job_start/avbd_job_tick below.

// The oracle (guest/avbd_fixture.cpp) on the CPU path.
static Variant avbd_fixture(String backend_s) {
	const std::string backend = backend_s.utf8();
	if (backend == "cpu") {
		return text(avbd_fixture_cpu());
	}
	return text("FAIL: backend must be cpu (rd: avbd_job_start(\"fixture\", \"rd\"), then avbd_job_tick per frame)");
}

// A pinned nx-by-ny panel dropped under gravity for `substeps` substeps of
// `iters` outer iterations each, on the CPU. Returns the final state's
// finiteness and lowest point; the host times the whole call.
static Variant avbd_bench(String backend_s, int nx, int ny, int substeps, int iters) {
	const std::string backend = backend_s.utf8();
	if (nx < 2 || ny < 2 || substeps < 1 || iters < 1) {
		return text("FAIL: nx, ny >= 2 and substeps, iters >= 1");
	}
	if (backend != "cpu") {
		return text("FAIL: backend must be cpu (rd: the jobs bench_fwd/bench_bwd through avbd_job_start/avbd_job_tick)");
	}
	AvbdCpu solver;
	ClothSimT<AvbdCpu> sim(solver);
	sim.setup(build_cloth_mesh(uint32_t(nx), uint32_t(ny), 1.0f, 1.0f, 1.0f, 0.5f));
	sim.solver.buildColoring();
	sim.iters = iters;
	for (int s = 0; s < substeps; ++s) {
		sim.step();
	}
	float ymin = 1e30f;
	for (uint32_t i = 0; i < sim.mesh.nVerts(); ++i) {
		ymin = ymin < sim.pos[3 * i + 1] ? ymin : sim.pos[3 * i + 1];
	}
	char b[160];
	std::snprintf(b, sizeof b, "cpu nv=%u colors=%u substeps=%d iters=%d finite=%s ymin=%.4f",
			sim.mesh.nVerts(), unsigned(sim.solver.numColors()), substeps, iters, sim.finite() ? "yes" : "NO", ymin);
	return text(b);
}

// Where did it get to? Named steps, not booleans.
static Variant rd_last_step() {
	return text(g_dev.step());
}

// The rule-4 guard (AGENTS.md rule 4): how many sync() calls landed in the
// same process frame as their submit(). A frame-driven host keeps this at 0.
static Variant rd_rule4() {
	return text("same_frame_syncs=" + std::to_string(g_dev.same_frame_syncs()) +
			" syncs=" + std::to_string(g_dev.syncs()) + " submits=" + std::to_string(g_dev.submits()));
}

// The guard's positive control: submit and sync in one call, which must raise
// same_frame_syncs by one.
static Variant rd_rule4_probe() {
	if (!g_dev.open()) {
		return text(g_dev.error());
	}
	const int64_t before = g_dev.same_frame_syncs();
	g_dev.list_begin();
	g_dev.list_end();
	g_dev.submit();
	g_dev.sync();
	const int64_t after = g_dev.same_frame_syncs();
	return text(std::string(after == before + 1 ? "PASS" : "FAIL") + " same_frame_syncs " +
			std::to_string(before) + " -> " + std::to_string(after));
}

// --- the job API: one job at a time, advanced one host frame per tick --------
//
// AGENTS.md rule 4 by construction: a job is a stage queue (guest/jobs.h)
// whose tick ends at every GPU submit, so its readbacks run on a later frame.
// The host starts a job, then calls avbd_job_tick once per _process with its
// own clock (the guest clock is not a clock) until the answer stops being
// "RUNNING k/N". A job is never destroyed in the frame its solver submitted:
// one replaced while in flight is parked and freed on a later frame.

static std::unique_ptr<jobs::Job> g_job;
static std::vector<std::unique_ptr<jobs::Job>> g_parked;
static int64_t g_parked_frame = -1;

// Through the device's method-name table (rd_compute.cpp): a literal here
// could share a host name-cache slot with a hot RenderingDevice name.
static int64_t process_frame() {
	return g_dev.process_frame();
}

static void free_parked() {
	if (!g_parked.empty() && process_frame() != g_parked_frame) {
		g_parked.clear();
	}
}

static Variant avbd_job_start(String name_s, String backend_s) {
	free_parked();
	const std::string name = name_s.utf8();
	const std::string backend = backend_s.utf8();
	if (g_job) {
		if (g_job->pending()) {
			g_parked.push_back(std::move(g_job));
			g_parked_frame = process_frame();
		}
		g_job.reset();
	}
	std::string err;
	g_job = make_avbd_job(name, backend, g_dev, err);
	if (!g_job) {
		return text("FAIL " + name + " " + backend + ": " + err);
	}
	return text("STARTED " + name + " " + backend);
}

static Variant avbd_job_tick(int64_t host_us) {
	free_parked();
	if (!g_job) {
		return text("FAIL: no job (avbd_job_start first)");
	}
	return text(g_job->tick(host_us));
}

static Variant avbd_job_names_api() {
	return text(avbd_job_names());
}

// --- the drape API: one session, advanced one host frame per drape_tick --------
//
// drape_open picks the backend (cpu, rd, or auto: rd from 160 vertices, rule
// 5); a scene call uploads; drape_config sets a knob (a material/topology
// knob re-uploads, and so rewinds, on the next drape_queue_forward);
// drape_queue_forward / drape_queue_backward queue stages which drape_tick
// runs, a GPU submit ending the tick (rule 4). drape_positions returns the
// last completed step and never syncs.

static std::string g_want = "auto";
static DrapeConfig g_cfg;
static DrapeScene g_scene;
static bool g_scene_set = false;
static bool g_dirty = false;
static std::unique_ptr<DrapeSession> g_sess;
static std::vector<std::unique_ptr<DrapeSession>> g_sess_parked;
static int64_t g_sess_parked_frame = -1;
static jobs::StageQueue g_q;

static void free_parked_sessions() {
	if (!g_sess_parked.empty() && process_frame() != g_sess_parked_frame) {
		g_sess_parked.clear();
	}
}

// An optimize's L-BFGS-B phase in flight (drape_queue_optimize, below).
static bool opt_pending();
static const std::string &drape_optimize_head();
// The last queued operation was an optimize: drape_tick reports its verdict.
static bool g_opt_last = false;

static bool session_busy() {
	return g_sess && (!g_q.empty() || g_sess->pending() || opt_pending());
}

static void drop_session() {
	if (g_sess && g_sess->pending()) {
		g_sess_parked.push_back(std::move(g_sess));
		g_sess_parked_frame = process_frame();
	}
	g_sess.reset();
	g_q = jobs::StageQueue();
}

// Upload g_scene with g_cfg on the backend drape_open asked for (auto picks
// by the vertex count), making the session if the backend changed.
static std::string load_scene() {
	const std::string b = drape_pick_backend(g_want, g_scene.nV);
	if (!g_sess || b != g_sess->backend()) {
		drop_session();
		std::string err;
		g_sess = make_drape_session(b, g_dev, err);
		if (!g_sess) {
			return "FAIL " + b + ": " + err;
		}
	}
	std::string err;
	if (!g_sess->load(g_scene, g_cfg, err)) {
		return "FAIL load: " + err;
	}
	g_dirty = false;
	return g_sess->result();
}

static Variant drape_open(String backend_s) {
	free_parked_sessions();
	const std::string b = backend_s.utf8();
	if (b != "cpu" && b != "rd" && b != "auto") {
		return text("FAIL: backend must be cpu, rd or auto");
	}
	if (session_busy()) {
		return text("BUSY: tick the queue empty first");
	}
	drop_session();
	g_want = b;
	g_scene_set = false;
	if (b == "auto") {
		return text("OPENED auto (rd from " + std::to_string(kDrapeAutoRdVerts) + " vertices)");
	}
	return text("OPENED " + b);
}

static Variant drape_scene_sphere_demo() {
	free_parked_sessions();
	if (session_busy()) {
		return text("BUSY: tick the queue empty first");
	}
	scene_sphere_demo(g_scene, g_cfg);
	g_scene_set = true;
	return text(load_scene());
}

// material: [density, kTri, kBend, kAttach], any prefix.
static Variant drape_scene_mesh(PackedFloat32Array pos_a, PackedInt32Array tris_a, PackedInt32Array pins_a,
		PackedFloat32Array material_a) {
	free_parked_sessions();
	if (session_busy()) {
		return text("BUSY: tick the queue empty first");
	}
	const std::vector<float> mat = material_a.fetch();
	const char *keys[4] = { "density", "kTri", "kBend", "kAttach" };
	for (size_t i = 0; i < mat.size() && i < 4; ++i) {
		if (!g_cfg.set(keys[i], mat[i])) {
			return text(std::string("FAIL: bad material ") + keys[i]);
		}
	}
	std::string err;
	if (!scene_mesh(g_scene, pos_a.fetch(), tris_a.fetch(), pins_a.fetch(), err)) {
		return text("FAIL: " + err);
	}
	g_scene_set = true;
	return text(load_scene());
}

// kind: sphere [cx,cy,cz, r, mu] | plane [cx,cy,cz, ulx,uly,ulz, urx,ury,urz, mu]
// | capsule [bx,by,bz, ax,ay,az, r, len, mu] | clear.
static Variant drape_primitive(String kind_s, PackedFloat32Array params_a) {
	const std::string kind = kind_s.utf8();
	const std::vector<float> p = params_a.fetch();
	auto v = [&](size_t i) { return v3d(p[i], p[i + 1], p[i + 2]); };
	if (kind == "clear") {
		g_scene.prims.clear();
	} else if (kind == "sphere" && p.size() >= 5) {
		g_scene.prims.push_back(make_sphere(v(0), p[3], p[4]));
	} else if (kind == "plane" && p.size() >= 10) {
		g_scene.prims.push_back(make_plane(v(0), v(3), v(6), p[9]));
	} else if (kind == "capsule" && p.size() >= 9) {
		g_scene.prims.push_back(make_capsule(v(0), v(3), p[6], p[7], p[8]));
	} else {
		return text("FAIL: kind sphere(5) | plane(10) | capsule(9) | clear, with that many params");
	}
	if (g_sess) {
		g_sess->setPrims(g_scene.prims);
	}
	return text(g_scene.describe());
}

// A triangle-mesh body collider (body_mesh.h), positions in drape units;
// params [skin, mu, band, depth], any prefix (0.1, 0.3, 0.1, 1.0).
static Variant drape_primitive_mesh(PackedFloat32Array pos_a, PackedInt32Array tris_a, PackedFloat32Array params_a) {
	const std::vector<float> p = params_a.fetch();
	double k[4] = { 0.1, 0.3, 0.1, 1.0 };
	for (size_t i = 0; i < p.size() && i < 4; ++i) {
		k[i] = p[i];
	}
	auto body = std::make_shared<BodyMesh>();
	std::string err;
	if (!body->build(pos_a.fetch(), tris_a.fetch(), 1.0, err)) {
		return text("FAIL: " + err);
	}
	g_scene.prims.push_back(make_mesh_collider(body, k[0], k[2], k[3], k[1]));
	if (g_sess) {
		g_sess->setPrims(g_scene.prims);
	}
	return text(g_scene.describe());
}

static Variant drape_config(String key_s, double value) {
	const std::string k = key_s.utf8();
	if (!g_cfg.set(k, value)) {
		return text("FAIL: unknown or refused key " + k);
	}
	static const char *upload[] = { "h", "density", "kTri", "kBend", "kAttach", "rawStiffness", "membrane", "bending",
		"colors", "selfK", "alGamma" };
	bool needs = false;
	for (const char *u : upload) {
		needs = needs || k == u;
	}
	if (needs) {
		g_dirty = true;
	} else if (g_sess) {
		g_sess->setLive(g_cfg);
	}
	if (k == "mu" && !g_scene.prims.empty()) {
		g_scene.prims[0].mu = value;
		if (g_sess) {
			g_sess->setPrims(g_scene.prims);
		}
	}
	return text(std::string(needs ? "SET (re-uploads and rewinds on the next forward) " : "SET ") + g_cfg.dump());
}

// steps > 0: queue that many steps after the current state; 0: rewind to x0.
static Variant drape_queue_forward(int steps) {
	free_parked_sessions();
	if (!g_sess || !g_scene_set) {
		return text("FAIL: no scene (drape_scene_sphere_demo or drape_scene_mesh)");
	}
	if (g_dirty) {
		if (session_busy()) {
			return text("BUSY: a changed knob needs a re-upload; tick the queue empty first");
		}
		const std::string r = load_scene();
		if (r.rfind("FAIL", 0) == 0) {
			return text(r);
		}
	}
	if (steps <= 0) {
		if (session_busy()) {
			return text("BUSY: tick the queue empty first");
		}
		g_sess->rewind();
		return text("REWOUND");
	}
	g_sess->enqueueForward(g_q, steps);
	g_opt_last = false;
	return text("QUEUED forward " + std::to_string(steps) + " on " + g_sess->backend());
}

// Cut 6d, the fit mode. The fit set of the loaded scene_mesh scene: one fit
// attachment per listed vertex (the loop lists every garment vertex not in
// its nofit set), pulling it toward the mesh collider's nearest surface point
// (+ gap along the outward normal, 0 in the loop) at kFit times the vertex's
// lumped area; params [kFit, gap, refresh, similarity, restEvery, settle,
// kAnchor], any prefix (600, 0, 1, 0, 4, 0, 100); similarity != 0 re-fits
// the rest shape every restEvery steps (drape_scene.h fitSimilarity) about
// the centroid of `anchor` (the waist loop, whose centre is held at its
// source position by a second attachment per vertex at kAnchor; empty = no
// hold, the garment's own centroid); settle = steps with the pull off after
// the fit. Re-uploads (the attachments change) on the next queue.
static Variant drape_fit_set(PackedInt32Array verts_a, PackedFloat32Array params_a, PackedInt32Array anchor_a) {
	free_parked_sessions();
	if (!g_scene_set || g_scene.name != "mesh") {
		return text("FAIL: no scene_mesh scene (drape_scene_mesh first)");
	}
	if (session_busy()) {
		return text("BUSY: tick the queue empty first");
	}
	const std::vector<float> p = params_a.fetch();
	double k = 600.0, gap = 0.0;
	int refresh = 1;
	if (p.size() > 0) {
		k = p[0];
	}
	if (p.size() > 1) {
		gap = p[1];
	}
	if (p.size() > 2) {
		refresh = int(p[2]);
	}
	const bool similarity = p.size() > 3 && p[3] != 0.0f;
	const int restEvery = p.size() > 4 ? int(p[4]) : 4;
	const int settle = p.size() > 5 ? int(p[5]) : 0;
	const double kAnchor = p.size() > 6 ? p[6] : 100.0;
	std::string err;
	if (!scene_fit_set(g_scene, verts_a.fetch(), k, gap, refresh, similarity, anchor_a.fetch(), restEvery, settle, kAnchor,
				err)) {
		return text("FAIL: " + err);
	}
	g_dirty = true;
	double amin = INFINITY, amax = 0.0, asum = 0.0;
	for (uint32_t a = g_scene.nPin(); a < g_scene.nAttach(); ++a) {
		const double av = g_scene.vertArea[g_scene.attachVert[a]];
		amin = std::min(amin, av);
		amax = std::max(amax, av);
		asum += av;
	}
	return text(drape_fmt("FIT SET fit=%u pins=%u kFit=%g gap=%g refresh=%d similarity=%d rest_every=%d settle=%d anchor=%u "
			  "kAnchor=%g vertex_area min=%.4g mean=%.4g max=%.4g (k per vertex = kFit x area; re-uploads on the next queue)",
			g_scene.nFit, g_scene.nPin(), k, gap, refresh, int(similarity), restEvery, settle, g_scene.nAnchorAtt, kAnchor,
			g_scene.nFit ? amin : 0.0, g_scene.nFit ? asum / g_scene.nFit : 0.0, amax));
}

// Queue the fit phase: up to max_steps steps with gravity off and damp 0,
// the fit targets refreshed every `refresh` steps from the mesh collider,
// stopping once the largest vertex move of a refresh cycle's last step is
// under tol (drape units). drape_tick runs it; drape_positions has the result.
static Variant drape_queue_fit(int max_steps, double tol) {
	free_parked_sessions();
	if (!g_sess || !g_scene_set) {
		return text("FAIL: no scene (drape_scene_mesh)");
	}
	if (max_steps < 1 || !(tol >= 0.0)) {
		return text("FAIL: max_steps >= 1 and tol >= 0");
	}
	if (g_scene.nFit == 0) {
		return text("FAIL: no fit set (drape_fit_set first)");
	}
	if (g_dirty) {
		if (session_busy()) {
			return text("BUSY: a changed knob needs a re-upload; tick the queue empty first");
		}
		const std::string r = load_scene();
		if (r.rfind("FAIL", 0) == 0) {
			return text(r);
		}
	}
	g_sess->enqueueFit(g_q, max_steps, tol);
	g_opt_last = false;
	return text("QUEUED fit " + std::to_string(max_steps) + " on " + g_sess->backend());
}

// kind: trajectory (the recorded frames become the MATCH_TRAJECTORY target)
// | points (verts, pos = 3 per vert, frame -1 = last) | clear.
static Variant drape_set_target(String kind_s, PackedInt32Array verts_a, PackedFloat32Array pos_a, int frame) {
	if (!g_sess) {
		return text("FAIL: no session");
	}
	if (session_busy()) {
		return text("BUSY: tick the queue empty first");
	}
	const std::string kind = kind_s.utf8();
	DrapeTarget &t = g_sess->target();
	if (kind == "trajectory") {
		g_sess->targetFromFrames();
		return text("TARGET trajectory frames=" + std::to_string(t.numFrames()));
	}
	if (kind == "points") {
		const std::vector<int32_t> vs = verts_a.fetch();
		const std::vector<float> ps = pos_a.fetch();
		if (vs.empty() || ps.size() != 3 * vs.size()) {
			return text("FAIL: points need 3 floats per vertex");
		}
		t.verts.assign(vs.begin(), vs.end());
		t.pos.assign(ps.begin(), ps.end());
		t.frame = frame;
		return text("TARGET points n=" + std::to_string(vs.size()) + " frame=" + std::to_string(frame));
	}
	if (kind == "clear") {
		t = DrapeTarget();
		return text("TARGET cleared");
	}
	return text("FAIL: kind trajectory | points | clear");
}

// loss: match_trajectory | target_points; mode: native | step | unrolled.
static Variant drape_queue_backward(String loss_s, String mode_s) {
	if (!g_sess) {
		return text("FAIL: no session");
	}
	const std::string l = loss_s.utf8(), m = mode_s.utf8();
	DrapeLoss loss;
	if (l == "match_trajectory") {
		loss = DrapeLoss::MatchTrajectory;
	} else if (l == "target_points") {
		loss = DrapeLoss::TargetPoints;
	} else {
		return text("FAIL: loss match_trajectory | target_points");
	}
	DrapeMode mode;
	if (m == "native") {
		mode = DrapeMode::Native;
	} else if (m == "step") {
		mode = DrapeMode::Step;
	} else if (m == "unrolled") {
		mode = DrapeMode::Unrolled;
	} else {
		return text("FAIL: mode native | step | unrolled");
	}
	g_sess->enqueueBackward(g_q, loss, mode);
	g_opt_last = false;
	return text("QUEUED backward " + l + " " + m);
}

static Variant drape_tick(int64_t host_us) {
	free_parked_sessions();
	if (!g_sess) {
		return text("IDLE no session");
	}
	g_sess->now_us = host_us;
	const bool cpu = std::string(g_sess->backend()) == "cpu";
	g_q.tick([]() { return g_sess->pending() || opt_pending(); }, cpu ? 4 : 256);
	if (!g_q.empty() || g_sess->pending() || opt_pending()) {
		return text("RUNNING " + std::to_string(g_q.done()) + "/" + std::to_string(g_q.total()));
	}
	const std::string &r = g_opt_last ? drape_optimize_head() : g_sess->result();
	return text("IDLE " + r.substr(0, r.find('\n')));
}

// The last completed step's positions (3 per vertex); never syncs.
static Variant drape_positions() {
	if (!g_sess) {
		return Variant(PackedFloat32Array(std::vector<float>()));
	}
	return Variant(PackedFloat32Array(g_sess->positions()));
}

static std::vector<float> frame_f32(DrapeSession *s, int i) {
	std::vector<float> out;
	if (s && i >= 0) {
		for (double d : s->frame(size_t(i))) {
			out.push_back(float(d));
		}
	}
	return out;
}

// Frame i (0 = x0, then each step's predictor, as the native OBJs).
static Variant drape_frame(int i) {
	return Variant(PackedFloat32Array(frame_f32(g_sess.get(), i)));
}

static Variant drape_faces() {
	std::vector<int32_t> f;
	const DrapeScene &sc = g_sess ? g_sess->scene() : g_scene;
	f.assign(sc.tri.begin(), sc.tri.end());
	return Variant(PackedInt32Array(f));
}

static Variant drape_result() {
	if (!g_sess) {
		return text("no session\n" + g_cfg.dump());
	}
	return text(g_sess->result() + "\n" + g_sess->config().dump() + "\n" + g_sess->scene().describe());
}

// --- drape_queue_optimize: L-BFGS-B over the session's parameters ----------------
//
// The in-guest L-BFGS-B (lbfgsb.h) with the session as its objective: each
// evaluation sets the parameters, rewinds, queues `steps` forward steps and a
// backward (loss and mode as drape_queue_backward) on the drape queue, and
// hands the loss and its gradient back. spec is "key=value ..." with
//   params=mu[,kTri,kBend,kAttach,density]   (default mu)
//   loss=match_trajectory|target_points      (default match_trajectory)
//   mode=native|step|unrolled                (default native)
//   steps=N                                  (default: the frames recorded now)
//   vec=cpu|rd|auto                          (the L-BFGS-B vectors; auto = cpu)
//   any LBFGSBParam key (m, epsilon, delta, max_linesearch, ...); the defaults
//   are DiffCloth's BackwardTaskSolver (m 10, delta 1e-3, max_linesearch 20).
// x0/lb/ub: one value per parameter (lb/ub empty: unbounded). maxIter 0 runs
// to convergence. The iterates stream to drape_optimize_result().

struct DrapeOpt {
	std::unique_ptr<lbv::Vec> vec;
	std::unique_ptr<Lbfgsb> drv;
	std::vector<std::string> params;
	DrapeLoss loss = DrapeLoss::MatchTrajectory;
	DrapeMode mode = DrapeMode::Native;
	int steps = 0;
	bool done = false;
	int64_t t0 = 0;
	std::string head, log;
	bool pending() const { return drv && drv->pending(); }
};
static std::unique_ptr<DrapeOpt> g_opt;
static std::vector<std::unique_ptr<DrapeOpt>> g_opt_parked;
static int64_t g_opt_parked_frame = -1;

static bool opt_pending() {
	return g_opt && g_opt->pending();
}

static const std::string &drape_optimize_head() {
	static const std::string none = "no optimize";
	return g_opt ? g_opt->head : none;
}

static void opt_handle(Lbfgsb::Status s);

static void opt_stage() {
	opt_handle(g_opt->drv->next());
}

static void opt_finish(const std::string &head) {
	DrapeOpt &o = *g_opt;
	o.done = true;
	char b[160];
	std::snprintf(b, sizeof b, " iterations=%d evaluations=%d wall_ms=%.1f", o.drv->iterations(), o.drv->nfev(),
			double(g_sess->now_us - o.t0) / 1000.0);
	o.head = head + b;
	o.log += o.head + "\n";
}

static std::string opt_x_line(const std::vector<float> &x) {
	std::string s;
	for (size_t i = 0; i < x.size() && i < g_opt->params.size(); ++i) {
		char b[64];
		std::snprintf(b, sizeof b, "%s%s=%.9g", i ? " " : "", g_opt->params[i].c_str(), double(x[i]));
		s += b;
	}
	return s;
}

// The collector after an evaluation's forward and backward: loss and
// gradient back to the driver.
static void opt_collect() {
	DrapeOpt &o = *g_opt;
	if (!g_sess->ok()) {
		const std::string &r = g_sess->result();
		opt_finish("FAIL optimize: " + r.substr(0, r.find('\n')));
		return;
	}
	const DrapeGrad &gr = g_sess->grad();
	std::vector<float> g(o.params.size(), 0.0f);
	for (size_t i = 0; i < o.params.size(); ++i) {
		const std::string &p = o.params[i];
		double d = gr.ddensity;
		if (p == "mu") {
			d = gr.dmu.empty() ? 0.0 : gr.dmu[0];
		} else if (p == "kTri") {
			d = gr.dkTri;
		} else if (p == "kBend") {
			d = gr.dkBend;
		} else if (p == "kAttach") {
			d = gr.dkAttach;
		}
		g[i] = float(d);
	}
	char b[200];
	std::snprintf(b, sizeof b, "  eval %d loss=%.9g grad[0]=%.9g\n", o.drv->nfev() + 1, gr.loss, double(g[0]));
	o.log += b;
	if (!o.drv->setGradient(g.data())) {
		opt_finish("FAIL optimize: setGradient");
		return;
	}
	opt_handle(o.drv->next(gr.loss));
}

// Apply x to the session, rewind, queue forward + backward + the collector.
static void opt_eval(const std::vector<float> &x) {
	DrapeOpt &o = *g_opt;
	bool reload = false;
	for (size_t i = 0; i < o.params.size(); ++i) {
		const double v = x[i];
		if (!g_cfg.set(o.params[i], v)) {
			opt_finish("FAIL optimize: " + o.params[i] + " refused " + std::to_string(v));
			return;
		}
		if (o.params[i] == "mu") {
			if (!g_scene.prims.empty()) {
				g_scene.prims[0].mu = v;
			}
		} else {
			reload = true;
		}
	}
	if (reload) {
		std::string err;
		if (!g_sess->load(g_scene, g_cfg, err)) {
			opt_finish("FAIL optimize: reload: " + err);
			return;
		}
	} else {
		g_sess->setLive(g_cfg);
		g_sess->setPrims(g_scene.prims);
		g_sess->rewind();
	}
	g_sess->enqueueForward(g_q, o.steps);
	g_sess->enqueueBackward(g_q, o.loss, o.mode);
	g_q.push([]() { opt_collect(); });
}

static void opt_handle(Lbfgsb::Status s) {
	DrapeOpt &o = *g_opt;
	for (;;) {
		switch (s) {
			case Lbfgsb::BUSY:
				g_q.next([]() { opt_stage(); });
				return;
			case Lbfgsb::NEED_EVAL:
			case Lbfgsb::TRY: {
				std::vector<float> x;
				o.drv->readX(x);
				o.log += std::string(s == Lbfgsb::TRY ? "  try " : "  x0 ") + opt_x_line(x) + "\n";
				opt_eval(x);
				return;
			}
			case Lbfgsb::ACCEPT: {
				std::vector<float> x;
				o.drv->readX(x);
				char b[160];
				std::snprintf(b, sizeof b, "iter %d f=%.9g pg=%.4g step=%.6g ", o.drv->iterations(), o.drv->fx(),
						o.drv->pgNorm(), o.drv->lastStep());
				o.log += b + opt_x_line(x) + "\n";
				s = o.drv->next();
				break;
			}
			case Lbfgsb::CONVERGED: {
				std::vector<float> x;
				o.drv->readX(x);
				char b[160];
				std::snprintf(b, sizeof b, "DONE optimize %s (%s) f=%.9g pg=%.4g ", o.vec->name(),
						o.drv->reason().c_str(), o.drv->fx(), o.drv->pgNorm());
				opt_finish(b + opt_x_line(x));
				return;
			}
			case Lbfgsb::FAIL:
				opt_finish("FAIL optimize: " + o.drv->error());
				return;
		}
	}
}

static Variant drape_queue_optimize(String spec_s, PackedFloat32Array x0_a, PackedFloat32Array lb_a,
		PackedFloat32Array ub_a, int max_iter) {
	free_parked_sessions();
	if (!g_opt_parked.empty() && process_frame() != g_opt_parked_frame) {
		g_opt_parked.clear();
	}
	if (!g_sess || !g_scene_set) {
		return text("FAIL: no scene (drape_scene_sphere_demo or drape_scene_mesh)");
	}
	if (session_busy()) {
		return text("BUSY: tick the queue empty first");
	}
	const std::string spec = spec_s.utf8();
	std::string lbkv, err, vecName = "auto", lossName = "match_trajectory", modeName = "native", params = "mu";
	int steps = int(g_sess->steps());
	LbfgsbParams prm;
	prm.m = 10;
	prm.delta = 1e-3;
	prm.max_linesearch = 20;
	std::istringstream ss(spec);
	std::string tok;
	while (ss >> tok) {
		const size_t eq = tok.find('=');
		if (eq == std::string::npos) {
			return text("FAIL: spec wants key=value, got " + tok);
		}
		const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
		if (k == "params") {
			params = v;
		} else if (k == "loss") {
			lossName = v;
		} else if (k == "mode") {
			modeName = v;
		} else if (k == "steps") {
			steps = std::atoi(v.c_str());
		} else if (k == "vec") {
			vecName = v;
		} else {
			lbkv += k + " " + v + " ";
		}
	}
	if (!prm.parse(lbkv, err)) {
		return text("FAIL: " + err);
	}
	prm.max_iterations = max_iter > 0 ? max_iter : 0;
	auto o = std::make_unique<DrapeOpt>();
	{
		std::istringstream ps(params);
		std::string p;
		while (std::getline(ps, p, ',')) {
			if (p != "mu" && p != "kTri" && p != "kBend" && p != "kAttach" && p != "density") {
				return text("FAIL: params are mu, kTri, kBend, kAttach, density; got " + p);
			}
			o->params.push_back(p);
		}
	}
	if (lossName == "match_trajectory") {
		o->loss = DrapeLoss::MatchTrajectory;
	} else if (lossName == "target_points") {
		o->loss = DrapeLoss::TargetPoints;
	} else {
		return text("FAIL: loss match_trajectory | target_points");
	}
	if (modeName == "native") {
		o->mode = DrapeMode::Native;
	} else if (modeName == "step") {
		o->mode = DrapeMode::Step;
	} else if (modeName == "unrolled") {
		o->mode = DrapeMode::Unrolled;
	} else {
		return text("FAIL: mode native | step | unrolled");
	}
	if (steps <= 0) {
		return text("FAIL: steps=N (no frames recorded to take the count from)");
	}
	o->steps = steps;
	const size_t n = o->params.size();
	std::vector<float> x0 = x0_a.fetch(), lb = lb_a.fetch(), ub = ub_a.fetch();
	if (n == 0 || x0.size() != n || (!lb.empty() && lb.size() != n) || (!ub.empty() && ub.size() != n)) {
		return text("FAIL: x0 (and lb, ub if given) need one value per parameter");
	}
	if (lb.empty()) {
		lb.assign(n, -INFINITY);
	}
	if (ub.empty()) {
		ub.assign(n, INFINITY);
	}
	// A handful of parameters: auto is the CPU vector backend (the L-BFGS-B
	// crossover is in gates/5-drape/lbfgsb; rd pays a submit per phase).
	o->vec = make_lbfgsb_vec(vecName == "auto" ? "cpu" : vecName, g_dev, err);
	if (!o->vec) {
		return text("FAIL: " + err);
	}
	o->drv = std::make_unique<Lbfgsb>(*o->vec);
	if (g_opt && g_opt->pending()) {
		g_opt_parked.push_back(std::move(g_opt));
		g_opt_parked_frame = process_frame();
	}
	g_opt = std::move(o);
	g_opt->t0 = g_sess->now_us;
	g_opt->log = "optimize " + spec + " | " + prm.dump() + "\n";
	const Lbfgsb::Status s = g_opt->drv->start(uint32_t(n), x0.data(), lb.data(), ub.data(), prm);
	g_q.push([s]() { opt_handle(s); });
	g_opt_last = true;
	return text("QUEUED optimize " + params + " on " + g_sess->backend() + " (vec " + g_opt->vec->name() + ")");
}

static Variant drape_optimize_result() {
	if (!g_opt) {
		return text("no optimize");
	}
	return text((g_opt->done ? g_opt->head : std::string("RUNNING")) + "\n" + g_opt->log);
}

// Hand the next drape job a data file (the Gate 5 oracle: comp_NN, prob_*,
// trace_*); key "clear" drops them all.
static Variant drape_job_data(String key_s, String text_s) {
	const std::string key = key_s.utf8();
	auto &d = lbfgsb_job_data();
	if (key == "clear") {
		d.clear();
		return text("CLEARED");
	}
	d[key] = text_s.utf8();
	return text("DATA " + key + " " + std::to_string(d[key].size()) + " bytes, " + std::to_string(d.size()) + " keys");
}

// --- the drape jobs (Gate 5): one at a time, like the avbd jobs ---------------------

static std::unique_ptr<jobs::Job> g_djob;
static std::vector<std::unique_ptr<jobs::Job>> g_djob_parked;
static int64_t g_djob_parked_frame = -1;

static void free_parked_djobs() {
	if (!g_djob_parked.empty() && process_frame() != g_djob_parked_frame) {
		g_djob_parked.clear();
	}
}

static Variant drape_job_start(String name_s, String backend_s, String args_s) {
	free_parked_djobs();
	const std::string name = name_s.utf8(), backend = backend_s.utf8(), args = args_s.utf8();
	if (g_djob) {
		if (g_djob->pending()) {
			g_djob_parked.push_back(std::move(g_djob));
			g_djob_parked_frame = process_frame();
		}
		g_djob.reset();
	}
	std::string err;
	g_djob = make_drape_job(name, backend, args, g_dev, err);
	if (!g_djob) {
		return text("FAIL " + name + " " + backend + ": " + err);
	}
	DrapeSession *s = drape_job_session(g_djob.get());
	return text(std::string("STARTED ") + name + " " + (s ? s->backend() : backend.c_str()) + " " + args);
}

static Variant drape_job_tick(int64_t host_us) {
	free_parked_djobs();
	if (!g_djob) {
		return text("FAIL: no job (drape_job_start first)");
	}
	return text(g_djob->tick(host_us));
}

static Variant drape_job_frame(int i) {
	return Variant(PackedFloat32Array(frame_f32(g_djob ? drape_job_session(g_djob.get()) : nullptr, i)));
}

static Variant drape_job_names_api() {
	return text(drape_job_names());
}

// The org's rotation fitter (vendor/sinew-align, the C port of sinew-mocap/
// solve's Align.lean) that the fit phase's similarity rest update uses,
// checked in the guest against AlignTest.lean's oracle: the 120-degree
// rotation of the unit quaternion (0.5, 0.5, 0.5, 0.5), as a matrix,
// recovered from (R b, b) pairs at N=5 (the covariance + ns30 path), N=2
// and N=1 (rodrigues), each to 1e-4 (the Lean test's bound); also the
// similarity fit's scale on the same pairs scaled by 0.7.
static Variant drape_sinew_align_test() {
	// Math.lean's quatToMat of (0.5, 0.5, 0.5, 0.5): the cyclic permutation
	// x -> y -> z -> x, row-major.
	const double w = 0.5, x = 0.5, y = 0.5, z = 0.5;
	const double R[9] = { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w),
		1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) };
	const double srcs[15] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0.3, -0.7, 0.5 };
	double tg[15];
	for (int i = 0; i < 5; ++i) {
		for (int r = 0; r < 3; ++r) {
			tg[3 * i + r] = R[3 * r] * srcs[3 * i] + R[3 * r + 1] * srcs[3 * i + 1] + R[3 * r + 2] * srcs[3 * i + 2];
		}
	}
	auto err = [&](const double *rec, size_t n) {
		double e = 0.0;
		for (size_t i = 0; i < n; ++i) {
			for (int r = 0; r < 3; ++r) {
				const double d = tg[3 * i + r] - (rec[3 * r] * srcs[3 * i] + rec[3 * r + 1] * srcs[3 * i + 1] +
														 rec[3 * r + 2] * srcs[3 * i + 2]);
				e = std::max(e, std::abs(d));
			}
		}
		return e;
	};
	double r5[9], r2[9], r1[9];
	sinew_align(tg, srcs, 5, r5);
	sinew_align(tg, srcs, 2, r2);
	sinew_align(tg + 3, srcs + 3, 1, r1);
	const double e5 = err(r5, 5), e2 = err(r2, 2);
	double e1 = 0.0;
	for (int r = 0; r < 3; ++r) {
		const double d = tg[3 + r] - (r1[3 * r] * srcs[3] + r1[3 * r + 1] * srcs[4] + r1[3 * r + 2] * srcs[5]);
		e1 = std::max(e1, std::abs(d));
	}
	// similarity_fit on the same pairs, the targets scaled by 0.7.
	float sf[15], df[15];
	for (int i = 0; i < 15; ++i) {
		sf[i] = float(srcs[i]);
		df[i] = float(0.7 * tg[i]);
	}
	Similarity S;
	const bool ok = similarity_fit(sf, df, 5, S);
	double eR = 0.0;
	for (int i = 0; i < 9; ++i) {
		eR = std::max(eR, std::abs(S.R[i] - R[i]));
	}
	const bool pass = e5 < 1e-4 && e2 < 1e-4 && e1 < 1e-4 && ok && std::abs(S.s - 0.7) < 1e-5 && eR < 1e-5;
	return text(drape_fmt("%s sinew_align oracle (AlignTest.lean, 120 deg): N=5 err %.3g N=2 err %.3g N=1 err %.3g "
						  "(want < 1e-4); similarity_fit s=%.7f (want 0.7) max|R-Rtrue|=%.3g rotated=%d",
			pass ? "PASS" : "FAIL", e5, e2, e1, S.s, eR, int(S.rotated)));
}

// Shutdown: drop every job and session (each solver frees the RIDs it made),
// then free the device. Refused while a submit is in flight, so it never
// syncs in a submit's frame; tick to the verdict first.
static Variant rd_close() {
	const int64_t f = process_frame();
	const bool inflight = (g_job && g_job->pending()) || (!g_parked.empty() && f == g_parked_frame) ||
			(g_djob && g_djob->pending()) || (!g_djob_parked.empty() && f == g_djob_parked_frame) ||
			(g_sess && g_sess->pending()) || (!g_sess_parked.empty() && f == g_sess_parked_frame) || opt_pending() ||
			(!g_opt_parked.empty() && f == g_opt_parked_frame);
	if (inflight) {
		return text("BUSY: a submit is in flight; tick the job to its verdict and close on a later frame");
	}
	g_parked.clear();
	g_job.reset();
	g_djob_parked.clear();
	g_djob.reset();
	g_sess_parked.clear();
	g_sess.reset();
	g_q = jobs::StageQueue();
	g_opt_parked.clear();
	g_opt.reset();
	const int64_t slots = g_dev.permanent_slots();
	const bool was_open = g_dev.ok();
	g_dev.close();
	return text(std::string(slots == 0 ? "CLOSED" : "CLOSED with leaked slots") + " device=" +
			(was_open ? "freed" : "none") + " permanent_slots=" + std::to_string(slots));
}

int main() {
	ADD_API_FUNCTION(avbd_fixture, "String", "String backend", "The AVBD oracle on cpu (rd: the fixture job)");
	ADD_API_FUNCTION(avbd_bench, "String", "String backend, int nx, int ny, int substeps, int iters",
			"Drop a pinned nx-by-ny panel on cpu; the host times it (rd: the bench jobs)");
	ADD_API_FUNCTION(rd_close, "String", "",
			"Drop the job and free this stage's RenderingDevice and its permanent RID slots");
	ADD_API_FUNCTION(rd_last_step, "String", "", "The last RenderingDevice step attempted");
	ADD_API_FUNCTION(rd_rule4, "String", "", "Syncs that landed in their submit's process frame");
	ADD_API_FUNCTION(rd_rule4_probe, "String", "", "Submit and sync in one call; must raise rd_rule4");
	ADD_API_FUNCTION(avbd_job_start, "String", "String name, String backend",
			"Start a Stage 2 gate job on cpu or rd; then tick it once per frame");
	ADD_API_FUNCTION(avbd_job_tick, "String", "int host_us",
			"Advance the job one frame (host clock in us); RUNNING k/N, then PASS or FAIL");
	add_sandbox_api_function("avbd_job_names", avbd_job_names_api, "String", "", "The job names");
	ADD_API_FUNCTION(drape_open, "String", "String backend", "Pick the drape backend: cpu, rd or auto (rd from 160 vertices)");
	ADD_API_FUNCTION(drape_scene_sphere_demo, "String", "", "Load DiffCloth's rotating-sphere demo (25x25, sphere r 2)");
	ADD_API_FUNCTION(drape_scene_mesh, "String",
			"PackedFloat32Array positions, PackedInt32Array triangles, PackedInt32Array pins, PackedFloat32Array material",
			"Load a host mesh; material = [density, kTri, kBend, kAttach]");
	ADD_API_FUNCTION(drape_primitive, "String", "String kind, PackedFloat32Array params",
			"Add a sphere(c,r,mu) / plane(c,ul,ur,mu) / capsule(b,axis,r,len,mu), or clear");
	ADD_API_FUNCTION(drape_primitive_mesh, "String", "PackedFloat32Array positions, PackedInt32Array triangles, PackedFloat32Array params",
			"Add a triangle-mesh body collider; params = [skin, mu, band, depth]");
	ADD_API_FUNCTION(drape_config, "String", "String key, double value", "Set a DrapeConfig knob; returns the config line");
	ADD_API_FUNCTION(drape_queue_forward, "String", "int steps", "Queue steps (0 rewinds to the initial state)");
	ADD_API_FUNCTION(drape_fit_set, "String", "PackedInt32Array verts, PackedFloat32Array params, PackedInt32Array anchor",
			"The fit set: vertices pulled to the body collider's surface at kFit x vertex area; params = [kFit, gap, refresh, similarity, restEvery, settle, kAnchor]; anchor = the loop whose centre is held at its source position");
	ADD_API_FUNCTION(drape_sinew_align_test, "String", "",
			"AlignTest.lean's oracle through the vendored sinew_align: a 120-degree rotation recovered at N=5, 2, 1");
	ADD_API_FUNCTION(drape_queue_fit, "String", "int max_steps, double tol",
			"Queue the fit phase (gravity off, targets refreshed) until the largest move per step is under tol");
	ADD_API_FUNCTION(drape_set_target, "String", "String kind, PackedInt32Array verts, PackedFloat32Array positions, int frame",
			"Target: trajectory (the recorded frames) | points | clear");
	ADD_API_FUNCTION(drape_queue_backward, "String", "String loss, String mode",
			"Queue a backward: match_trajectory | target_points, native | step | unrolled");
	ADD_API_FUNCTION(drape_tick, "String", "int host_us", "Advance the drape queue one frame; RUNNING k/N or IDLE <result>");
	ADD_API_FUNCTION(drape_positions, "PackedFloat32Array", "", "The last completed step's positions (never syncs)");
	ADD_API_FUNCTION(drape_frame, "PackedFloat32Array", "int i", "Frame i: x0, then each step's predictor");
	ADD_API_FUNCTION(drape_faces, "PackedInt32Array", "", "The scene's triangles");
	ADD_API_FUNCTION(drape_result, "String", "", "The last forward/backward result, the config and the scene");
	ADD_API_FUNCTION(drape_queue_optimize, "String",
			"String spec, PackedFloat32Array x0, PackedFloat32Array lb, PackedFloat32Array ub, int max_iter",
			"Queue L-BFGS-B over session parameters (spec: params=mu loss= mode= steps= vec= m= delta= ...)");
	ADD_API_FUNCTION(drape_optimize_result, "String", "", "The optimize verdict and its iterates");
	ADD_API_FUNCTION(drape_job_data, "String", "String key, String text",
			"Hand the drape jobs a data file (Gate 5 oracle: comp_NN, prob_*, trace_*); key clear drops all");
	ADD_API_FUNCTION(drape_job_start, "String", "String name, String backend, String args",
			"Start a Gate 5 drape job on cpu, rd or auto; then tick it once per frame");
	ADD_API_FUNCTION(drape_job_tick, "String", "int host_us", "Advance the drape job one frame; RUNNING k/N, then PASS or FAIL");
	ADD_API_FUNCTION(drape_job_frame, "PackedFloat32Array", "int i", "Frame i of the current drape job's session");
	add_sandbox_api_function("drape_job_names", drape_job_names_api, "String", "", "The drape job names");
	halt();
}
