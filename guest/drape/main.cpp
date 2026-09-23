// drape.elf -- the drape stage: the AVBD solver on both backends.
//
// One ELF per stage (AGENTS.md rule 6): this one owns the solver, its oracle
// and its bench, with its own RenderingDevice held in a guest static.
// dress_on.elf keeps the Stage 1 GPU-layer probes. Every ADD_API_FUNCTION here
// has a no-argument wrapper in project/main.gd (rule 8).

#include <api.hpp>

#include <cstdio>
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
#include "jobs.h"

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

// Shutdown: drop the job (its solver frees every RID it made, which releases
// their permanent slots), then free the device. Refused while a submit is in
// flight, so it never syncs in a submit's frame; tick the job to its verdict
// first. The host calls this before freeing the Sandbox.
static Variant rd_close() {
	if ((g_job && g_job->pending()) || (!g_parked.empty() && process_frame() == g_parked_frame)) {
		return text("BUSY: a submit is in flight; tick the job to its verdict and close on a later frame");
	}
	g_parked.clear();
	g_job.reset();
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
	halt();
}
