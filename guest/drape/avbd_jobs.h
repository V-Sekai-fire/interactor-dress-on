// The Stage 2 gate's jobs, templated over the backend (AvbdCpu, AvbdRd) and
// driven one process frame at a time through jobs::Job (guest/jobs.h), so no
// rd readback ever lands in the frame of its submit (AGENTS.md rule 4).
//
// Ports of cloth-dynamics' native tests (test_avbd_solver.cpp's backward
// smoke, test_avbd_gradcheck.cpp, test_avbd_stategrad.cpp) plus the checks
// only this port needs: cpu == rd on the backward, the padding negative
// control, the self-collision scan, the dual-snapshot check and the benches.
#pragma once

#include <memory>
#include <string>

#include "jobs.h"
#include "rd_compute.h"

// Build job `name` on backend "cpu" or "rd". two_vertex_bwd and
// two_vertex_bwd_nopad run both backends and ignore `backend`. Returns
// nullptr with `err` set for an unknown name or backend, or an rd solver
// that could not be built.
std::unique_ptr<jobs::Job> make_avbd_job(const std::string &name, const std::string &backend,
		rdc::Device &dev, std::string &err);

// The job names, space-separated.
const char *avbd_job_names();
