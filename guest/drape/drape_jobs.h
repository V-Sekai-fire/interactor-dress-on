// drape_jobs -- the drape session behind the drape_* API, and the Gate 5
// drape jobs (sphere_forward, sphere_backward, sim_gradcheck, bench_drape).
//
// A session is one backend's solver, a DrapeSimT over it, a target and the
// last result. Every operation is queued as stages on a jobs::StageQueue and
// advanced one host frame at a time: a stage that submits GPU work ends the
// tick, and the stage that reads it back runs on a later one (AGENTS.md rule
// 4). The API owns one session and one queue; each job owns its own session
// and queues onto its own jobs::Job queue.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "drape_session.h"
#include "jobs.h"
#include "rd_compute.h"


// "cpu" or "rd"; nullptr with err if the rd solver cannot be built.
std::unique_ptr<DrapeSession> make_drape_session(const std::string &backend, rdc::Device &dev, std::string &err);

// Rule 5: rd from 160 vertices. Gate 5 G9 (gates/5-drape/README.md) timed the
// full drape step (solve, contact, self-collision) frame-driven at 90 fps, the
// OpenXR display rate: cpu 20.0 ms/step at 144 vertices and 27.2 at 196, rd
// 22.2 (two frames per step) at every size up to 1024; the crossover
// interpolates to 160. rd's cost is frames, so uncapped (about 3 ms a frame)
// it crosses between 36 and 64 vertices instead.
constexpr uint32_t kDrapeAutoRdVerts = 160;
std::string drape_pick_backend(const std::string &want, uint32_t nV);

// The drape jobs. `args` is "key=value ..." (job keys, else DrapeConfig keys).
std::unique_ptr<jobs::Job> make_drape_job(const std::string &name, const std::string &backend,
		const std::string &args, rdc::Device &dev, std::string &err);
const char *drape_job_names();
// The session a job ran on (frames for the host's comparisons), or nullptr.
DrapeSession *drape_job_session(jobs::Job *job);
