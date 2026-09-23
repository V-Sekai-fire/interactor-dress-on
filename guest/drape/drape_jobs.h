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

// Rule 5: rd from 256 vertices (gates/2-avbd bench: rd 3.4 ms vs cpu 30 ms per
// substep at 16x16).
constexpr uint32_t kDrapeAutoRdVerts = 256;
std::string drape_pick_backend(const std::string &want, uint32_t nV);

// The drape jobs. `args` is "key=value ..." (job keys, else DrapeConfig keys).
std::unique_ptr<jobs::Job> make_drape_job(const std::string &name, const std::string &backend,
		const std::string &args, rdc::Device &dev, std::string &err);
const char *drape_job_names();
// The session a job ran on (frames for the host's comparisons), or nullptr.
DrapeSession *drape_job_session(jobs::Job *job);
