// lbfgsb_jobs -- the in-guest L-BFGS-B's Gate 5 jobs (G1 lbfgsb_components,
// G2 lbfgsb_problems), started through drape_job_start like the other drape
// jobs and ticked one host frame at a time (AGENTS.md rule 4: a GPU phase
// ends the tick; its scalars are read on the next).
//
// The oracle data (gates/5-drape/oracle) does not live in the ELF: the host
// hands each file over with drape_job_data(key, text) before starting a job
// (keys comp_NN, prob_<problem>, trace_<problem>_m<M>_<tag>), and
// project/gate_lbfgsb.gd (or main.gd's lbfgsb_load_oracle) does that.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <map>
#include <memory>
#include <string>

#include "jobs.h"
#include "lbfgsb_vec.h"
#include "rd_compute.h"

// key -> text, filled by drape_job_data.
std::map<std::string, std::string> &lbfgsb_job_data();

// "cpu" or "rd" (nullptr with err if the rd kernels cannot be built).
std::unique_ptr<lbv::Vec> make_lbfgsb_vec(const std::string &backend, rdc::Device &dev, std::string &err);

// lbfgsb_components | lbfgsb_problems on cpu or rd. args: "only=<substring>"
// runs the traces (or fixtures) whose key contains it.
std::unique_ptr<jobs::Job> make_lbfgsb_job(const std::string &name, const std::string &backend,
		const std::string &args, rdc::Device &dev, std::string &err);
