// inverse_jobs -- Gate 5 G3: cloth-dynamics' inverse_min (4 vertices, 4
// chained steps) recovered by the in-guest L-BFGS-B, on AvbdCpu or AvbdRd,
// against LBFGSpp's trace on the same objective (see inverse_jobs.cpp).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <memory>
#include <string>

#include "jobs.h"
#include "rd_compute.h"

// backend cpu | rd; args "vec=cpu|rd cases=01".
std::unique_ptr<jobs::Job> make_inverse_job(const std::string &backend, const std::string &args, rdc::Device &dev,
		std::string &err);
