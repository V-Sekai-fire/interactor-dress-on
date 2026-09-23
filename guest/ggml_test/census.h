// Census probes for ggml-rd's op families (Gate 3): the shapes the model
// census (skin-tokens, Pixal3D) actually dispatches, which test-backend-ops'
// small cases do not reach. Two probes, each a pump job:
//
//   census <OP|all>   each census row of OP: the op on ggml-rd and on the
//                     in-guest ggml-cpu (one thread) over the same random
//                     input, compared as test-backend-ops compares (NMSE,
//                     infinities must match in sign) and, for the unary
//                     ops, both against the formula in double precision.
//                     Prints "CENSUS ..." per row and RESULT.
//   perf <OP|all>     each census row of OP at its full census shape: a
//                     graph of R in-place applications (R - 1 barriers),
//                     buffer cleared, one warm-up, then T timed graphs, each
//                     graph_compute + synchronize followed by a COOP yield.
//                     The guest clock is not a clock (AGENTS.md), so the host
//                     times it: the interval from the WAIT_GPU yield of a
//                     submit to the end of the vmcall that syncs it
//                     (project/infer_host.gd, wait_us). Prints "PERF ..."
//                     per row and RESULT.
//
// Rows are family K1 (SILU, GELU, GELU_ERF, SIGMOID, NEG, SCALE,
// DIAG_MASK_INF) and K5 (ROPE NEOX), from the census's census_*.csv.
#pragma once

#include <string>

namespace census {

bool known(const std::string &probe); // "census" or "perf"
void run(const std::string &probe, const std::string &op);

} // namespace census
