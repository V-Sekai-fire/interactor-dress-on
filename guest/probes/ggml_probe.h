#pragma once

#include <string>

// n x n x n f16 x f32 ggml_mul_mat + ggml_soft_max on one ggml-cpu thread;
// returns one line of checksums ("n=... sum_abs=... ...") or "FAIL ...".
std::string ggml_probe_run(int n);
