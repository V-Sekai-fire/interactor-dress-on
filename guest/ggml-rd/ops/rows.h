// ggml-rd row kernels (lean/Ggml/SlangCodegen/Rows.lean): one work group per
// row of src0 (dim 0), ne1 * ne2 * ne3 rows. The kernel reads the row count
// from derived word 55 and its linear group from words 54 (gx); the grid is
// grid_1d over rows * threadgroup threads, so it has exactly `rows` groups,
// split (gx, gy) at 65535. The Serial siblings (one thread per group) take
// the same words and the same grid.
#pragma once

#include <cstdlib>

#include "../rd_pack.h"

namespace ggml_rd {

// Rows.wRows.
constexpr uint32_t W_ROWS = W_DERIVED + 2;
static_assert(W_ROWS == 55, "Rows.lean: word 55 is the row count");

// NORM, RMS_NORM and SOFT_MAX have a 64-thread kernel (<name>_t64).
// GGML_RD_ROW_THREADS=64 or 256 forces one (the rows_perf probe's sweep and
// the host tests' arms); otherwise rows of at most kShortRow elements take
// the 64-thread kernel. kShortRow is measured: probe rows_perf's sweep.
constexpr int64_t kShortRow = 1024;

inline const char *pick_threads(const ggml_tensor *op, const char *k256, const char *k64) {
	if (const char *e = std::getenv("GGML_RD_ROW_THREADS")) {
		if (std::atoi(e) == 64) {
			return k64;
		}
		if (std::atoi(e) == 256) {
			return k256;
		}
	}
	return op->src[0]->ne[0] <= kShortRow ? k64 : k256;
}

inline bool pack_rows(Pack &p, const char *kernel) {
	p.kernel = kernel_index(kernel);
	if (p.kernel < 0) {
		return false;
	}
	const ggml_tensor *a = p.node->src[0];
	const uint64_t rows = uint64_t(a->ne[1]) * uint64_t(a->ne[2]) * uint64_t(a->ne[3]);
	const uint64_t tg = kernel_desc(p.kernel).threadgroup[0];
	if (rows == 0 || rows > 0xFFFFFFFFull / tg) {
		return false;
	}
	p.w[W_ROWS] = uint32_t(rows);
	return grid_1d(p, rows * tg);
}

// One f32 source, an f32 destination, no other source, every stride in the
// params words.
inline bool rows_f32_unary(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

} // namespace ggml_rd
