// ggml-rd op: ARGSORT, f32 rows to i32 indices (ggml_argsort, and
// ggml_argsort_top_k, which is the descending sort under a [k, ..] view).
//
// The kernel is lean/Ggml/SlangCodegen/Argsort.lean (argsort_f32): one
// thread per dst element counts the row elements that sort before its own
// (ties by index) and writes its index at that rank, so the row's ranks are
// a permutation and every dst element is written exactly once. op_params
// word 37 is the order (ASC 0, DESC 1), read by the kernel. O(ne0^2) reads
// per row over ne0 threads: rows longer than kMaxRow are refused, so a
// sort a graph never meant for this kernel fails loudly instead of
// stalling the device.
//
// supports(): src0 f32, dst i32 of src0's shape, one source, ne0 <=
// kMaxRow, every stride in the words. A NaN in a row is undefined on both
// sides (ggml-cpu's std::sort is not a strict weak order on it).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

constexpr int64_t kMaxRow = 1 << 16;

bool supports_argsort_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_I32 || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, op)) {
		return false;
	}
	const int32_t order = reinterpret_cast<const int32_t *>(op->op_params)[0];
	if (order != int32_t(GGML_SORT_ORDER_ASC) && order != int32_t(GGML_SORT_ORDER_DESC)) {
		return false;
	}
	if (a->ne[0] > kMaxRow) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_argsort_f32(Pack &p) {
	p.kernel = kernel_index("argsort_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(argsort_f32, GGML_OP_ARGSORT, -1, supports_argsort_f32, pack_argsort_f32);
