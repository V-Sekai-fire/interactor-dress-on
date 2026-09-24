// ggml-rd op: TIMESTEP_EMBEDDING, f32 (ggml_timestep_embedding: dim in
// op_params[0], max_period in op_params[1]).
//
// The kernel is lean/Ggml/SlangCodegen/TimestepEmbedding.lean
// (timestep_embedding_f32): one thread per dst element of the [dim, N]
// output, cos(t * freq(j)) for j < dim / 2, sin for the next dim / 2, 0 in
// an odd dim's last column, freq(j) = exp(-log(max_period) * j / (dim / 2))
// as ggml-cpu's ggml_compute_forward_timestep_embedding_f32. dst through
// its strides; the timesteps through theirs.
//
// supports(): dst f32 [dim, N, 1, 1] with dim = op_params[0] >= 1, src0 f32
// [N, 1, 1, 1] (ggml-cpu reads timesteps[i] for i < src0.ne0 and writes
// row i), max_period >= 1 (its log is taken), one source.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_timestep_embedding_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	const int32_t dim = op->op_params[0], max_period = op->op_params[1];
	if (dim < 1 || max_period < 1 || op->ne[0] != dim) {
		return false;
	}
	if (op->ne[1] != a->ne[0] || op->ne[2] != 1 || op->ne[3] != 1 || a->ne[1] != 1 || a->ne[2] != 1 || a->ne[3] != 1) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_timestep_embedding_f32(Pack &p) {
	p.kernel = kernel_index("timestep_embedding_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(timestep_embedding_f32, GGML_OP_TIMESTEP_EMBEDDING, -1, supports_timestep_embedding_f32,
		pack_timestep_embedding_f32);
