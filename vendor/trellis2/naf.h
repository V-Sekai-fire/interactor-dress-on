// NAF (Neighborhood Attention Filtering, valeoai/NAF) forward as ggml graph
// code: Pixal3D's DINOv3 feature upsampler. Local to this vendored tree (not
// in pixal3d-ggml @1c22f5e); see CITATION.cff and
// gates/7-pixal3d/aux-models/naf.md for the op census this follows.
//
// Three stages, each a handful of small ggml graphs on one backend:
//   encode   guide image [3,S,S] in [0,1] -> the two conv branches, concat,
//            kept channel-major [S*S, 256]. 86-96% of the FLOPs. The encoder
//            of S = 1024 serves both shape_1024 (T 512) and tex_1024 (T 1024).
//   prepare  pool to T, block-major permutation, 2-D axial RoPE (two NEOX
//            ROPE passes, base 1, host freq_factors), KeyEncoder pool to hk;
//            upload the DINOv3 patch tokens as V.
//   attend   natten na2d in its exact block form, for any range of low-res
//            blocks: GET_ROWS window gather (host table) -> MUL_MAT ->
//            SOFT_MAX_EXT(scale 1/8) -> MUL_MAT. Output is block-major:
//            slot ((bi*hk + bj)*d + a)*d + b is pixel (bi*d + a, bj*d + b),
//            1024 floats each. tex_1024's full output is 4 GiB, so callers
//            evaluate block ranges (or only the blocks their taps touch).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct gguf_context;
struct ggml_gallocr;

namespace naf {

struct hparams {
    int dim = 256;          // query/key channels: 128 (1x1 branch) || 128 (3x3 branch)
    int enc_dim = 128;
    int heads = 4;          // Q/K head width 64, V head width C/4
    int kernel = 9;         // 9x9 low-res window
    int groups = 8;         // GroupNorm groups
    float eps = 1e-5f;
    float rope_base = 100.f;
    int enc_blocks = 2;
};

struct model {
    hparams hp;
    ggml_backend * backend = nullptr;
    ggml_context * ctx = nullptr;          // weights (metadata)
    gguf_context * gguf = nullptr;
    ggml_backend_buffer * buf = nullptr;   // weights (data)
    ggml_gallocr * alloc = nullptr;        // shared by every step graph
    std::vector<float> periods;            // image_encoder.rope.periods [16]
    int n_threads = 1;
    ggml_tensor * get(const std::string & name) const;
};

// Guide-image encoder output, persistent between prepare() calls.
struct encoded {
    int S = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer * buf = nullptr;
    ggml_tensor * enc = nullptr;           // [S*S, 256] channel-major (NCHW memory)
};

// One NAF evaluation at output size T on an hk x hk DINOv3 grid.
struct run {
    int S = 0, T = 0, hk = 0, d = 0, r = 0, C = 1024;
    std::vector<int32_t> window;           // [hk*hk*81] low-res window of each block (test hook)
    ggml_context * ctx = nullptr;
    ggml_backend_buffer * buf = nullptr;
    ggml_tensor * qb = nullptr;            // [64, d*d, 4, hk*hk] RoPE'd queries, per block
    ggml_tensor * k_lr = nullptr;          // [256, hk*hk] keys (block means of the queries)
    ggml_tensor * v_lr = nullptr;          // [C, hk*hk] DINOv3 patch tokens
    int blocks() const { return hk * hk; }
    int slots_per_block() const { return d * d; }
};

// Loads the GGUF written by tools/models/convert_naf_to_gguf.py onto the
// ggml-cpu backend with n_threads threads.
model * load(const std::string & path, int n_threads, std::string * err);
void free_model(model * m);

// img: [3, S, S] float in [0,1] (NCHW, not ImageNet-normalised). S % 16 == 0.
encoded * encode(model * m, const float * img, int S, std::string * err);
void free_encoded(encoded * e);

// tokens: DINOv3 patch tokens [hk*hk][C] row-major (Pixal3D's z[0, 5:], after
// the non-affine layer_norm), i.e. ggml [C, hk*hk]. T must be a multiple of hk
// and divide S; S/T is the encoder pool factor, T/hk the dilation d.
run * prepare(model * m, const encoded * e, int T, const float * tokens, int hk, int C,
              std::string * err);
void free_run(run * r);

// out: [(blk1 - blk0) * d*d][C] floats, block-major slots of blocks
// [blk0, blk1) (block index bi*hk + bj). max_blocks bounds one graph.
bool attend(model * m, run * r, int blk0, int blk1, float * out, std::string * err,
            int max_blocks = 256);

// Read back the keys [hk*hk][256] (ggml [256, hk*hk]).
bool read_keys(const run * r, float * out);

// Host tables (int32, as the graphs consume them).
std::vector<int32_t> window_table(int hk, int ks = 9);   // [hk*hk*ks*ks]
std::vector<int32_t> block_perm(int T, int d);           // slot -> raster pixel

}  // namespace naf

// ── C API (what the Stage 7 host GDExtension calls) ────────────────────────
// One handle = one model on ggml-vulkan (IDO_GGML_BACKEND=cpu forces ggml-cpu).
// Output is raster, channels-last: pixel (y, x) of the out_res x out_res map is
// out[((y - y0) * out_res + x) * C + c]. A full 1024^2 x 1024 map is 4 GiB, so
// naf_rows evaluates a band of rows; naf_upsample is the one-shot form.
extern "C" {
typedef struct naf_handle naf_handle;
naf_handle * naf_open(const char * gguf_path, int n_threads);   // NULL: see naf_last_error
void         naf_close(naf_handle * h);
const char * naf_last_error(void);
const char * naf_backend_name(const naf_handle * h);
// image: [3, S, S] float in [0,1] (NCHW); S % 16 == 0. Runs the encoder.
int naf_set_image(naf_handle * h, const float * image, int S);
// dino_lr: DINOv3 patch tokens [hk*hk][C] row-major (after the layer norm);
// out_res divides S, hk divides out_res. Runs prepare.
int naf_set_tokens(naf_handle * h, const float * dino_lr, int hk, int C, int out_res);
// rows [y0, y1) of the output into out ([(y1-y0) * out_res * C] floats).
int naf_rows(naf_handle * h, int y0, int y1, float * out);
// set_image + set_tokens + rows [0, out_res).
int naf_upsample(naf_handle * h, const float * dino_lr, int hk, int C, const float * image, int S,
                 int out_res, float * out);
}
