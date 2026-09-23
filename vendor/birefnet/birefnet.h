// BiRefNet (ZhengPeng7/BiRefNet, HR-matting weights) as ggml graph code.
//
// Swin-L backbone (run at full and half resolution, channel-concatenated),
// squeeze BasicDecBlk, and the ASPPDeformable decoder with dec_ipt_split
// patch inputs and gdt gating, at the 1024x1024 input Pixal3D uses.
// Census, op mapping and kernel specs: gates/7-pixal3d/aux-models/birefnet.md.
// Weights: tools/models/convert_birefnet_to_gguf.py.
//
// Layout: every activation is token-major, ggml ne [C, W*H] (ne0 = C,
// token t = y*W + x). Convolutions are a GET_ROWS im2col over host index
// tables (row H*W of the source is a zero row = padding) followed by
// MUL_MAT, banded over output rows so no column buffer exceeds a budget.
//
// Two ops are outside the RD backend's op set. Here, on the host, they are
// ggml custom ops (ggml_custom_4d / ggml_map_custom1) with exact C++ bodies;
// they are the HOST STAND-IN for Lean kernel K11 that Cut 3's ggml-rd
// backend must add (spec: birefnet.md section K11):
//   birefnet_bilinear_sample_zeros  (torchvision deform_conv2d sampler)
//   birefnet_relu                   (y = max(x, 0))
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct birefnet_model;

struct birefnet_load_params {
    // Keep >=2-D weights in the file's type (f16) instead of upcasting to f32
    // at load. With f16 weights ggml-cpu's MUL_MAT rounds the activations to
    // f16 too; the f32 upcast reproduces the fp32 torch reference.
    bool keep_f16 = false;
};

struct birefnet_run_params {
    int    n_threads   = 8;
    // Largest im2col / sampler column buffer per band (bytes). The 3x3 64->48
    // conv at 1024^2 alone is 2.4 GB unbanded.
    size_t band_bytes  = (size_t) 256 << 20;
    // Negative control: sample every deformable tap at its integer grid point
    // (offsets forced to 0, modulator kept).
    bool   zero_offsets = false;
    // Names of intermediates to read back (see birefnet.cpp tap names:
    // bb_full_x1..4, bb_half_x1..4, squeeze_in, squeeze_out,
    // dec_block4..1_in/_out, deform_sq_k7_{offset,mask,out},
    // deform_d1_k3_{offset,mask,out}). Values come back NCHW (C, H, W).
    std::vector<std::string> taps;
};

struct birefnet_tap {
    int C = 0, H = 0, W = 0;
    std::vector<float> data;  // NCHW
};

birefnet_model * birefnet_load(const std::string & path, const birefnet_load_params & lp,
                               std::string * error);
void birefnet_free(birefnet_model * m);

// x: the normalized input, NCHW (3, S, S), S = 1024 (the ipt grids are fixed
// at convert time for 1024). logits: S*S, row-major; alpha = sigmoid(logits).
bool birefnet_forward(birefnet_model * m, const float * x, int S, float * logits,
                      const birefnet_run_params & rp,
                      std::map<std::string, birefnet_tap> * taps, std::string * error);

// One modulated deformable conv (torchvision.ops.deform_conv2d, groups 1,
// stride 1, dilation 1, pad k/2, no bias) from given offset and mask, through
// the same graph code the model uses. x (C,H,W), offset (2K,H,W), mask
// (K,H,W), weight torch (Cout, C, k, k). out (Cout, H, W). All NCHW.
bool birefnet_deform_conv(const float * x, const float * offset, const float * mask,
                          const float * weight, int C, int H, int W, int Cout, int k,
                          float * out, int n_threads, size_t band_bytes, std::string * error);

// The two ops, for graph code and tests.
//   X ne [C, W, H] f32 contiguous (token-major image), P ne [2, N] f32
//   (P[0,n] = y, P[1,n] = x, pixel units)  ->  Y ne [C, N] f32.
ggml_tensor * birefnet_bilinear_sample_zeros(ggml_context * ctx, ggml_tensor * X, ggml_tensor * P);
ggml_tensor * birefnet_relu(ggml_context * ctx, ggml_tensor * x);
