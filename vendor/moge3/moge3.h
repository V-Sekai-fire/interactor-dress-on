// moge3.h -- MoGe-3 (Ruicheng/moge-3-vitl) camera field of view on ggml.
//
// The subgraph Pixal3D's get_camera_params_wild_moge needs and nothing more
// (gates/7-pixal3d/aux-models/moge3.md): DINOv2 ViT-L/14 at 840x840 (60x60
// tokens, pos-embed bicubic-interpolated at conversion), the neck, the points
// and mask heads (refine_steps = 0), then on the host a 64x64 bilinear sample
// of the head maps, MoGe's focal/shift least-squares solve and
// camera_angle_x = 2 atan(1 / (2 fx)).
//
// Weights: tools/models/convert_moge3_to_gguf.py. Input: the square S x S RGB
// image Pixal3D's preprocess_image produces (uint8, HWC). Output layouts of
// the taps are torch's (C, H, W), which is ggml's [W, H, C].
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct moge3_model;

// Load the GGUF. n_threads <= 0: MOGE3_N_THREADS, else 8.
moge3_model * moge3_load(const std::string & gguf_path, int n_threads, std::string * error);
void          moge3_free(moge3_model * m);

struct moge3_fov_result {
    float  focal        = 0;  // relative to the half diagonal (MoGe recover_focal_shift)
    float  shift        = 0;  // affine depth shift
    float  fx           = 0;  // normalised intrinsics (fx = fy for a square image)
    float  fy           = 0;
    double camera_angle_x = 0;  // radians; Pixal3D distance_from_fov default = fx
    int    n_valid      = 0;  // mask-valid samples of the 64x64 grid
    double ms_encoder = 0, ms_neck = 0, ms_heads = 0, ms_host = 0;
};

// Intermediates, filled when a pointer is passed.
struct moge3_taps {
    std::vector<float>   image14;       // [3, 840, 840] normalised ViT input
    std::vector<float>   features;      // [1024, 60, 60] encoder output (sum of 4 projections)
    std::vector<float>   neck[5];       // [C_l, 60*2^l, 60*2^l]
    std::vector<float>   points_raw;    // [3, 960, 960]  (x/z, y/z, log z)
    std::vector<float>   mask_raw;      // [1, 960, 960]  logits
    std::vector<float>   fov_points;    // [64, 64, 3] affine points at the samples
    std::vector<float>   fov_uv;        // [64, 64, 2]
    std::vector<uint8_t> fov_mask;      // [64, 64]
};

// rgb: S x S x 3 uint8 (row-major HWC). Only square inputs (Pixal3D's case).
bool moge3_fov(moge3_model * m, const uint8_t * rgb, int width, int height,
               moge3_fov_result & out, moge3_taps * taps, std::string * error);
