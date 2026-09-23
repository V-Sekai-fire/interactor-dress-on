// Stage 7a, host-native: image (with alpha) -> DINOv3 -> Pixal3D SS flow
// (proj attention, 12 Euler steps, CFG) -> SS decoder -> 64^3 occupancy ->
// marching cubes -> OBJ. Weights: the Aero-Ex Pixal3D-GGUF SS flow (arch
// 'flux', bf16) and the pixal3d-ggml conversions of DINOv3 and ss_dec.
// Camera: a fixed fov (--fov; default the pipeline's 0.8576 rad) until MoGe-3
// lands; distance from inference.py's distance_from_fov (extend_pixel 0).
//
//   pixal3d-7a --models DIR --image in.png --out mesh.obj [--fov RAD]
//              [--steps 12] [--seed 0] [--device auto|cpu]
// Prints one line per stage with its host wall time, then a summary line
// "7a: verts=V faces=F finite=1 occupied=K" and exits 0 iff faces > 0 and
// every vertex is finite.
#include "trellis2.h"
#include "marching_cubes.h"
#include "stb_image.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}
}

int main(int argc, char ** argv) {
    std::string models, image, out = "pixal3d_7a.obj", device = "auto";
    float fov = 0.8575560450553894f;
    int steps = 12;
    unsigned long long seed = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if      (a == "--models") models = argv[i + 1];
        else if (a == "--image")  image  = argv[i + 1];
        else if (a == "--out")    out    = argv[i + 1];
        else if (a == "--fov")    fov    = (float) std::atof(argv[i + 1]);
        else if (a == "--steps")  steps  = std::atoi(argv[i + 1]);
        else if (a == "--seed")   seed   = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--device") device = argv[i + 1];
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (models.empty() || image.empty()) {
        std::fprintf(stderr, "usage: pixal3d-7a --models DIR --image in.png [--out mesh.obj] [--fov RAD]\n");
        return 2;
    }
    const std::string conv = models + "/Pixal3D-GGUF-conv/";
    const std::string p_dino = conv + "dinov3_vitl16_f16.gguf";
    const std::string p_flow = models + "/Pixal3D-GGUF/Sparse/ss_flow_img_dit_1_3B_64_bf16.gguf";
    const std::string p_dec  = conv + "ss_dec_conv3d_16l8_f16.gguf";
    const char * dev = device.c_str();
    std::string e;

    auto t0 = clk::now();
    trellis2_dino_model * dino = trellis2_dino_load(p_dino, true, &e, dev);
    if (!dino) { std::fprintf(stderr, "dino load: %s\n", e.c_str()); return 1; }
    trellis2_ss_flow_model * flow = trellis2_ss_flow_load(p_flow, true, &e, dev);
    if (!flow) { std::fprintf(stderr, "ss_flow load: %s\n", e.c_str()); return 1; }
    trellis2_ss_dec_model * dec = trellis2_ss_dec_load(p_dec, true, &e, dev);
    if (!dec) { std::fprintf(stderr, "ss_dec load: %s\n", e.c_str()); return 1; }
    const trellis2_ss_flow_hparams & fhp = trellis2_ss_flow_hparams_of(flow);
    std::printf("stage load_ms=%.1f backend=\"%s\" proj=%d proj_in=%d res=%d blocks=%d\n",
                ms_since(t0), trellis2_ss_flow_backend_name(flow), fhp.image_attn_proj,
                fhp.proj_in_channels, fhp.resolution, fhp.num_blocks);
    std::fflush(stdout);

    // ── preprocess (alpha crop, black background, 512 px) ────────────────────
    t0 = clk::now();
    int w = 0, h = 0, comp = 0;
    unsigned char * px = stbi_load(image.c_str(), &w, &h, &comp, 4);
    if (!px) { std::fprintf(stderr, "cannot read %s\n", image.c_str()); return 1; }
    const int S = 512;
    std::vector<uint8_t> rgb;
    const bool pre_ok = trellis2_preprocess_rgba(px, w, h, S, rgb, &e);
    stbi_image_free(px);
    if (!pre_ok) { std::fprintf(stderr, "preprocess: %s\n", e.c_str()); return 1; }
    std::printf("stage preprocess_ms=%.1f in=%dx%d\n", ms_since(t0), w, h);

    // ── DINOv3 (affine-free LN'd last layer: 5 global + 32x32 patch tokens) ──
    t0 = clk::now();
    trellis2_dino_cond cond;
    if (!trellis2_dino_encode_rgb(dino, rgb.data(), S, cond, &e)) {
        std::fprintf(stderr, "dino: %s\n", e.c_str()); return 1;
    }
    const int C = (int) cond.channels(), T = (int) cond.tokens();
    const int side = S / 16;
    if (T != 5 + side * side) { std::fprintf(stderr, "dino tokens %d != 5+%d^2\n", T, side); return 1; }
    std::printf("stage dino_ms=%.1f tokens=%d channels=%d\n", ms_since(t0), T, C);

    // ── proj conditioning: lr patch map + camera ─────────────────────────────
    trellis2_proj_camera cam;
    cam.fov = fov;
    cam.mesh_scale = 1.0f;
    cam.image_size = S;
    cam.distance = trellis2_proj_distance(fov, cam.mesh_scale, S, 0);
    trellis2_proj_fmap lr;
    lr.data = cond.data.data() + (size_t) 5 * C;
    lr.h = side; lr.w = side; lr.c = C;
    if (!trellis2_ss_flow_set_proj(flow, cam, &lr, 1, &e)) {
        std::fprintf(stderr, "set_proj: %s\n", e.c_str()); return 1;
    }

    // ── SS flow: 12 steps, pipeline.json sparse_structure_sampler params ─────
    t0 = clk::now();
    trellis2_ss_sampler_params sp;
    sp.steps = steps;
    sp.guidance_strength = 7.5f;
    sp.guidance_rescale = 0.7f;
    sp.guidance_interval_min = 0.6f;
    sp.guidance_interval_max = 1.0f;
    sp.rescale_t = 5.0f;
    sp.sigma_min = 1e-5f;
    sp.seed = seed;
    sp.verbose = false;
    const int R = fhp.resolution;
    std::vector<float> latent((size_t) fhp.in_channels * R * R * R);
    // Global context = CLS + 4 registers (5 tokens); patches reach the DiT via proj.
    if (!trellis2_ss_flow_sample(flow, cond.data.data(), 5, C, &sp, nullptr, latent.data(), &e)) {
        std::fprintf(stderr, "ss_flow: %s\n", e.c_str()); return 1;
    }
    double lsum = 0; bool lfin = true;
    for (float v : latent) { lsum += (double) v * v; lfin = lfin && std::isfinite(v); }
    std::printf("stage ss_flow_ms=%.1f steps=%d latent_rms=%.4f finite=%d fov=%.4f d=%.4f\n",
                ms_since(t0), steps, std::sqrt(lsum / latent.size()), lfin ? 1 : 0, fov, cam.distance);

    // ── SS decoder -> 64^3 occupancy logits ──────────────────────────────────
    t0 = clk::now();
    const trellis2_ss_dec_hparams & dhp = trellis2_ss_dec_hparams_of(dec);
    const int Ro = dhp.res_out();
    std::vector<float> occ((size_t) dhp.out_channels * Ro * Ro * Ro);
    if (!trellis2_ss_dec_decode(dec, latent.data(), occ.data(), &e)) {
        std::fprintf(stderr, "ss_dec: %s\n", e.c_str()); return 1;
    }
    size_t occupied = 0;
    for (size_t i = 0; i < (size_t) Ro * Ro * Ro; ++i) occupied += occ[i] > 0.0f;
    std::printf("stage ss_dec_ms=%.1f res=%d occupied=%zu\n", ms_since(t0), Ro, occupied);

    // ── marching cubes on the occupancy (as trellis2_capi's coarse path) ─────
    t0 = clk::now();
    mc::Mesh mesh = mc::extract(occ.data(), Ro, Ro, Ro, 0.0f);
    const float inv = 1.0f / (float) Ro;
    bool finite = true;
    for (size_t i = 0; i < mesh.verts.size(); ++i) {
        mesh.verts[i] = mesh.verts[i] * inv - 0.5f;
        finite = finite && std::isfinite(mesh.verts[i]);
    }
    const size_t nv = mesh.verts.size() / 3, nf = mesh.tris.size() / 3;
    std::printf("stage mc_ms=%.1f\n", ms_since(t0));

    if (FILE * f = std::fopen(out.c_str(), "wb")) {
        for (size_t i = 0; i < nv; ++i)
            std::fprintf(f, "v %.6f %.6f %.6f\n", mesh.verts[3 * i], mesh.verts[3 * i + 1], mesh.verts[3 * i + 2]);
        for (size_t i = 0; i < nf; ++i)
            std::fprintf(f, "f %d %d %d\n", (int) mesh.tris[3 * i] + 1, (int) mesh.tris[3 * i + 1] + 1,
                         (int) mesh.tris[3 * i + 2] + 1);
        std::fclose(f);
    }
    std::printf("7a: verts=%zu faces=%zu finite=%d occupied=%zu out=%s\n", nv, nf, finite ? 1 : 0,
                occupied, out.c_str());

    trellis2_ss_dec_free(dec);
    trellis2_ss_flow_free(flow);
    trellis2_dino_free(dino);
    return (nf > 0 && finite) ? 0 : 1;
}
