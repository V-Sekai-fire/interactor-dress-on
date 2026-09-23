// One-block oracle hook for Pixal3D proj attention (Stage 7, A1 item 4).
// Runs trellis2_ss_flow_forward with only the first --blocks blocks on fixed
// inputs written by gates/7-pixal3d/proj/proj_block_ref.py, and writes the
// [out_channels, R^3] output (f32, channel-major) for the torch comparison.
//
//   pixal3d-proj-block --flow SS.gguf --in inputs.bin --out out.bin
//                      --fov F --dist D [--blocks 1] [--proj-zero 0]
//                      [--device auto|cpu]
// inputs.bin = x [8*R^3] | t [1] | global [5*1024] | fmap [32*32*1024] (f32).
#include "trellis2.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string flow_p, in_p, out_p, device = "auto";
    float fov = 0.8575560450553894f, dist = 2.0f;
    int blocks = 1, proj_zero = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if      (a == "--flow")      flow_p = argv[i + 1];
        else if (a == "--in")        in_p   = argv[i + 1];
        else if (a == "--out")       out_p  = argv[i + 1];
        else if (a == "--fov")       fov    = (float) std::atof(argv[i + 1]);
        else if (a == "--dist")      dist   = (float) std::atof(argv[i + 1]);
        else if (a == "--blocks")    blocks = std::atoi(argv[i + 1]);
        else if (a == "--proj-zero") proj_zero = std::atoi(argv[i + 1]);
        else if (a == "--device")    device = argv[i + 1];
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (flow_p.empty() || in_p.empty() || out_p.empty()) { std::fprintf(stderr, "missing args\n"); return 2; }

    std::string e;
    auto t0 = std::chrono::steady_clock::now();
    trellis2_ss_flow_model * m = trellis2_ss_flow_load(flow_p, true, &e, device.c_str());
    if (!m) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const trellis2_ss_flow_hparams & hp = trellis2_ss_flow_hparams_of(m);
    const int R = hp.resolution, N = R * R * R, Cc = hp.cond_channels, side = 32;
    const size_t nx = (size_t) hp.in_channels * N, ng = (size_t) 5 * Cc, nf = (size_t) side * side * Cc;

    std::vector<float> buf(nx + 1 + ng + nf);
    FILE * f = std::fopen(in_p.c_str(), "rb");
    if (!f || std::fread(buf.data(), sizeof(float), buf.size(), f) != buf.size()) {
        std::fprintf(stderr, "cannot read %zu floats from %s\n", buf.size(), in_p.c_str()); return 1;
    }
    std::fclose(f);
    const float * x = buf.data();
    const float t = buf[nx];
    const float * glob = buf.data() + nx + 1;
    const float * fmap = glob + ng;

    trellis2_proj_camera cam;
    cam.fov = fov; cam.distance = dist; cam.mesh_scale = 1.0f; cam.image_size = 512;
    trellis2_proj_fmap lr; lr.data = fmap; lr.h = side; lr.w = side; lr.c = Cc;
    if (!trellis2_ss_flow_set_proj(m, cam, &lr, 1, &e)) { std::fprintf(stderr, "set_proj: %s\n", e.c_str()); return 1; }
    trellis2_ss_flow_debug(m, blocks, proj_zero != 0);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::vector<float> out((size_t) hp.out_channels * N);
    t0 = std::chrono::steady_clock::now();
    if (!trellis2_ss_flow_forward(m, x, t, glob, 5, Cc, out.data(), &e)) {
        std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1;
    }
    const double fwd_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    f = std::fopen(out_p.c_str(), "wb");
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    std::printf("proj-block: backend=\"%s\" blocks=%d proj_zero=%d load_ms=%.1f forward_ms=%.1f\n",
                trellis2_ss_flow_backend_name(m), blocks, proj_zero, load_ms, fwd_ms);
    trellis2_ss_flow_free(m);
    return 0;
}
