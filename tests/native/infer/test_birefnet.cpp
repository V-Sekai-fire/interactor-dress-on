// G7.birefnet -- BiRefNet_HR-matting (vendor/birefnet) on ggml-vulkan, the
// deformable sampler as a ggml-cpu custom op, vs the CPU-torch fp32 oracle
// (gates/7-pixal3d/aux-models/birefnet_ref.py, its .npy outputs).
//
//   test_birefnet <gguf> <ref_bunny_dir> <ref_s14_dir> [threads]
//
//   BIREFNET_RELL2_TOL (1e-3)  alpha rel-L2 gate (f16 weights)
//   BIREFNET_CONTROL   0 skips the zero-offset negative control
//
// Checks:
//  [1] sampler op vs the documented torchvision semantics in float64, random
//      points plus exact integers, exactly -1, H, W and just inside/outside;
//  [2] one deformable conv through the graph code vs the oracle's two taps
//      (deform_sq_k7 @32^2, deform_d1_k3 @256^2) with the model's weights;
//  [3] Pixal3D preprocessing: PIL bilinear 1024^2 vs in_resized_u8, the
//      normalized tensor vs in_tensor, and (bunny) PIL bicubic mask back to
//      image size vs mask_fullres_u8;
//  [4] birefnet_alpha on both images: alpha at 1024 vs sigmoid(oracle
//      logits), rel-L2 <= tol;
//  [5] negative control: offsets forced to 0 must move alpha by > tol.
// Exit 77 = assets missing.
#include "birefnet.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Npy { std::string descr; std::vector<size_t> shape; std::vector<char> data; };

bool load_npy(const std::string & path, Npy & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint32_t hl = 0;
    if (magic[6] == 1) { uint16_t h16; f.read((char *) &h16, 2); hl = h16; }
    else f.read((char *) &hl, 4);
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    auto field = [&](const char * key) {
        size_t p = hdr.find(key);
        return p == std::string::npos ? std::string() : hdr.substr(p + std::strlen(key));
    };
    std::string d = field("'descr': '");
    out.descr = d.substr(0, d.find('\''));
    if (field("'fortran_order': ").rfind("False", 0) != 0) return false;
    std::string s = field("'shape': (");
    s = s.substr(0, s.find(')'));
    out.shape.clear();
    size_t n = 1;
    for (size_t i = 0; i < s.size();) {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',')) ++i;
        if (i >= s.size()) break;
        size_t v = std::strtoull(s.c_str() + i, nullptr, 10);
        out.shape.push_back(v);
        n *= v;
        while (i < s.size() && s[i] != ',') ++i;
    }
    size_t es = out.descr == "<f4" ? 4 : (out.descr == "|u1" || out.descr == "|b1") ? 1 : 0;
    if (!es) return false;
    out.data.resize(n * es);
    f.read(out.data.data(), (std::streamsize) out.data.size());
    return (bool) f;
}
std::vector<float> f32(const Npy & a) {
    std::vector<float> v(a.data.size() / 4);
    std::memcpy(v.data(), a.data.data(), a.data.size());
    return v;
}
std::vector<uint8_t> u8(const Npy & a) { return std::vector<uint8_t>(a.data.begin(), a.data.end()); }

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return std::sqrt(num / den);
}
double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs((double) a[i] - b[i]));
    return m;
}
std::vector<float> sigmoid(const std::vector<float> & x) {
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    return y;
}

int g_fail = 0;
void check(bool ok, const char * what) {
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

// [1] the sampler op vs torchvision's documented scalar semantics in float64
double sampler_f64_check(int n_threads) {
    const int C = 5, W = 9, H = 7;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uv(-1.f, 1.f), up(-2.5f, 10.5f);
    std::vector<float> X((size_t) C * W * H);   // token-major [C, W, H]
    for (auto & v : X) v = uv(rng);
    std::vector<float> P;
    for (int i = 0; i < 4000; ++i) { P.push_back(up(rng)); P.push_back(up(rng)); }
    const float sp[] = { -1.0f, -0.999999f, -1.000001f, 0.0f, 3.0f, (float) H - 1, (float) H, (float) H - 1e-6f,
                         (float) W - 1, (float) W, (float) W - 1e-6f, 0.5f, 2.25f };
    for (float a : sp) for (float b : sp) { P.push_back(a); P.push_back(b); }
    const int N = (int) P.size() / 2;

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tX = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, W, H);
    ggml_tensor * tP = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, N);
    ggml_tensor * tY = birefnet_bilinear_sample_zeros(ctx, tX, tP);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tY);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, n_threads);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(tX, X.data(), 0, X.size() * 4);
    ggml_backend_tensor_set(tP, P.data(), 0, P.size() * 4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> Y((size_t) C * N);
    ggml_backend_tensor_get(tY, Y.data(), 0, Y.size() * 4);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);

    double err = 0;
    int n_zero = 0;
    for (int n = 0; n < N; ++n) {
        const double y = P[2 * n], x = P[2 * n + 1];
        const bool oob = y <= -1 || y >= H || x <= -1 || x >= W;
        n_zero += oob;
        const int yl = (int) std::floor(y), xl = (int) std::floor(x), yh = yl + 1, xh = xl + 1;
        const double ly = y - yl, lx = x - xl, hy = 1 - ly, hx = 1 - lx;
        for (int c = 0; c < C; ++c) {
            auto at = [&](int yy, int xx) { return (double) X[(size_t) c + (size_t) C * ((size_t) yy * W + xx)]; };
            double r = 0;
            if (!oob) {
                const double v1 = (yl >= 0 && xl >= 0) ? at(yl, xl) : 0;
                const double v2 = (yl >= 0 && xh <= W - 1) ? at(yl, xh) : 0;
                const double v3 = (yh <= H - 1 && xl >= 0) ? at(yh, xl) : 0;
                const double v4 = (yh <= H - 1 && xh <= W - 1) ? at(yh, xh) : 0;
                r = hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
            }
            const double got = Y[(size_t) n * C + c];
            if (oob && got != 0.0) err = INFINITY;
            err = std::fmax(err, std::fabs(got - r));
        }
    }
    std::printf("[1] sampler op vs float64 torchvision semantics: %d points (%d out of bounds), max|err| = %.3e\n",
                N, n_zero, err);
    return err;
}

// regular_conv weight from the GGUF (ne [K*C, O], row kk*C + c) -> torch [O, C, k, k]
bool load_rc_weight(const std::string & gguf_path, const std::string & name, int C, int k, int O,
                    std::vector<float> & wt) {
    ggml_context * meta = nullptr;
    gguf_init_params gp; gp.no_alloc = false; gp.ctx = &meta;
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) return false;
    ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    const int K = k * k;
    bool ok = t && t->ne[0] == (int64_t) K * C && t->ne[1] == O;
    if (ok) {
        std::vector<float> w((size_t) K * C * O);
        if (t->type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, w.data(), (int64_t) w.size());
        else std::memcpy(w.data(), t->data, w.size() * 4);
        wt.resize(w.size());
        for (int o = 0; o < O; ++o)
            for (int c = 0; c < C; ++c)
                for (int kk = 0; kk < K; ++kk) wt[((size_t) o * C + c) * K + kk] = w[(size_t) o * K * C + (size_t) kk * C + c];
    }
    gguf_free(g);
    ggml_free(meta);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <gguf> <ref_bunny> <ref_s14> [threads]\n", argv[0]); return 2; }
    const std::string gguf_path = argv[1], rb = argv[2], rs = argv[3];
    const int nt = argc > 4 ? std::atoi(argv[4]) : 8;
    const double tol = std::getenv("BIREFNET_RELL2_TOL") ? std::atof(std::getenv("BIREFNET_RELL2_TOL")) : 1e-3;
    const bool control = !(std::getenv("BIREFNET_CONTROL") && std::atoi(std::getenv("BIREFNET_CONTROL")) == 0);
    {
        std::ifstream a(gguf_path), b(rb + "/logits.npy"), c(rs + "/logits.npy");
        if (!a || !b || !c) { std::printf("SKIP: assets missing (%s, %s, %s)\n", gguf_path.c_str(), rb.c_str(), rs.c_str()); return 77; }
    }

    // [1]
    check(sampler_f64_check(nt) <= 1e-6, "sampler op == float64 semantics (<= 1e-6)");

    // [2]
    std::printf("[2] deformable conv (graph code, ggml-cpu) vs oracle taps\n");
    struct Dc { const char * tap; const char * w; int k; };
    const Dc dcs[] = { { "deform_sq_k7", "squeeze_module.0.dec_att.aspp_deforms.2.atrous_conv.regular_conv.weight", 7 },
                       { "deform_d1_k3", "decoder.decoder_block1.dec_att.aspp_deforms.1.atrous_conv.regular_conv.weight", 3 } };
    for (const Dc & d : dcs) {
        Npy xi, of, mk, ou;
        const std::string b = rb + "/" + d.tap;
        if (!load_npy(b + "_in.npy", xi) || !load_npy(b + "_offset.npy", of) || !load_npy(b + "_mask.npy", mk) ||
            !load_npy(b + "_out.npy", ou)) { check(false, "deform oracle load"); continue; }
        const int C = (int) xi.shape[1], H = (int) xi.shape[2], W = (int) xi.shape[3], O = (int) ou.shape[1];
        std::vector<float> wt;
        if (!load_rc_weight(gguf_path, d.w, C, d.k, O, wt)) { check(false, "regular_conv weight"); continue; }
        std::vector<float> out((size_t) O * H * W), zo(of.data.size() / 4, 0.0f);
        std::string err;
        const auto X = f32(xi), OF = f32(of), MK = f32(mk), REF = f32(ou);
        const bool ok = birefnet_deform_conv(X.data(), OF.data(), MK.data(), wt.data(), C, H, W, O, d.k, out.data(), nt,
                                             (size_t) 256 << 20, &err);
        const double r = rel_l2(out, REF);
        std::vector<float> out0(out.size());
        birefnet_deform_conv(X.data(), zo.data(), MK.data(), wt.data(), C, H, W, O, d.k, out0.data(), nt, (size_t) 256 << 20, &err);
        const double r0 = rel_l2(out0, REF);
        std::printf("  %s (C %d, %dx%d, k %d -> %d): rel-L2 %.3e, max|err| %.3e; zero-offset control rel-L2 %.3e\n",
                    d.tap, C, W, H, d.k, O, r, max_abs(out, REF), r0);
        check(ok && r <= 1e-5, "deform conv == oracle tap (rel-L2 <= 1e-5)");
        check(r0 > 1e-3, "control: zero offsets differ from the tap (> 1e-3)");
    }

    // [3] preprocessing
    std::printf("[3] Pixal3D preprocessing\n");
    {
        Npy rgb, rz, it, mf, mu;
        if (load_npy(rb + "/in_rgb_u8.npy", rgb) && load_npy(rb + "/in_resized_u8.npy", rz) && load_npy(rb + "/in_tensor.npy", it)) {
            const int h = (int) rgb.shape[0], w = (int) rgb.shape[1];
            std::vector<uint8_t> r;
            birefnet_pil_resize((const uint8_t *) rgb.data.data(), w, h, 3, 1024, 1024, 1, r);
            size_t mism = 0;
            for (size_t i = 0; i < r.size(); ++i) mism += r[i] != (uint8_t) rz.data[i];
            std::printf("  bunny %dx%d -> PIL bilinear 1024^2: %zu / %zu bytes differ\n", w, h, mism, r.size());
            check(mism == 0, "PIL bilinear resize bit-exact");
        }
        if (load_npy(rb + "/mask_u8.npy", mu) && load_npy(rb + "/mask_fullres_u8.npy", mf)) {
            const int h = (int) mf.shape[0], w = (int) mf.shape[1];
            std::vector<uint8_t> r;
            birefnet_pil_resize((const uint8_t *) mu.data.data(), 1024, 1024, 1, w, h, 3, r);
            size_t mism = 0;
            for (size_t i = 0; i < r.size(); ++i) mism += r[i] != (uint8_t) mf.data[i];
            std::printf("  bunny mask 1024^2 -> PIL bicubic %dx%d: %zu / %zu bytes differ\n", w, h, mism, r.size());
            check(mism == 0, "PIL bicubic resize bit-exact");
        }
    }

    // [4] full model on ggml-vulkan
    std::string err;
    birefnet_load_params lp;
    lp.keep_f16 = true;
    birefnet_model * m = birefnet_load(gguf_path, lp, &err);
    if (!m) { std::printf("FAIL: load: %s\n", err.c_str()); return 1; }
    std::printf("[4] birefnet_alpha on ggml-vulkan (f16 weights), sampler on ggml-cpu x%d\n", nt);
    std::vector<float> alpha_bunny;
    for (const std::string & dir : { rb, rs }) {
        Npy rgb, lg;
        if (!load_npy(dir + "/in_rgb_u8.npy", rgb) || !load_npy(dir + "/logits.npy", lg)) { check(false, "oracle load"); continue; }
        const int h = (int) rgb.shape[0], w = (int) rgb.shape[1];
        const auto ref = sigmoid(f32(lg));
        for (int rep = 0; rep < (dir == rb ? 2 : 1); ++rep) {   // bunny twice: cold (pipeline compile) and warm
            birefnet_stats st;
            birefnet_run_params rp;
            rp.n_threads = nt;
            rp.stats = &st;
            std::vector<float> a;
            std::vector<uint8_t> a8;
            if (!birefnet_alpha(m, (const uint8_t *) rgb.data.data(), w, h, &a, &a8, rp, &err)) {
                std::printf("FAIL: forward: %s\n", err.c_str()); birefnet_free(m); return 1;
            }
            const double r = rel_l2(a, ref);
            size_t mism = 0;
            Npy mf;
            if (load_npy(dir + "/mask_fullres_u8.npy", mf))
                for (size_t i = 0; i < a8.size(); ++i) mism += a8[i] != (uint8_t) mf.data[i];
            std::printf("  %s (%dx%d) %s: alpha rel-L2 %.3e, max|err| %.3e; forward %.0f ms (compute %.0f ms, "
                        "CPU sampler %.0f ms / %d calls, %d splits, %d nodes)%s\n",
                        dir.c_str(), w, h, rep ? "warm" : "cold", r, max_abs(a, ref), st.total_ms, st.compute_ms,
                        st.sampler_ms, st.sampler_calls, st.n_splits, st.n_nodes,
                        mf.data.empty() ? "" : (std::string("; full-res u8 mask ") + std::to_string(mism) + " px differ").c_str());
            if (rep == 0) check(r <= tol, "alpha rel-L2 <= tol");
            if (dir == rb) alpha_bunny = a;
        }
    }

    // [5] negative control
    if (control) {
        Npy it, lg;
        if (load_npy(rb + "/in_tensor.npy", it) && load_npy(rb + "/logits.npy", lg)) {
            birefnet_run_params rp;
            rp.n_threads = nt;
            rp.zero_offsets = true;
            const auto x = f32(it);
            std::vector<float> l(1024 * 1024);
            if (birefnet_forward(m, x.data(), 1024, l.data(), rp, nullptr, &err)) {
                const double r = rel_l2(sigmoid(l), sigmoid(f32(lg)));
                std::printf("[5] zero-offset control: alpha rel-L2 %.3e vs oracle\n", r);
                check(r > tol, "control: zero offsets move alpha beyond tol");
            } else check(false, "control forward");
        }
    }
    birefnet_free(m);
    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
