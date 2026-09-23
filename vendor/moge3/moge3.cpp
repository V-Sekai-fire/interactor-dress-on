// moge3.cpp -- MoGe-3 camera FoV on ggml (see moge3.h).
//
// Port of V-Sekai-fire/MoGe @74fbce0 moge/model/{v3,v2}.py,
// modules/{dinov2_encoder,conv_stack}.py and utils/geometry_*.py
// (recover_focal_shift), restricted to the path Pixal3D uses.
//
// Op set: everything is the Cut 3 kernel families (K1-K8) except ReLU, which
// is a real GGML_UNARY_OP_RELU here (K11 in the plan gives ggml-rd its relu
// kernel). Non-ggml ops are decomposed:
//   * Conv2d 1x1, 3x3 and the 14x14/s14 patch embed: CONV_3D direct (KD = 1)
//     on [W, H, 1, C]; replicate padding = 4 CONCATs of 1-wide edge views.
//   * ConvTranspose2d k2 s2: a 1x1 conv to 4*Cout channels (rows (a, b, co),
//     baked by the converter), a CONT(PERMUTE) per output row parity and a
//     CONCAT that interleaves them.
//   * Upsample x2 bilinear (align_corners = False): per axis, clamped
//     neighbour views by CONCAT, 0.25/0.75 SCALE + ADD, CONCAT on a unit
//     inner dim to interleave. W pass first, then H (torch's order).
//   * The antialiased bilinear resize to 840 and (x - mean)/std run on the
//     host (float32 tables, as torch builds them); the 64x64 sample of the
//     head maps, exp/sigmoid and the 1-D least-squares solve are host work on
//     4096 values.
#include "moge3.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

namespace {

constexpr int kGrid  = 60;          // tokens per side (3600 tokens, square input)
constexpr int kPatch = 14;
constexpr int kImg   = kGrid * kPatch;  // 840
constexpr int kDim   = 1024;
constexpr int kHeads = 16;
constexpr int kHd    = 64;
constexpr int kLayers = 24;
constexpr int kTaps[4] = {5, 11, 17, 23};
constexpr int kLevelC[5] = {1024, 256, 128, 64, 32};
constexpr int kNeckRes[5] = {0, 2, 2, 2, 0};
constexpr int kHeadRes[5] = {0, 1, 1, 1, 0};
constexpr int kFovSamples = 64;

void set_error(std::string * e, const std::string & s) { if (e) *e = s; }

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

struct moge3_model {
    ggml_context *        ctx = nullptr;
    gguf_context *        gguf = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> t;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
    float ln_eps = 1e-6f;

    ggml_tensor * W(const std::string & name) const {
        auto it = t.find(name);
        if (it == t.end()) {
            std::fprintf(stderr, "moge3: missing tensor %s\n", name.c_str());
            std::abort();
        }
        return it->second;
    }
};

moge3_model * moge3_load(const std::string & path, int n_threads, std::string * error) {
    auto * m = new moge3_model();
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = &m->ctx;
    m->gguf = gguf_init_from_file(path.c_str(), gp);
    if (!m->gguf) { set_error(error, "gguf_init_from_file failed: " + path); delete m; return nullptr; }
    const int64_t arch = gguf_find_key(m->gguf, "general.architecture");
    if (arch < 0 || std::strcmp(gguf_get_val_str(m->gguf, arch), "moge3") != 0) {
        set_error(error, "not a moge3 GGUF: " + path);
        moge3_free(m);
        return nullptr;
    }
    const int64_t kg = gguf_find_key(m->gguf, "moge3.grid");
    if (kg < 0 || (int) gguf_get_val_u32(m->gguf, kg) != kGrid) {
        set_error(error, "moge3.grid must be 60 (pos-embed baked for 60x60 tokens)");
        moge3_free(m);
        return nullptr;
    }
    auto arr3 = [&](const char * key, float * dst) {
        const int64_t id = gguf_find_key(m->gguf, key);
        if (id >= 0 && gguf_get_arr_n(m->gguf, id) == 3) {
            const float * p = (const float *) gguf_get_arr_data(m->gguf, id);
            for (int i = 0; i < 3; ++i) dst[i] = p[i];
        }
    };
    arr3("moge3.image_mean", m->mean);
    arr3("moge3.image_std", m->stdv);
    const int64_t ke = gguf_find_key(m->gguf, "moge3.ln_eps");
    if (ke >= 0) m->ln_eps = gguf_get_val_f32(m->gguf, ke);

    for (ggml_tensor * x = ggml_get_first_tensor(m->ctx); x; x = ggml_get_next_tensor(m->ctx, x))
        m->t[x->name] = x;

    m->backend = ggml_backend_cpu_init();
    if (n_threads <= 0) {
        n_threads = 8;
        if (const char * e = std::getenv("MOGE3_N_THREADS")) { const int v = std::atoi(e); if (v > 0) n_threads = v; }
    }
    ggml_backend_cpu_set_n_threads(m->backend, n_threads);
    m->buf = ggml_backend_alloc_ctx_tensors(m->ctx, m->backend);
    if (!m->buf) { set_error(error, "weight allocation failed"); moge3_free(m); return nullptr; }

    std::ifstream fin(path, std::ios::binary);
    const size_t off = gguf_get_data_offset(m->gguf);
    std::vector<char> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(m->gguf); ++i) {
        ggml_tensor * x = m->t[gguf_get_tensor_name(m->gguf, i)];
        const size_t n = ggml_nbytes(x);
        tmp.resize(n);
        fin.seekg((std::streamoff) (off + gguf_get_tensor_offset(m->gguf, i)));
        fin.read(tmp.data(), (std::streamsize) n);
        if (!fin) { set_error(error, "short read: " + path); moge3_free(m); return nullptr; }
        ggml_backend_tensor_set(x, tmp.data(), 0, n);
    }
    return m;
}

void moge3_free(moge3_model * m) {
    if (!m) return;
    if (m->buf) ggml_backend_buffer_free(m->buf);
    if (m->backend) ggml_backend_free(m->backend);
    if (m->ctx) ggml_free(m->ctx);
    if (m->gguf) gguf_free(m->gguf);
    delete m;
}

namespace {

// ── graph helpers (conv layout: [W, H, C] f32, torch (C, H, W) memory) ─────

struct G {
    ggml_context * ctx;
    const moge3_model * m;

    ggml_tensor * bias3(ggml_tensor * x, const std::string & b) {
        ggml_tensor * bb = m->W(b);
        return ggml_add(ctx, x, ggml_reshape_3d(ctx, bb, 1, 1, bb->ne[0]));
    }

    // Conv with a torch [Cout, Cin, k, k] kernel (ggml [k, k, Cin, Cout]) on
    // an unpadded [W, H, Cin] input; stride s.
    ggml_tensor * conv(ggml_tensor * x, ggml_tensor * w, int s) {
        const int k = (int) w->ne[0];
        const int cin = (int) x->ne[2];
        const int cout = (int) (ggml_nelements(w) / ((int64_t) k * k * cin));
        ggml_tensor * x4 = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], 1, cin);
        ggml_tensor * k4 = ggml_reshape_4d(ctx, w, k, k, 1, (int64_t) cin * cout);
        ggml_tensor * y = ggml_conv_3d_direct(ctx, k4, x4, s, s, 1, 0, 0, 0, 1, 1, 1, cin, 1, cout);
        return ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], cout);
    }
    ggml_tensor * conv1x1(ggml_tensor * x, const std::string & p) {
        return bias3(conv(x, m->W(p + ".weight"), 1), p + ".bias");
    }
    // padding_mode = 'replicate', pad 1
    ggml_tensor * pad_rep(ggml_tensor * x) {
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2];
        const size_t es = ggml_element_size(x);
        ggml_tensor * l = ggml_view_3d(ctx, x, 1, h, c, x->nb[1], x->nb[2], 0);
        ggml_tensor * r = ggml_view_3d(ctx, x, 1, h, c, x->nb[1], x->nb[2], (w - 1) * es);
        ggml_tensor * y = ggml_concat(ctx, ggml_concat(ctx, l, x, 0), r, 0);
        ggml_tensor * t = ggml_view_3d(ctx, y, w + 2, 1, c, y->nb[1], y->nb[2], 0);
        ggml_tensor * b = ggml_view_3d(ctx, y, w + 2, 1, c, y->nb[1], y->nb[2], (h - 1) * y->nb[1]);
        return ggml_concat(ctx, ggml_concat(ctx, t, y, 1), b, 1);
    }
    ggml_tensor * conv3x3(ggml_tensor * x, const std::string & p) {
        return bias3(conv(pad_rep(x), m->W(p + ".weight"), 1), p + ".bias");
    }
    // ResidualConvBlock, no norms: x + conv(relu(conv(relu(x))))
    ggml_tensor * resblock(ggml_tensor * x, const std::string & p) {
        ggml_tensor * h = conv3x3(ggml_relu(ctx, x), p + ".layers.2");
        h = conv3x3(ggml_relu(ctx, h), p + ".layers.5");
        return ggml_add(ctx, x, h);
    }
    // ConvTranspose2d k2 s2 from the converter's weight_abco [4*Cout rows (a, b, co), Cin]
    ggml_tensor * convT(ggml_tensor * x, const std::string & p) {
        ggml_tensor * wm = m->W(p + ".weight_abco");
        const int64_t w = x->ne[0], h = x->ne[1], cin = x->ne[2];
        const int64_t cout = wm->ne[1] / 4;
        ggml_tensor * y = conv(x, ggml_reshape_4d(ctx, wm, 1, 1, cin, 4 * cout), 1);  // [w, h, 4*cout]
        ggml_tensor * rows[2];
        for (int a = 0; a < 2; ++a) {
            // dims (j, i, b, co) -> (b, j, i, co): output column 2j+b of row 2i+a
            ggml_tensor * v = ggml_view_4d(ctx, y, w, h, 2, cout, y->nb[1], cout * y->nb[2], y->nb[2],
                                           (size_t) a * 2 * cout * y->nb[2]);
            ggml_tensor * pr = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [2, w, h, cout]
            rows[a] = ggml_reshape_4d(ctx, pr, 2 * w, 1, h, cout);
        }
        ggml_tensor * o = ggml_concat(ctx, rows[0], rows[1], 1);                  // [2w, 2, h, cout]
        o = ggml_reshape_3d(ctx, o, 2 * w, 2 * h, cout);
        return bias3(o, p + ".bias");
    }
    // nn.Upsample(scale 2, bilinear, align_corners = False): W pass, then H pass
    ggml_tensor * up2(ggml_tensor * x) {
        const size_t es = ggml_element_size(x);
        {   // W
            const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2];
            ggml_tensor * c0 = ggml_view_3d(ctx, x, 1, h, c, x->nb[1], x->nb[2], 0);
            ggml_tensor * cl = ggml_view_3d(ctx, x, 1, h, c, x->nb[1], x->nb[2], (w - 1) * es);
            ggml_tensor * lo = ggml_view_3d(ctx, x, w - 1, h, c, x->nb[1], x->nb[2], 0);
            ggml_tensor * hi = ggml_view_3d(ctx, x, w - 1, h, c, x->nb[1], x->nb[2], es);
            ggml_tensor * left  = ggml_concat(ctx, c0, lo, 0);
            ggml_tensor * right = ggml_concat(ctx, hi, cl, 0);
            ggml_tensor * even = ggml_add(ctx, ggml_scale(ctx, left, 0.25f), ggml_scale(ctx, x, 0.75f));
            ggml_tensor * odd  = ggml_add(ctx, ggml_scale(ctx, x, 0.75f), ggml_scale(ctx, right, 0.25f));
            ggml_tensor * it = ggml_concat(ctx, ggml_reshape_4d(ctx, even, 1, w, h, c),
                                                ggml_reshape_4d(ctx, odd, 1, w, h, c), 0);
            x = ggml_reshape_3d(ctx, it, 2 * w, h, c);
        }
        {   // H
            const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2];
            ggml_tensor * r0 = ggml_view_3d(ctx, x, w, 1, c, x->nb[1], x->nb[2], 0);
            ggml_tensor * rl = ggml_view_3d(ctx, x, w, 1, c, x->nb[1], x->nb[2], (h - 1) * x->nb[1]);
            ggml_tensor * lo = ggml_view_3d(ctx, x, w, h - 1, c, x->nb[1], x->nb[2], 0);
            ggml_tensor * hi = ggml_view_3d(ctx, x, w, h - 1, c, x->nb[1], x->nb[2], x->nb[1]);
            ggml_tensor * up   = ggml_concat(ctx, r0, lo, 1);
            ggml_tensor * down = ggml_concat(ctx, hi, rl, 1);
            ggml_tensor * even = ggml_add(ctx, ggml_scale(ctx, up, 0.25f), ggml_scale(ctx, x, 0.75f));
            ggml_tensor * odd  = ggml_add(ctx, ggml_scale(ctx, x, 0.75f), ggml_scale(ctx, down, 0.25f));
            ggml_tensor * it = ggml_concat(ctx, ggml_reshape_4d(ctx, even, w, 1, h, c),
                                                ggml_reshape_4d(ctx, odd, w, 1, h, c), 1);
            x = ggml_reshape_3d(ctx, it, w, 2 * h, c);
        }
        return x;
    }
    ggml_tensor * resampler(ggml_tensor * x, const std::string & p, int i) {
        if (i < 3) x = convT(x, p + ".resamplers." + std::to_string(i) + ".0");
        else       x = up2(x);
        return conv3x3(x, p + ".resamplers." + std::to_string(i) + ".1");
    }

    // ── token layout helpers ([dim, N]) ──
    ggml_tensor * ln(ggml_tensor * x, const std::string & p) {
        x = ggml_norm(ctx, x, m->ln_eps);
        return ggml_add(ctx, ggml_mul(ctx, x, m->W(p + ".weight")), m->W(p + ".bias"));
    }
    ggml_tensor * linear(ggml_tensor * x, const std::string & p) {
        ggml_tensor * w = m->W(p + ".weight");
        if (ggml_n_dims(w) > 2) w = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]);
        return ggml_add(ctx, ggml_mul_mat(ctx, w, x), m->W(p + ".bias"));
    }
    ggml_tensor * block(ggml_tensor * x, int i) {
        const std::string p = "encoder.backbone.blocks." + std::to_string(i);
        const int64_t n = x->ne[1];
        ggml_tensor * qkv = linear(ln(x, p + ".norm1"), p + ".attn.qkv");  // [3072, N]
        const size_t es = ggml_element_size(qkv);
        auto head = [&](int k) {
            ggml_tensor * v = ggml_view_3d(ctx, qkv, kHd, kHeads, n, kHd * es, qkv->nb[1], (size_t) k * kDim * es);
            return ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));      // [hd, N, H]
        };
        ggml_tensor * o = ggml_flash_attn_ext(ctx, head(0), head(1), head(2), nullptr,
                                              1.0f / std::sqrt((float) kHd), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        o = ggml_reshape_2d(ctx, o, kDim, n);                              // [hd*H, N]
        o = linear(o, p + ".attn.proj");
        x = ggml_add(ctx, x, ggml_mul(ctx, o, m->W(p + ".ls1.gamma")));
        ggml_tensor * f = linear(ln(x, p + ".norm2"), p + ".mlp.fc1");
        f = linear(ggml_gelu_erf(ctx, f), p + ".mlp.fc2");
        return ggml_add(ctx, x, ggml_mul(ctx, f, m->W(p + ".ls2.gamma")));
    }
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    explicit Graph(size_t nodes = 16384) {
        ggml_init_params ip = {ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true};
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, nodes, false);
    }
    ~Graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * input(int64_t a, int64_t b, int64_t c = 1, int64_t d = 1) {
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d);
        ggml_set_input(x);
        return x;
    }
    void output(ggml_tensor * x) { ggml_set_output(x); ggml_build_forward_expand(gf, x); }
    bool run(ggml_backend_t be, std::string * error) {
        alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!ggml_gallocr_alloc_graph(alloc, gf)) { set_error(error, "ggml_gallocr_alloc_graph failed"); return false; }
        return true;
    }
    bool compute(ggml_backend_t be, std::string * error) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { set_error(error, "graph compute failed"); return false; }
        return true;
    }
};

void set(ggml_tensor * t, const std::vector<float> & v) { ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t)); }
std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// torch _upsample_bilinear2d_aa weights (align_corners = False), float32 like
// torch's opmath, as a sparse row list: out i <- sum_j w[i][j] * in[xmin[i] + j].
struct AaTable { std::vector<int> xmin; std::vector<std::vector<float>> w; };
AaTable aa_bilinear(int n_in, int n_out) {
    const float scale = (float) n_in / (float) n_out;
    const float support = scale >= 1.0f ? scale : 1.0f;
    const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    AaTable t;
    t.xmin.resize(n_out);
    t.w.resize(n_out);
    for (int i = 0; i < n_out; ++i) {
        const float center = scale * ((float) i + 0.5f);
        const int xmin = std::max((int) (center - support + 0.5f), 0);
        const int xmax = std::min((int) (center + support + 0.5f), n_in);
        std::vector<float> ws;
        float tot = 0.0f;
        for (int j = 0; j < xmax - xmin; ++j) {
            const float x = std::fabs(((float) j + (float) xmin - center + 0.5f) * invscale);
            const float wv = x < 1.0f ? 1.0f - x : 0.0f;
            ws.push_back(wv);
            tot += wv;
        }
        for (float & wv : ws) wv = tot != 0.0f ? wv / tot : wv;
        t.xmin[i] = xmin;
        t.w[i] = std::move(ws);
    }
    return t;
}

// torch bilinear, align_corners = False, antialias = False: 2 taps per output.
void bilinear_taps(int n_in, int n_out, std::vector<int> & i0, std::vector<int> & i1,
                   std::vector<float> & l0, std::vector<float> & l1) {
    const float scale = (float) n_in / (float) n_out;
    i0.resize(n_out); i1.resize(n_out); l0.resize(n_out); l1.resize(n_out);
    for (int d = 0; d < n_out; ++d) {
        float src = scale * ((float) d + 0.5f) - 0.5f;
        src = std::max(src, 0.0f);
        const int a = (int) src;
        i0[d] = a;
        i1[d] = a + (a < n_in - 1 ? 1 : 0);
        l1[d] = src - (float) a;
        l0[d] = 1.0f - l1[d];
    }
}

std::vector<int> nearest_idx(int n_in, int n_out) {
    std::vector<int> r(n_out);
    const float scale = (float) n_in / (float) n_out;
    for (int i = 0; i < n_out; ++i) r[i] = n_in == n_out ? i : std::min((int) std::floor((float) i * scale), n_in - 1);
    return r;
}

// torch.linspace(a, b, n) in float32
std::vector<float> linspace(float a, float b, int n) {
    std::vector<float> v(n);
    if (n == 1) { v[0] = a; return v; }
    const float step = (b - a) / (float) (n - 1);
    const int half = n / 2;
    for (int i = 0; i < n; ++i) v[i] = i < half ? a + step * (float) i : b - step * (float) (n - 1 - i);
    return v;
}

// normalized_view_plane_uv(width, height) as [2, h, w] (u plane, v plane)
std::vector<float> uv_planes(int w, int h) {
    const double ar = (double) w / h;
    const float sx = (float) (ar / std::sqrt(1 + ar * ar)), sy = (float) (1 / std::sqrt(1 + ar * ar));
    const std::vector<float> u = linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w);
    const std::vector<float> v = linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h);
    std::vector<float> r((size_t) 2 * w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            r[(size_t) y * w + x] = u[x];
            r[(size_t) w * h + (size_t) y * w + x] = v[y];
        }
    return r;
}

// MoGe recover_focal_shift: min_s sum |f(s) xy/(z+s) - uv|^2, f(s) closed form;
// 1-D Levenberg-Marquardt run to convergence from s = 0 (moge3_ref.py).
void solve_focal_shift(const std::vector<float> & uv, const std::vector<float> & xyz, float & shift, float & focal) {
    const size_t n = uv.size() / 2;
    auto resid = [&](double s, std::vector<double> & r) {
        r.resize(2 * n);
        double pu = 0, pp = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = xyz[3 * i + 2] + s;
            const double px = xyz[3 * i] / d, py = xyz[3 * i + 1] / d;
            pu += px * uv[2 * i] + py * uv[2 * i + 1];
            pp += px * px + py * py;
        }
        const double f = pu / pp;
        for (size_t i = 0; i < n; ++i) {
            const double d = xyz[3 * i + 2] + s;
            r[2 * i]     = f * xyz[3 * i] / d - uv[2 * i];
            r[2 * i + 1] = f * xyz[3 * i + 1] / d - uv[2 * i + 1];
        }
    };
    auto dot = [](const std::vector<double> & a, const std::vector<double> & b) {
        double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
    };
    double s = 0, lam = 1e-3;
    std::vector<double> r, rp, rm, rn, J(2 * n);
    resid(s, r);
    double E = dot(r, r);
    for (int it = 0; it < 500; ++it) {
        const double h = 1e-7 * std::max(1.0, std::fabs(s));
        resid(s + h, rp); resid(s - h, rm);
        for (size_t i = 0; i < J.size(); ++i) J[i] = (rp[i] - rm[i]) / (2 * h);
        const double step = -dot(J, r) / (dot(J, J) * (1 + lam));
        resid(s + step, rn);
        const double En = dot(rn, rn);
        if (En < E) {
            s += step; r.swap(rn); E = En; lam *= 0.3;
            if (std::fabs(step) < 1e-13 * std::max(1.0, std::fabs(s))) break;
        } else {
            lam *= 10;
            if (lam > 1e12) break;
        }
    }
    shift = (float) s;
    // upstream: optim_focal from float32 xy/z and the float32 shift
    float pu = 0, pp = 0;
    for (size_t i = 0; i < n; ++i) {
        const float d = xyz[3 * i + 2] + shift;
        const float px = xyz[3 * i] / d, py = xyz[3 * i + 1] / d;
        pu += px * uv[2 * i] + py * uv[2 * i + 1];
        pp += px * px + py * py;
    }
    focal = pu / pp;
}

} // namespace

bool moge3_fov(moge3_model * m, const uint8_t * rgb, int W, int H,
               moge3_fov_result & out, moge3_taps * taps, std::string * error) {
    if (W != H) { set_error(error, "moge3_fov: square input only (Pixal3D's preprocess makes it so)"); return false; }
    auto t0 = std::chrono::steady_clock::now();

    // ── host: [0,1] CHW, antialiased bilinear to 840 (W pass, then H), normalise ──
    std::vector<float> img14((size_t) 3 * kImg * kImg);
    {
        const AaTable tx = aa_bilinear(W, kImg), ty = aa_bilinear(H, kImg);
        std::vector<float> tmp((size_t) H * kImg);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < kImg; ++x) {
                    const std::vector<float> & w = tx.w[x];
                    float acc = 0.0f;
                    for (size_t j = 0; j < w.size(); ++j)
                        acc += (float) rgb[((size_t) y * W + tx.xmin[x] + j) * 3 + c] / 255.0f * w[j];
                    tmp[(size_t) y * kImg + x] = acc;
                }
            for (int y = 0; y < kImg; ++y)
                for (int x = 0; x < kImg; ++x) {
                    const std::vector<float> & w = ty.w[y];
                    float acc = 0.0f;
                    for (size_t j = 0; j < w.size(); ++j) acc += tmp[(size_t) (ty.xmin[y] + j) * kImg + x] * w[j];
                    img14[((size_t) c * kImg + y) * kImg + x] = (acc - m->mean[c]) / m->stdv[c];
                }
        }
    }
    if (taps) taps->image14 = img14;
    out.ms_host = ms_since(t0);

    // ── encoder: DINOv2 ViT-L/14 -> 4 taps -> LN -> 4 x 1x1 -> sum ──
    std::vector<float> feat;
    {
        auto t1 = std::chrono::steady_clock::now();
        Graph g;
        G b{g.ctx, m};
        ggml_tensor * img = g.input(kImg, kImg, 3);
        ggml_tensor * pe = b.conv(img, m->W("encoder.backbone.patch_embed.proj.weight"), kPatch);   // [60, 60, 1024]
        pe = b.bias3(pe, "encoder.backbone.patch_embed.proj.bias");
        ggml_tensor * tok = ggml_cont(g.ctx, ggml_transpose(g.ctx, ggml_reshape_2d(g.ctx, pe, kGrid * kGrid, kDim)));  // [1024, 3600]
        ggml_tensor * cls = ggml_reshape_2d(g.ctx, m->W("encoder.backbone.cls_token"), kDim, 1);
        ggml_tensor * x = ggml_add(g.ctx, ggml_concat(g.ctx, cls, tok, 1), m->W("encoder.backbone.pos_embed_60x60"));
        ggml_tensor * sum = nullptr;
        int tap = 0;
        for (int i = 0; i < kLayers; ++i) {
            x = b.block(x, i);
            if (tap < 4 && i == kTaps[tap]) {
                ggml_tensor * n = b.ln(x, "encoder.backbone.norm");
                ggml_tensor * patches = ggml_view_2d(g.ctx, n, kDim, kGrid * kGrid, n->nb[1], n->nb[1]);
                ggml_tensor * p = b.linear(patches, "encoder.output_projections." + std::to_string(tap));
                sum = sum ? ggml_add(g.ctx, sum, p) : p;
                ++tap;
            }
        }
        ggml_tensor * f = ggml_cont(g.ctx, ggml_transpose(g.ctx, sum));  // [3600, 1024] = [60, 60, 1024]
        g.output(f);
        if (!g.run(m->backend, error)) return false;
        set(img, img14);
        if (!g.compute(m->backend, error)) return false;
        feat = get(f);
        out.ms_encoder = ms_since(t1);
    }
    if (taps) taps->features = feat;

    // ── neck ──
    std::vector<float> neck[5];
    {
        auto t1 = std::chrono::steady_clock::now();
        Graph g;
        G b{g.ctx, m};
        ggml_tensor * f0 = g.input(kGrid, kGrid, kDim);
        ggml_tensor * uv[5];
        for (int l = 0; l < 5; ++l) uv[l] = g.input(kGrid << l, kGrid << l, 2);
        ggml_tensor * x = b.conv1x1(ggml_concat(g.ctx, f0, uv[0], 2), "neck.input_blocks.0");
        ggml_tensor * outs[5];
        outs[0] = x;
        for (int l = 1; l < 5; ++l) {
            x = b.resampler(x, "neck", l - 1);
            x = ggml_add(g.ctx, x, b.conv1x1(uv[l], "neck.input_blocks." + std::to_string(l)));
            for (int r = 0; r < kNeckRes[l]; ++r)
                x = b.resblock(x, "neck.res_blocks." + std::to_string(l) + "." + std::to_string(r));
            outs[l] = x;
        }
        for (int l = 0; l < 5; ++l) g.output(outs[l]);
        if (!g.run(m->backend, error)) return false;
        set(f0, feat);
        for (int l = 0; l < 5; ++l) set(uv[l], uv_planes(kGrid << l, kGrid << l));
        if (!g.compute(m->backend, error)) return false;
        for (int l = 0; l < 5; ++l) neck[l] = get(outs[l]);
        out.ms_neck = ms_since(t1);
    }
    if (taps) for (int l = 0; l < 5; ++l) taps->neck[l] = neck[l];

    // ── points and mask heads (one graph each) ──
    auto t2 = std::chrono::steady_clock::now();
    auto run_head = [&](const std::string & p, std::vector<float> & res) {
        Graph g;
        G b{g.ctx, m};
        ggml_tensor * in[5];
        for (int l = 0; l < 5; ++l) in[l] = g.input(kGrid << l, kGrid << l, kLevelC[l]);
        ggml_tensor * x = b.conv1x1(in[0], p + ".input_blocks.0");
        for (int l = 1; l < 5; ++l) {
            x = b.resampler(x, p, l - 1);
            x = ggml_add(g.ctx, x, b.conv1x1(in[l], p + ".input_blocks." + std::to_string(l)));
            for (int r = 0; r < kHeadRes[l]; ++r)
                x = b.resblock(x, p + ".res_blocks." + std::to_string(l) + "." + std::to_string(r));
        }
        x = b.conv1x1(x, p + ".output_blocks.4");
        g.output(x);
        if (!g.run(m->backend, error)) return false;
        for (int l = 0; l < 5; ++l) set(in[l], neck[l]);
        if (!g.compute(m->backend, error)) return false;
        res = get(x);
        return true;
    };
    std::vector<float> praw, mraw;
    if (!run_head("points_head", praw) || !run_head("mask_head", mraw)) return false;
    out.ms_heads = ms_since(t2);

    // ── host: 64x64 nearest grid of the S x S bilinear resize, exp / sigmoid, solve ──
    auto t3 = std::chrono::steady_clock::now();
    const int R = kGrid << 4;  // 960
    std::vector<int> j0, j1, i0, i1;
    std::vector<float> m0, m1, l0, l1;
    bilinear_taps(R, H, j0, j1, m0, m1);
    bilinear_taps(R, W, i0, i1, l0, l1);
    const std::vector<int> iy = nearest_idx(H, kFovSamples), ix = nearest_idx(W, kFovSamples);
    const std::vector<float> uvS = uv_planes(W, H);
    const size_t plane = (size_t) R * R;
    auto sample = [&](const std::vector<float> & raw, int c, int r, int k) {
        const float * p = raw.data() + (size_t) c * plane;
        const int y = iy[r], x = ix[k];
        const float a = p[(size_t) j0[y] * R + i0[x]] * l0[x] + p[(size_t) j0[y] * R + i1[x]] * l1[x];
        const float bb = p[(size_t) j1[y] * R + i0[x]] * l0[x] + p[(size_t) j1[y] * R + i1[x]] * l1[x];
        return a * m0[y] + bb * m1[y];
    };
    std::vector<float> pts((size_t) kFovSamples * kFovSamples * 3), uvs((size_t) kFovSamples * kFovSamples * 2);
    std::vector<uint8_t> msk((size_t) kFovSamples * kFovSamples);
    std::vector<float> vuv, vxyz;
    for (int r = 0; r < kFovSamples; ++r)
        for (int k = 0; k < kFovSamples; ++k) {
            const size_t s = (size_t) r * kFovSamples + k;
            const float z = std::exp(sample(praw, 2, r, k));
            pts[3 * s] = sample(praw, 0, r, k) * z;
            pts[3 * s + 1] = sample(praw, 1, r, k) * z;
            pts[3 * s + 2] = z;
            uvs[2 * s] = uvS[(size_t) iy[r] * W + ix[k]];
            uvs[2 * s + 1] = uvS[(size_t) W * H + (size_t) iy[r] * W + ix[k]];
            const float lg = sample(mraw, 0, r, k);
            msk[s] = 1.0f / (1.0f + std::exp(-lg)) > 0.5f;
            if (msk[s]) {
                vuv.insert(vuv.end(), {uvs[2 * s], uvs[2 * s + 1]});
                vxyz.insert(vxyz.end(), {pts[3 * s], pts[3 * s + 1], pts[3 * s + 2]});
            }
        }
    out.n_valid = (int) (vuv.size() / 2);
    if (out.n_valid < 2) { set_error(error, "moge3_fov: fewer than 2 mask-valid samples"); return false; }
    solve_focal_shift(vuv, vxyz, out.shift, out.focal);
    const float ar = (float) W / (float) H;
    out.fx = out.focal / 2 * std::sqrt(1 + ar * ar) / ar;
    out.fy = out.focal / 2 * std::sqrt(1 + ar * ar);
    out.camera_angle_x = 2.0 * std::atan((double) W / (2.0 * ((double) out.fx * W)));
    out.ms_host += ms_since(t3);
    if (taps) {
        taps->points_raw = std::move(praw);
        taps->mask_raw = std::move(mraw);
        taps->fov_points = std::move(pts);
        taps->fov_uv = std::move(uvs);
        taps->fov_mask = std::move(msk);
    }
    return true;
}
