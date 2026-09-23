// BiRefNet as ggml graph code. See birefnet.h for layout and the op contract,
// gates/7-pixal3d/aux-models/birefnet.md for the census and the K11 spec.

#include "birefnet.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>

namespace {

constexpr int WS  = 12;         // Swin window
constexpr int WN  = WS * WS;    // tokens per window
constexpr int ZMAX = 8192;      // widest padded activation (squeeze in: 5760)

void set_err(std::string * e, const std::string & s) { if (e) *e = s; }

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// HOST STAND-IN for Lean kernel K11 (birefnet.md section K11). Cut 3's ggml-rd
// backend must provide these two ops from Lean -> Slang; until then the host
// runs these exact C++ bodies as ggml custom ops.
// ─────────────────────────────────────────────────────────────────────────────

// bilinear_sample_zeros: transcription of torchvision
// ops/cpu/deform_conv2d_kernel.cpp::bilinear_interpolate (scalar_t = float).
// FP contraction is off so the float result is the literal expression order.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
void bsz_op(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * X = dst->src[0];
    const ggml_tensor * P = dst->src[1];
    const int64_t C = X->ne[0], W = X->ne[1], H = X->ne[2];
    const int64_t N = P->ne[1];
    const float * xd = (const float *) X->data;
    const char  * pd = (const char *) P->data;
    float * yd = (float *) dst->data;
    const float fH = (float) H, fW = (float) W;
    const int64_t per = (N + nth - 1) / nth;
    const int64_t n0 = per * ith, n1 = std::min<int64_t>(N, n0 + per);
    for (int64_t n = n0; n < n1; ++n) {
        const float * pn = (const float *) (pd + n * P->nb[1]);
        const float y = pn[0];
        const float x = *(const float *) ((const char *) pn + P->nb[0]);
        float * out = yd + n * C;
        if (y <= -1.0f || fH <= y || x <= -1.0f || fW <= x) {
            std::memset(out, 0, (size_t) C * sizeof(float));
            continue;
        }
        const int y_low = (int) std::floor(y);
        const int x_low = (int) std::floor(x);
        const int y_high = y_low + 1;
        const int x_high = x_low + 1;
        const float ly = y - (float) y_low;
        const float lx = x - (float) x_low;
        const float hy = 1.0f - ly, hx = 1.0f - lx;
        const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        const float * r1 = (y_low >= 0 && x_low >= 0)              ? xd + ((int64_t) y_low  * W + x_low)  * C : nullptr;
        const float * r2 = (y_low >= 0 && x_high <= W - 1)         ? xd + ((int64_t) y_low  * W + x_high) * C : nullptr;
        const float * r3 = (y_high <= H - 1 && x_low >= 0)         ? xd + ((int64_t) y_high * W + x_low)  * C : nullptr;
        const float * r4 = (y_high <= H - 1 && x_high <= W - 1)    ? xd + ((int64_t) y_high * W + x_high) * C : nullptr;
        for (int64_t c = 0; c < C; ++c) {
            const float v1 = r1 ? r1[c] : 0.0f;
            const float v2 = r2 ? r2[c] : 0.0f;
            const float v3 = r3 ? r3[c] : 0.0f;
            const float v4 = r4 ? r4[c] : 0.0f;
            out[c] = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
        }
    }
}

// relu: y = max(x, 0), elementwise f32 (contiguous).
void relu_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * /*userdata*/) {
    const int64_t n = ggml_nelements(dst);
    const int64_t per = (n + nth - 1) / nth;
    const int64_t i0 = per * ith, i1 = std::min<int64_t>(n, i0 + per);
    const float * s = (const float *) a->data;
    float * d = (float *) dst->data;
    for (int64_t i = i0; i < i1; ++i) d[i] = s[i] > 0.0f ? s[i] : 0.0f;
}
#if defined(__clang__)
#pragma clang fp contract(on)
#endif

} // namespace

ggml_tensor * birefnet_bilinear_sample_zeros(ggml_context * ctx, ggml_tensor * X, ggml_tensor * P) {
    GGML_ASSERT(X->type == GGML_TYPE_F32 && P->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(X) && P->ne[0] == 2 && P->ne[2] == 1 && P->ne[3] == 1);
    ggml_tensor * args[2] = { X, P };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, X->ne[0], P->ne[1], 1, 1, args, 2, bsz_op,
                          GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * birefnet_relu(ggml_context * ctx, ggml_tensor * x) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    return ggml_map_custom1(ctx, x, relu_op, GGML_N_TASKS_MAX, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// model
// ─────────────────────────────────────────────────────────────────────────────

struct birefnet_model {
    ggml_backend_t        backend = nullptr;
    ggml_context        * wctx = nullptr;   // weights
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_context        * cctx = nullptr;   // derived constants
    ggml_backend_buffer_t cbuf = nullptr;
    std::map<std::string, ggml_tensor *> w;
    int embed = 192;
    int depths[4] = { 2, 2, 18, 2 };
    int heads[4]  = { 6, 12, 24, 48 };
    int ipt_grid[5] = { 32, 16, 8, 4, 1 };  // ipt_blk5..1
    int input_size = 1024;
    std::vector<ggml_tensor *> relbias[4];  // [144k, 144q, nH] per block
    ggml_tensor * zeros = nullptr;          // [ZMAX] f32 zeros
};

void birefnet_free(birefnet_model * m) {
    if (!m) return;
    if (m->wbuf) ggml_backend_buffer_free(m->wbuf);
    if (m->cbuf) ggml_backend_buffer_free(m->cbuf);
    if (m->wctx) ggml_free(m->wctx);
    if (m->cctx) ggml_free(m->cctx);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

birefnet_model * birefnet_load(const std::string & path, const birefnet_load_params & lp,
                               std::string * error) {
    ggml_context * meta = nullptr;
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = &meta;
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { set_err(error, "gguf_init_from_file failed: " + path); return nullptr; }

    auto fail = [&](const std::string & s) -> birefnet_model * {
        set_err(error, s);
        gguf_free(g);
        if (meta) ggml_free(meta);
        return nullptr;
    };
    {
        const int64_t id = gguf_find_key(g, "general.architecture");
        if (id < 0 || std::strcmp(gguf_get_val_str(g, id), "birefnet") != 0)
            return fail("not a birefnet GGUF: " + path);
    }
    auto * m = new birefnet_model();
    auto kv_arr = [&](const char * key, int * dst, int n) {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0) return;
        GGML_ASSERT((int) gguf_get_arr_n(g, id) == n);
        for (int i = 0; i < n; ++i) dst[i] = ((const int32_t *) gguf_get_arr_data(g, id))[i];
    };
    kv_arr("birefnet.depths", m->depths, 4);
    kv_arr("birefnet.num_heads", m->heads, 4);
    kv_arr("birefnet.ipt_grid", m->ipt_grid, 5);
    {
        const int64_t id = gguf_find_key(g, "birefnet.embed_dim");
        if (id >= 0) m->embed = (int) gguf_get_val_u32(g, id);
        const int64_t is = gguf_find_key(g, "birefnet.input_size");
        if (is >= 0) m->input_size = (int) gguf_get_val_u32(g, is);
        const int64_t iw = gguf_find_key(g, "birefnet.window_size");
        if (iw >= 0 && gguf_get_val_u32(g, iw) != WS) { birefnet_free(m); return fail("window_size != 12"); }
    }

    const int64_t nt = gguf_get_n_tensors(g);
    ggml_init_params ip = { (size_t) (nt + 8) * ggml_tensor_overhead(), nullptr, true };
    m->wctx = ggml_init(ip);
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(meta, name);
        ggml_type t = src->type;
        if (t == GGML_TYPE_F16 && (!lp.keep_f16 || ggml_n_dims(src) == 1)) t = GGML_TYPE_F32;
        ggml_tensor * d = ggml_new_tensor(m->wctx, t, GGML_MAX_DIMS, src->ne);
        ggml_set_name(d, name);
        m->w[name] = d;
    }
    m->backend = ggml_backend_cpu_init();
    m->wbuf = ggml_backend_alloc_ctx_tensors(m->wctx, m->backend);
    if (!m->wbuf) { birefnet_free(m); return fail("weight alloc failed"); }

    std::ifstream fin(path, std::ios::binary);
    if (!fin) { birefnet_free(m); return fail("cannot reopen " + path); }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> raw;
    std::vector<float> f32;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(meta, name);
        ggml_tensor * d = m->w[name];
        raw.resize(ggml_nbytes(src));
        fin.seekg((std::streamoff) (data_off + gguf_get_tensor_offset(g, i)));
        fin.read((char *) raw.data(), (std::streamsize) raw.size());
        if (!fin) { birefnet_free(m); return fail(std::string("short read: ") + name); }
        if (src->type == d->type) {
            ggml_backend_tensor_set(d, raw.data(), 0, raw.size());
        } else {
            GGML_ASSERT(src->type == GGML_TYPE_F16 && d->type == GGML_TYPE_F32);
            f32.resize(ggml_nelements(src));
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) f32.size());
            ggml_backend_tensor_set(d, f32.data(), 0, f32.size() * sizeof(float));
        }
    }
    gguf_free(g);
    ggml_free(meta);

    // derived constants: relative-position bias per block, zeros
    int nblk = 0;
    for (int s = 0; s < 4; ++s) nblk += m->depths[s];
    ggml_init_params cp = { (size_t) (nblk + 4) * ggml_tensor_overhead(), nullptr, true };
    m->cctx = ggml_init(cp);
    m->zeros = ggml_new_tensor_1d(m->cctx, GGML_TYPE_F32, ZMAX);
    for (int s = 0; s < 4; ++s)
        for (int b = 0; b < m->depths[s]; ++b)
            m->relbias[s].push_back(ggml_new_tensor_3d(m->cctx, GGML_TYPE_F32, WN, WN, m->heads[s]));
    m->cbuf = ggml_backend_alloc_ctx_tensors(m->cctx, m->backend);
    {
        std::vector<float> z(ZMAX, 0.0f);
        ggml_backend_tensor_set(m->zeros, z.data(), 0, z.size() * sizeof(float));
    }
    for (int s = 0; s < 4; ++s) {
        const int nH = m->heads[s];
        for (int b = 0; b < m->depths[s]; ++b) {
            char key[128];
            std::snprintf(key, sizeof key, "bb.layers.%d.blocks.%d.attn.relative_position_bias_table", s, b);
            auto it = m->w.find(key);
            if (it == m->w.end()) { birefnet_free(m); set_err(error, std::string("missing ") + key); return nullptr; }
            std::vector<float> tab((size_t) (2 * WS - 1) * (2 * WS - 1) * nH);
            ggml_backend_tensor_get(it->second, tab.data(), 0, tab.size() * sizeof(float));
            std::vector<float> bias((size_t) WN * WN * nH);
            // torch: attn[h, q, k] += table[((qy-ky+11)*23 + (qx-kx+11)), h]
            for (int h = 0; h < nH; ++h)
                for (int q = 0; q < WN; ++q)
                    for (int k = 0; k < WN; ++k) {
                        const int r = (q / WS - k / WS + WS - 1) * (2 * WS - 1) + (q % WS - k % WS + WS - 1);
                        bias[(size_t) k + WN * ((size_t) q + WN * h)] = tab[(size_t) r * nH + h];
                    }
            ggml_backend_tensor_set(m->relbias[s][b], bias.data(), 0, bias.size() * sizeof(float));
        }
    }
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// graph builder
// ─────────────────────────────────────────────────────────────────────────────
namespace {

struct Act { ggml_tensor * t; int C, W, H; };   // t ne [C, W*H]

struct G {
    ggml_context * ctx = nullptr;
    birefnet_model * m = nullptr;
    size_t band_bytes = (size_t) 256 << 20;
    bool zero_offsets = false;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> host;
    std::map<std::string, ggml_tensor *> cache;
    std::set<std::string> want;
    std::map<std::string, Act> taps;
    std::map<std::string, std::string> dcn_tap;  // atrous_conv prefix -> tap base

    ggml_tensor * W(const std::string & n) const {
        auto it = m->w.find(n);
        if (it == m->w.end()) { std::fprintf(stderr, "birefnet: missing tensor %s\n", n.c_str()); GGML_ABORT("missing tensor"); }
        return it->second;
    }
    bool has(const std::string & n) const { return m && m->w.count(n) != 0; }

    template <class T>
    ggml_tensor * input(ggml_type type, std::vector<T> && v, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        GGML_ASSERT((int64_t) v.size() == ne0 * ne1 * ne2 * ne3);
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        ggml_set_input(t);
        std::vector<uint8_t> bytes(v.size() * sizeof(T));
        std::memcpy(bytes.data(), v.data(), bytes.size());
        host.emplace_back(t, std::move(bytes));
        return t;
    }
    ggml_tensor * cached(const std::string & key, const std::function<ggml_tensor *()> & make) {
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        ggml_tensor * t = make();
        cache[key] = t;
        return t;
    }
    void tap(const std::string & name, const Act & a) {
        if (!want.count(name)) return;
        ggml_set_output(a.t);
        taps[name] = a;
    }

    ggml_tensor * zero_row(int C) {
        GGML_ASSERT(C <= ZMAX);
        return ggml_view_2d(ctx, m->zeros, C, 1, (size_t) C * sizeof(float), 0);
    }

    // ── elementwise / norms ─────────────────────────────────────────────────
    ggml_tensor * layer_norm(ggml_tensor * x, const std::string & p) {
        x = ggml_norm(ctx, x, 1e-5f);
        return ggml_add(ctx, ggml_mul(ctx, x, W(p + ".weight")), W(p + ".bias"));
    }
    ggml_tensor * bn(ggml_tensor * x, const std::string & p) {
        return ggml_add(ctx, ggml_mul(ctx, x, W(p + ".bn_scale")), W(p + ".bn_shift"));
    }
    ggml_tensor * relu(ggml_tensor * x) { return birefnet_relu(ctx, x); }
    ggml_tensor * linear(ggml_tensor * x, const std::string & p, bool bias = true) {
        ggml_tensor * y = ggml_mul_mat(ctx, W(p + ".weight"), x);
        return bias ? ggml_add(ctx, y, W(p + ".bias")) : y;
    }

    ggml_tensor * concat_list(std::vector<ggml_tensor *> v, int dim) {
        while (v.size() > 1) {
            std::vector<ggml_tensor *> nv;
            for (size_t i = 0; i + 1 < v.size(); i += 2) nv.push_back(ggml_concat(ctx, v[i], v[i + 1], dim));
            if (v.size() & 1) nv.push_back(v.back());
            v.swap(nv);
        }
        return v[0];
    }

    // ── convolution: gather-im2col (host table) + MUL_MAT, banded ────────────
    ggml_tensor * im2col_table(int k, int s, int p, int Wi, int Hi, int Wo, int Ho) {
        char key[96];
        std::snprintf(key, sizeof key, "im2col k%d s%d p%d %dx%d", k, s, p, Wi, Hi);
        return cached(key, [&]() {
            const int K = k * k;
            std::vector<int32_t> idx((size_t) K * Wo * Ho);
            const int32_t zrow = Wi * Hi;
            for (int oy = 0; oy < Ho; ++oy)
                for (int ox = 0; ox < Wo; ++ox)
                    for (int ky = 0; ky < k; ++ky)
                        for (int kx = 0; kx < k; ++kx) {
                            const int iy = oy * s - p + ky, ix = ox * s - p + kx;
                            const bool ok = iy >= 0 && iy < Hi && ix >= 0 && ix < Wi;
                            idx[(size_t) (ky * k + kx) + (size_t) K * ((size_t) ox + (size_t) Wo * oy)] =
                                ok ? iy * Wi + ix : zrow;
                        }
            return input(GGML_TYPE_I32, std::move(idx), (int64_t) K * Wo * Ho);
        });
    }

    // Several convs sharing one input and one kernel geometry share the columns.
    std::vector<Act> conv_multi(const Act & x, const std::vector<std::pair<std::string, std::string>> & wb,
                                int k, int s, int p) {
        const int K = k * k;
        const int Wo = (x.W + 2 * p - k) / s + 1, Ho = (x.H + 2 * p - k) / s + 1;
        std::vector<Act> out;
        std::vector<ggml_tensor *> Ws;
        for (auto & e : wb) {
            ggml_tensor * w = W(e.first);
            GGML_ASSERT(w->ne[0] == (int64_t) K * x.C);
            Ws.push_back(w);
        }
        std::vector<std::vector<ggml_tensor *>> bands(wb.size());
        if (k == 1 && s == 1 && p == 0) {
            for (size_t i = 0; i < Ws.size(); ++i) bands[i].push_back(ggml_mul_mat(ctx, Ws[i], x.t));
        } else {
            ggml_tensor * idx = im2col_table(k, s, p, x.W, x.H, Wo, Ho);
            ggml_tensor * xe = ggml_concat(ctx, x.t, zero_row(x.C), 1);
            const size_t row_bytes = (size_t) K * x.C * Wo * sizeof(float);
            const int rows = (int) std::max<size_t>(1, band_bytes / row_bytes);
            for (int y0 = 0; y0 < Ho; y0 += rows) {
                const int nr = std::min(rows, Ho - y0);
                const int64_t n = (int64_t) nr * Wo;
                ggml_tensor * iv = ggml_view_1d(ctx, idx, (int64_t) K * n, (size_t) K * Wo * y0 * sizeof(int32_t));
                ggml_tensor * cols = ggml_get_rows(ctx, xe, iv);                  // [C, K*n]
                cols = ggml_reshape_2d(ctx, cols, (int64_t) K * x.C, n);           // row k*C + c
                for (size_t i = 0; i < Ws.size(); ++i) bands[i].push_back(ggml_mul_mat(ctx, Ws[i], cols));
            }
        }
        for (size_t i = 0; i < Ws.size(); ++i) {
            ggml_tensor * y = concat_list(bands[i], 1);
            if (!wb[i].second.empty()) y = ggml_add(ctx, y, W(wb[i].second));
            out.push_back({ y, (int) Ws[i]->ne[1], Wo, Ho });
        }
        return out;
    }
    Act conv(const Act & x, const std::string & p, int k, int s, int pad, bool bias = true) {
        return conv_multi(x, { { p + ".weight", bias ? p + ".bias" : std::string() } }, k, s, pad)[0];
    }

    // ── bilinear, align_corners=True (torch upsample_bilinear2d, W then H) ──
    void ac_coeffs(int in, int out, std::vector<int> & i0, std::vector<int> & i1,
                   std::vector<float> & l0, std::vector<float> & l1) {
        const float scale = out > 1 ? (float) (in - 1) / (float) (out - 1) : 0.0f;
        i0.resize(out); i1.resize(out); l0.resize(out); l1.resize(out);
        for (int j = 0; j < out; ++j) {
            const float src = scale * (float) j;
            const int a = (int) src;
            i0[j] = a;
            i1[j] = a + (a < in - 1 ? 1 : 0);
            l1[j] = std::min(std::max(src - (float) a, 0.0f), 1.0f);
            l0[j] = 1.0f - l1[j];
        }
    }
    Act interp_ac(const Act & x, int Wo, int Ho) {
        Act cur = x;
        std::vector<int> i0, i1; std::vector<float> l0, l1;
        if (Wo != cur.W) {
            const int Wi = cur.W, H = cur.H;
            ac_coeffs(Wi, Wo, i0, i1, l0, l1);
            char key[96];
            std::snprintf(key, sizeof key, "acW %d>%d h%d", Wi, Wo, H);
            ggml_tensor * t0 = cached(std::string(key) + " i0", [&]() {
                std::vector<int32_t> v((size_t) Wo * H);
                for (int h = 0; h < H; ++h) for (int j = 0; j < Wo; ++j) v[(size_t) j + (size_t) Wo * h] = i0[j] + Wi * h;
                return input(GGML_TYPE_I32, std::move(v), (int64_t) Wo * H); });
            ggml_tensor * t1 = cached(std::string(key) + " i1", [&]() {
                std::vector<int32_t> v((size_t) Wo * H);
                for (int h = 0; h < H; ++h) for (int j = 0; j < Wo; ++j) v[(size_t) j + (size_t) Wo * h] = i1[j] + Wi * h;
                return input(GGML_TYPE_I32, std::move(v), (int64_t) Wo * H); });
            ggml_tensor * L0 = cached(std::string(key) + " l0", [&]() { auto v = l0; return input(GGML_TYPE_F32, std::move(v), 1, Wo); });
            ggml_tensor * L1 = cached(std::string(key) + " l1", [&]() { auto v = l1; return input(GGML_TYPE_F32, std::move(v), 1, Wo); });
            ggml_tensor * g0 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, cur.t, t0), cur.C, Wo, H);
            ggml_tensor * g1 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, cur.t, t1), cur.C, Wo, H);
            ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, g0, L0), ggml_mul(ctx, g1, L1));
            cur = { ggml_reshape_2d(ctx, y, cur.C, (int64_t) Wo * H), cur.C, Wo, H };
        }
        if (Ho != cur.H) {
            const int Hi = cur.H, Wc = cur.W;
            ac_coeffs(Hi, Ho, i0, i1, l0, l1);
            char key[96];
            std::snprintf(key, sizeof key, "acH %d>%d w%d", Hi, Ho, Wc);
            ggml_tensor * t0 = cached(std::string(key) + " i0", [&]() {
                std::vector<int32_t> v((size_t) Wc * Ho);
                for (int j = 0; j < Ho; ++j) for (int w = 0; w < Wc; ++w) v[(size_t) w + (size_t) Wc * j] = w + Wc * i0[j];
                return input(GGML_TYPE_I32, std::move(v), (int64_t) Wc * Ho); });
            ggml_tensor * t1 = cached(std::string(key) + " i1", [&]() {
                std::vector<int32_t> v((size_t) Wc * Ho);
                for (int j = 0; j < Ho; ++j) for (int w = 0; w < Wc; ++w) v[(size_t) w + (size_t) Wc * j] = w + Wc * i1[j];
                return input(GGML_TYPE_I32, std::move(v), (int64_t) Wc * Ho); });
            ggml_tensor * L0 = cached(std::string(key) + " l0", [&]() { auto v = l0; return input(GGML_TYPE_F32, std::move(v), 1, 1, Ho); });
            ggml_tensor * L1 = cached(std::string(key) + " l1", [&]() { auto v = l1; return input(GGML_TYPE_F32, std::move(v), 1, 1, Ho); });
            ggml_tensor * g0 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, cur.t, t0), cur.C, Wc, Ho);
            ggml_tensor * g1 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, cur.t, t1), cur.C, Wc, Ho);
            ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, g0, L0), ggml_mul(ctx, g1, L1));
            cur = { ggml_reshape_2d(ctx, y, cur.C, (int64_t) Wc * Ho), cur.C, Wc, Ho };
        }
        return cur;
    }

    // ── Swin ────────────────────────────────────────────────────────────────
    ggml_tensor * swin_fwd_table(int H, int Wd, int s) {
        char key[64]; std::snprintf(key, sizeof key, "swin fwd %dx%d s%d", Wd, H, s);
        return cached(key, [&]() {
            const int Hp = (H + WS - 1) / WS * WS, Wp = (Wd + WS - 1) / WS * WS;
            const int nWw = Wp / WS, nW = (Hp / WS) * nWw;
            std::vector<int32_t> v((size_t) nW * WN);
            for (int hq = 0; hq < Hp; ++hq)
                for (int wq = 0; wq < Wp; ++wq) {
                    const int hp = (hq + s) % Hp, wp = (wq + s) % Wp;       // roll(-s)
                    const int row = ((hq / WS) * nWw + wq / WS) * WN + (hq % WS) * WS + wq % WS;
                    v[row] = (hp < H && wp < Wd) ? hp * Wd + wp : H * Wd;   // pad -> zero row
                }
            return input(GGML_TYPE_I32, std::move(v), (int64_t) nW * WN);
        });
    }
    ggml_tensor * swin_inv_table(int H, int Wd, int s) {
        char key[64]; std::snprintf(key, sizeof key, "swin inv %dx%d s%d", Wd, H, s);
        return cached(key, [&]() {
            const int Hp = (H + WS - 1) / WS * WS, Wp = (Wd + WS - 1) / WS * WS;
            const int nWw = Wp / WS;
            std::vector<int32_t> v((size_t) H * Wd);
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < Wd; ++w) {
                    const int hq = (h - s + Hp) % Hp, wq = (w - s + Wp) % Wp;  // roll(+s), crop
                    v[(size_t) h * Wd + w] = ((hq / WS) * nWw + wq / WS) * WN + (hq % WS) * WS + wq % WS;
                }
            return input(GGML_TYPE_I32, std::move(v), (int64_t) H * Wd);
        });
    }
    ggml_tensor * swin_mask(int H, int Wd) {
        char key[64]; std::snprintf(key, sizeof key, "swin mask %dx%d", Wd, H);
        return cached(key, [&]() {
            const int Hp = (H + WS - 1) / WS * WS, Wp = (Wd + WS - 1) / WS * WS;
            const int nWh = Hp / WS, nWw = Wp / WS, nW = nWh * nWw, sh = WS / 2;
            auto region = [&](int c, int n) { return c < n - WS ? 0 : (c < n - sh ? 1 : 2); };
            std::vector<float> v((size_t) WN * WN * nW);
            std::vector<int> lab(WN);
            for (int wy = 0; wy < nWh; ++wy)
                for (int wx = 0; wx < nWw; ++wx) {
                    for (int i = 0; i < WN; ++i)
                        lab[i] = region(wy * WS + i / WS, Hp) * 3 + region(wx * WS + i % WS, Wp);
                    float * d = v.data() + (size_t) (wy * nWw + wx) * WN * WN;
                    for (int q = 0; q < WN; ++q)
                        for (int k = 0; k < WN; ++k) d[(size_t) q * WN + k] = lab[q] != lab[k] ? -100.0f : 0.0f;
                }
            return input(GGML_TYPE_F32, std::move(v), WN, WN, 1, nW);
        });
    }

    Act swin_block(const Act & x, int st, int bi) {
        char pb[64]; std::snprintf(pb, sizeof pb, "bb.layers.%d.blocks.%d", st, bi);
        const std::string p = pb;
        const int C = x.C, nH = m->heads[st], hd = C / nH;
        const int s = (bi % 2) ? WS / 2 : 0;
        const int Hp = (x.H + WS - 1) / WS * WS, Wp = (x.W + WS - 1) / WS * WS;
        const int nW = (Hp / WS) * (Wp / WS);

        ggml_tensor * xn = layer_norm(x.t, p + ".norm1");
        ggml_tensor * xe = ggml_concat(ctx, xn, zero_row(C), 1);
        ggml_tensor * xw = ggml_get_rows(ctx, xe, swin_fwd_table(x.H, x.W, s));      // [C, nW*144]
        ggml_tensor * qkv = linear(xw, p + ".attn.qkv");                               // [3C, nW*144]
        const size_t es = sizeof(float);
        auto head_view = [&](int which) {
            return ggml_view_4d(ctx, qkv, hd, nH, WN, nW, hd * es, 3 * C * es, (size_t) 3 * C * WN * es,
                                (size_t) which * C * es);
        };
        ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, head_view(0), 0, 2, 1, 3));   // [hd, 144, nH, nW]
        q = ggml_scale(ctx, q, (float) std::pow((double) hd, -0.5));
        ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, head_view(1), 0, 2, 1, 3));
        ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, head_view(2), 1, 2, 0, 3));   // [144, hd, nH, nW]
        ggml_tensor * a = ggml_mul_mat(ctx, k, q);                                      // [144k, 144q, nH, nW]
        a = ggml_add(ctx, a, m->relbias[st][bi]);
        if (s > 0) a = ggml_add(ctx, a, swin_mask(x.H, x.W));
        a = ggml_soft_max(ctx, a);
        ggml_tensor * o = ggml_mul_mat(ctx, v, a);                                      // [hd, 144q, nH, nW]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                           // [hd, nH, 144, nW]
        o = ggml_reshape_2d(ctx, o, C, (int64_t) WN * nW);
        o = linear(o, p + ".attn.proj");
        o = ggml_get_rows(ctx, o, swin_inv_table(x.H, x.W, s));                         // [C, H*W]
        ggml_tensor * y = ggml_add(ctx, x.t, o);
        ggml_tensor * h = layer_norm(y, p + ".norm2");
        h = ggml_gelu_erf(ctx, linear(h, p + ".mlp.fc1"));
        h = linear(h, p + ".mlp.fc2");
        return { ggml_add(ctx, y, h), C, x.W, x.H };
    }

    Act patch_merge(const Act & x, int st) {
        const int W2 = x.W / 2, H2 = x.H / 2;
        GGML_ASSERT(x.W % 2 == 0 && x.H % 2 == 0);
        char key[64]; std::snprintf(key, sizeof key, "merge %dx%d", x.W, x.H);
        ggml_tensor * idx = cached(key, [&]() {
            std::vector<int32_t> v((size_t) 4 * W2 * H2);
            static const int dh[4] = { 0, 1, 0, 1 }, dw[4] = { 0, 0, 1, 1 };   // x0 x1 x2 x3
            for (int h = 0; h < H2; ++h)
                for (int w = 0; w < W2; ++w)
                    for (int j = 0; j < 4; ++j)
                        v[(size_t) j + 4 * ((size_t) h * W2 + w)] = (2 * h + dh[j]) * x.W + 2 * w + dw[j];
            return input(GGML_TYPE_I32, std::move(v), (int64_t) 4 * W2 * H2);
        });
        ggml_tensor * g = ggml_reshape_2d(ctx, ggml_get_rows(ctx, x.t, idx), (int64_t) 4 * x.C, (int64_t) W2 * H2);
        char p[64]; std::snprintf(p, sizeof p, "bb.layers.%d.downsample", st);
        g = layer_norm(g, std::string(p) + ".norm");
        g = linear(g, std::string(p) + ".reduction", false);
        return { g, 2 * x.C, W2, H2 };
    }

    std::vector<Act> backbone(const Act & x) {
        Act cur = conv(x, "bb.patch_embed.proj", 4, 4, 0);
        cur.t = layer_norm(cur.t, "bb.patch_embed.norm");
        std::vector<Act> outs;
        for (int st = 0; st < 4; ++st) {
            for (int b = 0; b < m->depths[st]; ++b) cur = swin_block(cur, st, b);
            char n[16]; std::snprintf(n, sizeof n, "bb.norm%d", st);
            outs.push_back({ layer_norm(cur.t, n), cur.C, cur.W, cur.H });
            if (st < 3) cur = patch_merge(cur, st);
        }
        return outs;
    }

    // ── deformable conv (torchvision deform_conv2d, modulated) ─────────────
    ggml_tensor * dcn_base(int k, int Wd, int H) {
        char key[64]; std::snprintf(key, sizeof key, "dcn base k%d %dx%d", k, Wd, H);
        return cached(key, [&]() {
            const int K = k * k, p = k / 2;
            std::vector<float> v((size_t) 2 * K * Wd * H);
            for (int oy = 0; oy < H; ++oy)
                for (int ox = 0; ox < Wd; ++ox)
                    for (int kk = 0; kk < K; ++kk) {
                        const size_t n = (size_t) kk + (size_t) K * ((size_t) oy * Wd + ox);
                        v[2 * n + 0] = (float) (oy - p + kk / k);   // integer part, exact
                        v[2 * n + 1] = (float) (ox - p + kk % k);
                    }
            return input(GGML_TYPE_F32, std::move(v), 2, (int64_t) K * Wd * H);
        });
    }
    // x [C, W*H]; off [2K, W*H] (dy on even rows); mask [K, W*H]; Wr [K*C, Cout]
    ggml_tensor * dcn_core(const Act & x, ggml_tensor * off, ggml_tensor * mask, ggml_tensor * Wr, int k) {
        const int K = k * k;
        const int64_t HW = (int64_t) x.W * x.H;
        GGML_ASSERT(Wr->ne[0] == (int64_t) K * x.C);
        ggml_tensor * base = dcn_base(k, x.W, x.H);
        ggml_tensor * P = zero_offsets ? base
                                       : ggml_add(ctx, base, ggml_reshape_2d(ctx, off, 2, (int64_t) K * HW));
        ggml_tensor * X3 = ggml_reshape_3d(ctx, x.t, x.C, x.W, x.H);
        const size_t row_bytes = (size_t) K * x.C * x.W * sizeof(float);
        const int rows = (int) std::max<size_t>(1, band_bytes / row_bytes);
        std::vector<ggml_tensor *> bands;
        for (int y0 = 0; y0 < x.H; y0 += rows) {
            const int nr = std::min(rows, x.H - y0);
            const int64_t n = (int64_t) nr * x.W, pix0 = (int64_t) y0 * x.W;
            ggml_tensor * Pv = ggml_view_2d(ctx, P, 2, (int64_t) K * n, P->nb[1], (size_t) pix0 * K * P->nb[1]);
            ggml_tensor * Y = birefnet_bilinear_sample_zeros(ctx, X3, Pv);                       // [C, K*n]
            ggml_tensor * Mv = ggml_view_3d(ctx, mask, 1, K, n, sizeof(float), mask->nb[1],
                                            (size_t) pix0 * mask->nb[1]);
            Y = ggml_mul(ctx, ggml_reshape_3d(ctx, Y, x.C, K, n), Mv);                            // mask * val
            bands.push_back(ggml_mul_mat(ctx, Wr, ggml_reshape_2d(ctx, Y, (int64_t) K * x.C, n))); // [Cout, n]
        }
        return concat_list(bands, 1);
    }
    Act dcn(const Act & x, const std::string & p, int k) {
        auto om = conv_multi(x, { { p + ".offset_conv.weight", p + ".offset_conv.bias" },
                                  { p + ".modulator_conv.weight", p + ".modulator_conv.bias" } }, k, 1, k / 2);
        ggml_tensor * mask = ggml_scale(ctx, ggml_sigmoid(ctx, om[1].t), 2.0f);
        ggml_tensor * y = dcn_core(x, om[0].t, mask, W(p + ".regular_conv.weight"), k);
        Act out = { y, (int) W(p + ".regular_conv.weight")->ne[1], x.W, x.H };
        auto it = dcn_tap.find(p);
        if (it != dcn_tap.end()) {
            tap(it->second + "_in", x);
            tap(it->second + "_offset", om[0]);
            tap(it->second + "_mask", { mask, om[1].C, x.W, x.H });
            tap(it->second + "_out", out);
        }
        return out;
    }

    Act aspp_deformable(const Act & x, const std::string & p) {
        std::vector<ggml_tensor *> br;
        static const int ks[3] = { 1, 3, 7 };
        {
            Act b = dcn(x, p + ".aspp1.atrous_conv", 1);
            br.push_back(relu(bn(b.t, p + ".aspp1.bn")));
        }
        for (int i = 0; i < 3; ++i) {
            const std::string q = p + ".aspp_deforms." + std::to_string(i);
            Act b = dcn(x, q + ".atrous_conv", ks[i]);
            br.push_back(relu(bn(b.t, q + ".bn")));
        }
        // global average pool -> 1x1 conv -> BN -> relu -> broadcast
        ggml_tensor * gap = ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, x.t)));    // [1, C]
        gap = ggml_reshape_2d(ctx, gap, x.C, 1);
        ggml_tensor * g = ggml_mul_mat(ctx, W(p + ".global_avg_pool.1.weight"), gap);    // [256, 1]
        g = relu(bn(g, p + ".global_avg_pool.2"));
        br.push_back(ggml_repeat(ctx, g, br[0]));
        ggml_tensor * cat = concat_list(br, 0);                                          // [1280, HW]
        ggml_tensor * y = ggml_mul_mat(ctx, W(p + ".conv1.weight"), cat);
        y = relu(bn(y, p + ".bn1"));
        return { y, (int) y->ne[0], x.W, x.H };
    }

    Act dec_blk(const Act & x, const std::string & p) {
        Act h = conv(x, p + ".conv_in", 3, 1, 1);
        h.t = relu(bn(h.t, p + ".bn_in"));
        h = aspp_deformable(h, p + ".dec_att");
        Act o = conv(h, p + ".conv_out", 3, 1, 1);
        o.t = bn(o.t, p + ".bn_out");
        return o;
    }

    Act simple_convs(const Act & x, const std::string & p) {
        return conv(conv(x, p + ".conv1", 3, 1, 1), p + ".conv_out", 3, 1, 1);
    }

    // rearrange 'b c (hg h) (wg w) -> b (c hg wg) h w', channel order permuted
    // to (hg wg c) (the converter permutes ipt conv1's input axis to match).
    Act patches(const Act & x, int g) {
        if (g == 1) return x;
        const int S = x.W, h = S / g;
        GGML_ASSERT(x.W == x.H && S % g == 0);
        char key[48]; std::snprintf(key, sizeof key, "patch g%d S%d", g, S);
        ggml_tensor * idx = cached(key, [&]() {
            std::vector<int32_t> v((size_t) S * S);
            for (int ih = 0; ih < h; ++ih)
                for (int iw = 0; iw < h; ++iw)
                    for (int ihg = 0; ihg < g; ++ihg)
                        for (int iwg = 0; iwg < g; ++iwg)
                            v[(size_t) (ihg * g + iwg) + (size_t) g * g * ((size_t) ih * h + iw)] =
                                (ihg * h + ih) * S + iwg * h + iw;
            return input(GGML_TYPE_I32, std::move(v), (int64_t) S * S);
        });
        ggml_tensor * t = ggml_reshape_2d(ctx, ggml_get_rows(ctx, x.t, idx), (int64_t) x.C * g * g, (int64_t) h * h);
        return { t, x.C * g * g, h, h };
    }

    Act gdt(const Act & p, int n) {
        const std::string d = "decoder.";
        Act g = conv(p, d + "gdt_convs_" + std::to_string(n) + ".0", 3, 1, 1);
        g.t = relu(bn(g.t, d + "gdt_convs_" + std::to_string(n) + ".1"));
        ggml_tensor * a = ggml_sigmoid(ctx, linear(g.t, d + "gdt_convs_attn_" + std::to_string(n) + ".0"));  // [1, HW]
        return { ggml_mul(ctx, p.t, a), p.C, p.W, p.H };
    }

    Act cat(const Act & a, const Act & b) {
        GGML_ASSERT(a.W == b.W && a.H == b.H);
        return { ggml_concat(ctx, a.t, b.t, 0), a.C + b.C, a.W, a.H };
    }

    Act forward(const Act & x) {
        const int S = x.W;
        std::vector<Act> f = backbone(x);
        std::vector<Act> hf = backbone(interp_ac(x, S / 2, S / 2));
        for (int i = 0; i < 4; ++i) {
            tap("bb_full_x" + std::to_string(i + 1), f[i]);
            tap("bb_half_x" + std::to_string(i + 1), hf[i]);
        }
        Act xs[4];
        for (int i = 0; i < 4; ++i) xs[i] = cat(f[i], interp_ac(hf[i], f[i].W, f[i].H));
        const int s4 = xs[3].W;
        Act x4 = cat(cat(cat(interp_ac(xs[0], s4, s4), interp_ac(xs[1], s4, s4)), interp_ac(xs[2], s4, s4)), xs[3]);
        tap("squeeze_in", x4);
        x4 = dec_blk(x4, "squeeze_module.0");
        tap("squeeze_out", x4);

        const std::string D = "decoder.";
        Act in4 = cat(x4, simple_convs(patches(x, m->ipt_grid[0]), D + "ipt_blk5"));
        tap("dec_block4_in", in4);
        Act p4 = dec_blk(in4, D + "decoder_block4");
        tap("dec_block4_out", p4);
        p4 = gdt(p4, 4);
        Act u3 = interp_ac(p4, xs[2].W, xs[2].H);
        Act l3 = conv(xs[2], D + "lateral_block4.conv", 1, 1, 0);
        Act in3 = cat({ ggml_add(ctx, u3.t, l3.t), u3.C, u3.W, u3.H },
                      simple_convs(patches(x, m->ipt_grid[1]), D + "ipt_blk4"));
        tap("dec_block3_in", in3);
        Act p3 = dec_blk(in3, D + "decoder_block3");
        tap("dec_block3_out", p3);
        p3 = gdt(p3, 3);
        Act u2 = interp_ac(p3, xs[1].W, xs[1].H);
        Act l2 = conv(xs[1], D + "lateral_block3.conv", 1, 1, 0);
        Act in2 = cat({ ggml_add(ctx, u2.t, l2.t), u2.C, u2.W, u2.H },
                      simple_convs(patches(x, m->ipt_grid[2]), D + "ipt_blk3"));
        tap("dec_block2_in", in2);
        Act p2 = dec_blk(in2, D + "decoder_block2");
        tap("dec_block2_out", p2);
        p2 = gdt(p2, 2);
        Act u1 = interp_ac(p2, xs[0].W, xs[0].H);
        Act l1 = conv(xs[0], D + "lateral_block2.conv", 1, 1, 0);
        Act in1 = cat({ ggml_add(ctx, u1.t, l1.t), u1.C, u1.W, u1.H },
                      simple_convs(patches(x, m->ipt_grid[3]), D + "ipt_blk2"));
        tap("dec_block1_in", in1);
        Act p1 = dec_blk(in1, D + "decoder_block1");
        tap("dec_block1_out", p1);
        Act up = interp_ac(p1, S, S);
        Act fin = cat(up, simple_convs(patches(x, m->ipt_grid[4]), D + "ipt_blk1"));
        Act lg = conv(fin, D + "conv_out1.0", 1, 1, 0);
        tap("logits", lg);
        return lg;
    }
};

// token-major [C, W*H] <-> NCHW
void nchw_to_tm(const float * s, int C, int HW, std::vector<float> & d) {
    d.resize((size_t) C * HW);
    for (int c = 0; c < C; ++c) for (int i = 0; i < HW; ++i) d[(size_t) i * C + c] = s[(size_t) c * HW + i];
}
void tm_to_nchw(const float * s, int C, int HW, float * d) {
    for (int i = 0; i < HW; ++i) for (int c = 0; c < C; ++c) d[(size_t) c * HW + i] = s[(size_t) i * C + c];
}

bool run(ggml_backend_t be, ggml_cgraph * gf, G & g, const std::function<void()> & read, std::string * err,
         const char * tag) {
    const double t0 = now_ms();
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(al, gf)) {
        ggml_gallocr_free(al);
        set_err(err, "ggml_gallocr_alloc_graph failed");
        return false;
    }
    const double t1 = now_ms();
    for (auto & h : g.host) ggml_backend_tensor_set(h.first, h.second.data(), 0, h.second.size());
    const double t2 = now_ms();
    const ggml_status st = ggml_backend_graph_compute(be, gf);
    const double t3 = now_ms();
    if (st == GGML_STATUS_SUCCESS) read();
    size_t host_bytes = 0;
    for (auto & h : g.host) host_bytes += h.second.size();
    std::fprintf(stderr, "[birefnet %s] nodes %d, compute buffer %.1f MiB, host tables %.1f MiB; "
                 "alloc %.0f ms, upload %.0f ms, compute %.0f ms\n",
                 tag, ggml_graph_n_nodes(gf), ggml_gallocr_get_buffer_size(al, 0) / 1048576.0,
                 host_bytes / 1048576.0, t1 - t0, t2 - t1, t3 - t2);
    ggml_gallocr_free(al);
    if (st != GGML_STATUS_SUCCESS) { set_err(err, "graph compute failed"); return false; }
    return true;
}

} // namespace

bool birefnet_forward(birefnet_model * m, const float * x, int S, float * logits,
                      const birefnet_run_params & rp,
                      std::map<std::string, birefnet_tap> * taps, std::string * error) {
    if (S != m->input_size) { set_err(error, "input size must be " + std::to_string(m->input_size)); return false; }
    ggml_backend_cpu_set_n_threads(m->backend, rp.n_threads);
    const size_t n_graph = 65536;
    ggml_init_params ip = { n_graph * ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(n_graph, false),
                            nullptr, true };
    G g;
    g.ctx = ggml_init(ip);
    g.m = m;
    g.band_bytes = rp.band_bytes;
    g.zero_offsets = rp.zero_offsets;
    g.want.insert(rp.taps.begin(), rp.taps.end());
    g.dcn_tap["squeeze_module.0.dec_att.aspp_deforms.2.atrous_conv"] = "deform_sq_k7";
    g.dcn_tap["decoder.decoder_block1.dec_att.aspp_deforms.1.atrous_conv"] = "deform_d1_k3";

    std::vector<float> xtm;
    nchw_to_tm(x, 3, S * S, xtm);
    Act xin = { g.input(GGML_TYPE_F32, std::move(xtm), 3, (int64_t) S * S), 3, S, S };
    Act out = g.forward(xin);
    ggml_set_output(out.t);
    ggml_cgraph * gf = ggml_new_graph_custom(g.ctx, n_graph, false);
    ggml_build_forward_expand(gf, out.t);
    for (auto & t : g.taps) ggml_build_forward_expand(gf, t.second.t);

    bool ok = run(m->backend, gf, g, [&]() {
        ggml_backend_tensor_get(out.t, logits, 0, (size_t) S * S * sizeof(float));
        if (taps) {
            std::vector<float> tmp;
            for (auto & kv : g.taps) {
                const Act & a = kv.second;
                const int HW = a.W * a.H;
                tmp.resize((size_t) a.C * HW);
                ggml_backend_tensor_get(a.t, tmp.data(), 0, tmp.size() * sizeof(float));
                birefnet_tap & t = (*taps)[kv.first];
                t.C = a.C; t.H = a.H; t.W = a.W;
                t.data.resize(tmp.size());
                tm_to_nchw(tmp.data(), a.C, HW, t.data.data());
            }
        }
    }, error, rp.zero_offsets ? "forward zero-offsets" : "forward");
    ggml_free(g.ctx);
    return ok;
}

bool birefnet_deform_conv(const float * x, const float * offset, const float * mask,
                          const float * weight, int C, int H, int W, int Cout, int k,
                          float * out, int n_threads, size_t band_bytes, std::string * error) {
    const int K = k * k, HW = H * W;
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, n_threads);
    ggml_init_params ip = { 4096 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    G g;
    g.ctx = ggml_init(ip);
    g.band_bytes = band_bytes;
    std::vector<float> xt, ot, mt, wt((size_t) Cout * K * C);
    nchw_to_tm(x, C, HW, xt);
    nchw_to_tm(offset, 2 * K, HW, ot);
    nchw_to_tm(mask, K, HW, mt);
    for (int o = 0; o < Cout; ++o)   // torch [Cout, C, k, k] -> [Cout, k, k, C]
        for (int c = 0; c < C; ++c)
            for (int kk = 0; kk < K; ++kk) wt[(size_t) o * K * C + (size_t) kk * C + c] = weight[((size_t) o * C + c) * K + kk];
    Act xa = { g.input(GGML_TYPE_F32, std::move(xt), C, HW), C, W, H };
    ggml_tensor * off = g.input(GGML_TYPE_F32, std::move(ot), 2 * K, HW);
    ggml_tensor * msk = g.input(GGML_TYPE_F32, std::move(mt), K, HW);
    ggml_tensor * Wr  = g.input(GGML_TYPE_F32, std::move(wt), (int64_t) K * C, Cout);
    ggml_tensor * y = g.dcn_core(xa, off, msk, Wr, k);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, y);
    bool ok = run(be, gf, g, [&]() {
        std::vector<float> tmp((size_t) Cout * HW);
        ggml_backend_tensor_get(y, tmp.data(), 0, tmp.size() * sizeof(float));
        tm_to_nchw(tmp.data(), Cout, HW, out);
    }, error, "deform_conv");
    ggml_free(g.ctx);
    ggml_backend_free(be);
    return ok;
}
