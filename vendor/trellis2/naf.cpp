// NAF forward as ggml graph code. See naf.h and
// gates/7-pixal3d/aux-models/naf.md (section 3 is the op chain followed here).
#include "naf.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>

namespace naf {

namespace {

void set_err(std::string * err, const std::string & s) {
    if (err) *err = s;
}

// ggml-vulkan (the host GPU) unless IDO_GGML_BACKEND=cpu or no GPU device is
// registered; ggml-cpu then runs n_threads threads.
ggml_backend * pick_backend(int n_threads) {
    const char * e = std::getenv("IDO_GGML_BACKEND");
    ggml_backend * b = nullptr;
    if (!(e && std::strcmp(e, "cpu") == 0)) b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!b) {
        b = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(b, n_threads);
    }
    return b;
}

// Every node must run on the one backend: name the first that cannot.
bool check_ops(ggml_backend * be, ggml_cgraph * gf, std::string * err) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(be, n)) {
            set_err(err, std::string("naf: ") + ggml_backend_name(be) + " does not support " + ggml_op_desc(n) +
                             " (" + n->name + ")");
            return false;
        }
    }
    return true;
}

// One step = one small graph on the model's backend: build, allocate with the
// shared gallocr, upload inputs, compute. Inputs/outputs that must outlive the
// step live in persistent buffers (encoded/run) and are reached through views.
struct step {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<std::pair<ggml_tensor *, const void *>> uploads;

    explicit step(size_t n_tensors = 512) {
        ggml_init_params p;
        p.mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(4096, false);
        p.mem_buffer = nullptr;
        p.no_alloc = true;
        ctx = ggml_init(p);
        gf = ggml_new_graph_custom(ctx, 4096, false);
    }
    ~step() { ggml_free(ctx); }

    ggml_tensor * input_i32(const std::vector<int32_t> & v) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) v.size());
        ggml_set_input(t);
        uploads.emplace_back(t, v.data());
        return t;
    }
    ggml_tensor * input_f32(const std::vector<float> & v) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) v.size());
        ggml_set_input(t);
        uploads.emplace_back(t, v.data());
        return t;
    }
    void expand(ggml_tensor * t) { ggml_build_forward_expand(gf, t); }

    bool run(model * m, std::string * err) {
        if (!ggml_gallocr_alloc_graph(m->alloc, gf)) {
            set_err(err, "naf: ggml_gallocr_alloc_graph failed");
            return false;
        }
        if (!check_ops(m->backend, gf, err)) return false;
        for (auto & u : uploads) ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
        if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) {
            set_err(err, "naf: graph compute failed");
            return false;
        }
        return true;
    }
};

// A context + backend buffer holding tensors that persist across steps.
bool alloc_persistent(model * m, ggml_context ** ctx, ggml_backend_buffer ** buf,
                      const std::function<void(ggml_context *)> & make, std::string * err) {
    ggml_init_params p;
    p.mem_size = ggml_tensor_overhead() * 16;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    *ctx = ggml_init(p);
    make(*ctx);
    *buf = ggml_backend_alloc_ctx_tensors(*ctx, m->backend);
    if (!*buf) {
        set_err(err, "naf: persistent buffer allocation failed");
        return false;
    }
    return true;
}

// PyTorch 'reflect' padding by one pixel: -1 -> 1, n -> n-2.
inline int reflect1(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i); }

// Conv2d stride 1 (1x1, or 3x3 with 1-px reflect padding), banded over output
// pixels: [GET_ROWS(in_cl, im2col-reflect table) ->] RESHAPE [9Ci, b] ->
// MUL_MAT(cols, W[9Ci, O]) -> [b, O] channel-major -> ADD bias -> CPY into
// rows [s, s+b) of out (channel-major [P, *], channels c0..c0+O).
bool conv(model * m, ggml_tensor * in_cl, int S, int k, ggml_tensor * W, ggml_tensor * bias,
          ggml_tensor * out, int c0, std::string * err) {
    const int64_t P = (int64_t) S * S;
    const int64_t Ci = in_cl->ne[0];
    const int64_t O = W->ne[1];
    // ~256 MiB of im2col columns per band.
    int64_t band = std::max<int64_t>(S, ((int64_t) 1 << 26) / (Ci * k * k));
    band = std::min<int64_t>(P, band / S * S);
    std::vector<int32_t> idx;
    for (int64_t s = 0; s < P; s += band) {
        const int64_t b = std::min(band, P - s);
        step st;
        ggml_tensor * cols;
        if (k == 1) {
            cols = ggml_view_2d(st.ctx, in_cl, Ci, b, in_cl->nb[1], (size_t) s * in_cl->nb[1]);
        } else {
            idx.resize((size_t) b * 9);
            for (int64_t p = 0; p < b; ++p) {
                const int y = (int) ((s + p) / S), x = (int) ((s + p) % S);
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx)
                        idx[(size_t) p * 9 + ky * 3 + kx] =
                            reflect1(y + ky - 1, S) * S + reflect1(x + kx - 1, S);
            }
            cols = ggml_get_rows(st.ctx, in_cl, st.input_i32(idx));      // [Ci, 9b]
            cols = ggml_reshape_2d(st.ctx, cols, Ci * 9, b);               // [9Ci, b]
        }
        ggml_tensor * y = ggml_mul_mat(st.ctx, cols, W);                   // [b, O]
        y = ggml_add(st.ctx, y, ggml_reshape_2d(st.ctx, bias, 1, O));
        ggml_tensor * dst = ggml_view_2d(st.ctx, out, b, O, out->nb[1],
                                         (size_t) s * out->nb[0] + (size_t) c0 * out->nb[1]);
        st.expand(ggml_cpy(st.ctx, y, dst));
        if (!st.run(m, err)) return false;
    }
    return true;
}

// GroupNorm(G) on channel-major [P, C] (group g is contiguous: RESHAPE
// [C/G*P, G] -> NORM -> RESHAPE) -> MUL gamma -> ADD beta -> SILU, written
// transposed into the channels-last [C, P] input of the next conv.
bool gn_silu(model * m, ggml_tensor * x_cm, ggml_tensor * g, ggml_tensor * b, ggml_tensor * out_cl,
             std::string * err) {
    const int64_t P = x_cm->ne[0], C = x_cm->ne[1];
    const int G = m->hp.groups;
    step st;
    ggml_tensor * y = ggml_reshape_2d(st.ctx, x_cm, P * (C / G), G);
    y = ggml_norm(st.ctx, y, m->hp.eps);
    y = ggml_reshape_2d(st.ctx, y, P, C);
    y = ggml_mul(st.ctx, y, ggml_reshape_2d(st.ctx, g, 1, C));
    y = ggml_add(st.ctx, y, ggml_reshape_2d(st.ctx, b, 1, C));
    y = ggml_silu(st.ctx, y);
    st.expand(ggml_cpy(st.ctx, ggml_transpose(st.ctx, y), out_cl));
    return st.run(m, err);
}

}  // namespace

ggml_tensor * model::get(const std::string & name) const {
    return ggml_get_tensor(ctx, name.c_str());
}

model * load(const std::string & path, int n_threads, std::string * err) {
    model * m = new model();
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = &m->ctx;
    m->gguf = gguf_init_from_file(path.c_str(), gp);
    if (!m->gguf) {
        set_err(err, "naf: cannot read GGUF " + path);
        delete m;
        return nullptr;
    }
    auto u32 = [&](const char * k, int def) {
        const int64_t id = gguf_find_key(m->gguf, k);
        return id < 0 ? def : (int) gguf_get_val_u32(m->gguf, id);
    };
    auto f32 = [&](const char * k, float def) {
        const int64_t id = gguf_find_key(m->gguf, k);
        return id < 0 ? def : gguf_get_val_f32(m->gguf, id);
    };
    const int64_t aid = gguf_find_key(m->gguf, "general.architecture");
    if (aid < 0 || std::strcmp(gguf_get_val_str(m->gguf, aid), "naf") != 0) {
        set_err(err, "naf: " + path + " is not a NAF GGUF (general.architecture != naf)");
        free_model(m);
        return nullptr;
    }
    hparams & hp = m->hp;
    hp.dim = u32("naf.dim", 256);
    hp.enc_dim = u32("naf.enc_dim", 128);
    hp.heads = u32("naf.num_heads", 4);
    hp.kernel = u32("naf.kernel_size", 9);
    hp.groups = u32("naf.gn_groups", 8);
    hp.eps = f32("naf.gn_eps", 1e-5f);
    hp.rope_base = f32("naf.rope_base", 100.f);
    hp.enc_blocks = u32("naf.enc_blocks", 2);
    if (hp.dim != 256 || hp.heads != 4 || hp.kernel != 9 || hp.dim != 2 * hp.enc_dim) {
        set_err(err, "naf: unsupported config (dim 256, 4 heads of 64, kernel 9 expected)");
        free_model(m);
        return nullptr;
    }

    m->n_threads = n_threads > 0 ? n_threads : 1;
    m->backend = pick_backend(m->n_threads);
    m->buf = ggml_backend_alloc_ctx_tensors(m->ctx, m->backend);
    if (!m->buf) {
        set_err(err, "naf: weight buffer allocation failed");
        free_model(m);
        return nullptr;
    }
    std::ifstream f(path, std::ios::binary);
    const size_t off = gguf_get_data_offset(m->gguf);
    std::vector<char> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(m->gguf); ++i) {
        ggml_tensor * t = m->get(gguf_get_tensor_name(m->gguf, i));
        if (t->type != GGML_TYPE_F32) {
            set_err(err, std::string("naf: tensor ") + t->name + " is not f32");
            free_model(m);
            return nullptr;
        }
        tmp.resize(ggml_nbytes(t));
        f.seekg((std::streamoff) (off + gguf_get_tensor_offset(m->gguf, i)));
        f.read(tmp.data(), (std::streamsize) tmp.size());
        if (!f) {
            set_err(err, "naf: short read in " + path);
            free_model(m);
            return nullptr;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
    }
    ggml_tensor * per = m->get("image_encoder.rope.periods");
    if (!per || per->ne[0] != 16) {
        set_err(err, "naf: missing image_encoder.rope.periods [16]");
        free_model(m);
        return nullptr;
    }
    m->periods.resize(16);
    ggml_backend_tensor_get(per, m->periods.data(), 0, 16 * sizeof(float));
    m->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    return m;
}

void free_model(model * m) {
    if (!m) return;
    if (m->alloc) ggml_gallocr_free(m->alloc);
    if (m->buf) ggml_backend_buffer_free(m->buf);
    if (m->ctx) ggml_free(m->ctx);
    if (m->gguf) gguf_free(m->gguf);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

encoded * encode(model * m, const float * img, int S, std::string * err) {
    const int64_t P = (int64_t) S * S;
    const int E = m->hp.enc_dim;
    encoded * e = new encoded();
    e->S = S;
    ggml_context * sctx = nullptr;
    ggml_backend_buffer * sbuf = nullptr;
    ggml_tensor *img_cl = nullptr, *x_cm = nullptr, *x_cl = nullptr;
    bool ok = alloc_persistent(m, &e->ctx, &e->buf, [&](ggml_context * c) {
        e->enc = ggml_new_tensor_2d(c, GGML_TYPE_F32, P, 2 * E);
    }, err) && alloc_persistent(m, &sctx, &sbuf, [&](ggml_context * c) {
        img_cl = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3, P);
        x_cm = ggml_new_tensor_2d(c, GGML_TYPE_F32, P, E);
        x_cl = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, P);
    }, err);
    if (ok) {
        std::vector<float> t((size_t) P * 3);
        for (int64_t p = 0; p < P; ++p)
            for (int c = 0; c < 3; ++c) t[(size_t) p * 3 + c] = img[(size_t) c * P + p];
        ggml_backend_tensor_set(img_cl, t.data(), 0, t.size() * sizeof(float));
    }
    const char * branch[2] = {"image_encoder.encoder.", "image_encoder.sem_encoder."};
    const int ks[2] = {1, 3};
    for (int br = 0; ok && br < 2; ++br) {
        const std::string pre = branch[br];
        const int k = ks[br];
        const bool single = m->hp.enc_blocks == 0;
        ok = conv(m, img_cl, S, k, m->get(pre + "0.weight"), m->get(pre + "0.bias"),
                  single ? e->enc : x_cm, single ? br * E : 0, err);
        for (int blk = 1; ok && blk <= m->hp.enc_blocks; ++blk) {
            const std::string p = pre + std::to_string(blk) + ".";
            const bool last = blk == m->hp.enc_blocks;
            ok = gn_silu(m, x_cm, m->get(p + "norm1.weight"), m->get(p + "norm1.bias"), x_cl, err) &&
                 conv(m, x_cl, S, k, m->get(p + "conv1.weight"), m->get(p + "conv1.bias"), x_cm, 0, err) &&
                 gn_silu(m, x_cm, m->get(p + "norm2.weight"), m->get(p + "norm2.bias"), x_cl, err) &&
                 conv(m, x_cl, S, k, m->get(p + "conv2.weight"), m->get(p + "conv2.bias"),
                      last ? e->enc : x_cm, last ? br * E : 0, err);  // CONCAT: branch br -> channels [br*E, br*E+E)
        }
    }
    if (sbuf) ggml_backend_buffer_free(sbuf);
    if (sctx) ggml_free(sctx);
    if (!ok) {
        free_encoded(e);
        return nullptr;
    }
    return e;
}

void free_encoded(encoded * e) {
    if (!e) return;
    if (e->buf) ggml_backend_buffer_free(e->buf);
    if (e->ctx) ggml_free(e->ctx);
    delete e;
}

std::vector<int32_t> window_table(int hk, int ks) {
    std::vector<int32_t> w((size_t) hk * hk * ks * ks);
    size_t n = 0;
    for (int bi = 0; bi < hk; ++bi)
        for (int bj = 0; bj < hk; ++bj) {
            const int sh = std::min(std::max(bi - ks / 2, 0), hk - ks);
            const int sw = std::min(std::max(bj - ks / 2, 0), hk - ks);
            for (int u = 0; u < ks; ++u)
                for (int v = 0; v < ks; ++v) w[n++] = (sh + u) * hk + sw + v;
        }
    return w;
}

std::vector<int32_t> block_perm(int T, int d) {
    const int hk = T / d;
    std::vector<int32_t> p((size_t) T * T);
    size_t n = 0;
    for (int bi = 0; bi < hk; ++bi)
        for (int bj = 0; bj < hk; ++bj)
            for (int a = 0; a < d; ++a)
                for (int b = 0; b < d; ++b) p[n++] = (bi * d + a) * T + bj * d + b;
    return p;
}

run * prepare(model * m, const encoded * e, int T, const float * tokens, int hk, int C,
              std::string * err) {
    const int S = e->S;
    if (T <= 0 || hk <= 0 || S % T || T % hk || hk < m->hp.kernel || C % m->hp.heads) {
        set_err(err, "naf: need T | S, hk | T, hk >= kernel, heads | C");
        return nullptr;
    }
    run * r = new run();
    r->S = S; r->T = T; r->hk = hk; r->d = T / hk; r->r = S / T; r->C = C;
    const int d = r->d, rr = r->r, D = m->hp.dim, H = m->hp.heads, hd = D / H;
    const int64_t P = (int64_t) T * T, Bk = (int64_t) hk * hk;
    r->window = window_table(hk, m->hp.kernel);
    if (!alloc_persistent(m, &r->ctx, &r->buf, [&](ggml_context * c) {
            r->qb = ggml_new_tensor_4d(c, GGML_TYPE_F32, hd, (int64_t) d * d, H, Bk);
            r->k_lr = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, Bk);
            r->v_lr = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, Bk);
        }, err)) {
        free_run(r);
        return nullptr;
    }
    ggml_backend_tensor_set(r->v_lr, tokens, 0, ggml_nbytes(r->v_lr));

    // Host tables: block-major permutation, RoPE positions and freq factors.
    const std::vector<int32_t> perm = block_perm(T, d);
    std::vector<int32_t> pos_h((size_t) P), pos_w((size_t) P);
    for (int64_t q = 0; q < P; ++q) {
        const int y = perm[q] / T, x = perm[q] % T;
        pos_h[q] = 2 * y + 1 - T;
        pos_w[q] = 2 * x + 1 - T;
    }
    // theta_i = pos / ff_i = 2*pi*coord/period with coord = pos/T; 1e30 makes the
    // other axis' half an identity rotation (below fp32 epsilon).
    const int nf = hd / 2, half = nf / 2;
    std::vector<float> ff_h((size_t) nf), ff_w((size_t) nf);
    for (int i = 0; i < half; ++i) {
        const float f = (float) ((double) T * m->periods[i] / 6.283185307179586);
        ff_h[i] = f;            ff_h[half + i] = 1e30f;
        ff_w[i] = 1e30f;        ff_w[half + i] = f;
    }

    step st;
    ggml_tensor * x = e->enc;                                           // [S*S, 256]
    if (rr > 1) {                                                       // adaptive_avg_pool2d, integer r
        x = ggml_mean(st.ctx, ggml_reshape_4d(st.ctx, x, rr, S / rr, S, D));         // [1, S/r, S, D]
        x = ggml_reshape_4d(st.ctx, x, S / rr, rr, S / rr, D);
        x = ggml_mean(st.ctx, ggml_cont(st.ctx, ggml_permute(st.ctx, x, 1, 0, 2, 3))); // [1, S/r, S/r, D]
        x = ggml_reshape_2d(st.ctx, x, P, D);
    }
    ggml_tensor * xb = ggml_get_rows(st.ctx, ggml_cont(st.ctx, ggml_transpose(st.ctx, x)),
                                     st.input_i32(perm));               // [D, P] block-major
    ggml_tensor * q = ggml_reshape_3d(st.ctx, xb, hd, H, P);
    q = ggml_rope_ext(st.ctx, q, st.input_i32(pos_h), st.input_f32(ff_h), hd, GGML_ROPE_TYPE_NEOX,
                      0, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_rope_ext(st.ctx, q, st.input_i32(pos_w), st.input_f32(ff_w), hd, GGML_ROPE_TYPE_NEOX,
                      0, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    // KeyEncoder: mean over each d x d block = d*d consecutive slots.
    ggml_tensor * k = ggml_reshape_3d(st.ctx, q, D, (int64_t) d * d, Bk);
    k = ggml_mean(st.ctx, ggml_cont(st.ctx, ggml_permute(st.ctx, k, 1, 0, 2, 3)));   // [1, D, Bk]
    st.expand(ggml_cpy(st.ctx, ggml_reshape_2d(st.ctx, k, D, Bk), r->k_lr));
    ggml_tensor * qv = ggml_reshape_4d(st.ctx, q, hd, H, (int64_t) d * d, Bk);
    st.expand(ggml_cpy(st.ctx, ggml_permute(st.ctx, qv, 0, 2, 1, 3), r->qb));        // [hd, d*d, H, Bk]
    if (!st.run(m, err)) {
        free_run(r);
        return nullptr;
    }
    return r;
}

void free_run(run * r) {
    if (!r) return;
    if (r->buf) ggml_backend_buffer_free(r->buf);
    if (r->ctx) ggml_free(r->ctx);
    delete r;
}

bool attend(model * m, run * r, int blk0, int blk1, float * out, std::string * err, int max_blocks) {
    const int H = m->hp.heads, hd = m->hp.dim / H, D = m->hp.dim, C = r->C, Cv = C / H;
    const int KK = m->hp.kernel * m->hp.kernel;
    const int64_t d2 = (int64_t) r->d * r->d;
    if (blk0 < 0 || blk1 > r->blocks() || blk0 > blk1 || (int64_t) r->window.size() != (int64_t) r->blocks() * KK) {
        set_err(err, "naf: bad block range or window table");
        return false;
    }
    const float scale = 1.0f / std::sqrt((float) hd);
    std::vector<int32_t> idx;
    for (int s = blk0; s < blk1; s += max_blocks) {
        const int nb = std::min(max_blocks, blk1 - s);
        idx.assign(r->window.begin() + (size_t) s * KK, r->window.begin() + (size_t) (s + nb) * KK);
        step st;
        ggml_tensor * w = st.input_i32(idx);
        ggml_tensor * Kw = ggml_reshape_4d(st.ctx, ggml_get_rows(st.ctx, r->k_lr, w), hd, H, KK, nb);
        Kw = ggml_cont(st.ctx, ggml_permute(st.ctx, Kw, 0, 2, 1, 3));                // [hd, KK, H, nb]
        ggml_tensor * Q = ggml_view_4d(st.ctx, r->qb, hd, d2, H, nb, r->qb->nb[1], r->qb->nb[2],
                                       r->qb->nb[3], (size_t) s * r->qb->nb[3]);       // [hd, d2, H, nb]
        ggml_tensor * A = ggml_soft_max_ext(st.ctx, ggml_mul_mat(st.ctx, Kw, Q), nullptr, scale, 0.0f);  // [KK, d2, H, nb]
        ggml_tensor * Vw = ggml_reshape_4d(st.ctx, ggml_get_rows(st.ctx, r->v_lr, w), Cv, H, KK, nb);
        Vw = ggml_cont(st.ctx, ggml_permute(st.ctx, Vw, 1, 2, 0, 3));                // [KK, Cv, H, nb]
        ggml_tensor * O = ggml_mul_mat(st.ctx, Vw, A);                                  // [Cv, d2, H, nb]
        O = ggml_cont(st.ctx, ggml_permute(st.ctx, O, 0, 2, 1, 3));                    // [Cv, H, d2, nb] = [C, d2*nb]
        ggml_set_output(O);
        st.expand(O);
        if (!st.run(m, err)) return false;
        ggml_backend_tensor_get(O, out + (size_t) (s - blk0) * d2 * C, 0, ggml_nbytes(O));
    }
    (void) D;
    return true;
}

bool read_keys(const run * r, float * out) {
    ggml_backend_tensor_get(r->k_lr, out, 0, ggml_nbytes(r->k_lr));
    return true;
}

}  // namespace naf

// ── C API ──────────────────────────────────────────────────────────────────
struct naf_handle {
    naf::model * m = nullptr;
    naf::encoded * e = nullptr;
    naf::run * r = nullptr;
};

namespace {
thread_local std::string g_naf_err;
int naf_fail(const std::string & s) { g_naf_err = s; return 0; }
}  // namespace

extern "C" {

naf_handle * naf_open(const char * gguf_path, int n_threads) {
    std::string err;
    naf::model * m = naf::load(gguf_path ? gguf_path : "", n_threads, &err);
    if (!m) { g_naf_err = err; return nullptr; }
    naf_handle * h = new naf_handle();
    h->m = m;
    return h;
}

void naf_close(naf_handle * h) {
    if (!h) return;
    naf::free_run(h->r);
    naf::free_encoded(h->e);
    naf::free_model(h->m);
    delete h;
}

const char * naf_last_error(void) { return g_naf_err.c_str(); }

const char * naf_backend_name(const naf_handle * h) {
    return h && h->m && h->m->backend ? ggml_backend_name(h->m->backend) : "";
}

int naf_set_image(naf_handle * h, const float * image, int S) {
    if (!h || !image || S <= 0 || S % 16) return naf_fail("naf_set_image: need image and S % 16 == 0");
    naf::free_run(h->r); h->r = nullptr;
    naf::free_encoded(h->e); h->e = nullptr;
    std::string err;
    h->e = naf::encode(h->m, image, S, &err);
    return h->e ? 1 : naf_fail(err);
}

int naf_set_tokens(naf_handle * h, const float * dino_lr, int hk, int C, int out_res) {
    if (!h || !h->e || !dino_lr) return naf_fail("naf_set_tokens: call naf_set_image first");
    naf::free_run(h->r); h->r = nullptr;
    std::string err;
    h->r = naf::prepare(h->m, h->e, out_res, dino_lr, hk, C, &err);
    return h->r ? 1 : naf_fail(err);
}

int naf_rows(naf_handle * h, int y0, int y1, float * out) {
    if (!h || !h->r || !out) return naf_fail("naf_rows: call naf_set_tokens first");
    naf::run * r = h->r;
    const int T = r->T, d = r->d, hk = r->hk, C = r->C;
    if (y0 < 0 || y1 > T || y0 >= y1) return naf_fail("naf_rows: bad row range");
    std::vector<float> blk((size_t) hk * d * d * C);
    std::string err;
    for (int bi = y0 / d; bi * d < y1; ++bi) {
        if (!naf::attend(h->m, r, bi * hk, (bi + 1) * hk, blk.data(), &err)) return naf_fail(err);
        for (int a = 0; a < d; ++a) {
            const int y = bi * d + a;
            if (y < y0 || y >= y1) continue;
            float * row = out + (size_t) (y - y0) * T * C;
            for (int bj = 0; bj < hk; ++bj)
                for (int b = 0; b < d; ++b)
                    std::memcpy(row + ((size_t) bj * d + b) * C, blk.data() + (((size_t) bj * d + a) * d + b) * C,
                                sizeof(float) * C);
        }
    }
    return 1;
}

int naf_upsample(naf_handle * h, const float * dino_lr, int hk, int C, const float * image, int S,
                 int out_res, float * out) {
    return naf_set_image(h, image, S) && naf_set_tokens(h, dino_lr, hk, C, out_res) && naf_rows(h, 0, out_res, out);
}

}  // extern "C"
