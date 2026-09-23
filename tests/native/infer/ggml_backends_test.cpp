// The host oracle's two backends agree: one small graph (MUL_MAT, ADD,
// SOFT_MAX, MUL_MAT -- the shape of an attention block) evaluated on
// ggml-cpu and on the first GPU device (ggml-vulkan on the host; never
// shipped, the guest runs ggml-rd), compared elementwise.
//
//   ggml_backends_test [require-gpu]   (require-gpu: 1 = a GPU device must exist)
//
// Exit 0 pass, 1 mismatch or missing required GPU. Deliberately small: every
// host run is under a 300 s hard cap (tests/native/infer/CMakeLists.txt), and
// a comparison that needs a big graph on the CPU shrinks the problem instead.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int D = 64;    // head dim
constexpr int NQ = 96;   // queries
constexpr int NK = 160;  // keys

struct run_result {
    std::vector<float> out;
    double ms = 0.0;
    int nodes = 0;
};

bool run(ggml_backend_t backend, const std::vector<float> & q, const std::vector<float> & k,
         const std::vector<float> & v, const std::vector<float> & bias, run_result & r) {
    ggml_init_params p{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * tq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NQ);
    ggml_tensor * tk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NK);
    ggml_tensor * tvT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NK, D);
    ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NK, NQ);
    ggml_tensor * s = ggml_mul_mat(ctx, tk, tq);                 // [NK, NQ]
    s = ggml_add(ctx, s, tb);
    s = ggml_soft_max_ext(ctx, s, nullptr, 1.0f / std::sqrt((float) D), 0.0f);
    ggml_tensor * o = ggml_mul_mat(ctx, tvT, s);                 // [D, NQ]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(tq, q.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk, k.data(), 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tvT, v.data(), 0, ggml_nbytes(tvT));
    ggml_backend_tensor_set(tb, bias.data(), 0, ggml_nbytes(tb));
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r.nodes = ggml_graph_n_nodes(gf);
    r.out.resize(ggml_nelements(o));
    if (ok) ggml_backend_tensor_get(o, r.out.data(), 0, ggml_nbytes(o));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const bool require_gpu = argc > 1 && std::atoi(argv[1]) != 0;
    std::printf("ggml devices: %zu\n", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        std::printf("  [%zu] %s (%s) reg=%s\n", i, ggml_backend_dev_name(d),
                    ggml_backend_dev_description(d),
                    ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)));
    }

    unsigned seed = 12345u;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xffff) / 32768.0f - 1.0f; };
    std::vector<float> q(D * NQ), k(D * NK), v(NK * D), bias(NK * NQ);
    for (auto * a : {&q, &k, &v, &bias}) for (float & x : *a) x = rnd();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 1);
    run_result rc;
    if (!run(cpu, q, k, v, bias, rc)) { std::printf("FAIL: cpu compute\n"); return 1; }
    std::printf("cpu: %d nodes, %.3f ms\n", rc.nodes, rc.ms);

    ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gdev) {
        std::printf("%s: no GPU device\n", require_gpu ? "FAIL" : "SKIP");
        ggml_backend_free(cpu);
        return require_gpu ? 1 : 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(gdev, nullptr);
    if (!gpu) { std::printf("FAIL: GPU init\n"); return 1; }
    run_result rg;
    if (!run(gpu, q, k, v, bias, rg)) { std::printf("FAIL: gpu compute\n"); return 1; }
    run(gpu, q, k, v, bias, rg);  // second run: pipelines compiled
    std::printf("%s: %d nodes, %.3f ms (2nd run)\n", ggml_backend_name(gpu), rg.nodes, rg.ms);

    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < rc.out.size(); ++i) {
        max_abs = std::fmax(max_abs, std::fabs((double) rc.out[i] - rg.out[i]));
        max_ref = std::fmax(max_ref, std::fabs((double) rc.out[i]));
    }
    const double rel = max_abs / (max_ref > 0 ? max_ref : 1.0);
    // The GPU MUL_MAT rounds through f16 (measured 1.5e-3 on the RTX 4090);
    // 5e-3 of the output's range.
    const bool pass = rel < 5e-3;
    std::printf("%s: max|cpu-gpu| = %.3e, max|cpu| = %.3e, rel = %.3e (tol 5e-3)\n",
                pass ? "PASS" : "FAIL", max_abs, max_ref, rel);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return pass ? 0 : 1;
}
