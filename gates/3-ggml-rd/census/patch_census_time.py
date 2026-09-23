# Second scratch patch on C:/b/ggml-census: per-op wall time on thread 0 (the
# time from one node's start to the barrier after it), accumulated per op name
# (UNARY split by unary op) and rewritten to $GGML_CENSUS_FILE.time after every graph.
import sys, pathlib

p = pathlib.Path(sys.argv[1]) / "src/ggml-cpu/ggml-cpu.c"
s = p.read_text(encoding="utf-8")

a1 = "static thread_ret_t ggml_graph_compute_thread(void * data) {\n"
assert s.count(a1) == 1
glob = r'''
/* ---- op census timing (scratch patch) ---- */
#define CENSUS_SLOTS (GGML_OP_COUNT + GGML_UNARY_OP_COUNT)
static int64_t census_us[CENSUS_SLOTS];
static int64_t census_n[CENSUS_SLOTS];
static int census_time_on = -1;
static int census_slot(const struct ggml_tensor * t) {
    if (t->op == GGML_OP_UNARY) return GGML_OP_COUNT + (int) ggml_get_unary_op(t);
    return (int) t->op;
}
static void census_time_dump(void) {
    const char * path = getenv("GGML_CENSUS_FILE");
    if (!path || !path[0]) return;
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s.time", path);
    FILE * f = fopen(buf, "w");
    if (!f) return;
    for (int i = 0; i < CENSUS_SLOTS; i++) {
        if (!census_n[i]) continue;
        const char * name = i < GGML_OP_COUNT ? ggml_op_name((enum ggml_op) i) : ggml_unary_op_name((enum ggml_unary_op) (i - GGML_OP_COUNT));
        fprintf(f, "%s%s\t%lld\t%lld\n", i < GGML_OP_COUNT ? "" : "UNARY:", name, (long long) census_n[i], (long long) census_us[i]);
    }
    fclose(f);
}
/* ---- end timing ---- */

'''
s = s.replace(a1, glob + a1)

a2 = """        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }
"""
assert s.count(a2) == 1
b2 = """        const int census_node = node_n;
        const int64_t census_t0 = (state->ith == 0 && census_time_on == 1) ? ggml_time_us() : 0;
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }
"""
s = s.replace(a2, b2)

a3 = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }
"""
assert s.count(a3) == 1, s.count(a3)
b3 = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
        if (state->ith == 0 && census_time_on == 1) {
            const int sl = census_slot(cgraph->nodes[census_node]);
            census_us[sl] += ggml_time_us() - census_t0;
            census_n[sl] += 1;
        }
    }
"""
s = s.replace(a3, b3)

a4 = "    census_graph(cgraph);\n"
assert s.count(a4) == 1
b4 = """    if (census_time_on < 0) { const char * cp = getenv("GGML_CENSUS_FILE"); census_time_on = (cp && cp[0]) ? 1 : 0; }
    census_graph(cgraph);
"""
s = s.replace(a4, b4)

# dump after the graph has run: find the return at the end of ggml_graph_compute
a5 = "enum ggml_status ggml_graph_compute_with_ctx("
assert s.count(a5) == 1
i = s.index(a5)
j = s.rindex("    return ret;\n", 0, i)
s = s[:j] + "    if (census_time_on == 1) census_time_dump();\n" + s[j:]
p.write_text(s, encoding="utf-8")
print("patched timing")
