# Scratch-only op-census logging patch for a COPY of ggml (C:/b/ggml-census).
# Every node that reaches the CPU backend's ggml_graph_compute is appended, one
# TSV line per node, to the file named by GGML_CENSUS_FILE (no env -> no-op).
import sys, pathlib

p = pathlib.Path(sys.argv[1]) / "src/ggml-cpu/ggml-cpu.c"
s = p.read_text(encoding="utf-8")
anchor = "enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {\n    ggml_cpu_init();\n"
assert s.count(anchor) == 1, "anchor"

helper = r'''
/* ---- op census (scratch patch) ---- */
static const char * census_contig(const struct ggml_tensor * t) {
    if (ggml_is_contiguous(t)) return "C";
    if (ggml_is_contiguous_rows(t)) return "R";   /* rows contiguous, strided outer */
    if (t->nb[0] == ggml_type_size(t->type)) return "R";
    return "N";
}
static void census_tensor(FILE * f, const struct ggml_tensor * t) {
    if (!t) { fprintf(f, "\t-\t-\t-\t-"); return; }
    fprintf(f, "\t%s\t%s\t%lld,%lld,%lld,%lld\t%zu,%zu,%zu,%zu", ggml_type_name(t->type), census_contig(t),
        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
}
static void census_graph(const struct ggml_cgraph * g) {
    static FILE * f = NULL;
    static int inited = 0;
    static long long graph_id = 0;
    if (!inited) {
        inited = 1;
        const char * path = getenv("GGML_CENSUS_FILE");
        if (path && path[0]) f = fopen(path, "a");
    }
    if (!f) return;
    graph_id++;
    const int n = ggml_graph_n_nodes((struct ggml_cgraph *) g);
    for (int i = 0; i < n; i++) {
        const struct ggml_tensor * t = ggml_graph_node((struct ggml_cgraph *) g, i);
        const char * sub = "-";
        if (t->op == GGML_OP_UNARY) sub = ggml_unary_op_name(ggml_get_unary_op(t));
        else if (t->op == GGML_OP_GLU) sub = ggml_glu_op_name(ggml_get_glu_op(t));
        fprintf(f, "%lld\t%d\t%s\t%s", graph_id, i, ggml_op_name(t->op), sub);
        census_tensor(f, t);
        for (int k = 0; k < 4; k++) census_tensor(f, t->src[k]);
        fprintf(f, "\t");
        const int32_t * op = (const int32_t *) t->op_params;
        int last = -1;
        for (int k = 0; k < (int)(GGML_MAX_OP_PARAMS / sizeof(int32_t)); k++) if (op[k]) last = k;
        for (int k = 0; k <= last; k++) fprintf(f, "%s%d", k ? "," : "", op[k]);
        if (last < 0) fprintf(f, "-");
        fprintf(f, "\n");
    }
    fflush(f);
}
/* ---- end op census ---- */

'''
s = s.replace(anchor, helper + anchor + "    census_graph(cgraph);\n")
p.write_text(s, encoding="utf-8")
print("patched", p)
