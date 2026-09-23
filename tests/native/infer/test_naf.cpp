// NAF on ggml (vendor/trellis2/naf.{h,cpp}) vs the CPU-torch oracle of
// gates/7-pixal3d/aux-models/naf_ref.py (naf_oracle/<config>/*.npy).
//
//   test_naf <naf.gguf> <oracle root> <config>[,<config>...] [threads]
//
// configs: shape_512 (S 512, T 512), shape_1024 (S 1024, T 512),
// tex_1024 (S 1024, T 1024). Configs with the same S share one encoder run.
// The hr output is evaluated in block rows (tex_1024's is 4 GiB) and compared
// against the raster hr_features.npy read in the same slabs.
// Checks per config (f32):
//   host tables == tbl_block_perm / tbl_window81 (exact)
//   keys_lr rel-L2, hr_features rel-L2 <= 1e-4
//   invariance: permuting the 81 slots inside every window leaves the output
//     equal (attention is a set operation over the window) -- rel-L2 <= 1e-4
//   negative control: the window table shuffled across blocks must FAIL
//     (rel-L2 > 1e-2) on the first block rows.
// Exit 0 pass, 1 fail, 77 skipped (no GGUF / oracle).
#include "naf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct npy {
    FILE * f = nullptr;
    long long off = 0;
    std::string descr;
    std::vector<long long> shape;
    long long count() const {
        long long n = 1;
        for (long long s : shape) n *= s;
        return n;
    }
    ~npy() { if (f) std::fclose(f); }
    bool open(const std::string & path) {
        f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[8];
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
        size_t hl = 0;
        if (magic[6] == 1) {
            uint16_t h; if (std::fread(&h, 2, 1, f) != 1) return false; hl = h; off = 10 + hl;
        } else {
            uint32_t h; if (std::fread(&h, 4, 1, f) != 1) return false; hl = h; off = 12 + hl;
        }
        std::string hdr(hl, ' ');
        if (std::fread(&hdr[0], 1, hl, f) != hl) return false;
        size_t p = hdr.find("'descr'");
        p = hdr.find('\'', hdr.find(':', p) + 1);
        descr = hdr.substr(p + 1, hdr.find('\'', p + 1) - p - 1);
        if (hdr.find("'fortran_order': False") == std::string::npos) return false;
        p = hdr.find('(', hdr.find("'shape'"));
        const size_t e = hdr.find(')', p);
        std::string s = hdr.substr(p + 1, e - p - 1);
        for (size_t i = 0; i < s.size();) {
            while (i < s.size() && (s[i] == ' ' || s[i] == ',')) ++i;
            if (i >= s.size()) break;
            shape.push_back(std::strtoll(s.c_str() + i, nullptr, 10));
            while (i < s.size() && s[i] != ',') ++i;
        }
        return true;
    }
    template <class T> bool read(long long elem, long long n, T * dst) {
        if (_fseeki64(f, off + elem * (long long) sizeof(T), SEEK_SET) != 0) return false;
        return std::fread(dst, sizeof(T), (size_t) n, f) == (size_t) n;
    }
    template <class T> std::vector<T> all() {
        std::vector<T> v((size_t) count());
        if (!read(0, count(), v.data())) v.clear();
        return v;
    }
};

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct l2 {
    double num = 0, den = 0, maxabs = 0;
    void add(double a, double b) {
        num += (a - b) * (a - b);
        den += b * b;
        maxabs = std::max(maxabs, std::fabs(a - b));
    }
    double rel() const { return std::sqrt(num / std::max(den, 1e-300)); }
};

struct cfg { const char * name; int S, T; };
const cfg CFGS[] = {{"shape_512", 512, 512}, {"shape_1024", 1024, 512}, {"tex_1024", 1024, 1024}};

// Evaluate NAF over block rows [bi0, bi1) and compare with the raster oracle.
bool compare_rows(naf::model * m, naf::run * r, npy & ref, int bi0, int bi1, l2 & acc, double & t_attend) {
    const int T = r->T, d = r->d, hk = r->hk, C = r->C;
    const size_t slots = (size_t) hk * d * d;
    std::vector<float> out(slots * C), slab((size_t) C * d * T);
    std::string err;
    for (int bi = bi0; bi < bi1; ++bi) {
        const double t0 = now_s();
        if (!naf::attend(m, r, bi * hk, (bi + 1) * hk, out.data(), &err)) {
            std::fprintf(stderr, "attend: %s\n", err.c_str());
            return false;
        }
        t_attend += now_s() - t0;
        // hr_features [1, C, T, T]: rows bi*d .. bi*d+d-1 of every channel.
        for (int c = 0; c < C; ++c)
            if (!ref.read((long long) c * T * T + (long long) bi * d * T, (long long) d * T, slab.data() + (size_t) c * d * T))
                return false;
        for (int bj = 0; bj < hk; ++bj)
            for (int a = 0; a < d; ++a)
                for (int b = 0; b < d; ++b) {
                    const float * o = out.data() + (((size_t) bj * d + a) * d + b) * C;
                    const size_t px = (size_t) a * T + (size_t) bj * d + b;
                    for (int c = 0; c < C; ++c) acc.add(o[c], slab[(size_t) c * d * T + px]);
                }
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: test_naf <naf.gguf> <oracle root> <configs> [threads]\n");
        return 2;
    }
    const std::string gguf = argv[1], root = argv[2];
    const int threads = argc > 4 ? std::atoi(argv[4]) : 8;
    std::vector<cfg> want;
    for (std::string s = argv[3]; !s.empty();) {
        const size_t c = s.find(',');
        const std::string n = s.substr(0, c);
        bool found = false;
        for (const cfg & k : CFGS) if (n == k.name) { want.push_back(k); found = true; }
        if (!found) { std::fprintf(stderr, "unknown config %s\n", n.c_str()); return 2; }
        s = c == std::string::npos ? "" : s.substr(c + 1);
    }
    if (FILE * f = std::fopen(gguf.c_str(), "rb")) std::fclose(f);
    else { std::printf("SKIP: no GGUF at %s\n", gguf.c_str()); return 77; }
    for (const cfg & k : want) {
        npy t;
        if (!t.open(root + "/" + k.name + "/hr_features.npy")) {
            std::printf("SKIP: no oracle %s/%s\n", root.c_str(), k.name);
            return 77;
        }
    }

    std::string err;
    naf::model * m = naf::load(gguf, threads, &err);
    if (!m) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("NAF %s: dim %d, heads %d, kernel %d, threads %d\n", gguf.c_str(), m->hp.dim, m->hp.heads, m->hp.kernel, threads);

    bool pass = true;
    naf::encoded * enc = nullptr;
    for (const cfg & k : want) {
        const std::string dir = root + "/" + k.name + "/";
        npy img, tok, keys, hr, tperm, twin;
        if (!img.open(dir + "image_naf.npy") || !tok.open(dir + "dino_tokens.npy") || !keys.open(dir + "keys_lr.npy") ||
            !hr.open(dir + "hr_features.npy") || !tperm.open(dir + "tbl_block_perm.npy") || !twin.open(dir + "tbl_window81.npy")) {
            std::fprintf(stderr, "[%s] missing oracle arrays\n", k.name);
            return 1;
        }
        const int S = k.S, T = k.T, hk = S / 16, C = (int) tok.shape[2], d = T / hk;
        const long long Bk = (long long) hk * hk;
        if (img.shape[2] != S || tok.shape[1] != 5 + Bk || hr.shape[1] != C || hr.shape[2] != T) {
            std::fprintf(stderr, "[%s] oracle shapes do not match S %d T %d\n", k.name, S, T);
            return 1;
        }
        double t_enc = 0;
        if (!enc || enc->S != S) {
            naf::free_encoded(enc);
            std::vector<float> im = img.all<float>();
            const double t0 = now_s();
            enc = naf::encode(m, im.data(), S, &err);
            t_enc = now_s() - t0;
            if (!enc) { std::fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
        }
        std::vector<float> tokens((size_t) Bk * C);
        tok.read(5LL * C, Bk * C, tokens.data());
        double t0 = now_s();
        naf::run * r = naf::prepare(m, enc, T, tokens.data(), hk, C, &err);
        const double t_prep = now_s() - t0;
        if (!r) { std::fprintf(stderr, "prepare: %s\n", err.c_str()); return 1; }

        // host tables vs the oracle's
        const std::vector<int32_t> perm = naf::block_perm(T, d);
        const bool tbl_ok = tperm.all<int32_t>() == perm && twin.all<int32_t>() == r->window;

        // keys: ours [Bk][256], oracle [1, 256, hk, hk]
        std::vector<float> kk((size_t) Bk * 256), kr = keys.all<float>();
        naf::read_keys(r, kk.data());
        l2 ek;
        for (long long p = 0; p < Bk; ++p)
            for (int c = 0; c < 256; ++c) ek.add(kk[(size_t) p * 256 + c], kr[(size_t) c * Bk + p]);

        // full output, block row by block row
        l2 eo;
        double t_att = 0;
        t0 = now_s();
        if (!compare_rows(m, r, hr, 0, hk, eo, t_att)) { std::fprintf(stderr, "[%s] compare failed\n", k.name); return 1; }
        const double t_cmp = now_s() - t0;

        // invariance: permute the 81 slots inside every window (first 2 block rows)
        const std::vector<int32_t> good = r->window;
        std::mt19937 rng(1234);
        for (size_t b = 0; b < good.size(); b += 81) std::shuffle(r->window.begin() + b, r->window.begin() + b + 81, rng);
        l2 ei;
        double tdummy = 0;
        compare_rows(m, r, hr, 0, 2, ei, tdummy);
        // negative control: shuffle the whole table across blocks (first 2 block rows)
        r->window = good;
        std::shuffle(r->window.begin(), r->window.end(), rng);
        l2 en;
        compare_rows(m, r, hr, 0, 2, en, tdummy);
        r->window = good;

        const bool ok = tbl_ok && ek.rel() <= 1e-4 && eo.rel() <= 1e-4 && ei.rel() <= 1e-4 && en.rel() > 1e-2;
        pass = pass && ok;
        std::printf("[%s] S %d T %d hk %d d %d r %d | tables %s | keys rel-L2 %.3e | hr rel-L2 %.3e (max abs %.3e, %lld px x %d) | "
                    "window-slot permutation %.3e | shuffled table (control) %.3e | encode %.2f s%s, prepare %.2f s, attend %.2f s (%d block rows), "
                    "attend+compare %.2f s | %s\n",
                    k.name, S, T, hk, d, S / T, tbl_ok ? "exact" : "MISMATCH", ek.rel(), eo.rel(), eo.maxabs, (long long) T * T, C,
                    ei.rel(), en.rel(), t_enc, t_enc == 0 ? " (shared)" : "", t_prep, t_att, hk, t_cmp, ok ? "PASS" : "FAIL");
        std::fflush(stdout);
        naf::free_run(r);
    }
    naf::free_encoded(enc);
    naf::free_model(m);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
