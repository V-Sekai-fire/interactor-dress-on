// G7.moge3 -- MoGe-3 camera FoV on ggml-cpu vs the CPU-torch fp32 oracle
// (gates/7-pixal3d/aux-models/moge3_ref.py, its .npy outputs).
//
//   MOGE3_GGUF    GGUF from tools/models/convert_moge3_to_gguf.py
//   MOGE3_REF     moge3_ref_out/ (input_rgb_u8.npy, intrinsics.npy, intermediates)
//   MOGE3_CONTROL 0 skips the flipped-image negative controls
//   MOGE3_ANGLE_TOL_DEG (0.05), MOGE3_RELL2_TOL (1e-3)
//
// Pass: |camera_angle_x - oracle| <= 0.05 deg and rel-L2 <= 1e-3 for the raw
// points head map and the 64x64 affine points the solve uses. Controls: the
// horizontally and the vertically flipped image each move the FoV by more
// than the tolerance (the gate can see a wrong input). Exit 77 = no assets.
#include "moge3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    size_t es = out.descr == "<f4" ? 4 : (out.descr == "|u1" || out.descr == "|b1") ? 1 : out.descr == "<i8" ? 8 : 0;
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
double deg(double r) { return r * 180.0 / M_PI; }

} // namespace

int main() {
    const char * gguf = std::getenv("MOGE3_GGUF");
    const char * refd = std::getenv("MOGE3_REF");
    if (!gguf || !refd) { std::printf("SKIP: set MOGE3_GGUF and MOGE3_REF\n"); return 77; }
    const std::string R = std::string(refd) + "/";
    const double tol_deg = std::getenv("MOGE3_ANGLE_TOL_DEG") ? std::atof(std::getenv("MOGE3_ANGLE_TOL_DEG")) : 0.05;
    const double tol_l2 = std::getenv("MOGE3_RELL2_TOL") ? std::atof(std::getenv("MOGE3_RELL2_TOL")) : 1e-3;
    const bool control = !(std::getenv("MOGE3_CONTROL") && std::string(std::getenv("MOGE3_CONTROL")) == "0");

    Npy rgb, K;
    if (!load_npy(R + "input_rgb_u8.npy", rgb) || !load_npy(R + "intrinsics.npy", K)) {
        std::printf("SKIP: oracle npy missing under %s\n", refd);
        return 77;
    }
    const int H = (int) rgb.shape[0], W = (int) rgb.shape[1];
    const double fx_ref = f32(K)[0];
    const double ang_ref = 2.0 * std::atan(1.0 / (2.0 * fx_ref));

    std::string err;
    moge3_model * m = moge3_load(gguf, 0, &err);
    if (!m) { std::printf("FAIL: load: %s\n", err.c_str()); return 1; }

    moge3_taps taps;
    moge3_fov_result r;
    if (!moge3_fov(m, (const uint8_t *) rgb.data.data(), W, H, r, &taps, &err)) {
        std::printf("FAIL: moge3_fov: %s\n", err.c_str());
        return 1;
    }
    std::printf("time ms: host %.0f encoder %.0f neck %.0f heads %.0f\n", r.ms_host, r.ms_encoder, r.ms_neck, r.ms_heads);

    struct Cmp { const char * npy; const std::vector<float> * ours; };
    const Cmp cmps[] = {
        {"image14_norm", &taps.image14}, {"encoder_features", &taps.features},
        {"neck_level0", &taps.neck[0]}, {"neck_level1", &taps.neck[1]}, {"neck_level2", &taps.neck[2]},
        {"neck_level3", &taps.neck[3]}, {"neck_level4", &taps.neck[4]},
        {"points_head_raw", &taps.points_raw}, {"mask_head_raw", &taps.mask_raw},
        {"fov_points_lr", &taps.fov_points}, {"fov_uv_lr", &taps.fov_uv},
    };
    double l2_points_raw = INFINITY, l2_fov_points = INFINITY;
    for (const Cmp & c : cmps) {
        Npy a;
        if (!load_npy(R + c.npy + ".npy", a)) { std::printf("  %-18s (no oracle)\n", c.npy); continue; }
        const std::vector<float> ref = f32(a);
        const double l2 = rel_l2(*c.ours, ref), mx = max_abs(*c.ours, ref);
        std::printf("  %-18s rel-L2 %.3e  maxabs %.3e  (n %zu)\n", c.npy, l2, mx, ref.size());
        if (!std::strcmp(c.npy, "points_head_raw")) l2_points_raw = l2;
        if (!std::strcmp(c.npy, "fov_points_lr")) l2_fov_points = l2;
    }
    int mask_diff = -1;
    {
        Npy a;
        if (load_npy(R + "fov_mask_lr.npy", a)) {
            mask_diff = 0;
            for (size_t i = 0; i < taps.fov_mask.size(); ++i) mask_diff += (taps.fov_mask[i] != 0) != (a.data[i] != 0);
        }
    }
    const double dang = deg(std::fabs(r.camera_angle_x - ang_ref));
    std::printf("focal %.6f shift %.5f fx %.6f valid %d/4096 (mask samples differing from oracle: %d)\n",
                r.focal, r.shift, r.fx, r.n_valid, mask_diff);
    std::printf("camera_angle_x %.4f deg (oracle %.4f deg, fx %.6f)  |d| %.4f deg  tol %.3f\n",
                deg(r.camera_angle_x), deg(ang_ref), fx_ref, dang, tol_deg);

    bool pass = dang <= tol_deg && l2_points_raw <= tol_l2 && l2_fov_points <= tol_l2;
    std::printf("GATE angle %s, points_head_raw rel-L2 %s, fov_points rel-L2 %s\n",
                dang <= tol_deg ? "PASS" : "FAIL", l2_points_raw <= tol_l2 ? "PASS" : "FAIL",
                l2_fov_points <= tol_l2 ? "PASS" : "FAIL");

    if (control) {
        const uint8_t * src = (const uint8_t *) rgb.data.data();
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<uint8_t> flip((size_t) W * H * 3);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const int sy = mode ? H - 1 - y : y, sx = mode ? x : W - 1 - x;
                    std::memcpy(&flip[((size_t) y * W + x) * 3], &src[((size_t) sy * W + sx) * 3], 3);
                }
            moge3_fov_result rc;
            if (!moge3_fov(m, flip.data(), W, H, rc, nullptr, &err)) { std::printf("FAIL: control: %s\n", err.c_str()); return 1; }
            const double d = deg(std::fabs(rc.camera_angle_x - ang_ref));
            const bool seen = d > tol_deg;
            std::printf("CONTROL %s flip: camera_angle_x %.4f deg, |d| vs oracle %.4f deg -> %s\n",
                        mode ? "vertical" : "horizontal", deg(rc.camera_angle_x), d,
                        seen ? "detected (gate would fail)" : "NOT detected");
            pass = pass && seen;
        }
    }
    moge3_free(m);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
