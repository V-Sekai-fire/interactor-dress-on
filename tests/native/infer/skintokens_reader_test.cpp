// skin-tokens: the bundle_reader / upload path (interactor-dress-on, the
// guest's way in: no filesystem) against the unpatched path (model::load of a
// directory: istream manifest + gguf_init_from_file + ifstream tensor reads),
// on the FoxGirl fixture avatar, host ggml-cpu.
//
//   skintokens-reader-test [BUNDLE_DIR] [MESH_OBJ] [ORACLE_DIR]
//
// Defaults: $SKINTOKENS_GGUF_DIR or <repo>/models/SkinTokens-GGUF/F16,
// $SKINTOKENS_MESH or <repo>/project/fixtures/foxgirl/avatar.obj,
// $SKINTOKENS_ORACLE_DIR or ./skintokens-oracle. Exit 77 (skip) without the
// weights. SKINTOKENS_THREADS (default 8) sets ggml-cpu threads.
//
// Both paths rig the same mesh greedily (beams 1, top_k 1, temperature 0,
// seed 0). PASS needs: identical token streams (the first 120 and the whole),
// identical skeleton (names, parents, rest positions bitwise), identical
// JOINTS_0 and bitwise-identical WEIGHTS_0; and, as the controls, the reader
// actually carried every tensor byte (read and upload counters), and a reader
// that negates one tensor on upload (llm.norm.w; SKINTOKENS_READER_CONTROL=1,
// default on) is caught -- its tokens or weights differ. The unpatched run's output
// is written to ORACLE_DIR as the G4b oracle (tokens.i32, joints.u16,
// weights.f32, rest.f32, parents.i32, names.txt, oracle.txt).
#include <skintokens/skintokens.hpp>

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

std::string env_or(const char * name, const std::string & fallback) {
    const char * value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::string{value} : fallback;
}

bool load_obj(const std::string & path, skintokens::mesh & output) {
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string tag;
        fields >> tag;
        if (tag == "v") {
            skintokens::vec3 value;
            fields >> value.x >> value.y >> value.z;
            output.vertices.push_back(value);
        } else if (tag == "f") {
            std::vector<std::uint32_t> polygon;
            std::string corner;
            while (fields >> corner) {
                const long index = std::stol(corner.substr(0, corner.find('/')));
                const long count = static_cast<long>(output.vertices.size());
                polygon.push_back(static_cast<std::uint32_t>(index < 0 ? count + index : index - 1));
            }
            for (std::size_t i = 2; i < polygon.size(); ++i)
                output.faces.push_back({polygon[0], polygon[i - 1], polygon[i]});
        }
    }
    return !output.vertices.empty() && !output.faces.empty();
}

// The reader under test, wrapped to count what flows through it and,
// for the control, to corrupt one tensor's bytes on upload.
class counting_reader final : public skintokens::bundle_reader {
public:
    explicit counting_reader(std::unique_ptr<skintokens::bundle_reader> inner, std::string corrupt = {})
        : inner_(std::move(inner)), corrupt_(std::move(corrupt)) {}
    bool exists(std::string_view component) override { ++exists_calls; return inner_->exists(component); }
    std::uint64_t size(std::string_view component) override { return inner_->size(component); }
    std::size_t read(std::string_view component, void * output, std::uint64_t offset,
                     std::size_t bytes) override {
        ++read_calls;
        const std::size_t count = inner_->read(component, output, offset, bytes);
        read_bytes += count;
        return count;
    }
    bool upload(const skintokens::tensor_upload & tensor) override {
        ++upload_calls;
        upload_bytes += tensor.bytes;
        if (!bundle_reader::upload(tensor)) return false;
        if (!corrupt_.empty() && tensor.name == corrupt_) {
            // Re-upload the tensor negated (an F32 norm weight: every logit
            // flips sign, so greedy decoding must take other tokens).
            auto * value = static_cast<ggml_tensor *>(tensor.tensor);
            if (value->type != GGML_TYPE_F32) return false;
            std::vector<float> data(tensor.bytes / sizeof(float));
            ggml_backend_tensor_get(value, data.data(), 0U, tensor.bytes);
            for (auto & x : data) x = -x;
            ggml_backend_tensor_set(value, data.data(), 0U, tensor.bytes);
            corrupted = true;
        }
        return true;
    }
    std::size_t exists_calls = 0, read_calls = 0, upload_calls = 0;
    std::uint64_t read_bytes = 0, upload_bytes = 0;
    bool corrupted = false;

private:
    std::unique_ptr<skintokens::bundle_reader> inner_;
    std::string corrupt_;
};

struct run_result {
    skintokens::skin binding;
    double load_s = 0.0, rig_s = 0.0;
};

bool rig_with(const char * label, skintokens::result<skintokens::model> loaded, double load_s,
              const skintokens::mesh & source, const skintokens::generation_options & options,
              run_result & output) {
    if (!loaded) {
        std::fprintf(stderr, "%s: load failed: %s\n", label, loaded.error().message.c_str());
        return false;
    }
    output.load_s = load_s;
    const auto start = clock_type::now();
    auto rigged = loaded->rig(source, options);
    output.rig_s = seconds_since(start);
    if (!rigged) {
        std::fprintf(stderr, "%s: rig failed: %s\n", label, rigged.error().message.c_str());
        return false;
    }
    output.binding = std::move(*rigged);
    std::printf("%s: backend \"%.*s\", load %.2f s, rig %.2f s, %zu tokens, %zu joints\n", label,
                static_cast<int>(loaded->backend_name().size()), loaded->backend_name().data(),
                output.load_s, output.rig_s, output.binding.tokens.size(), output.binding.rig.names.size());
    std::fflush(stdout);
    return true;
}

template<class T> bool same_bytes(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

std::size_t common_prefix(const std::vector<std::int32_t> & a, const std::vector<std::int32_t> & b) {
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
    return i;
}

template<class T> bool write_raw(const std::string & path, const std::vector<T> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    return static_cast<bool>(output);
}

} // namespace

int main(int argc, char ** argv) {
    const std::string root = IDO_ROOT_DIR;
    const std::string bundle = argc > 1 ? argv[1] : env_or("SKINTOKENS_GGUF_DIR", root + "/models/SkinTokens-GGUF/F16");
    const std::string mesh_path = argc > 2 ? argv[2] : env_or("SKINTOKENS_MESH", root + "/project/fixtures/foxgirl/avatar.obj");
    const std::string oracle = argc > 3 ? argv[3] : env_or("SKINTOKENS_ORACLE_DIR", "skintokens-oracle");
    const bool control = env_or("SKINTOKENS_READER_CONTROL", "1") != "0";
    const auto threads = static_cast<std::uint32_t>(std::stoul(env_or("SKINTOKENS_THREADS", "8")));

    {
        std::ifstream probe(bundle + "/tokenrig.gguf", std::ios::binary);
        if (!probe) {
            std::printf("SKIP: no SkinTokens bundle at %s\n", bundle.c_str());
            return 77;
        }
    }
    skintokens::mesh source;
    if (!load_obj(mesh_path, source)) {
        std::fprintf(stderr, "cannot read mesh %s\n", mesh_path.c_str());
        return 1;
    }
    std::printf("mesh %s: %zu vertices, %zu triangles; bundle %s; %u threads\n", mesh_path.c_str(),
                source.vertices.size(), source.faces.size(), bundle.c_str(), threads);

    skintokens::runtime_options runtime;
    runtime.device = skintokens::device_kind::cpu;
    runtime.threads = threads;
    skintokens::generation_options greedy;
    greedy.seed = 0;
    greedy.top_k = 1;
    greedy.top_p = 1.0F;
    greedy.temperature = 0.0F;
    greedy.beams = 1;

    // 1. Unpatched: the directory path.
    run_result unpatched;
    {
        const auto start = clock_type::now();
        auto loaded = skintokens::model::load(bundle, runtime);
        if (!rig_with("unpatched", std::move(loaded), seconds_since(start), source, greedy, unpatched)) return 1;
    }
    // 2. The reader path.
    run_result patched;
    std::uint64_t file_bytes = 0;
    counting_reader reader{skintokens::make_file_bundle_reader(bundle)};
    {
        for (const char * name : {"mesh-encoder.gguf", "tokenrig.gguf", "skin-vae.gguf"}) file_bytes += reader.size(name);
        const auto start = clock_type::now();
        auto loaded = skintokens::model::load(reader, runtime);
        if (!rig_with("reader", std::move(loaded), seconds_since(start), source, greedy, patched)) return 1;
    }
    std::printf("reader: %zu exists, %zu reads (%.1f MiB), %zu uploads (%.1f MiB) of %.1f MiB on disk\n",
                reader.exists_calls, reader.read_calls, static_cast<double>(reader.read_bytes) / 1048576.0,
                reader.upload_calls, static_cast<double>(reader.upload_bytes) / 1048576.0,
                static_cast<double>(file_bytes) / 1048576.0);

    const auto & a = unpatched.binding;
    const auto & b = patched.binding;
    const std::size_t prefix = common_prefix(a.tokens, b.tokens);
    const bool tokens_120 = a.tokens.size() >= 120U && prefix >= 120U;
    const bool tokens_all = a.tokens == b.tokens;
    const bool skeleton_same = a.rig.names == b.rig.names && a.rig.parents == b.rig.parents &&
                               same_bytes(a.rig.rest_positions, b.rig.rest_positions);
    const bool joints_same = same_bytes(a.joints, b.joints);
    const bool weights_same = same_bytes(a.weights, b.weights);
    // The reader carried every tensor: uploads cover all but the three GGUF
    // headers, and reads at least the uploads.
    const bool carried = reader.upload_bytes > file_bytes * 99U / 100U && reader.read_bytes >= reader.upload_bytes &&
                         reader.upload_calls > 0U;
    double worst_sum = 0.0;
    for (const auto & weights : a.weights) {
        double sum = 0.0;
        for (const float value : weights) sum += value;
        worst_sum = std::max(worst_sum, std::abs(sum - 1.0));
    }
    std::printf("G tokens: %zu vs %zu, common prefix %zu -> first-120 %s, all %s\n", a.tokens.size(),
                b.tokens.size(), prefix, tokens_120 ? "IDENTICAL" : "DIFFER", tokens_all ? "IDENTICAL" : "DIFFER");
    std::printf("G rig: skeleton %s (%zu joints), JOINTS_0 %s, WEIGHTS_0 %s (bitwise); max |sum w - 1| = %.3g\n",
                skeleton_same ? "IDENTICAL" : "DIFFER", a.rig.names.size(), joints_same ? "IDENTICAL" : "DIFFER",
                weights_same ? "IDENTICAL" : "DIFFER", worst_sum);
    std::printf("G carried: %s\n", carried ? "yes" : "NO");

    // The G4b oracle: the unpatched host run.
    {
        std::vector<float> rest;
        for (const auto & value : a.rig.rest_positions) { rest.push_back(value.x); rest.push_back(value.y); rest.push_back(value.z); }
        std::vector<std::uint16_t> joints;
        for (const auto & value : a.joints) joints.insert(joints.end(), value.begin(), value.end());
        std::vector<float> weights;
        for (const auto & value : a.weights) weights.insert(weights.end(), value.begin(), value.end());
        std::ofstream names(oracle + "/names.txt", std::ios::trunc);
        for (const auto & name : a.rig.names) names << name << '\n';
        std::ofstream meta(oracle + "/oracle.txt", std::ios::trunc);
        meta << "mesh=" << mesh_path << "\nvertices=" << source.vertices.size() << "\ntriangles=" << source.faces.size()
             << "\nbundle=" << bundle << "\nbackend=ggml-cpu threads=" << threads
             << "\ngeneration=seed 0, beams 1, top_k 1, top_p 1, temperature 0, repetition_penalty "
             << greedy.repetition_penalty << ", max_tokens " << greedy.max_tokens
             << "\ntokens=" << a.tokens.size() << " (tokens.i32: skeleton tokens from the start pair, then skin codes + 267)"
             << "\njoints=" << a.rig.names.size() << " (names.txt, parents.i32, rest.f32 [J,3] Y-up mesh frame)"
             << "\ninfluences=" << a.joints.size() << " x 4 (joints.u16, weights.f32)"
             << "\nmax_abs_weight_sum_minus_1=" << worst_sum << "\nload_s=" << unpatched.load_s
             << "\nrig_s=" << unpatched.rig_s << '\n';
        const bool written = write_raw(oracle + "/tokens.i32", a.tokens) && write_raw(oracle + "/parents.i32", a.rig.parents) &&
                             write_raw(oracle + "/rest.f32", rest) && write_raw(oracle + "/joints.u16", joints) &&
                             write_raw(oracle + "/weights.f32", weights) && static_cast<bool>(names) && static_cast<bool>(meta);
        std::printf("oracle -> %s: %s\n", oracle.c_str(), written ? "written" : "WRITE FAILED (does the directory exist?)");
        if (!written) return 1;
    }

    bool caught = true;
    if (control) {
        // Control: the same reader, one tensor corrupted on upload (the
        // final norm, negated: the tokens themselves must move). The gate
        // must see it. Its token budget is the good run's plus 64, so a
        // runaway stream ends as a failed rig rather than 2048 steps.
        counting_reader bad{skintokens::make_file_bundle_reader(bundle), "llm.norm.w"};
        auto bounded = greedy;
        bounded.max_tokens = std::max<std::size_t>(a.tokens.size() + 64U, 8U);
        run_result corrupted;
        const auto start = clock_type::now();
        auto loaded = skintokens::model::load(bad, runtime);
        const bool ran = rig_with("control", std::move(loaded), seconds_since(start), source, bounded, corrupted);
        const auto & c = corrupted.binding;
        caught = !bad.corrupted ? false : !ran || c.tokens != a.tokens || !same_bytes(c.weights, a.weights);
        std::printf("G control (llm.norm.w negated on upload: %s): %s%s\n", bad.corrupted ? "yes" : "NO",
                    caught ? "CAUGHT" : "NOT CAUGHT",
                    ran ? (" -- common token prefix " + std::to_string(common_prefix(a.tokens, c.tokens))).c_str() : " -- rig failed");
    }

    const bool pass = tokens_120 && tokens_all && skeleton_same && joints_same && weights_same && carried && caught;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
