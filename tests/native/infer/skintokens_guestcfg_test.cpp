// skin-tokens built with SKINTOKENS_GUEST, run on the host: the guest's load
// contract, before there is a guest. The library has no file reader in this
// configuration, so this test brings its own (a stand-in for the guest's
// READ/UPLOAD requests), and checks:
//   1. model::load(path) refuses (the guest has no filesystem);
//   2. a reader missing a component is refused (exists() is asked first);
//   3. automatic and rd with no "RD" registry fail as backend_unavailable --
//      after all three GGUF manifests were accepted -- with no CPU fallback;
//   4. an explicit cpu loads, every tensor byte through upload().
//
//   skintokens-guestcfg-test BUNDLE_DIR      (exit 77 without the weights)
#include <skintokens/skintokens.hpp>

#include <cstdio>
#include <string>
#include <unordered_map>

#if !defined(SKINTOKENS_GUEST)
#error "build against skintokens_guestcfg (SKINTOKENS_GUEST)"
#endif

namespace {

class stdio_reader final : public skintokens::bundle_reader {
public:
    explicit stdio_reader(std::string directory, std::string hide = {})
        : directory_(std::move(directory)), hide_(std::move(hide)) {}
    ~stdio_reader() override {
        for (auto & [name, file] : files_) if (file != nullptr) std::fclose(file);
    }
    bool exists(std::string_view component) override { return file(component) != nullptr; }
    std::uint64_t size(std::string_view component) override {
        auto * f = file(component);
        if (f == nullptr || _fseeki64(f, 0, SEEK_END) != 0) return 0U;
        const long long end = _ftelli64(f);
        return end < 0 ? 0U : static_cast<std::uint64_t>(end);
    }
    std::size_t read(std::string_view component, void * output, std::uint64_t offset, std::size_t bytes) override {
        auto * f = file(component);
        if (f == nullptr || _fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) return 0U;
        const std::size_t count = std::fread(output, 1U, bytes, f);
        read_bytes += count;
        return count;
    }
    bool upload(const skintokens::tensor_upload & tensor) override {
        uploaded += tensor.bytes;
        ++uploads;
        return bundle_reader::upload(tensor);
    }
    std::uint64_t read_bytes = 0, uploaded = 0, uploads = 0;

private:
    std::FILE * file(std::string_view component) {
        const std::string name{component};
        if (name == hide_) return nullptr;
        auto & slot = files_[name];
        if (slot == nullptr) slot = std::fopen((directory_ + "/" + name).c_str(), "rb");
        return slot;
    }
    std::string directory_, hide_;
    std::unordered_map<std::string, std::FILE *> files_;
};

int failures = 0;
void check(bool ok, const char * what, const std::string & detail = {}) {
    std::printf("%s: %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) ++failures;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    const std::string bundle = argv[1];
    {
        stdio_reader probe{bundle};
        if (!probe.exists("tokenrig.gguf")) {
            std::printf("SKIP: no SkinTokens bundle at %s\n", bundle.c_str());
            return 77;
        }
    }
    skintokens::runtime_options cpu;
    cpu.device = skintokens::device_kind::cpu;
    cpu.threads = 8;  // the guest configuration must ignore this (1 thread)

    {
        auto loaded = skintokens::model::load(std::filesystem::path{bundle}, cpu);
        check(!loaded && loaded.error().code == skintokens::error_code::io, "load(path) refused",
              loaded ? "loaded" : loaded.error().message);
    }
    {
        stdio_reader reader{bundle, "skin-vae.gguf"};
        auto loaded = skintokens::model::load(reader, cpu);
        check(!loaded && loaded.error().code == skintokens::error_code::io && reader.read_bytes == 0U,
              "missing component refused before any read", loaded ? "loaded" : loaded.error().message);
    }
    for (const auto device : {skintokens::device_kind::automatic, skintokens::device_kind::rd}) {
        stdio_reader reader{bundle};
        skintokens::runtime_options options;
        options.device = device;
        auto loaded = skintokens::model::load(reader, options);
        check(!loaded && loaded.error().code == skintokens::error_code::backend_unavailable &&
                  reader.read_bytes > 0U && reader.uploads == 0U,
              device == skintokens::device_kind::rd ? "rd without an RD registry: backend_unavailable"
                                                    : "automatic without RD: backend_unavailable, no CPU fallback",
              (loaded ? std::string{"loaded on "} + std::string{loaded->backend_name()} : loaded.error().message) +
                  ", manifest bytes read " + std::to_string(reader.read_bytes));
    }
    {
        stdio_reader reader{bundle};
        const std::uint64_t on_disk = reader.size("mesh-encoder.gguf") + reader.size("tokenrig.gguf") +
                                      reader.size("skin-vae.gguf");
        auto loaded = skintokens::model::load(reader, cpu);
        check(static_cast<bool>(loaded) && reader.uploaded > on_disk * 99U / 100U && reader.uploaded <= on_disk,
              "cpu loads through upload()",
              loaded ? "backend \"" + std::string{loaded->backend_name()} + "\", " + std::to_string(reader.uploads) +
                           " tensors, " + std::to_string(reader.uploaded) + " of " + std::to_string(on_disk) + " bytes"
                     : loaded.error().message);
    }
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
