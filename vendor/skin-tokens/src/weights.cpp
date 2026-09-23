#include "internal.hpp"

#include <algorithm>
#include <fstream>

namespace skintokens {

// ── interactor-dress-on: the reader path ───────────────────────────────────
bundle_reader::~bundle_reader() = default;

bool bundle_reader::upload(const tensor_upload & value) {
    auto * tensor = static_cast<ggml_tensor *>(value.tensor);
    if (tensor == nullptr) return false;
    std::vector<char> scratch(std::min<std::size_t>(value.bytes, 8U << 20U));
    for (std::size_t done = 0; done < value.bytes;) {
        const std::size_t amount = std::min(scratch.size(), value.bytes - done);
        if (read(value.component, scratch.data(), value.offset + done, amount) != amount) return false;
        ggml_backend_tensor_set(tensor, scratch.data(), done, amount);
        done += amount;
    }
    return true;
}

#if !defined(SKINTOKENS_GUEST)
namespace {

// One FILE* per component, opened on first use; the last offset is kept so
// sequential reads (the GGUF header, then each tensor in order) never seek.
class file_bundle_reader final : public bundle_reader {
public:
    explicit file_bundle_reader(std::filesystem::path directory) : directory_(std::move(directory)) {}
    file_bundle_reader(const file_bundle_reader &) = delete;
    file_bundle_reader & operator=(const file_bundle_reader &) = delete;
    ~file_bundle_reader() override {
        for (auto & [name, value] : files_)
            if (value.file != nullptr) std::fclose(value.file);
    }

    bool exists(std::string_view component) override { return open(component) != nullptr; }

    std::uint64_t size(std::string_view component) override {
        auto * value = open(component);
        return value == nullptr ? 0U : value->size;
    }

    std::size_t read(std::string_view component, void * output, std::uint64_t offset,
                     std::size_t bytes) override {
        auto * value = open(component);
        if (value == nullptr || bytes == 0U || offset >= value->size) return 0U;
        if (value->position != offset) {
            if (!seek(value->file, offset)) return 0U;
            value->position = offset;
        }
        const std::size_t count = std::fread(output, 1U, bytes, value->file);
        value->position += count;
        return count;
    }

private:
    struct open_file {
        std::FILE * file = nullptr;
        std::uint64_t size = 0U;
        std::uint64_t position = 0U;
    };

    static bool seek(std::FILE * file, std::uint64_t offset) {
#if defined(_WIN32)
        return _fseeki64(file, static_cast<long long>(offset), SEEK_SET) == 0;
#else
        return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
    }

    open_file * open(std::string_view component) {
        // A component is a bare file name; nothing may escape the directory.
        if (component.empty() || component.find_first_of("/\\:") != std::string_view::npos ||
            component == "." || component == "..")
            return nullptr;
        const std::string key{component};
        if (auto found = files_.find(key); found != files_.end())
            return found->second.file == nullptr ? nullptr : &found->second;
        open_file value;
        const auto path = directory_ / key;
        value.file = ggml_fopen(path.string().c_str(), "rb");
        if (value.file != nullptr) {
            if (!seek_end(value.file, value.size)) {
                std::fclose(value.file);
                value.file = nullptr;
            } else {
                value.position = value.size;
            }
        }
        auto & stored = files_[key] = value;
        return stored.file == nullptr ? nullptr : &stored;
    }

    static bool seek_end(std::FILE * file, std::uint64_t & size) {
#if defined(_WIN32)
        if (_fseeki64(file, 0, SEEK_END) != 0) return false;
        const long long end = _ftelli64(file);
#else
        if (fseeko(file, 0, SEEK_END) != 0) return false;
        const off_t end = ftello(file);
#endif
        if (end < 0) return false;
        size = static_cast<std::uint64_t>(end);
        return true;
    }

    std::filesystem::path directory_;
    std::unordered_map<std::string, open_file> files_;
};

} // namespace

std::unique_ptr<bundle_reader> make_file_bundle_reader(const std::filesystem::path & directory) {
    return std::make_unique<file_bundle_reader>(directory);
}
#endif // !SKINTOKENS_GUEST

} // namespace skintokens

namespace skintokens::detail {

weight_component::~weight_component() {
    if (buffer != nullptr) ggml_backend_buffer_free(buffer);
    if (file != nullptr) gguf_free(file);
    if (context != nullptr) ggml_free(context);
}

ggml_tensor * weight_component::tensor(std::string_view name) const {
    return context == nullptr ? nullptr : ggml_get_tensor(context, std::string{name}.c_str());
}

namespace {

struct gguf_source {
    bundle_reader * reader;
    std::string_view component;
};

std::size_t gguf_source_read(void * user, void * output, std::uint64_t offset, std::size_t bytes) {
    auto & source = *static_cast<gguf_source *>(user);
    return source.reader->read(source.component, output, offset, bytes);
}

} // namespace

result<std::unique_ptr<weight_component>> open_component(
    bundle_reader & reader, std::string_view component, bundle_metadata & metadata) {
    const std::string name{component};
    if (!reader.exists(component))
        return std::unexpected(fail(error_code::io, "cannot open GGUF component: " + name));
    const std::uint64_t total = reader.size(component);
    if (total == 0U) return std::unexpected(fail(error_code::io, "empty GGUF component: " + name));
    auto output = std::make_unique<weight_component>();
    output->path = name;
    gguf_source source{&reader, component};
    gguf_init_params parameters{true, &output->context};
    // 1 MiB per callback bounds a guest READ request; the header is far smaller.
    output->file = gguf_init_from_callback(gguf_source_read, &source, 1U << 20U, total, parameters);
    if (output->file == nullptr || output->context == nullptr)
        return std::unexpected(fail(error_code::invalid_format, "truncated GGUF header: " + name));
    auto inspected = inspect_gguf(output->file);
    if (!inspected) return std::unexpected(inspected.error());
    metadata = std::move(*inspected);
    return output;
}

result<void> upload_component(bundle_reader & reader, std::string_view component,
                              weight_component & weights, ggml_backend_t backend) {
    const std::string name{component};
    weights.buffer = ggml_backend_alloc_ctx_tensors(weights.context, backend);
    if (weights.buffer == nullptr)
        return std::unexpected(fail(error_code::allocation, "cannot allocate model component on backend: " + name));
    const std::uint64_t total = reader.size(component);
    const std::size_t data_offset = gguf_get_data_offset(weights.file);
    for (std::int64_t i = 0; i < gguf_get_n_tensors(weights.file); ++i) {
        const char * tensor_name = gguf_get_tensor_name(weights.file, i);
        auto * tensor = ggml_get_tensor(weights.context, tensor_name);
        if (tensor == nullptr || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16))
            return std::unexpected(fail(error_code::incompatible_model, "GGUF component has unsupported tensor type"));
        tensor_upload value;
        value.component = component;
        value.name = tensor_name;
        value.offset = static_cast<std::uint64_t>(data_offset) + gguf_get_tensor_offset(weights.file, i);
        value.bytes = ggml_nbytes(tensor);
        value.tensor = tensor;
        if (value.offset > total || value.bytes > total - value.offset || !reader.upload(value))
            return std::unexpected(fail(error_code::invalid_format, "GGUF tensor data is truncated"));
    }
    return {};
}

#if !defined(SKINTOKENS_GUEST)
result<std::unique_ptr<weight_component>> load_component(
    const std::filesystem::path & path, ggml_backend_t backend) {
    auto output = std::make_unique<weight_component>();
    output->path = path;
    gguf_init_params parameters{true, &output->context};
    output->file = gguf_init_from_file(path.string().c_str(), parameters);
    if (output->file == nullptr || output->context == nullptr)
        return std::unexpected(fail(error_code::invalid_format, "cannot map GGUF tensors: " + path.string()));
    output->buffer = ggml_backend_alloc_ctx_tensors(output->context, backend);
    if (output->buffer == nullptr)
        return std::unexpected(fail(error_code::allocation, "cannot allocate model component on backend: " + path.string()));

    std::ifstream stream(path, std::ios::binary);
    if (!stream) return std::unexpected(fail(error_code::io, "cannot reopen GGUF component"));
    const std::size_t data_offset = gguf_get_data_offset(output->file);
    std::vector<char> scratch(8U << 20U);
    for (std::int64_t i = 0; i < gguf_get_n_tensors(output->file); ++i) {
        const char * name = gguf_get_tensor_name(output->file, i);
        auto * tensor = ggml_get_tensor(output->context, name);
        if (tensor == nullptr || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16))
            return std::unexpected(fail(error_code::incompatible_model, "GGUF component has unsupported tensor type"));
        const std::size_t bytes = ggml_nbytes(tensor);
        const std::size_t offset = gguf_get_tensor_offset(output->file, i);
        stream.seekg(static_cast<std::streamoff>(data_offset + offset));
        for (std::size_t done = 0; done < bytes;) {
            const std::size_t amount = std::min(scratch.size(), bytes - done);
            stream.read(scratch.data(), static_cast<std::streamsize>(amount));
            if (!stream) return std::unexpected(fail(error_code::invalid_format, "GGUF tensor data is truncated"));
            ggml_backend_tensor_set(tensor, scratch.data(), done, amount);
            done += amount;
        }
    }
    return output;
}
#else
result<std::unique_ptr<weight_component>> load_component(const std::filesystem::path &, ggml_backend_t) {
    return std::unexpected(fail(error_code::io, "the guest has no filesystem; load through a bundle_reader"));
}
#endif // !SKINTOKENS_GUEST

} // namespace skintokens::detail
