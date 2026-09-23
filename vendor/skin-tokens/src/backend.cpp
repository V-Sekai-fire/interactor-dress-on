#include "internal.hpp"

#include <algorithm>
#if !defined(SKINTOKENS_GUEST)
#include <mutex>
#include <set>
#include <thread>
#endif

namespace skintokens::detail {

backend_handle::~backend_handle() {
    if (value != nullptr) ggml_backend_free(value);
}

namespace {

#if defined(SKINTOKENS_GUEST)
// interactor-dress-on: the guest ELF links its backends in (ggml-cpu, and
// ggml-rd registered by the guest before load); there is no directory to
// scan and nothing to dlopen, so ggml_backend_load_all is never called.
void load_backends(const std::filesystem::path &) {}
#else
void load_backends(const std::filesystem::path & directory) {
    static std::mutex mutex;
    static std::set<std::string> loaded;
    const std::string key = directory.empty() ? std::string{} : directory.lexically_normal().string();
    std::scoped_lock lock{mutex};
    if (!loaded.insert(key).second) return;
    if (directory.empty()) ggml_backend_load_all();
    else ggml_backend_load_all_from_path(directory.string().c_str());
}
#endif

bool registry_is(ggml_backend_dev_t device, std::string_view wanted) {
    auto * registry = ggml_backend_dev_backend_reg(device);
    const char * name = registry == nullptr ? nullptr : ggml_backend_reg_name(registry);
    return name != nullptr && std::string_view{name} == wanted;
}

bool matches(device_kind wanted, ggml_backend_dev_t device) {
    const auto type = ggml_backend_dev_type(device);
#if defined(SKINTOKENS_GUEST)
    // The guest's GPU is Godot's RenderingDevice, reached only through the
    // registry named "RD" (ggml-rd); CPU stays selectable for host tests of
    // the guest configuration. Nothing else is accepted.
    if (wanted == device_kind::automatic || wanted == device_kind::rd) return registry_is(device, "RD");
    if (wanted == device_kind::cpu) return type == GGML_BACKEND_DEVICE_TYPE_CPU;
    return false;
#else
    if (wanted == device_kind::automatic) return type == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (wanted == device_kind::cpu) return type == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (wanted == device_kind::rd) return registry_is(device, "RD");
    return registry_is(device, "Vulkan");
#endif
}

} // namespace

result<std::unique_ptr<backend_handle>> make_backend(const runtime_options & options) {
    load_backends(options.backend_directory);
    ggml_backend_dev_t selected = nullptr;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (matches(options.device, candidate)) {
            selected = candidate;
            break;
        }
    }
#if defined(SKINTOKENS_GUEST)
    // No silent CPU fallback in the guest: without RD, automatic fails (the
    // guest runs inference on ggml-rd only; cpu must be asked for by name).
    constexpr bool fall_back_to_cpu = false;
#else
    constexpr bool fall_back_to_cpu = true;
#endif
    if (fall_back_to_cpu && selected == nullptr && options.device == device_kind::automatic) {
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * candidate = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                selected = candidate;
                break;
            }
        }
    }
    if (selected == nullptr) {
        return std::unexpected(fail(error_code::backend_unavailable,
            "requested GGML backend is unavailable"));
    }
    auto output = std::make_unique<backend_handle>();
    output->value = ggml_backend_dev_init(selected, nullptr);
    if (output->value == nullptr) {
        return std::unexpected(fail(error_code::backend_unavailable,
            "GGML failed to initialize the selected backend"));
    }
    const char * description = ggml_backend_dev_description(selected);
    output->name = description == nullptr ? ggml_backend_dev_name(selected) : description;
    if (ggml_backend_dev_type(selected) == GGML_BACKEND_DEVICE_TYPE_CPU) {
#if defined(SKINTOKENS_GUEST)
        // One hart, no thread pool in the sandbox.
        const std::uint32_t requested = 1U;
#else
        const std::uint32_t requested = options.threads == 0
            ? std::max(1U, std::thread::hardware_concurrency()) : options.threads;
#endif
        auto * registry = ggml_backend_dev_backend_reg(selected);
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
        if (set_threads != nullptr) {
            set_threads(output->value, static_cast<int>(std::min<std::uint32_t>(
                requested, static_cast<std::uint32_t>(std::numeric_limits<int>::max()))));
        }
    }
    return output;
}

} // namespace skintokens::detail
