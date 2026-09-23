#include "core/object/object.h"

#include "core/object/worker_thread_pool.h"
#include "core/os/os.h"
#include "core/os/time.h"

#include <chrono>
#include <cstdlib>

namespace gdl {

Variant Object::callp(const StringName &p_method, const Variant **, int, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	ERR_FAIL_V_MSG(Variant(), "Method \"" + p_method + "\" is not callable by name on " + get_class() + " (godot-lite has no ClassDB; use callable_mp).");
}

WorkerThreadPool *WorkerThreadPool::get_singleton() {
	static WorkerThreadPool singleton;
	return &singleton;
}

OS *OS::get_singleton() {
	static OS singleton;
	return &singleton;
}

bool OS::has_environment(const String &p_var) const {
	return getenv(p_var.get_data()) != nullptr;
}

String OS::get_environment(const String &p_var) const {
	const char *v = getenv(p_var.get_data());
	return v ? String(v) : String();
}

Time *Time::get_singleton() {
	static Time singleton;
	return &singleton;
}

uint64_t Time::get_ticks_usec() const {
	static const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

} // namespace gdl
