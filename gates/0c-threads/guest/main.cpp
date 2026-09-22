// GATE 0C — does a godot-sandbox guest actually run std::thread?
//
// libpthread.a is in the riscv64 sysroot, so thread code LINKS. Whether
// libriscv executes it -- concurrently, sequentially, or not at all -- is a
// separate question, and it decides two things at once: whether Geogram's
// parallel Delaunay (PDEL, which uses GEO::Thread over pthreads) can be used,
// and whether the sub-island parallel drape idea has anywhere to run.
//
// Reports: how many threads observably ran, and whether any two overlapped.
#include <api.hpp>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

static Variant thread_probe(int n) {
	std::atomic<int> started{0}, finished{0}, max_live{0}, live{0};
	std::vector<std::thread> pool;
	bool spawn_ok = true;
	std::string err;
	try {
		for (int i = 0; i < n; ++i) {
			pool.emplace_back([&] {
				++started;
				int l = ++live;
				int m = max_live.load();
				while (l > m && !max_live.compare_exchange_weak(m, l)) {}
				// busy-wait so overlapping threads are observable
				auto t0 = std::chrono::steady_clock::now();
				while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(20)) {}
				--live;
				++finished;
			});
		}
	} catch (const std::exception &e) {
		spawn_ok = false; err = e.what();
	} catch (...) {
		spawn_ok = false; err = "unknown exception";
	}
	for (auto &t : pool) if (t.joinable()) t.join();

	std::string r = spawn_ok ? "spawn=ok " : ("spawn=FAIL(" + err + ") ");
	r += "hw=" + std::to_string(std::thread::hardware_concurrency());
	r += " started=" + std::to_string(started.load());
	r += " finished=" + std::to_string(finished.load());
	r += " max_live=" + std::to_string(max_live.load());
	if (finished.load() == n && max_live.load() > 1)      r = "PASS-CONCURRENT " + r;
	else if (finished.load() == n)                        r = "PASS-SEQUENTIAL " + r;
	else                                                  r = "FAIL " + r;
	return Variant(String(r));
}

int main() {
	ADD_API_FUNCTION(thread_probe, "String", "int n", "Spawn n std::threads and report what ran");
	halt();
}
