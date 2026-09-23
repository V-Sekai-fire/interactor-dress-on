// fit_clock: an instruction clock for fit.elf's profiling build
// (FIT_INSTRET_CLOCK=ON, e.g. project/fit_prof.elf; gates/6-fit/budget).
//
// The guest clock is not a clock (AGENTS.md: it jumps between time bases), so
// polysolve's stopwatches print negative seconds in fit.elf. Linked with
// --wrap=clock_gettime, every clock the ELF reads (std::chrono's steady and
// system clocks, spdlog's timestamps) returns libriscv's retired-instruction
// counter as nanoseconds: 1 "second" = 1e9 instructions. polysolve's
// [timing] lines then split a solve's instructions into assembly, linear
// solve, line search, broad- and narrow-phase CCD, and each log line's
// timestamp is the instruction count at that line. Nothing in the solve reads
// a clock to decide anything, so the numerics are fit.elf's (the budget's
// profile run is held to the loop's fitted garment bit for bit).
#include <cstdint>
#include <ctime>

extern "C" int __wrap_clock_gettime(clockid_t, struct timespec *ts) {
	uint64_t n;
	asm volatile("rdinstret %0" : "=r"(n));
	if (ts) {
		ts->tv_sec = time_t(n / 1000000000ull);
		ts->tv_nsec = long(n % 1000000000ull);
	}
	return 0;
}
