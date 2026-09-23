// cassie_checks: Gate 4's checks (guest/curvenet/checks.cpp) natively -- the
// flat control for curvenet.elf. Same TUs, same strict FP; std types only
// here, so no godot-lite prelude.
//
//   cassie_checks            every check, then "checks: <passed>/<total>"
//   cassie_checks <name>...  the named checks
#include "curvenet_api.h"

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
	const auto t0 = std::chrono::steady_clock::now();
	int fail = 0;
	if (argc > 1) {
		for (int i = 1; i < argc; ++i) {
			const std::string line = cn::check(argv[i]);
			fail += line.rfind("PASS ", 0) == 0 ? 0 : 1;
			std::printf("%s\n", line.c_str());
		}
	} else {
		const std::string all = cn::check_all();
		std::printf("%s\n", all.c_str());
		fail = all.find("\nFAIL") != std::string::npos || all.rfind("FAIL", 0) == 0 ? 1 : 0;
	}
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::fprintf(stderr, "cassie_checks: %.1f ms\n", ms);
	return fail == 0 ? 0 : 1;
}
