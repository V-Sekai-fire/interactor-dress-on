// Host-native twin of probes.elf's ggml_probe: the same guest/probes/
// ggml_probe.cpp, built with llvm-mingw against the same ggml tree.
#include "ggml_probe.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
	const int n = argc > 1 ? std::atoi(argv[1]) : 256;
	std::printf("%s\n", ggml_probe_run(n).c_str());
	return 0;
}
