// resolve_specs: build-time tool for the specs fit.elf embeds.
//
// cloth-fit's init() (optimize.cpp:1379-1399) reads json-specs/input-spec.json
// and expands its jse "include" rules from files at run time; polysolve's
// Solver::create reads nonlinear-/linear-solver-spec.json by absolute
// build-machine path. The guest has no files, so this tool does the reads on
// the build machine instead: it resolves input-spec.json's includes against
// the same directories init() searches (cloth-fit json-specs, then the
// polysolve source dir) and re-dumps the two polysolve specs, all as
// json.dump(2) + "\n" (deterministic: nlohmann sorts keys).
//
//   resolve_specs <cloth-fit json-specs dir> <polysolve dir> <out dir>
//       writes <out>/input-spec.resolved.json, nonlinear-solver-spec.json,
//       linear-solver-spec.json
//   resolve_specs <json-specs dir> <polysolve dir> <out dir> --check <committed dir>
//       also compares each output with the committed copy byte for byte;
//       exit 1 names the stale file (regenerate into guest/fit/specs).
//
// apply_default_solver (which depends on the solvers compiled in) is left to
// the driver at run time, as init() does it after the include step.
#include <jse/jse.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using json = nlohmann::json;

static bool read_text(const std::string &path, std::string *out) {
	std::ifstream f(path, std::ios::binary);
	if (!f)
		return false;
	std::stringstream ss;
	ss << f.rdbuf();
	*out = ss.str();
	return true;
}

static bool write_text(const std::string &path, const std::string &text) {
	std::ofstream f(path, std::ios::binary);
	f << text;
	return bool(f);
}

int main(int argc, char **argv) {
	if (argc != 4 && !(argc == 6 && std::string(argv[4]) == "--check")) {
		std::fprintf(stderr, "usage: resolve_specs <json-specs dir> <polysolve dir> <out dir> [--check <committed dir>]\n");
		return 2;
	}
	const std::string specs = argv[1], polysolve = argv[2], out = argv[3];
	const std::string committed = argc == 6 ? argv[5] : "";

	struct Item {
		std::string name;
		std::string text;
	};
	Item items[3];
	try {
		std::string src;
		if (!read_text(specs + "/input-spec.json", &src)) {
			std::fprintf(stderr, "resolve_specs: cannot read %s/input-spec.json\n", specs.c_str());
			return 1;
		}
		jse::JSE jse;
		jse.include_directories.push_back(specs);
		jse.include_directories.push_back(polysolve);
		const json rules = jse.inject_include(json::parse(src));
		items[0] = {"input-spec.resolved.json", rules.dump(2) + "\n"};

		const char *ps[2] = {"nonlinear-solver-spec.json", "linear-solver-spec.json"};
		for (int i = 0; i < 2; i++) {
			if (!read_text(polysolve + "/" + ps[i], &src)) {
				std::fprintf(stderr, "resolve_specs: cannot read %s/%s\n", polysolve.c_str(), ps[i]);
				return 1;
			}
			items[1 + i] = {ps[i], json::parse(src).dump(2) + "\n"};
		}
	} catch (const std::exception &e) {
		std::fprintf(stderr, "resolve_specs: %s\n", e.what());
		return 1;
	}

	int stale = 0;
	for (const Item &it : items) {
		if (!write_text(out + "/" + it.name, it.text)) {
			std::fprintf(stderr, "resolve_specs: cannot write %s/%s\n", out.c_str(), it.name.c_str());
			return 1;
		}
		if (!committed.empty()) {
			std::string have;
			if (!read_text(committed + "/" + it.name, &have) || have != it.text) {
				std::fprintf(stderr, "resolve_specs: %s/%s is stale; copy %s/%s over it\n",
							 committed.c_str(), it.name.c_str(), out.c_str(), it.name.c_str());
				++stale;
			}
		}
		std::printf("resolve_specs: %s (%zu bytes)\n", it.name.c_str(), it.text.size());
	}
	return stale ? 1 : 0;
}
