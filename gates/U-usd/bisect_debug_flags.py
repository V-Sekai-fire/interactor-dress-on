"""Gate U finding 1, resume step 1: adds usd_debug(flags) to the guest, the stage,
main.gd and probe_usd_push.gd (README.md). Run once from the repo root, then rebuild usd.elf."""
import os
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repo root

p = W + "/guest/usd/usd_core.h"
s = open(p, encoding="utf-8").read()
s = s.replace("std::string init();\n", "std::string init();\n// Bisecting flags (gates/U-usd): 1 = on an open error keep the document's asset\n// (no close), 2 = never erase assets on close, 4 = clear() the tables without\n// giving the memory back, 8 = no TfErrorMark around the open.\nvoid set_debug(int flags);\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = W + "/guest/usd/usd_core.cpp"
s = open(p, encoding="utf-8").read()
s = s.replace("std::string g_pkg_path; // the current asset's path in the resolver (\"\" when closed)\n",
              "std::string g_pkg_path; // the current asset's path in the resolver (\"\" when closed)\nint g_debug = 0;\n")
s = s.replace('''	// Give the memory back to the guest heap: the next document may be bigger
	// and the heap's ceiling is the whole sandbox.
	std::vector<float>().swap(g_points);''', '''	// Give the memory back to the guest heap: the next document may be bigger
	// and the heap's ceiling is the whole sandbox.
	if (g_debug & 4)
		return;
	std::vector<float>().swap(g_points);''')
s = s.replace('''std::string init() {
	return usdmem::init();
}
''', '''std::string init() {
	return usdmem::init();
}

void set_debug(int flags) {
	g_debug = flags;
}
''')
# error paths: optionally keep the document (no close)
s = s.replace('''		if (!layer) {
			std::string e = "ERR: " + std::string(ext) + " open failed: " + first_error(mark);
			close();
			return e;
		}''', '''		if (!layer) {
			std::string e = "ERR: " + std::string(ext) + " open failed: " + first_error(mark);
			if (!(g_debug & 1))
				close();
			return e;
		}''')
s = s.replace('''void close() {
	if (!g_pkg_path.empty()) {
		usdmem::erase_asset(g_pkg_path);
		g_pkg_path.clear();
	}
	clear_tables();
}''', '''void close() {
	if (!g_pkg_path.empty()) {
		if (!(g_debug & 2))
			usdmem::erase_asset(g_pkg_path);
		g_pkg_path.clear();
	}
	clear_tables();
}''')
s = s.replace('''	TfErrorMark mark;
	size_t prims = 0;
	std::string warn;''', '''	std::unique_ptr<TfErrorMark> markp((g_debug & 8) ? nullptr : new TfErrorMark());
	TfErrorMark &mark = markp ? *markp : *new TfErrorMark(); // 8: a mark that is never cleared (leaked on purpose)
	size_t prims = 0;
	std::string warn;''')
s = s.replace("#include <cstdio>\n#include <cstring>\n#include <map>\n", "#include <cstdio>\n#include <cstring>\n#include <map>\n#include <memory>\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = W + "/guest/usd/main.cpp"
s = open(p, encoding="utf-8").read()
s = s.replace('''static Variant usd_open(PackedArray<uint8_t> package) {''', '''static Variant usd_debug(int flags) {
	usdg::set_debug(flags);
	return text("debug=" + std::to_string(flags));
}

static Variant usd_open(PackedArray<uint8_t> package) {''')
s = s.replace('''	ADD_API_FUNCTION(usd_open, "String", "PackedByteArray package",''', '''	ADD_API_FUNCTION(usd_debug, "String", "int flags", "Bisecting flags (gates/U-usd finding 1); 0 in use");
	ADD_API_FUNCTION(usd_open, "String", "PackedByteArray package",''')
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = W + "/project/probe_usd_push.gd"
s = open(p, encoding="utf-8").read()
s = s.replace("var _seq := PackedStringArray()\n", "var _seq := PackedStringArray()\nvar _debug := 0\n")
s = s.replace('''		elif a.begins_with("--seq="):''', '''		elif a.begins_with("--debug="):
			_debug = int(a.substr(8))
		elif a.begins_with("--seq="):''')
s = s.replace('''	if _arm == "seq":
		for f in _seq:''', '''	if _arm == "seq":
		for f in _seq:
			if _debug != 0:
				_say("debug -> %s" % str(_stage.call_now("usd_debug", [_debug])))''')
open(p, "w", encoding="utf-8", newline="\n").write(s)

# the stage: usd_debug wrapper (rule 8) and its main.gd delegate
p = W + "/project/stages/usd_stage.gd"
s = open(p, encoding="utf-8").read()
s = s.replace('''func usd_init() -> String: return init()
''', '''func usd_init() -> String: return init()
func usd_debug(flags: int = 0) -> String: return _s("usd_debug", [flags])
''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
p = W + "/project/main.gd"
s = open(p, encoding="utf-8").read()
s = s.replace('''func usd_init() -> String: return usd.usd_init()
''', '''func usd_init() -> String: return usd.usd_init()
func usd_debug(flags: int = 0) -> String: return usd.usd_debug(flags)
''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
