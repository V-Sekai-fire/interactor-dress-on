# wrapper_audit -- AGENTS.md rule 8 read from the source text: every guest
# entry point (ADD_API_FUNCTION(name, ...) or add_sandbox_api_function("name",
# ...) in guest/**/main.cpp) must be reached from a public function of
# project/main.gd whose parameters all have defaults.
#
# main.gd is a thin root: a wrapper there is a one-line delegate such as
#   func cn_get_param(name: String = "snap_radius") -> String: return curvenet.cn_get_param(name)
# so a wrapper reaches the guest name "n" when the string literal "n" is in
# its own body, or in the body of the stage function it delegates to
# (<stage>.<f>(...)), or in any function of that stage file those call, to any
# depth. A literal counts only for the stage it goes through, so drape.elf's
# rd_close (drape_rd_close) and dress_on.elf's (rd_close) are told apart.
#
# Used by tests/probe_main_wrappers.gd (every guest) and gate_curvenet.gd
# (Gate 4 check 7, curvenet only).
extends RefCounted

# Guest source -> the /root/Main member (stage node) that owns its Sandbox.
const GUESTS := {
	"guest/main.cpp": "dress_on",
	"guest/probes/main.cpp": "dress_on",
	"guest/curvenet/main.cpp": "curvenet",
	"guest/drape/main.cpp": "drape",
	"guest/fit/main.cpp": "fit",
	"guest/ggml_test/main.cpp": "ggml",
}
const STAGE_FILES := {
	"dress_on": "res://stages/dress_on_stage.gd",
	"curvenet": "res://stages/curvenet_stage.gd",
	"drape": "res://stages/drape_stage.gd",
	"fit": "res://stages/fit_stage.gd",
	"ggml": "res://stages/ggml_stage.gd",
}

static func read(path: String) -> String:
	var f := FileAccess.open(ProjectSettings.globalize_path(path), FileAccess.READ)
	if f == null:
		return ""
	var t := f.get_as_text()
	f.close()
	return t.replace("\r", "")

static func api_names(cpp: String) -> PackedStringArray:
	var out := PackedStringArray()
	var re := RegEx.new()
	re.compile("(?m)^\\s*ADD_API_FUNCTION\\(\\s*(\\w+)\\s*,")
	for m in re.search_all(cpp):
		out.append(m.get_string(1))
	re.compile("add_sandbox_api_function\\(\\s*\"(\\w+)\"")
	for m in re.search_all(cpp):
		if not out.has(m.get_string(1)):
			out.append(m.get_string(1))
	return out

# {stage member: Array of guest names} for the given guest files
# (repo-relative), read from res://../<file>.
static func api_by_stage(guests: Dictionary = GUESTS) -> Dictionary:
	var out := {}
	for g in guests:
		var names := api_names(read("res://../" + g))
		var sv: String = guests[g]
		if not out.has(sv):
			out[sv] = []
		for n in names:
			if not out[sv].has(n):
				out[sv].append(n)
	return out

# Top-level funcs of a GDScript source: {name: {name, params, body}}. A
# parameter list may span lines; a column-0 comment neither ends a body nor
# counts in one; any other column-0 line (var, const, func) ends it.
static func gd_funcs(src: String) -> Dictionary:
	var funcs := {}
	var cur = null
	var header := ""
	var depth := 0
	for line in src.split("\n"):
		if header != "":
			header += " " + line.strip_edges()
			depth += _depth(line)
			if depth <= 0:
				cur = _func_from_header(header)
				funcs[cur.name] = cur
				header = ""
			continue
		if line.begins_with("func ") or line.begins_with("static func "):
			depth = _depth(line)
			if depth > 0:
				header = line
				continue
			cur = _func_from_header(line)
			funcs[cur.name] = cur
		elif line.begins_with("#"):
			continue
		elif cur != null and (line.is_empty() or line.begins_with("\t") or line.begins_with(" ")):
			cur.body += line + "\n"
		else:
			cur = null
	return funcs

static func _depth(line: String) -> int:
	var d := 0
	var in_str := false
	for i in line.length():
		var ch := line[i]
		if ch == "\"":
			in_str = not in_str
		elif in_str:
			continue
		elif ch == "(" or ch == "[" or ch == "{":
			d += 1
		elif ch == ")" or ch == "]" or ch == "}":
			d -= 1
		elif ch == "#":
			break
	return d

static func _func_from_header(line: String) -> Dictionary:
	var at := line.find("func ") + 5
	var open := line.find("(", at)
	var name := line.substr(at, open - at).strip_edges()
	# Brackets and commas inside a string default ("-o ADD,MUL -b RD0") are
	# text, not syntax.
	var d := 0
	var close := -1
	var in_str := false
	for i in range(open, line.length()):
		var ch := line[i]
		if ch == "\"":
			in_str = not in_str
		elif in_str:
			continue
		elif ch == "(" or ch == "[" or ch == "{":
			d += 1
		elif ch == ")" or ch == "]" or ch == "}":
			d -= 1
			if d == 0:
				close = i
				break
	var params := PackedStringArray()
	var inner := line.substr(open + 1, close - open - 1)
	d = 0
	in_str = false
	var start := 0
	for i in inner.length():
		var ch := inner[i]
		if ch == "\"":
			in_str = not in_str
		elif in_str:
			continue
		elif ch == "(" or ch == "[" or ch == "{":
			d += 1
		elif ch == ")" or ch == "]" or ch == "}":
			d -= 1
		elif ch == "," and d == 0:
			params.append(inner.substr(start, i - start).strip_edges())
			start = i + 1
	if not inner.substr(start).strip_edges().is_empty():
		params.append(inner.substr(start).strip_edges())
	return {"name": name, "params": params, "body": line + "\n"}

# The body of f plus every function of the same file it calls, to any depth.
static func _closure(fname: String, funcs: Dictionary, seen: Dictionary) -> String:
	if seen.has(fname) or not funcs.has(fname):
		return ""
	seen[fname] = true
	var body: String = funcs[fname].body
	var out := body
	var re := RegEx.new()
	re.compile("\\b([A-Za-z_]\\w*)\\s*\\(")
	for m in re.search_all(body):
		var callee := m.get_string(1)
		if callee != fname and funcs.has(callee):
			out += _closure(callee, funcs, seen)
	return out

# [missing ("<stage>:<name>"), no-default ("wrapper(param)"), wrapper names].
# api: {stage member: names}; main_src: main.gd's text; stage_srcs: {member: text}.
static func audit(api: Dictionary, main_src: String, stage_srcs: Dictionary) -> Array:
	var mains := gd_funcs(main_src)
	var stages := {}
	for sv in stage_srcs:
		stages[sv] = gd_funcs(stage_srcs[sv])
	var calls := RegEx.new()
	calls.compile("\\b(%s)\\.(\\w+)\\s*\\(" % "|".join(PackedStringArray(stage_srcs.keys())))
	# reach[wrapper][stage member] = the text that wrapper reaches in that stage
	var reach := {}
	for w in mains:
		if w.begins_with("_"):
			continue
		var body: String = mains[w].body
		for m in calls.search_all(body):
			var sv := m.get_string(1)
			var r: Dictionary = reach.get(w, {})
			r[sv] = str(r.get(sv, "")) + body + _closure(m.get_string(2), stages[sv], {})
			reach[w] = r
	var missing := []
	var wrappers := {}
	for sv in api:
		for n in api[sv]:
			var found := false
			for w in reach:
				if str(reach[w].get(sv, "")).contains("\"%s\"" % n):
					found = true
					wrappers[w] = true
			if not found:
				missing.append("%s:%s" % [sv, n])
	var nodefault := []
	for w in wrappers:
		for p in mains[w].params:
			if not p.contains("="):
				nodefault.append("%s(%s)" % [w, p])
	return [missing, nodefault, wrappers.keys()]

static func stage_sources() -> Dictionary:
	var out := {}
	for sv in STAGE_FILES:
		out[sv] = read(STAGE_FILES[sv])
	return out

# Wrapper names that are not methods of the compiled main.gd taking no
# required argument (a source audit cannot see a parse error).
static func not_callable(wrappers: Array, script: Script) -> Array:
	var have := {}
	for m in script.get_script_method_list():
		have[m.name] = m.args.size() - m.default_args.size()
	var bad := []
	for w in wrappers:
		if have.get(w, -1) != 0:
			bad.append(w)
	return bad
