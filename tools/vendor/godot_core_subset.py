#!/usr/bin/env python3
"""Extract the Godot core subset that Cassie needs into vendor/godot-core-subset/.

Source: V-Sekai-fire/entities-godot (a Godot fork, MIT), pinned below. Every
file is copied verbatim, then a fixed list of edits is applied and asserted:

  * the whole body is wrapped in `namespace gdl { ... }` (includes stay
    outside, so a vendored header and a hand-written guest/godot_lite header
    can include each other freely);
  * `operator String()` declarations/definitions are removed, and uses of them
    inside error messages become a fixed placeholder string;
  * Variant `*_bind` helpers (plane, aabb) are removed;
  * Curve3D is cut out of curve.{h,cpp} with its ClassDB/_bind_methods and
    PropertyListHelper code removed;
  * geometry_3d is reduced to the one function Cassie calls.

Every edit states how many times its anchor must match; a mismatch aborts
before anything is written. The script is idempotent: it wipes and rewrites
the destination tree (except CITATION.cff). LICENSE.txt is copied as is.

    python tools/vendor/godot_core_subset.py [--src C:/contract-manifest/4-entities/godot]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

PIN = "c165a519d2836f3ded600948abb9d8f799cd4c5f"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DST = os.path.join(ROOT, "vendor", "godot-core-subset")
BANNER = (
    "// gdl: extracted from V-Sekai-fire/entities-godot@" + PIN[:10]
    + " by tools/vendor/godot_core_subset.py; edits listed in CITATION.cff. Do not edit.\n"
)


class EditError(Exception):
    pass


def sub(text, pattern, repl, expect, what, flags=re.M):
    new, n = re.subn(pattern, repl, text, flags=flags)
    if expect is not None and n != expect:
        raise EditError(f"{what}: pattern {pattern!r} matched {n}x, expected {expect}")
    return new


def literal(text, old, new, expect, what):
    n = text.count(old)
    if n != expect:
        raise EditError(f"{what}: anchor {old!r} matched {n}x, expected {expect}")
    return text.replace(old, new)


def _brace_end(text, open_idx):
    """Index just past the brace that closes text[open_idx] == '{'."""
    depth = 0
    i = open_idx
    in_str = None
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c == "/" and text.startswith("//", i):
            i = text.index("\n", i)
            continue
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise EditError("unbalanced braces")


def function_span(text, header_regex):
    """Spans (start, end) of every top-level function whose first line matches."""
    spans = []
    for m in re.finditer(header_regex, text, flags=re.M):
        start = m.start()
        brace = text.index("{", m.start())
        end = _brace_end(text, brace)
        # Swallow the newline and one following blank line.
        while end < len(text) and text[end] == "\n":
            end += 1
            if end < len(text) and text[end] != "\n":
                break
        spans.append((start, end))
    return spans


def remove_functions(text, header_regex, expect, what):
    spans = function_span(text, header_regex)
    if len(spans) != expect:
        raise EditError(f"{what}: function {header_regex!r} matched {len(spans)}x, expected {expect}")
    for s, e in reversed(spans):
        text = text[:s] + text[e:]
    return text


def wrap_namespace(text, what):
    lines = text.split("\n")
    last_inc = -1
    for i, ln in enumerate(lines):
        if re.match(r"\s*#\s*include\b", ln):
            last_inc = i
    # Preprocessor depth at the last include (e.g. `#ifdef _MSC_VER #include <intrin.h> #endif`).
    depth = 0
    for ln in lines[: last_inc + 1]:
        if re.match(r"\s*#\s*if", ln):
            depth += 1
        elif re.match(r"\s*#\s*endif", ln):
            depth -= 1
    j = last_inc + 1
    while depth > 0:
        if re.match(r"\s*#\s*if", lines[j]):
            depth += 1
        elif re.match(r"\s*#\s*endif", lines[j]):
            depth -= 1
        j += 1
    if last_inc < 0:
        # No includes: open after `#pragma once` (or at the top of the body).
        j = next((k + 1 for k, ln in enumerate(lines) if ln.startswith("#pragma once")), 0)
    # Every later include would land inside the namespace: refuse.
    for ln in lines[j:]:
        if re.match(r"\s*#\s*include\b", ln):
            raise EditError(f"{what}: include after body start: {ln}")
    while text.endswith("\n\n"):
        text = text[:-1]
    body = "\n".join(lines[j:]).rstrip("\n")
    head = "\n".join(lines[:j])
    return head + "\nnamespace gdl {\n" + body + "\n\n} // namespace gdl\n"


# --- operator String / _bind stripping ---------------------------------------

def strip_string_ops(text, what, decls, defs, uses):
    text = sub(text, r"^[ \t]*explicit operator String\(\) const;\n", "", decls, what + " decl")
    text = remove_functions(text, r"^\w+::operator String\(\) const \{", defs, what + " def")
    # Remaining uses sit in error messages: `"...Vector3 " + p_normal.operator String() + "..."`.
    text = sub(
        text,
        r"(?<![\w:])(?:([A-Za-z_]\w*)\.)?operator String\(\)",
        lambda m: 'String("<' + (m.group(1) or "this") + '>")',
        uses,
        what + " uses",
    )
    return text


def strip_binds(text, what, decls, defs):
    text = sub(text, r"^[ \t]*// For Variant bindings\.\n", "", None, what)
    text = sub(text, r"^[ \t]*Variant \w+_bind\([^;{]*\) const;\n", "", decls, what + " bind decl")
    text = remove_functions(text, r"^Variant \w+::\w+_bind\(", defs, what + " bind def")
    return text


# --- per-file recipes ----------------------------------------------------------

def plain(text, what):
    return text


def recipe_string(decls, defs, uses):
    return lambda t, w: strip_string_ops(t, w, decls, defs, uses)


def r_plane_h(t, w):
    t = strip_string_ops(t, w, 1, 0, 0)
    return strip_binds(t, w, 3, 0)


def r_plane_cpp(t, w):
    t = strip_string_ops(t, w, 0, 1, 0)
    t = strip_binds(t, w, 0, 3)
    return literal(t, '#include "core/variant/variant.h"\n', "", 1, w)


def r_aabb_h(t, w):
    t = strip_string_ops(t, w, 1, 0, 0)
    return strip_binds(t, w, 2, 0)


def r_aabb_cpp(t, w):
    t = strip_string_ops(t, w, 0, 1, 0)
    t = strip_binds(t, w, 0, 2)
    return literal(t, '#include "core/variant/variant.h"\n', "", 1, w)


def r_math_funcs_cpp(t, w):
    # Random numbers need RandomPCG (and thirdparty/pcg); Cassie draws none.
    t = literal(t, '#include "core/math/random_pcg.h"\n', "", 1, w)
    t = literal(t, "static RandomPCG default_rand;\n\n", "", 1, w)
    t = remove_functions(t, r"^uint32_t Math::rand_from_seed\(", 1, w)
    t = remove_functions(t, r"^void Math::(seed|randomize)\(", 2, w)
    t = remove_functions(t, r"^uint32_t Math::rand\(\)", 1, w)
    t = remove_functions(t, r"^double Math::randfn\(", 1, w)
    t = remove_functions(t, r"^(double|float|int) Math::random\(", 3, w)
    return t


def r_geometry_3d_h(src_text, w):
    # Keep the license block, then declare only what Cassie calls.
    lic_end = src_text.index("#pragma once")
    decl = re.search(r"^void get_closest_points_between_segments\([^;]*\);\n", src_text, re.M)
    if not decl:
        raise EditError(w + ": declaration not found")
    return (
        src_text[:lic_end]
        + "#pragma once\n\n"
        + '#include "core/math/vector3.h"\n\n'
        + "namespace Geometry3D {\n"
        + decl.group(0)
        + "} // namespace Geometry3D\n"
    )


def r_geometry_3d_cpp(src_text, w):
    lic_end = src_text.index('#include "geometry_3d.h"')
    spans = function_span(src_text, r"^void Geometry3D::get_closest_points_between_segments\(")
    if len(spans) != 1:
        raise EditError(w + ": function not found once")
    s, e = spans[0]
    return src_text[:lic_end] + '#include "geometry_3d.h"\n\n' + src_text[s:e].rstrip("\n") + "\n"


def r_curve_h(t, w):
    lic_end = t.index("#pragma once")
    m = re.search(r"^class Curve3D : public Resource \{", t, re.M)
    if not m:
        raise EditError(w + ": class Curve3D not found")
    end = _brace_end(t, t.index("{", m.start()))
    cls = t[m.start(): end] + ";\n"
    cls = literal(cls, "\tstatic inline PropertyListHelper base_property_helper;\n\tPropertyListHelper property_helper;\n\n", "", 1, w)
    cls = literal(cls, "\tbool _filter_property(const String &p_name, int p_index) const;\n", "", 1, w)
    cls = sub(cls, r"^\tbool _set\(.*\n\tbool _get\(.*\n\tvoid _get_property_list\(.*\n\tbool _property_can_revert\(.*\n\tbool _property_get_revert\(.*\n\n", "", 1, w + " property helper forwarders", re.M)
    cls = literal(cls, "\tstatic void _bind_methods();\n", "", 1, w)
    cls = literal(cls, "protected:\n\npublic:\n", "public:\n", 1, w)
    return (
        t[:lic_end]
        + "#pragma once\n\n"
        + '#include "core/io/resource.h"\n'
        + '#include "core/templates/rb_map.h"\n\n'
        + cls
    )


def r_curve_cpp(t, w):
    head_end = t.index("#include <cfloat>")
    head = t[: head_end] + "#include <cfloat> // FLT_EPSILON\n\n"
    start = t.index("int Curve3D::get_point_count() const {")
    ctor = t.index("Curve3D::Curve3D() {")
    end = _brace_end(t, t.index("{", ctor))
    body = t[start:end] + "\n"
    body = remove_functions(body, r"^bool Curve3D::_filter_property\(", 1, w)
    body = remove_functions(body, r"^void Curve3D::_bind_methods\(\) \{", 1, w)
    body = literal(body, "Curve3D::Curve3D() {\n\tproperty_helper.setup_for_instance(base_property_helper, this);\n}", "Curve3D::Curve3D() {\n}", 1, w)
    return head + body


FILES = [
    # (path, recipe, wrap): wrap None copies the file untouched, no banner.
    ("LICENSE.txt", plain, None),
    ("core/typedefs.h", plain, True),
    ("core/error/error_list.h", plain, True),
    ("core/math/math_defs.h", plain, True),
    ("core/math/math_funcs.h", plain, True),
    ("core/math/math_funcs.cpp", r_math_funcs_cpp, True),
    ("core/math/math_funcs_binary.h", plain, True),
    ("core/math/vector2.h", recipe_string(1, 0, 0), True),
    ("core/math/vector2.cpp", recipe_string(0, 1, 2), True),
    ("core/math/vector2i.h", recipe_string(1, 0, 0), True),
    ("core/math/vector2i.cpp", recipe_string(0, 1, 0), True),
    ("core/math/vector3.h", recipe_string(1, 0, 2), True),
    ("core/math/vector3.cpp", recipe_string(0, 1, 0), True),
    ("core/math/vector3i.h", recipe_string(1, 0, 0), True),
    ("core/math/vector3i.cpp", recipe_string(0, 1, 0), True),
    ("core/math/basis.h", recipe_string(1, 0, 0), True),
    ("core/math/basis.cpp", recipe_string(0, 1, 2), True),
    ("core/math/quaternion.h", recipe_string(1, 0, 1), True),
    ("core/math/quaternion.cpp", recipe_string(0, 1, 11), True),
    ("core/math/transform_3d.h", recipe_string(1, 0, 0), True),
    ("core/math/transform_3d.cpp", recipe_string(0, 1, 0), True),
    ("core/math/plane.h", r_plane_h, True),
    ("core/math/plane.cpp", r_plane_cpp, True),
    ("core/math/aabb.h", r_aabb_h, True),
    ("core/math/aabb.cpp", r_aabb_cpp, True),
    ("core/math/dynamic_bvh.h", plain, True),
    ("core/math/dynamic_bvh.cpp", plain, True),
    ("core/math/geometry_3d.h", r_geometry_3d_h, True),
    ("core/math/geometry_3d.cpp", r_geometry_3d_cpp, True),
    ("core/templates/hashfuncs.h", plain, True),
    ("core/templates/hashfuncs.cpp", plain, True),
    ("core/templates/hash_map.h", plain, True),
    ("core/templates/hash_set.h", plain, True),
    ("core/templates/local_vector.h", plain, True),
    ("core/templates/pair.h", plain, True),
    ("core/templates/sort_array.h", plain, True),
    ("core/templates/sort_list.h", plain, True),
    ("core/templates/list.h", plain, True),
    ("core/templates/paged_allocator.h", plain, True),
    ("core/templates/rb_map.h", plain, True),
    ("core/templates/span.h", plain, True),
    ("scene/resources/curve.h", r_curve_h, True),
    ("scene/resources/curve.cpp", r_curve_cpp, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("GODOT_SRC", "C:/contract-manifest/4-entities/godot"))
    ap.add_argument("--allow-unpinned", action="store_true")
    args = ap.parse_args()

    head = subprocess.run(["git", "-C", args.src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != PIN and not args.allow_unpinned:
        sys.exit(f"error: {args.src} is at {head or '?'}, expected {PIN} (pass --allow-unpinned to override)")

    out = {}
    try:
        for rel, recipe, wrap in FILES:
            with open(os.path.join(args.src, rel), encoding="utf-8", newline="") as f:
                text = f.read().replace("\r\n", "\n")
            text = recipe(text, rel)
            if wrap is None:  # copied byte for byte (the licence)
                out[rel] = text
                continue
            if wrap:
                text = wrap_namespace(text, rel)
            out[rel] = BANNER + text
    except EditError as e:
        sys.exit(f"error: {e}")

    for entry in os.listdir(DST) if os.path.isdir(DST) else []:
        if entry != "CITATION.cff":
            p = os.path.join(DST, entry)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for rel, text in out.items():
        p = os.path.join(DST, rel)
        os.makedirs(os.path.dirname(p) or DST, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(f"wrote {len(out)} files to {os.path.relpath(DST, ROOT)} from {args.src}@{PIN[:10]}")


if __name__ == "__main__":
    main()
