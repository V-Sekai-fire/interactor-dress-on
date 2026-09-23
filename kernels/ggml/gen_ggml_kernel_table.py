#!/usr/bin/env python3
"""The ggml-rd kernel table, and the check that every kernel has the fixed layout.

Every ggml-rd kernel must bind exactly (lean/Ggml/SlangCodegen/Common.lean):

    set 0  b0  params  storage buffer, read-only
           b1  s0      storage buffer, read-write
           b2  s1      storage buffer, read-write
           b3  s2      storage buffer, read-write
           b4  dst     storage buffer, read-write
    set 1  b0  slot    uniform buffer

and nothing else (no push constants), because the backend builds one set-0
uniform set per tuple of buffers and binds it under every pipeline, and one
set-1 set per params slot. Godot keys a uniform set's format on (binding,
type, writable, length), so a kernel that dropped an unused binding, or
declared a source read-only, would get a different format and its dispatch
would be refused (or, for read-only, recorded wrongly: see Common.lean).

The check reads the SPIR-V itself, not only slangc's reflection JSON: at the
default -O1, slangc strips an unused binding from the SPIR-V while the JSON
still lists it (the reason gen.sh compiles with -O0 -preserve-params). Both
must agree. Any mismatch exits non-zero and writes nothing.

    python gen_ggml_kernel_table.py --spv-dir build/spv-ggml --out GgmlKernelTable.inc k1 k2 ...
    python gen_ggml_kernel_table.py --spv-dir DIR --check-only k   # exit 0 iff k has the layout
"""

import argparse
import json
import os
import struct
import sys

# name -> (set, binding, kind, writable); kind is "storage" or "uniform".
LAYOUT = {
    "params": (0, 0, "storage", False),
    "s0": (0, 1, "storage", True),
    "s1": (0, 2, "storage", True),
    "s2": (0, 3, "storage", True),
    "dst": (0, 4, "storage", True),
    "slot": (1, 0, "uniform", None),
}

# SPIR-V opcodes / enums used below.
OP_NAME, OP_MEMBER_NAME, OP_ENTRY_POINT, OP_EXECUTION_MODE = 5, 6, 15, 16
OP_TYPE_POINTER, OP_VARIABLE, OP_DECORATE, OP_MEMBER_DECORATE = 32, 59, 71, 72
DEC_BLOCK, DEC_NONWRITABLE, DEC_BINDING, DEC_SET = 2, 24, 33, 34
SC_UNIFORM, SC_PUSH_CONSTANT, SC_STORAGE_BUFFER = 2, 9, 12
EM_LOCAL_SIZE = 17


def _string(words):
    raw = struct.pack("<%dI" % len(words), *words)
    return raw.split(b"\0", 1)[0].decode("utf-8")


def parse_spirv(path):
    """(variables {name: (set, binding, kind, writable)}, local size, push constants)."""
    data = open(path, "rb").read()
    if len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
        raise ValueError("%s is not little-endian SPIR-V" % path)
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    names, decos, member_nonwritable = {}, {}, set()
    pointers, variables = {}, {}
    local_size = None
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        if count == 0:
            raise ValueError("%s: zero-length instruction at word %d" % (path, i))
        ops = words[i + 1:i + count]
        if op == OP_NAME:
            names[ops[0]] = _string(ops[1:])
        elif op == OP_DECORATE:
            decos.setdefault(ops[0], {})[ops[1]] = ops[2:]
        elif op == OP_MEMBER_DECORATE and ops[2] == DEC_NONWRITABLE:
            member_nonwritable.add(ops[0])
        elif op == OP_TYPE_POINTER:
            pointers[ops[0]] = (ops[1], ops[2])  # id -> (storage class, pointee)
        elif op == OP_VARIABLE:
            variables[ops[1]] = (ops[0], ops[2])  # id -> (pointer type, storage class)
        elif op == OP_EXECUTION_MODE and ops[1] == EM_LOCAL_SIZE:
            local_size = list(ops[2:5])
        i += count
    out, push = {}, []
    for vid, (ptype, sc) in variables.items():
        if sc == SC_PUSH_CONSTANT:
            push.append(names.get(vid, "%" + str(vid)))
            continue
        d = decos.get(vid, {})
        if DEC_BINDING not in d:
            continue  # builtins and private variables
        name = names.get(vid, "%" + str(vid))
        if sc == SC_STORAGE_BUFFER:
            kind = "storage"
        elif sc == SC_UNIFORM:
            kind = "uniform"
        else:
            kind = "storage-class-%d" % sc
        pointee = pointers.get(ptype, (None, None))[1]
        nonwritable = DEC_NONWRITABLE in d or pointee in member_nonwritable
        writable = None if kind == "uniform" else not nonwritable
        out[name] = (d.get(DEC_SET, [0])[0], d[DEC_BINDING][0], kind, writable)
    return out, local_size, push


def parse_json(path):
    """{name: (set, binding, kind, writable)} from slangc -reflection-json."""
    refl = json.load(open(path))
    out = {}
    for p in refl.get("parameters", []):
        b = p.get("binding", {})
        if b.get("kind") != "descriptorTableSlot":
            raise ValueError("parameter %r binds as %r (push constants?)" % (p["name"], b.get("kind")))
        t = p["type"]
        if t.get("kind") == "constantBuffer":
            kind, writable = "uniform", None
        elif t.get("kind") == "resource" and t.get("baseShape") == "structuredBuffer":
            kind, writable = "storage", t.get("access", "read") == "readWrite"
        else:
            raise ValueError("parameter %r is a %r" % (p["name"], t.get("kind")))
        out[p["name"]] = (b.get("space", 0), b["index"], kind, writable)
    return out


def check(spv_dir, kernel):
    """The kernel's thread-group size, or raise ValueError naming the difference."""
    spv = os.path.join(spv_dir, kernel + ".spv")
    js = os.path.join(spv_dir, kernel + ".refl.json")
    if not os.path.exists(spv):
        raise ValueError("%s: missing %s" % (kernel, spv))
    got, local_size, push = parse_spirv(spv)
    if push:
        raise ValueError("%s: push constants %s" % (kernel, push))
    if got != LAYOUT:
        missing = sorted(set(LAYOUT) - set(got))
        extra = sorted(set(got) - set(LAYOUT))
        wrong = sorted(n for n in set(got) & set(LAYOUT) if got[n] != LAYOUT[n])
        raise ValueError("%s: SPIR-V layout differs: missing %s, extra %s, wrong %s" % (
            kernel, missing, extra, ["%s=%s want %s" % (n, got[n], LAYOUT[n]) for n in wrong]))
    if os.path.exists(js):
        refl = parse_json(js)
        if refl != LAYOUT:
            raise ValueError("%s: reflection JSON layout differs: %s" % (kernel, refl))
    if not local_size:
        raise ValueError("%s: no LocalSize execution mode" % kernel)
    return local_size


def emit(kernels, out_path):
    lines = [
        "// GENERATED by kernels/ggml/gen_ggml_kernel_table.py -- DO NOT EDIT.",
        "//",
        "// The ggml-rd kernels in kernels/ggml/kernels.txt order; a kernel's id",
        "// (params word 0) is its index here. Every entry was checked to bind",
        "// exactly the fixed layout of lean/Ggml/SlangCodegen/Common.lean.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace ggml_rd {",
        "",
        "struct KernelDesc {",
        "\tconst char *name; // <name>.spv in ggml_kernels::kEntries",
        "\tuint32_t threadgroup[3];",
        "};",
        "",
        "static const KernelDesc kKernels[] = {",
    ]
    for name, tg in kernels:
        lines.append('\t{ "%s", { %d, %d, %d } },' % (name, tg[0], tg[1], tg[2]))
    lines += [
        "};",
        "static const uint32_t kKernelCount = sizeof(kKernels) / sizeof(kKernels[0]);",
        "",
        "} // namespace ggml_rd",
        "",
    ]
    with open(out_path, "w", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spv-dir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("kernels", nargs="+")
    a = ap.parse_args()
    parsed, errors = [], []
    for k in a.kernels:
        try:
            parsed.append((k, check(a.spv_dir, k)))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        for e in errors:
            print("layout error: " + e, file=sys.stderr)
        sys.exit(1)
    if a.check_only:
        print("layout ok: %s" % ", ".join(a.kernels))
        return
    if not a.out:
        sys.exit("--out is required unless --check-only")
    emit(parsed, a.out)
    print("wrote %s: %d kernels, all on the fixed layout" % (a.out, len(parsed)))


if __name__ == "__main__":
    main()
