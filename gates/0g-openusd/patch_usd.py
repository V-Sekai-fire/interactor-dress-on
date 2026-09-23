"""Gate 0G: the local adaptations applied to a pristine OpenUSD v26.05 checkout
(read-only upstream, outside the repo). Idempotent; `git diff` in the checkout
afterwards is gates/0g-openusd/openusd-26.05-rv64.patch.

  python patch_usd.py C:/b/usd2605
"""
import sys, os

root = sys.argv[1]


def edit(rel, pairs):
    p = os.path.join(root, rel)
    s = open(p, newline="").read()
    crlf = "\r\n" in s
    s = s.replace("\r\n", "\n")
    for a, b in pairs:
        if b in s:
            continue
        assert a in s, (rel, a[:60])
        s = s.replace(a, b, 1)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "w", newline="").write(s)


# 1. arch: riscv64 is a 64-bit little-endian CPU with sincos() in glibc.
edit("pxr/base/arch/defines.h", [
    ("#define ARCH_CPU_ARM\n#endif",
     "#define ARCH_CPU_ARM\n#elif defined(__riscv) && (__riscv_xlen == 64)\n#define ARCH_CPU_RISCV\n#endif"),
    ("defined(_M_ARM64) || defined(__wasm64__)",
     "defined(_M_ARM64) || defined(__wasm64__) || defined(ARCH_CPU_RISCV)"),
])
edit("pxr/base/arch/math.h", [
    ("#if defined (ARCH_CPU_INTEL) || defined (ARCH_CPU_ARM) ||  \\\n",
     "#if defined (ARCH_CPU_INTEL) || defined (ARCH_CPU_ARM) || defined(ARCH_CPU_RISCV) || \\\n"),
])
# The non-locking execve for crash reports: the Linux riscv64 syscall ABI
# (a7 = 221 execve, ecall).
edit("pxr/base/arch/stackTrace.cpp", [
    ("        result = __file_result;\n    }\n#elif defined(ARCH_CPU_INTEL) && defined(ARCH_BITS_64)",
     "        result = __file_result;\n    }\n#elif defined(ARCH_CPU_RISCV)\n"
     "    {\n"
     "        register long a0 asm (\"a0\") = (long)file;\n"
     "        register char* const* a1 asm (\"a1\") = argv;\n"
     "        register char* const* a2 asm (\"a2\") = envp;\n"
     "        register long a7 asm (\"a7\") = 221;\n"
     "        __asm__ __volatile__ (\"ecall\" : \"+r\" (a0)\n"
     "            : \"r\" (a1), \"r\" (a2), \"r\" (a7) : \"memory\");\n"
     "        result = a0;\n"
     "    }\n"
     "#elif defined(ARCH_CPU_INTEL) && defined(ARCH_BITS_64)"),
])

# 2. plug: plugInfo.json contents from memory (the guest has no filesystem).
edit("pxr/base/plug/info.cpp", [
    ("#include <fstream>\n", "#include \"pxr/base/plug/api.h\"\n#include <fstream>\n#include <sstream>\n"),
    ("PXR_NAMESPACE_OPEN_SCOPE\n\nnamespace {",
     "PXR_NAMESPACE_OPEN_SCOPE\n\n"
     "// Gate 0G (interactor-dress-on): a guest without a filesystem registers its\n"
     "// plugInfo.json contents from memory. When set, the hook is asked first for\n"
     "// every plugInfo path; a non-null return is that file's contents.\n"
     "PLUG_API const char* (*Plug_InMemoryPlugInfoHook)(const char* pathname) = nullptr;\n\n"
     "namespace {"),
    ("    // The file may not exist or be readable.\n    std::ifstream ifs;\n",
     "    const char* inMemory = Plug_InMemoryPlugInfoHook\n"
     "        ? Plug_InMemoryPlugInfoHook(pathname.c_str()) : nullptr;\n"
     "    std::istringstream iss(inMemory ? inMemory : \"\");\n"
     "    // The file may not exist or be readable.\n    std::ifstream ifs;\n    if (!inMemory) {\n"),
    ("            Msg(\"Failed to open plugin info %s\\n\", pathname.c_str());\n        return false;\n    }\n",
     "            Msg(\"Failed to open plugin info %s\\n\", pathname.c_str());\n        return false;\n    }\n    }\n"
     "    std::istream& in = inMemory ? static_cast<std::istream&>(iss) : ifs;\n"),
    ("    while (getline(ifs, line)) {", "    while (getline(in, line)) {"),
])
print("patched", root)
