import LeanSlang

/-!
# `Ggml.SlangCodegen.Common` — the one layout every ggml-rd kernel shares

Not a kernel. Every ggml op kernel (`Ggml.SlangCodegen.*`) is built from
the pieces here, so all of them reflect the same descriptor layout and a
set-0 uniform set built for one pipeline binds under every other
(`-preserve-params` keeps the bindings a kernel does not touch).
`kernels/ggml/gen_ggml_kernel_table.py` refuses any kernel whose
reflection differs.

## Bindings

    set 0  b0  StructuredBuffer<uint>     params  the per-dispatch table, 64 words (256 B) per slot
           b1  RWStructuredBuffer<T0>     s0      source 0
           b2  RWStructuredBuffer<T1>     s1      source 1
           b3  RWStructuredBuffer<T2>     s2      source 2
           b4  RWStructuredBuffer<Td>     dst     destination
    set 1  b0  ConstantBuffer<Slot>       slot    { uint base; pad x3 }: which 64 words are this dispatch's

There are no push constants (the guest reaches the GPU only through
`RenderingDevice`, and one `buffer_update` of the whole table before
`compute_list_begin` replaces a push per dispatch). A dispatch finds its
words at `params[slot.base + k]`; `slot` is a pooled 16-byte uniform
buffer per slot index, bound as set 1.

The element types `T0 T1 T2 Td` are the kernel's to choose; the layout
check compares binding, set, kind and access, which is all Godot's
uniform-set format keys on (a runtime array reflects length 0 whatever
its element).

**The sources are declared read-write although kernels only read them.**
Godot's render graph registers a buffer's usage in a compute list from the
first binding that names it (`RenderingDeviceGraph::add_compute_list_usage`
drops every later usage of the same tracker; only a DEV build reports it),
and a uniform set lists its trackers in binding order. With read-only
sources, a ggml buffer holding both a source and the destination (the
usual case: one ggml buffer is one RD buffer) is recorded as only *read*
by that list, the write is never tracked, and the next list is not ordered
after it: Gate 0F's in-place counters lost 80% of their increments that
way. Read-write sources make every list's usage of a data buffer a write,
so consecutive lists that share a buffer are always ordered.

## Words of a slot (`uint`, strides and offsets in elements)

    0       kernel id (the kernel's index in kernels/ggml/kernels.txt)
    1-4     dst ne     5-8   dst nb     9   dst offset
    10-13   s0 ne      14-17 s0 nb      18  s0 offset
    19-22   s1 ne      23-26 s1 nb      27  s1 offset
    28-31   s2 ne      32-35 s2 nb      36  s2 offset
    37-52   op_params, raw (ggml's 64 bytes)
    53-63   derived by the op's packer; 53 = thread count, 54 = groups along x

`guest/ggml-rd/ggml_rd_params.h` is generated from the same numbers
(`paramsHeader`, written by `lake exe emit_ggml`), so the packer and the
kernels cannot disagree about a word.

## Grid

Work-group counts stay at or below 65535 per dimension (the Vulkan
minimum), so a large 1-D launch is split as `(gx, gy, 1)` and a kernel
recovers its linear group as `gid.y * gx + gid.x`, with `gx` read from
word 54.
-/

namespace Ggml.SlangCodegen.Common

open LeanSlang

/-! ## Word layout -/

def wordsPerSlot : Nat := 64
def slotBytes : Nat := 4 * wordsPerSlot

/-- Word 0: the kernel id. -/
def wKernel : Nat := 0
/-- First word of the destination's tensor block. -/
def wDst : Nat := 1
/-- First word of source `i`'s tensor block (`i` in 0..2). -/
def wSrc (i : Nat) : Nat := 10 + 9 * i
/-- Inside a tensor block: `ne` at +0..3, `nb` at +4..7, offset at +8. -/
def tNe : Nat := 0
def tNb : Nat := 4
def tOff : Nat := 8
def tensorBlockWords : Nat := 9
def wOpParams : Nat := 37
def nOpParams : Nat := 16
def wDerived : Nat := 53
def nDerived : Nat := 11
/-- Derived word 53: how many threads do work (the rest return). -/
def wThreads : Nat := 53
/-- Derived word 54: work groups along x in the 2-D split of a 1-D grid. -/
def wGroupsX : Nat := 54
def maxGroupsPerDim : Nat := 65535

example : wSrc 2 + tensorBlockWords = wOpParams := by native_decide
example : wOpParams + nOpParams = wDerived := by native_decide
example : wDerived + nDerived = wordsPerSlot := by native_decide
example : wDst + tensorBlockWords = wSrc 0 := by native_decide

/-! ## Declarations -/

/-- `struct Slot { uint base; uint pad0; uint pad1; uint pad2; };` — 16
    bytes, a uniform buffer's natural size. -/
def slotStruct : SlangStructDecl :=
  { name := "Slot"
  , fields :=
      [ ⟨"base", .scalar .uint, Semantic.none, none, none, .qIn⟩
      , ⟨"pad0", .scalar .uint, Semantic.none, none, none, .qIn⟩
      , ⟨"pad1", .scalar .uint, Semantic.none, none, none, .qIn⟩
      , ⟨"pad2", .scalar .uint, Semantic.none, none, none, .qIn⟩ ] }

/-- The globals with the sources read-write (`srcRw`, every kernel) or
    read-only (only the control kernels of `Ggml.controls`, which exist to
    show the render-graph hazard the read-write sources avoid). -/
def globalsWith (srcRw : Bool) (t0 t1 t2 td : Scalar) : List SlangBinding :=
  let src := fun (t : Scalar) => if srcRw then SlangType.rwBuf (.scalar t) else SlangType.roBuf (.scalar t)
  [ ⟨"params", .roBuf (.scalar .uint), Semantic.none, some 0, some 0, .qIn⟩
  , ⟨"s0",     src t0,                 Semantic.none, some 1, some 0, .qIn⟩
  , ⟨"s1",     src t1,                 Semantic.none, some 2, some 0, .qIn⟩
  , ⟨"s2",     src t2,                 Semantic.none, some 3, some 0, .qIn⟩
  , ⟨"dst",    .rwBuf (.scalar td),    Semantic.none, some 4, some 0, .qIn⟩
  , ⟨"slot",   .const "Slot",          Semantic.none, some 0, some 1, .qIn⟩ ]

/-- The fixed globals, with the element type of each storage binding. -/
def globals (t0 t1 t2 td : Scalar) : List SlangBinding :=
  globalsWith true t0 t1 t2 td

/-! ## Expression shorthands -/

def u (n : Nat) : SlangExpr := .litUint n
def v (s : String) : SlangExpr := .var s
def add (a b : SlangExpr) : SlangExpr := .bin "+" a b
def mul (a b : SlangExpr) : SlangExpr := .bin "*" a b
def urem (a b : SlangExpr) : SlangExpr := .bin "%" a b
def udiv (a b : SlangExpr) : SlangExpr := .bin "/" a b
/-- `pw(k)`: word `k` of this dispatch's slot. -/
def pwE (k : SlangExpr) : SlangExpr := .call "pw" [k]
def pwN (k : Nat) : SlangExpr := pwE (u k)
/-- `pf(k)`: word `k` as a float (op_params hold floats bit for bit). -/
def pfN (k : Nat) : SlangExpr := .call "pf" [u k]

private def uintParam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩
private def uintOut (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qOut⟩

/-! ## Helper functions (each kernel lists the ones it calls) -/

/-- `uint pw(uint k) { return params[slot.base + k]; }` -/
def fnPw : SlangFunctionDecl :=
  { retType := .scalar .uint
  , name := "pw"
  , params := [uintParam "k"]
  , body := [ .ret (some (.index (v "params") (add (.member (v "slot") "base") (v "k")))) ] }

/-- `float pf(uint k) { return asfloat(pw(k)); }` -/
def fnPf : SlangFunctionDecl :=
  { retType := .scalar .float
  , name := "pf"
  , params := [uintParam "k"]
  , body := [ .ret (some (.call "asfloat" [pwE (v "k")])) ] }

/-- `unravel4(e, w, i0..i3)`: the 4-D index of linear element `e` of the
    tensor whose block starts at word `w` (ggml order: dim 0 fastest). -/
def fnUnravel4 : SlangFunctionDecl :=
  { name := "unravel4"
  , params := [ uintParam "e", uintParam "w"
              , uintOut "i0", uintOut "i1", uintOut "i2", uintOut "i3" ]
  , body :=
      [ .declInit (.scalar .uint) "n0" (pwE (v "w"))
      , .declInit (.scalar .uint) "n1" (pwE (add (v "w") (u 1)))
      , .declInit (.scalar .uint) "n2" (pwE (add (v "w") (u 2)))
      , .assign (v "i0") (urem (v "e") (v "n0"))
      , .declInit (.scalar .uint) "r" (udiv (v "e") (v "n0"))
      , .assign (v "i1") (urem (v "r") (v "n1"))
      , .assign (v "r") (udiv (v "r") (v "n1"))
      , .assign (v "i2") (urem (v "r") (v "n2"))
      , .assign (v "i3") (udiv (v "r") (v "n2")) ] }

/-- `off4(w, i0..i3)`: element offset of index (i0..i3) in the tensor whose
    block starts at word `w`: its offset plus the strided sum. -/
def fnOff4 : SlangFunctionDecl :=
  { retType := .scalar .uint
  , name := "off4"
  , params := [ uintParam "w", uintParam "i0", uintParam "i1", uintParam "i2", uintParam "i3" ]
  , body :=
      [ .ret (some
          (add (add (add (add
            (pwE (add (v "w") (u tOff)))
            (mul (v "i0") (pwE (add (v "w") (u tNb)))))
            (mul (v "i1") (pwE (add (v "w") (u (tNb + 1))))))
            (mul (v "i2") (pwE (add (v "w") (u (tNb + 2))))))
            (mul (v "i3") (pwE (add (v "w") (u (tNb + 3))))))) ] }

/-- The 16-bit half `e` of a buffer of `uint` words, in the low bits. -/
private def halfOf (buf : String) : List SlangStmt :=
  [ .declInit (.scalar .uint) "w" (.index (v buf) (.bin ">>" (v "e") (u 1)))
  , .declInit (.scalar .uint) "h"
      (.ternary (.bin "!=" (.bin "&" (v "e") (u 1)) (u 0))
        (.bin ">>" (v "w") (u 16))
        (.bin "&" (v "w") (u 65535))) ]

/-- `float ld_bf16_<buf>(uint e)`: element `e` of a bf16 tensor held in a
    `uint` word buffer, widened exactly (bf16 is the top half of an f32). -/
def fnLdBf16 (buf : String) : SlangFunctionDecl :=
  { retType := .scalar .float
  , name := "ld_bf16_" ++ buf
  , params := [uintParam "e"]
  , body := halfOf buf ++ [ .ret (some (.call "asfloat" [.bin "<<" (v "h") (u 16)])) ] }

/-- `float ld_f16_<buf>(uint e)`: element `e` of an f16 tensor held in a
    `uint` word buffer (no 16-bit storage capability needed). -/
def fnLdF16 (buf : String) : SlangFunctionDecl :=
  { retType := .scalar .float
  , name := "ld_f16_" ++ buf
  , params := [uintParam "e"]
  , body := halfOf buf ++ [ .ret (some (.call "f16tof32" [v "h"])) ] }

/-! ## Entry point -/

/-- The linear thread index of a 1-D launch split as `(gx, gy, 1)` groups
    of `tg` threads: `(gid.y * gx + gid.x) * tg + lid.x`. -/
def linearThread (tg : Nat) : SlangExpr :=
  add (mul (add (mul (.member (v "gid") "y") (pwN wGroupsX)) (.member (v "gid") "x")) (u tg))
      (.member (v "lid") "x")

/-- A 1-D kernel entry: `e` is this thread's element; threads at or past
    word 53 return. -/
def entry1D (tg : Nat) (body : List SlangStmt) : SlangFunctionDecl :=
  { attrs := [.shaderCompute, .numthreads tg 1 1]
  , name := "main"
  , params := [ ⟨"gid", .vec .uint 3, .svGroupId, none, none, .qIn⟩
              , ⟨"lid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩ ]
  , body :=
      [ .declInit (.scalar .uint) "e" (linearThread tg)
      , .ifNoElse (.bin ">=" (v "e") (pwN wThreads)) [ .ret none ] ] ++ body }

/-- A kernel module: the Slot struct, the fixed globals, then `helpers`
    and the entry. -/
def kernelModule (t0 t1 t2 td : Scalar) (helpers : List SlangFunctionDecl)
    (main : SlangFunctionDecl) : SlangShaderModule :=
  { structs := [slotStruct]
  , globals := globals t0 t1 t2 td
  , functions := helpers ++ [main] }

/-- A control kernel: the same, with read-only sources. Never in
    kernels/ggml/kernels.txt; the layout check refuses it. -/
def controlModuleRoSources (t0 t1 t2 td : Scalar) (helpers : List SlangFunctionDecl)
    (main : SlangFunctionDecl) : SlangShaderModule :=
  { structs := [slotStruct]
  , globals := globalsWith false t0 t1 t2 td
  , functions := helpers ++ [main] }

/-! ## Pinned emission of the shared pieces -/

example : LeanSlang.emitFunction fnPw =
"uint pw(uint k) {
  return params[(slot.base + k)];
}" := by native_decide

example : LeanSlang.emitFunction fnOff4 =
"uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}" := by native_decide

example : LeanSlang.emitFunction (fnLdBf16 "s0") =
"float ld_bf16_s0(uint e) {
  uint w = s0[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return asfloat((h << 16u));
}" := by native_decide

/-! ## The C++ side of the same layout -/

private def cu (name : String) (n : Nat) : String :=
  "constexpr uint32_t " ++ name ++ " = " ++ toString n ++ ";\n"

/-- `guest/ggml-rd/ggml_rd_params.h`, written by `lake exe emit_ggml`. -/
def paramsHeader : String :=
  "// GENERATED by `lake exe emit_ggml` from lean/Ggml/SlangCodegen/Common.lean.\n" ++
  "// Do not edit; regenerate with kernels/ggml/gen.sh.\n" ++
  "//\n" ++
  "// The word layout of one params slot and the descriptor layout every\n" ++
  "// ggml-rd kernel shares. Strides and offsets are in elements.\n" ++
  "#pragma once\n\n#include <cstdint>\n\nnamespace ggml_rd {\n\n" ++
  cu "kWordsPerSlot" wordsPerSlot ++
  cu "kSlotBytes" slotBytes ++
  cu "W_KERNEL" wKernel ++
  cu "W_DST" wDst ++
  cu "W_SRC0" (wSrc 0) ++
  cu "W_SRC1" (wSrc 1) ++
  cu "W_SRC2" (wSrc 2) ++
  cu "T_NE" tNe ++
  cu "T_NB" tNb ++
  cu "T_OFF" tOff ++
  cu "W_OP_PARAMS" wOpParams ++
  cu "N_OP_PARAMS" nOpParams ++
  cu "W_DERIVED" wDerived ++
  cu "N_DERIVED" nDerived ++
  cu "W_THREADS" wThreads ++
  cu "W_GROUPS_X" wGroupsX ++
  cu "kMaxGroupsPerDim" maxGroupsPerDim ++
  "\n// Descriptor layout (set, binding).\n" ++
  cu "SET_TENSORS" 0 ++
  cu "SET_SLOT" 1 ++
  cu "B_PARAMS" 0 ++
  cu "B_SRC0" 1 ++
  cu "B_SRC1" 2 ++
  cu "B_SRC2" 3 ++
  cu "B_DST" 4 ++
  cu "B_SLOT" 0 ++
  "\n} // namespace ggml_rd\n"

def paramsHeaderExpected : String :=
"// GENERATED by `lake exe emit_ggml` from lean/Ggml/SlangCodegen/Common.lean.
// Do not edit; regenerate with kernels/ggml/gen.sh.
//
// The word layout of one params slot and the descriptor layout every
// ggml-rd kernel shares. Strides and offsets are in elements.
#pragma once

#include <cstdint>

namespace ggml_rd {

constexpr uint32_t kWordsPerSlot = 64;
constexpr uint32_t kSlotBytes = 256;
constexpr uint32_t W_KERNEL = 0;
constexpr uint32_t W_DST = 1;
constexpr uint32_t W_SRC0 = 10;
constexpr uint32_t W_SRC1 = 19;
constexpr uint32_t W_SRC2 = 28;
constexpr uint32_t T_NE = 0;
constexpr uint32_t T_NB = 4;
constexpr uint32_t T_OFF = 8;
constexpr uint32_t W_OP_PARAMS = 37;
constexpr uint32_t N_OP_PARAMS = 16;
constexpr uint32_t W_DERIVED = 53;
constexpr uint32_t N_DERIVED = 11;
constexpr uint32_t W_THREADS = 53;
constexpr uint32_t W_GROUPS_X = 54;
constexpr uint32_t kMaxGroupsPerDim = 65535;

// Descriptor layout (set, binding).
constexpr uint32_t SET_TENSORS = 0;
constexpr uint32_t SET_SLOT = 1;
constexpr uint32_t B_PARAMS = 0;
constexpr uint32_t B_SRC0 = 1;
constexpr uint32_t B_SRC1 = 2;
constexpr uint32_t B_SRC2 = 3;
constexpr uint32_t B_DST = 4;
constexpr uint32_t B_SLOT = 0;

} // namespace ggml_rd
"

/-- The header is pinned: a change to the word layout has to change this
    text too, and so shows up in review next to the kernels it moves. -/
example : paramsHeader = paramsHeaderExpected := by native_decide

end Ggml.SlangCodegen.Common
