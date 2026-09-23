import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.Move` — the data-movement kernels' shared pieces

Not a kernel. CPY/DUP/CONT (`Cpy`), GET_ROWS (`GetRows`), CONCAT
(`Concat`) and REPEAT (`Repeat`) move elements without arithmetic, so
every one of them is "for each destination element, find its source
element, convert it, store it". Each kernel supplies one function

    uint val(uint i0, uint i1, uint i2, uint i3)

that returns the destination element at index (i0..i3), already in the
destination's bit pattern (32 bits, or 16 in the low half), and one of the
two entries below stores it.

## Everything is a `uint` word

All four storage bindings are `RWStructuredBuffer<uint>`: an f32 or i32
element is a word, an f16, bf16 or i16 element is half of one. No float
ever passes through a float register on the way, so a same-type copy is a
bit copy (test-backend-ops wants those exact, NMSE 0), and the conversions
are integer code with ggml-cpu's rounding:

- f16 -> f32: exact widening, subnormals normalised by a shift loop, NaN
  payload kept (as ggml's `ggml_compute_fp16_to_fp32`).
- f32 -> f16: round to nearest even; >= 65520 is inf; NaN is `0x7E00 | sign`
  (ggml's `ggml_compute_fp32_to_fp16`).
- bf16 -> f32: `h << 16` (ggml's).
- f32 -> bf16: ggml's `ggml_compute_fp32_to_bf16`: round to nearest even by
  `(u + 0x7fff + ((u >> 16) & 1)) >> 16`, NaN is `(u >> 16) | 64`.

## The two entries

`entry32` (32-bit destination): one thread per destination element,
`dst[off(i)] = val(i)`.

`entry16` (16-bit destination): a 16-bit element shares its word with a
neighbour, and the cpp target has no InterlockedAnd/Or. So the thread of the
element at the EVEN address owns the word: it writes its own half and, when
the element at the next address is also in the destination, that one's
half too, in one store; when it is not, it keeps the other half
(read-modify-write, no other thread of the dispatch touches that word). The
thread of an element at an ODD address returns if the previous address is
in the destination (its owner writes it), else it owns the word the same way.
Every word is then written by exactly one thread.

"Is the next / previous address in the destination" is decided from the
iteration order the packer (`guest/ggml-rd/ops/move.h`) arranges: it sorts
dst's dimensions by stride (size-1 dimensions last) and permutes the dst and
source blocks alike, and the packer supports a 16-bit destination only when
the sorted layout is nested (`nb[k+1] >= ne[k] * nb[k]`). Word 55 (`wMerge`)
then holds the merge chain: bit 0 set when `nb[0] = 1`, bit k set when bit
k-1 is and `nb[k] = ne[k-1] * nb[k-1]`. Under nesting the element after
(i0..i3) in memory is (i0+1, ..) if bit 0 and i0+1 < ne0, else (0, i1+1, ..)
if bit 1 and i1+1 < ne1, and so on (a carry); the previous one exists iff
some dimension k with bit k set has i_k > 0.

## Derived words

    55       wMerge       entry16's merge chain
    56..59   wLinW        CPY: the original destination linear weight of
                          each iteration dimension (1, ne0, ne0*ne1, ..
                          permuted), so a reshaping copy finds its source
-/

namespace Ggml.SlangCodegen.Move

open LeanSlang
open Ggml.SlangCodegen.Common

def threadgroup : Nat := 256

/-- Derived word 55: entry16's merge chain. -/
def wMerge : Nat := 55
/-- Derived words 56..59: CPY's linear weights of the iteration dims. -/
def wLinW (k : Nat) : Nat := 56 + k

example : wDerived + 2 = wMerge := by native_decide
example : wLinW 3 < wordsPerSlot := by native_decide

/-! ## Expression shorthands -/

def band (a b : SlangExpr) : SlangExpr := .bin "&" a b
def bor (a b : SlangExpr) : SlangExpr := .bin "|" a b
def shr (a b : SlangExpr) : SlangExpr := .bin ">>" a b
def shl (a b : SlangExpr) : SlangExpr := .bin "<<" a b
def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
def land (a b : SlangExpr) : SlangExpr := .bin "&&" a b
def lor (a b : SlangExpr) : SlangExpr := .bin "||" a b
def eq (a b : SlangExpr) : SlangExpr := .bin "==" a b
def neq (a b : SlangExpr) : SlangExpr := .bin "!=" a b
def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
def gt (a b : SlangExpr) : SlangExpr := .bin ">" a b

def tU : SlangType := .scalar .uint
def tB : SlangType := .scalar .bool

def uparam (n : String) : SlangBinding :=
  ⟨n, .scalar .uint, Semantic.none, none, none, .qIn⟩

def idx4 : List SlangBinding := [uparam "i0", uparam "i1", uparam "i2", uparam "i3"]
def iv : List SlangExpr := [v "i0", v "i1", v "i2", v "i3"]

/-! ## Loads and conversions (all integer) -/

/-- `uint ld16_<buf>(uint e)`: 16-bit element `e` of a `uint` word buffer. -/
def fnLd16 (buf : String) : SlangFunctionDecl :=
  { retType := tU
  , name := "ld16_" ++ buf
  , params := [uparam "e"]
  , body := [ .ret (some (band (shr (.index (v buf) (shr (v "e") (u 1))) (mul (band (v "e") (u 1)) (u 16)))
                               (u 65535))) ] }

/-- `uint f16_to_f32(uint h)`: the f32 bits of f16 bits `h`, exact. -/
def fnF16ToF32 : SlangFunctionDecl :=
  { retType := tU
  , name := "f16_to_f32"
  , params := [uparam "h"]
  , body :=
      [ .declInit tU "s" (shl (band (v "h") (u 32768)) (u 16))
      , .declInit tU "x" (band (shr (v "h") (u 10)) (u 31))
      , .declInit tU "m" (band (v "h") (u 1023))
      , .ifNoElse (eq (v "x") (u 31))
          [ .ret (some (bor (bor (v "s") (u 2139095040)) (shl (v "m") (u 13)))) ]
      , .ifNoElse (neq (v "x") (u 0))
          [ .ret (some (bor (bor (v "s") (shl (add (v "x") (u 112)) (u 23))) (shl (v "m") (u 13)))) ]
      , .ifNoElse (eq (v "m") (u 0)) [ .ret (some (v "s")) ]
      , .declInit tU "x2" (u 113)
      , .whileLoop (eq (band (v "m") (u 1024)) (u 0))
          [ .assign (v "m") (shl (v "m") (u 1))
          , .assign (v "x2") (sub (v "x2") (u 1)) ]
      , .ret (some (bor (bor (v "s") (shl (v "x2") (u 23))) (shl (band (v "m") (u 1023)) (u 13)))) ] }

/-- `uint f32_to_f16(uint b)`: the f16 bits of f32 bits `b`, round to
    nearest even, as ggml-cpu. -/
def fnF32ToF16 : SlangFunctionDecl :=
  { retType := tU
  , name := "f32_to_f16"
  , params := [uparam "b"]
  , body :=
      [ .declInit tU "s" (band (shr (v "b") (u 16)) (u 32768))
      , .declInit tU "a" (band (v "b") (u 2147483647))
      , .ifNoElse (gt (v "a") (u 2139095040)) [ .ret (some (bor (v "s") (u 32256))) ]
      , .ifNoElse (ge (v "a") (u 1199566848)) [ .ret (some (bor (v "s") (u 31744))) ]
      , .ifNoElse (ge (v "a") (u 947912704))
          [ .declInit tU "m" (sub (v "a") (u 939524096))
          , .ret (some (bor (v "s")
              (shr (add (add (v "m") (u 4095)) (band (shr (v "m") (u 13)) (u 1))) (u 13)))) ]
      , .declInit tU "x" (shr (v "a") (u 23))
      , .ifNoElse (lt (v "x") (u 102)) [ .ret (some (v "s")) ]
      , .declInit tU "q" (bor (band (v "a") (u 8388607)) (u 8388608))
      , .declInit tU "sh" (sub (u 126) (v "x"))
      , .ret (some (bor (v "s")
          (shr (add (add (v "q") (sub (shl (u 1) (sub (v "sh") (u 1))) (u 1)))
                    (band (shr (v "q") (v "sh")) (u 1)))
               (v "sh")))) ] }

/-- `uint f32_to_bf16(uint b)`: ggml's `ggml_compute_fp32_to_bf16`. -/
def fnF32ToBf16 : SlangFunctionDecl :=
  { retType := tU
  , name := "f32_to_bf16"
  , params := [uparam "b"]
  , body :=
      [ .ifNoElse (gt (band (v "b") (u 2147483647)) (u 2139095040))
          [ .ret (some (bor (shr (v "b") (u 16)) (u 64))) ]
      , .ret (some (shr (add (v "b") (add (u 32767) (band (shr (v "b") (u 16)) (u 1)))) (u 16))) ] }

/-! ## Element kinds -/

/-- How a kernel's source elements are stored and what the destination wants. -/
inductive Kind
  | b32   -- a 32-bit word, moved as bits (f32, i32)
  | b16   -- a 16-bit half, moved as bits (f16, bf16, i16)
  | f16
  | bf16
  | f32
deriving BEq, Repr

/-- Load element `e` of `buf` as its raw bits (32 or 16). -/
def load (k : Kind) (buf : String) (e : SlangExpr) : SlangExpr :=
  match k with
  | .b32 | .f32 => .index (v buf) e
  | _ => .call ("ld16_" ++ buf) [e]

def is16 : Kind → Bool
  | .b16 | .f16 | .bf16 => true
  | _ => false

/-- Convert raw source bits of kind `s` to raw destination bits of kind `d`. -/
def conv (s d : Kind) (x : SlangExpr) : SlangExpr :=
  let toF32 : SlangExpr := match s with
    | .f16 => .call "f16_to_f32" [x]
    | .bf16 => shl x (u 16)
    | _ => x
  match s, d with
  | .b32, _ | .b16, _ => x
  | _, .f16 => if s == .f16 then x else .call "f32_to_f16" [toF32]
  | _, .bf16 => if s == .bf16 then x else .call "f32_to_bf16" [toF32]
  | _, _ => toF32

/-- The helper functions `conv s d` and the loads of `bufs` need. -/
def convHelpers (s d : Kind) (bufs : List String) : List SlangFunctionDecl :=
  (if is16 s then bufs.map fnLd16 else []) ++
  (if s == .f16 && d != .f16 then [fnF16ToF32] else []) ++
  (if d == .f16 && s != .f16 && s != .b16 then [fnF32ToF16] else []) ++
  (if d == .bf16 && s != .bf16 && s != .b16 then [fnF32ToBf16] else [])

/-- `uint val(uint i0, uint i1, uint i2, uint i3) { body }` -/
def fnVal (body : List SlangStmt) : SlangFunctionDecl :=
  { retType := tU, name := "val", params := idx4, body := body }

/-! ## Entries -/

def declIdx (names : List String) : List SlangStmt := names.map (fun n => .decl tU n)

/-- One thread per destination element: `dst[off(i)] = val(i)`. -/
def entry32 : SlangFunctionDecl :=
  entry1D threadgroup
    (declIdx ["i0", "i1", "i2", "i3"] ++
     [ .expr (.call "unravel4" ([v "e", u wDst] ++ iv))
     , .assign (.index (v "dst") (.call "off4" ([u wDst] ++ iv))) (.call "val" iv) ])

/-- The chain bit `k` of word 55 is set. -/
def mbit (k : Nat) : SlangExpr := neq (band (v "m") (u (2 ^ k))) (u 0)

/-- One thread per destination element; the thread at the even address of
    a word writes it (see the module doc). -/
def entry16 : SlangFunctionDecl :=
  let w := shr (v "a") (u 1)
  let dw := .index (v "dst") w
  let hasPrev :=
    lor (lor (lor (land (mbit 0) (gt (v "i0") (u 0))) (land (mbit 1) (gt (v "i1") (u 0))))
             (land (mbit 2) (gt (v "i2") (u 0))))
        (land (mbit 3) (gt (v "i3") (u 0)))
  let nx (k : Nat) := lt (add (v ("i" ++ toString k)) (u 1)) (pwN (wDst + tNe + k))
  entry1D threadgroup
    (declIdx ["i0", "i1", "i2", "i3"] ++
     [ .expr (.call "unravel4" ([v "e", u wDst] ++ iv))
     , .declInit tU "a" (.call "off4" ([u wDst] ++ iv))
     , .declInit tU "m" (pwN wMerge)
     , .ifNoElse (neq (band (v "a") (u 1)) (u 0))
         [ .ifNoElse hasPrev [ .ret none ]
         , .assign dw (bor (band dw (u 65535)) (shl (.call "val" iv) (u 16)))
         , .ret none ]
     , .declInit tU "lo" (.call "val" iv)
     , .declInit tU "k0" (v "i0")
     , .declInit tU "k1" (v "i1")
     , .declInit tU "k2" (v "i2")
     , .declInit tU "k3" (v "i3")
     , .declInit tB "hasNext" (.litBool true)
     , .ifThen (land (mbit 0) (nx 0))
         [ .assign (v "k0") (add (v "i0") (u 1)) ]
         [ .ifThen (land (mbit 1) (nx 1))
             [ .assign (v "k0") (u 0), .assign (v "k1") (add (v "i1") (u 1)) ]
             [ .ifThen (land (mbit 2) (nx 2))
                 [ .assign (v "k0") (u 0), .assign (v "k1") (u 0), .assign (v "k2") (add (v "i2") (u 1)) ]
                 [ .ifThen (land (mbit 3) (nx 3))
                     [ .assign (v "k0") (u 0), .assign (v "k1") (u 0), .assign (v "k2") (u 0)
                     , .assign (v "k3") (add (v "i3") (u 1)) ]
                     [ .assign (v "hasNext") (.litBool false) ] ] ] ]
     , .ifThen (v "hasNext")
         [ .assign dw (bor (v "lo") (shl (.call "val" [v "k0", v "k1", v "k2", v "k3"]) (u 16))) ]
         [ .assign dw (bor (band dw (u 4294901760)) (v "lo")) ] ])

/-- A move kernel: all-`uint` bindings, the shared helpers, `extra`
    helpers, `val`, then the entry for the destination's width. -/
def moveModule (dst16 : Bool) (extra : List SlangFunctionDecl) (valFn : SlangFunctionDecl) :
    SlangShaderModule :=
  kernelModule .uint .uint .uint .uint
    ([fnPw, fnUnravel4, fnOff4] ++ extra ++ [valFn])
    (if dst16 then entry16 else entry32)

/-! ## Pinned text of the shared pieces -/

def ld16S0Text : String :=
"uint ld16_s0(uint e) {
  return ((s0[(e >> 1u)] >> ((e & 1u) * 16u)) & 65535u);
}"
example : emitFunction (fnLd16 "s0") = ld16S0Text := by native_decide
example : emitFunction (fnLd16 "s1") = ld16S0Text.replace "s0" "s1" := by native_decide

def f16ToF32Text : String :=
"uint f16_to_f32(uint h) {
  uint s = ((h & 32768u) << 16u);
  uint x = ((h >> 10u) & 31u);
  uint m = (h & 1023u);
  if ((x == 31u)) {
    return ((s | 2139095040u) | (m << 13u));
  }
  if ((x != 0u)) {
    return ((s | ((x + 112u) << 23u)) | (m << 13u));
  }
  if ((m == 0u)) {
    return s;
  }
  uint x2 = 113u;
  while (((m & 1024u) == 0u)) {
    m = (m << 1u);
    x2 = (x2 - 1u);
  }
  return ((s | (x2 << 23u)) | ((m & 1023u) << 13u));
}"
example : emitFunction fnF16ToF32 = f16ToF32Text := by native_decide

def f32ToF16Text : String :=
"uint f32_to_f16(uint b) {
  uint s = ((b >> 16u) & 32768u);
  uint a = (b & 2147483647u);
  if ((a > 2139095040u)) {
    return (s | 32256u);
  }
  if ((a >= 1199566848u)) {
    return (s | 31744u);
  }
  if ((a >= 947912704u)) {
    uint m = (a - 939524096u);
    return (s | (((m + 4095u) + ((m >> 13u) & 1u)) >> 13u));
  }
  uint x = (a >> 23u);
  if ((x < 102u)) {
    return s;
  }
  uint q = ((a & 8388607u) | 8388608u);
  uint sh = (126u - x);
  return (s | (((q + ((1u << (sh - 1u)) - 1u)) + ((q >> sh) & 1u)) >> sh));
}"
example : emitFunction fnF32ToF16 = f32ToF16Text := by native_decide

def f32ToBf16Text : String :=
"uint f32_to_bf16(uint b) {
  if (((b & 2147483647u) > 2139095040u)) {
    return ((b >> 16u) | 64u);
  }
  return ((b + (32767u + ((b >> 16u) & 1u))) >> 16u);
}"
example : emitFunction fnF32ToBf16 = f32ToBf16Text := by native_decide

def entry32Text : String :=
"[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  dst[off4(1u, i0, i1, i2, i3)] = val(i0, i1, i2, i3);
}"
example : emitFunction entry32 = entry32Text := by native_decide

def entry16Text : String :=
"[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint a = off4(1u, i0, i1, i2, i3);
  uint m = pw(55u);
  if (((a & 1u) != 0u)) {
    if (((((((m & 1u) != 0u) && (i0 > 0u)) || (((m & 2u) != 0u) && (i1 > 0u))) || (((m & 4u) != 0u) && (i2 > 0u))) || (((m & 8u) != 0u) && (i3 > 0u)))) {
      return;
    }
    dst[(a >> 1u)] = ((dst[(a >> 1u)] & 65535u) | (val(i0, i1, i2, i3) << 16u));
    return;
  }
  uint lo = val(i0, i1, i2, i3);
  uint k0 = i0;
  uint k1 = i1;
  uint k2 = i2;
  uint k3 = i3;
  bool hasNext = true;
  if ((((m & 1u) != 0u) && ((i0 + 1u) < pw(1u)))) {
    k0 = (i0 + 1u);
  } else {
    if ((((m & 2u) != 0u) && ((i1 + 1u) < pw(2u)))) {
      k0 = 0u;
      k1 = (i1 + 1u);
    } else {
      if ((((m & 4u) != 0u) && ((i2 + 1u) < pw(3u)))) {
        k0 = 0u;
        k1 = 0u;
        k2 = (i2 + 1u);
      } else {
        if ((((m & 8u) != 0u) && ((i3 + 1u) < pw(4u)))) {
          k0 = 0u;
          k1 = 0u;
          k2 = 0u;
          k3 = (i3 + 1u);
        } else {
          hasNext = false;
        }
      }
    }
  }
  if (hasNext) {
    dst[(a >> 1u)] = (lo | (val(k0, k1, k2, k3) << 16u));
  } else {
    dst[(a >> 1u)] = ((dst[(a >> 1u)] & 4294901760u) | lo);
  }
}"
example : emitFunction entry16 = entry16Text := by native_decide

/-- The text every move kernel starts with: Slot, the all-`uint`
    globals, pw, unravel4, off4. -/
def preludeText : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}"

/-- A move kernel's full text from pinned pieces: the prelude, the helper
    texts, the val text, the entry text. -/
def assemble (helpers : List String) (valText : String) (dst16 : Bool) : String :=
  String.intercalate "\n\n" ([preludeText] ++ helpers ++ [valText, if dst16 then entry16Text else entry32Text])

end Ggml.SlangCodegen.Move
