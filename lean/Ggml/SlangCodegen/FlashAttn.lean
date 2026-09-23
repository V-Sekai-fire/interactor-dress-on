import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.FlashAttn` — ggml FLASH_ATTN_EXT, no mask (family K8)

    dst[:, h, i, b] = softmax_j( cap(scale · q_{i,h,b} · k_{j,hk,bk}) ) · v_{:,hv,bv}

Q (s0) is f32 `[D, N, Hq, B]`; K (s1) and V (s2) are `[D, Lk, Hk, Bk]`,
both f32 or both f16 (a `uint` word buffer read with `ld_f16_*`); the
destination is f32 `[D, Hq, N, B]` (ggml's permuted output). Grouped-query
attention is ggml's broadcast: head `h` reads K head `h / (Hq / Hk)` and
batch `b` reads K batch `b / (B / Bk)` (words 55-58). `cap(s)` is ggml's
logit softcap: `softcap · tanh(s)`, with `scale` already divided by
`softcap` on the host (word 59), exactly as ggml-cpu does it; word 60 is
`softcap`, 0 for none. Masks, ALiBi and sinks are not here: the packer
(`guest/ggml-rd/ops/flash_attn_ext.cpp`) refuses a node that has a mask or
sinks (ALiBi only scales a mask).

With f16 K, ggml-cpu rounds Q to f16 before the dot product (its K
`vec_dot_type`); the f16 kernels do the same, so both sides multiply the
same operands. Accumulation is f32 (ggml-cpu's f16 V accumulates in f16;
test-backend-ops' NMSE 5e-4 covers the difference).

## Tiled kernel (`flash_attn_ext_<kv>_d<D>`, the GPU path)

One work group = 16 queries of one head and batch, 128 threads as 16 rows
× 8 lanes (`r = t / 8`, `l = t % 8`); grid `(ceil(N / 16), Hq, B)`.
Groupshared: the Q tile `Qs[16][D+1]`, a K/V tile of 32 keys `Ks[32][D+1]`,
`Vs[32][D]`, and the scores `Ss[16][33]` (43,264 bytes at D = 128; the pads
keep the lanes of a row and the rows of a warp on different banks). Per K/V
tile: all 128 threads load it (coalesced), lane `l` of row `r` scores keys
`l, l+8, l+16, l+24` (a full D-long dot product each), then every lane of
the row reads the 32 scores, takes the new running max `m`, rescales its
`l` sum and its accumulators by `exp(m_old - m)` and adds `exp(s_j - m) ·
V[j][c]` for its own D/8 columns `c = l + 8 i` (the online softmax of
Milakov & Gimelshein / FlashAttention). The 8 lanes of a row compute the
same max and sum in the same order, so they agree bit for bit. Keys past
Lk score -inf and load zeros. The accumulators are unrolled here in Lean
(`acc0 .. acc{D/8-1}`), so the GPU keeps them in registers.

## Serial sibling (`flash_attn_ext_<kv>_d<D>_serial`)

`slangc -target cpp` refuses `GroupMemoryBarrierWithGroupSync` (E36107,
`Cloth.SlangCodegen.DotReduceSerial`), so the tiled kernel has no cpp
emit. Its sibling computes the same thing with the same operation order
(32-key tiles, the same dot products, max, rescale and sums), one thread
per query row and no groupshared: it is what the host L2 test
(`tests/ggml_rd_kernels`) runs through the cpp emit, and on the GPU it is
the A/B for the tiled kernel (`GGML_RD_SERIAL=1`).
-/

namespace Ggml.SlangCodegen.FlashAttn

open LeanSlang
open Ggml.SlangCodegen.Common

/-- K/V element type. -/
inductive Kv
  | f32
  | f16
deriving BEq, DecidableEq

def Kv.name : Kv → String
  | .f32 => "f32"
  | .f16 => "f16"

/-- Queries per work group (rows) and lanes per row. -/
def rows : Nat := 16
def lanes : Nat := 8
def threadgroup : Nat := rows * lanes
/-- Keys per K/V tile. -/
def tileKeys : Nat := 32
/-- Threads per work group of the serial sibling. -/
def serialThreadgroup : Nat := 64

/-- Derived words (the op's own, 55-63). -/
def wRk2 : Nat := 55
def wRk3 : Nat := 56
def wRv2 : Nat := 57
def wRv3 : Nat := 58
def wScale : Nat := 59
def wSoftcap : Nat := 60

example : wDerived + 2 = wRk2 := by native_decide
example : wSoftcap < wordsPerSlot := by native_decide

/-! ## Shorthands -/

private def fT : SlangType := .scalar .float
private def uT : SlangType := .scalar .uint
private def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
private def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
private def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
private def f0 : SlangExpr := .litFloat 0.0
/-- -inf as a float: `asfloat(0xFF800000u)`. -/
private def negInf : SlangExpr := .call "asfloat" [u 4286578688]
private def ix2 (a : String) (i j : SlangExpr) : SlangExpr := .index (.index (v a) i) j
private def off4E (w : Nat) (i0 i1 i2 i3 : SlangExpr) : SlangExpr :=
  .call "off4" [u w, i0, i1, i2, i3]

/-- Element `e` of K (`s1`) or V (`s2`) as a float. -/
def ldKv (kv : Kv) (buf : String) (e : SlangExpr) : SlangExpr :=
  match kv with
  | .f32 => .index (v buf) e
  | .f16 => .call ("ld_f16_" ++ buf) [e]

/-- Q as the dot product sees it: f32, or rounded to f16 for f16 K. -/
def qAsKv (kv : Kv) (x : SlangExpr) : SlangExpr :=
  match kv with
  | .f32 => x
  | .f16 => .call "f16tof32" [.call "f32tof16" [x]]

/-- The words both kernels start from: Lk, the K/V head and batch of
    (h, b), the scale and the softcap, and the element offsets of Q's row
    `(iq, h, b)` and of key 0 of K's and V's `(hk, bk)` / `(hv, bv)` with
    their key strides. Rows are contiguous (the packer requires nb0 = the
    type size), so element c of a row is its offset plus c. -/
def headWords : List SlangStmt :=
  [ .declInit uT "lk" (pwN (wSrc 1 + tNe + 1))
  , .declInit uT "hk" (udiv (v "h") (pwN wRk2))
  , .declInit uT "bk" (udiv (v "b") (pwN wRk3))
  , .declInit uT "hv" (udiv (v "h") (pwN wRv2))
  , .declInit uT "bv" (udiv (v "b") (pwN wRv3))
  , .declInit fT "scale" (pfN wScale)
  , .declInit fT "cap" (pfN wSoftcap)
  , .declInit uT "qb" (off4E (wSrc 0) (u 0) (v "iq") (v "h") (v "b"))
  , .declInit uT "kb" (off4E (wSrc 1) (u 0) (u 0) (v "hk") (v "bk"))
  , .declInit uT "kn" (pwN (wSrc 1 + tNb + 1))
  , .declInit uT "vb" (off4E (wSrc 2) (u 0) (u 0) (v "hv") (v "bv"))
  , .declInit uT "vn" (pwN (wSrc 2 + tNb + 1)) ]

/-- `s = dot * scale`, then the softcap when there is one. -/
def scoreFinish (s : String) : List SlangStmt :=
  [ .assign (v s) (mul (v s) (v "scale"))
  , .ifNoElse (.bin "!=" (v "cap") f0)
      [ .assign (v s) (mul (v "cap") (.call "tanh" [v s])) ] ]

/-- Offset of K (or V) element c of key `key`. -/
def kOff (c key : SlangExpr) : SlangExpr := add (add (v "kb") (mul key (v "kn"))) c
def vOff (c key : SlangExpr) : SlangExpr := add (add (v "vb") (mul key (v "vn"))) c
/-- Offset of Q element c of this row. -/
def qOff (c : SlangExpr) : SlangExpr := add (v "qb") c

/-! ## The tiled kernel -/

def tiledShared (d : Nat) : List SlangGroupSharedDecl :=
  [ { name := "Qs", elemType := fT, dims := [rows, d + 1] }
  , { name := "Ks", elemType := fT, dims := [tileKeys, d + 1] }
  , { name := "Vs", elemType := fT, dims := [tileKeys, d] }
  , { name := "Ss", elemType := fT, dims := [rows, tileKeys + 1] } ]

/-- Groupshared bytes of the tiled kernel. -/
def tiledSharedBytes (d : Nat) : Nat :=
  4 * (rows * (d + 1) + tileKeys * (d + 1) + tileKeys * d + rows * (tileKeys + 1))

example : tiledSharedBytes 128 = 43264 := by native_decide
example : tiledSharedBytes 128 ≤ 49152 := by native_decide

private def barrier : SlangStmt := .expr (.call "GroupMemoryBarrierWithGroupSync" [])

/-- `acc<i>`, the accumulator of column `l + 8 i`. -/
private def acc (i : Nat) : String := "acc" ++ toString i
private def colOf (i : Nat) : SlangExpr :=
  if i == 0 then v "l" else add (v "l") (u (lanes * i))

/-- `d<q>`, the score of key `l + 8 q` of this lane. -/
private def dk (q : Nat) : String := "d" ++ toString q
private def keyOf (q : Nat) : SlangExpr := colOf q
private def keysOf : List Nat := List.range (tileKeys / lanes)

def tiledMain (kv : Kv) (d : Nat) : SlangFunctionDecl :=
  let nc := d / lanes
  let cols := List.range nc
  { attrs := [.shaderCompute, .numthreads threadgroup 1 1]
  , name := "main"
  , params := [ ⟨"gid", .vec .uint 3, .svGroupId, none, none, .qIn⟩
              , ⟨"lid", .vec .uint 3, .svGroupThreadId, none, none, .qIn⟩ ]
  , body :=
      [ .declInit uT "t" (.member (v "lid") "x")
      , .declInit uT "r" (udiv (v "t") (u lanes))
      , .declInit uT "l" (urem (v "t") (u lanes))
      , .declInit uT "iq" (add (mul (.member (v "gid") "x") (u rows)) (v "r"))
      , .declInit uT "h" (.member (v "gid") "y")
      , .declInit uT "b" (.member (v "gid") "z") ] ++
      headWords ++
      [ .declInit (.scalar .bool) "valid" (lt (v "iq") (pwN (wSrc 0 + tNe + 1)))
      -- The Q tile: row r, columns l + 8 i (zeros past N).
      , .forCount "i" (u 0) (u nc)
          [ .declInit uT "c" (add (v "l") (mul (v "i") (u lanes)))
          , .declInit fT "x" f0
          , .ifNoElse (v "valid")
              [ .assign (v "x") (qAsKv kv (.index (v "s0") (qOff (v "c")))) ]
          , .assign (ix2 "Qs" (v "r") (v "c")) (v "x") ]
      , .declInit fT "m" negInf
      , .declInit fT "ls" f0 ] ++
      cols.map (fun i => .declInit fT (acc i) f0) ++
      [ .declInit uT "nt" (udiv (add (v "lk") (u (tileKeys - 1))) (u tileKeys))
      , .forCount "tt" (u 0) (v "nt") (
          [ .declInit uT "k0" (mul (v "tt") (u tileKeys))
          -- The last tile's readers are done (and, the first time, Q is written).
          , barrier
          , .forCount "i" (u 0) (u (tileKeys * d / threadgroup))
              [ .declInit uT "f" (add (v "t") (mul (v "i") (u threadgroup)))
              , .declInit uT "j" (udiv (v "f") (u d))
              , .declInit uT "c" (urem (v "f") (u d))
              , .declInit uT "key" (add (v "k0") (v "j"))
              , .declInit fT "kx" f0
              , .declInit fT "vx" f0
              , .ifNoElse (lt (v "key") (v "lk"))
                  [ .assign (v "kx") (ldKv kv "s1" (kOff (v "c") (v "key")))
                  , .assign (v "vx") (ldKv kv "s2" (vOff (v "c") (v "key"))) ]
              , .assign (ix2 "Ks" (v "j") (v "c")) (v "kx")
              , .assign (ix2 "Vs" (v "j") (v "c")) (v "vx") ]
          , barrier
          -- Scores: lane l of row r takes keys l, l + 8, l + 16, l + 24,
          -- one pass over c (each Q element read once for the four).
          ] ++
          keysOf.map (fun q => .declInit fT (dk q) f0) ++
          [ .forCount "c" (u 0) (u d)
              ([ .declInit fT "qc" (ix2 "Qs" (v "r") (v "c")) ] ++
               keysOf.map (fun q => .assign (v (dk q))
                 (add (v (dk q)) (mul (v "qc") (ix2 "Ks" (keyOf q) (v "c")))))) ] ++
          keysOf.flatMap (fun q =>
            scoreFinish (dk q) ++
            [ .ifNoElse (ge (add (v "k0") (keyOf q)) (v "lk")) [ .assign (v (dk q)) negInf ]
            , .assign (ix2 "Ss" (v "r") (keyOf q)) (v (dk q)) ]) ++
          [ barrier
          -- The online softmax, every lane of the row alike.
          , .declInit fT "mt" (v "m")
          , .forCount "j" (u 0) (u tileKeys)
              [ .assign (v "mt") (.call "max" [v "mt", ix2 "Ss" (v "r") (v "j")]) ]
          , .declInit fT "corr" (.call "exp" [sub (v "m") (v "mt")]) ] ++
          cols.map (fun i => .assign (v (acc i)) (mul (v (acc i)) (v "corr"))) ++
          [ .declInit fT "ps" f0
          , .forCount "j" (u 0) (u tileKeys)
              ([ .declInit fT "p" (.call "exp" [sub (ix2 "Ss" (v "r") (v "j")) (v "mt")])
               , .assign (v "ps") (add (v "ps") (v "p")) ] ++
               cols.map (fun i => .assign (v (acc i))
                 (add (v (acc i)) (mul (v "p") (ix2 "Vs" (v "j") (colOf i))))))
          , .assign (v "ls") (add (mul (v "ls") (v "corr")) (v "ps"))
          , .assign (v "m") (v "mt") ])
      , .ifNoElse (v "valid")
          ([ .declInit fT "inv" (.ternary (.bin "==" (v "ls") f0) f0 (.bin "/" (.litFloat 1.0) (v "ls"))) ] ++
           cols.map (fun i => .assign (.index (v "dst") (off4E wDst (colOf i) (v "h") (v "iq") (v "b")))
             (mul (v (acc i)) (v "inv")))) ] }

def helpers (kv : Kv) : List SlangFunctionDecl :=
  match kv with
  | .f32 => [fnPw, fnPf, fnOff4]
  | .f16 => [fnPw, fnPf, fnOff4, fnLdF16 "s1", fnLdF16 "s2"]

def kvScalar : Kv → Scalar
  | .f32 => .float
  | .f16 => .uint

def tiled (kv : Kv) (d : Nat) : SlangShaderModule :=
  { kernelModule .float (kvScalar kv) (kvScalar kv) .float (helpers kv) (tiledMain kv d) with
    groupShared := tiledShared d }

/-! ## The serial sibling -/

def serialBody (kv : Kv) (d : Nat) : List SlangStmt :=
  [ .declInit uT "nq" (pwN (wSrc 0 + tNe + 1))
  , .declInit uT "iq" (urem (v "e") (v "nq"))
  , .declInit uT "rr" (udiv (v "e") (v "nq"))
  , .declInit uT "h" (urem (v "rr") (pwN (wSrc 0 + tNe + 2)))
  , .declInit uT "b" (udiv (v "rr") (pwN (wSrc 0 + tNe + 2))) ] ++
  headWords ++
  [ .declareArray fT "qv" d
  , .declareArray fT "acc" d
  , .declareArray fT "sc" tileKeys
  , .forCount "c" (u 0) (u d)
      [ .assign (.index (v "qv") (v "c"))
          (qAsKv kv (.index (v "s0") (qOff (v "c"))))
      , .assign (.index (v "acc") (v "c")) f0 ]
  , .declInit fT "m" negInf
  , .declInit fT "ls" f0
  , .declInit uT "nt" (udiv (add (v "lk") (u (tileKeys - 1))) (u tileKeys))
  , .forCount "tt" (u 0) (v "nt")
      [ .declInit uT "k0" (mul (v "tt") (u tileKeys))
      , .forCount "j" (u 0) (u tileKeys)
          [ .declInit uT "key" (add (v "k0") (v "j"))
          , .declInit fT "s" negInf
          , .ifNoElse (lt (v "key") (v "lk"))
              ([ .assign (v "s") f0
               , .forCount "c" (u 0) (u d)
                   [ .assign (v "s") (add (v "s") (mul (.index (v "qv") (v "c")) (ldKv kv "s1" (kOff (v "c") (v "key"))))) ] ] ++
               scoreFinish "s")
          , .assign (.index (v "sc") (v "j")) (v "s") ]
      , .declInit fT "mt" (v "m")
      , .forCount "j" (u 0) (u tileKeys)
          [ .assign (v "mt") (.call "max" [v "mt", .index (v "sc") (v "j")]) ]
      , .declInit fT "corr" (.call "exp" [sub (v "m") (v "mt")])
      , .forCount "c" (u 0) (u d)
          [ .assign (.index (v "acc") (v "c")) (mul (.index (v "acc") (v "c")) (v "corr")) ]
      , .declInit fT "ps" f0
      , .forCount "j" (u 0) (u tileKeys)
          [ .declInit fT "p" (.call "exp" [sub (.index (v "sc") (v "j")) (v "mt")])
          , .assign (v "ps") (add (v "ps") (v "p"))
          , .declInit uT "key" (add (v "k0") (v "j"))
          , .ifNoElse (lt (v "key") (v "lk"))
              [ .forCount "c" (u 0) (u d)
                  [ .assign (.index (v "acc") (v "c"))
                      (add (.index (v "acc") (v "c")) (mul (v "p") (ldKv kv "s2" (vOff (v "c") (v "key"))))) ] ] ]
      , .assign (v "ls") (add (mul (v "ls") (v "corr")) (v "ps"))
      , .assign (v "m") (v "mt") ]
  , .declInit fT "inv" (.ternary (.bin "==" (v "ls") f0) f0 (.bin "/" (.litFloat 1.0) (v "ls")))
  , .forCount "c" (u 0) (u d)
      [ .assign (.index (v "dst") (off4E wDst (v "c") (v "h") (v "iq") (v "b")))
          (mul (.index (v "acc") (v "c")) (v "inv")) ] ]

def serial (kv : Kv) (d : Nat) : SlangShaderModule :=
  kernelModule .float (kvScalar kv) (kvScalar kv) .float (helpers kv)
    (entry1D serialThreadgroup (serialBody kv d))

/-! ## The kernels -/

def name (kv : Kv) (d : Nat) : String := "flash_attn_ext_" ++ kv.name ++ "_d" ++ toString d

def kernels : List (String × SlangShaderModule) :=
  [Kv.f32, Kv.f16].flatMap fun kv =>
    [64, 128].flatMap fun d =>
      [ (name kv d, tiled kv d), (name kv d ++ "_serial", serial kv d) ]

example : (kernels.map Prod.fst) =
    [ "flash_attn_ext_f32_d64", "flash_attn_ext_f32_d64_serial"
    , "flash_attn_ext_f32_d128", "flash_attn_ext_f32_d128_serial"
    , "flash_attn_ext_f16_d64", "flash_attn_ext_f16_d64_serial"
    , "flash_attn_ext_f16_d128", "flash_attn_ext_f16_d128_serial" ] := by native_decide

/-! ## Pinned emission

The f32 kernels are pinned in full (the tiled D = 128 one is the census
kernel); the f16 kernels are the f32 text with the K/V bindings as `uint`
words, the two `ld_f16_*` readers after `off4`, K and V read through them,
and Q rounded to f16; the D = 128 serial sibling is the D = 64 one with its
column loops and arrays widened. -/

/-- The f32 tiled kernel, D = 64. -/
def expectedTiledF32D64 : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float Qs[16][65];
groupshared float Ks[32][65];
groupshared float Vs[32][64];
groupshared float Ss[16][33];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint t = lid.x;
  uint r = (t / 8u);
  uint l = (t % 8u);
  uint iq = ((gid.x * 16u) + r);
  uint h = gid.y;
  uint b = gid.z;
  uint lk = pw(20u);
  uint hk = (h / pw(55u));
  uint bk = (b / pw(56u));
  uint hv = (h / pw(57u));
  uint bv = (b / pw(58u));
  float scale = pf(59u);
  float cap = pf(60u);
  uint qb = off4(10u, 0u, iq, h, b);
  uint kb = off4(19u, 0u, 0u, hk, bk);
  uint kn = pw(24u);
  uint vb = off4(28u, 0u, 0u, hv, bv);
  uint vn = pw(33u);
  bool valid = (iq < pw(11u));
  for (uint i = 0u; i < 8u; ++i) {
    uint c = (l + (i * 8u));
    float x = 0.000000;
    if (valid) {
      x = s0[(qb + c)];
    }
    Qs[r][c] = x;
  }
  float m = asfloat(4286578688u);
  float ls = 0.000000;
  float acc0 = 0.000000;
  float acc1 = 0.000000;
  float acc2 = 0.000000;
  float acc3 = 0.000000;
  float acc4 = 0.000000;
  float acc5 = 0.000000;
  float acc6 = 0.000000;
  float acc7 = 0.000000;
  uint nt = ((lk + 31u) / 32u);
  for (uint tt = 0u; tt < nt; ++tt) {
    uint k0 = (tt * 32u);
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0u; i < 16u; ++i) {
      uint f = (t + (i * 128u));
      uint j = (f / 64u);
      uint c = (f % 64u);
      uint key = (k0 + j);
      float kx = 0.000000;
      float vx = 0.000000;
      if ((key < lk)) {
        kx = s1[((kb + (key * kn)) + c)];
        vx = s2[((vb + (key * vn)) + c)];
      }
      Ks[j][c] = kx;
      Vs[j][c] = vx;
    }
    GroupMemoryBarrierWithGroupSync();
    float d0 = 0.000000;
    float d1 = 0.000000;
    float d2 = 0.000000;
    float d3 = 0.000000;
    for (uint c = 0u; c < 64u; ++c) {
      float qc = Qs[r][c];
      d0 = (d0 + (qc * Ks[l][c]));
      d1 = (d1 + (qc * Ks[(l + 8u)][c]));
      d2 = (d2 + (qc * Ks[(l + 16u)][c]));
      d3 = (d3 + (qc * Ks[(l + 24u)][c]));
    }
    d0 = (d0 * scale);
    if ((cap != 0.000000)) {
      d0 = (cap * tanh(d0));
    }
    if (((k0 + l) >= lk)) {
      d0 = asfloat(4286578688u);
    }
    Ss[r][l] = d0;
    d1 = (d1 * scale);
    if ((cap != 0.000000)) {
      d1 = (cap * tanh(d1));
    }
    if (((k0 + (l + 8u)) >= lk)) {
      d1 = asfloat(4286578688u);
    }
    Ss[r][(l + 8u)] = d1;
    d2 = (d2 * scale);
    if ((cap != 0.000000)) {
      d2 = (cap * tanh(d2));
    }
    if (((k0 + (l + 16u)) >= lk)) {
      d2 = asfloat(4286578688u);
    }
    Ss[r][(l + 16u)] = d2;
    d3 = (d3 * scale);
    if ((cap != 0.000000)) {
      d3 = (cap * tanh(d3));
    }
    if (((k0 + (l + 24u)) >= lk)) {
      d3 = asfloat(4286578688u);
    }
    Ss[r][(l + 24u)] = d3;
    GroupMemoryBarrierWithGroupSync();
    float mt = m;
    for (uint j = 0u; j < 32u; ++j) {
      mt = max(mt, Ss[r][j]);
    }
    float corr = exp((m - mt));
    acc0 = (acc0 * corr);
    acc1 = (acc1 * corr);
    acc2 = (acc2 * corr);
    acc3 = (acc3 * corr);
    acc4 = (acc4 * corr);
    acc5 = (acc5 * corr);
    acc6 = (acc6 * corr);
    acc7 = (acc7 * corr);
    float ps = 0.000000;
    for (uint j = 0u; j < 32u; ++j) {
      float p = exp((Ss[r][j] - mt));
      ps = (ps + p);
      acc0 = (acc0 + (p * Vs[j][l]));
      acc1 = (acc1 + (p * Vs[j][(l + 8u)]));
      acc2 = (acc2 + (p * Vs[j][(l + 16u)]));
      acc3 = (acc3 + (p * Vs[j][(l + 24u)]));
      acc4 = (acc4 + (p * Vs[j][(l + 32u)]));
      acc5 = (acc5 + (p * Vs[j][(l + 40u)]));
      acc6 = (acc6 + (p * Vs[j][(l + 48u)]));
      acc7 = (acc7 + (p * Vs[j][(l + 56u)]));
    }
    ls = ((ls * corr) + ps);
    m = mt;
  }
  if (valid) {
    float inv = ((ls == 0.000000) ? 0.000000 : (1.000000 / ls));
    dst[off4(1u, l, h, iq, b)] = (acc0 * inv);
    dst[off4(1u, (l + 8u), h, iq, b)] = (acc1 * inv);
    dst[off4(1u, (l + 16u), h, iq, b)] = (acc2 * inv);
    dst[off4(1u, (l + 24u), h, iq, b)] = (acc3 * inv);
    dst[off4(1u, (l + 32u), h, iq, b)] = (acc4 * inv);
    dst[off4(1u, (l + 40u), h, iq, b)] = (acc5 * inv);
    dst[off4(1u, (l + 48u), h, iq, b)] = (acc6 * inv);
    dst[off4(1u, (l + 56u), h, iq, b)] = (acc7 * inv);
  }
}"

/-- The f32 tiled kernel, D = 128: the census kernel. -/
def expectedTiledF32D128 : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float Qs[16][129];
groupshared float Ks[32][129];
groupshared float Vs[32][128];
groupshared float Ss[16][33];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint t = lid.x;
  uint r = (t / 8u);
  uint l = (t % 8u);
  uint iq = ((gid.x * 16u) + r);
  uint h = gid.y;
  uint b = gid.z;
  uint lk = pw(20u);
  uint hk = (h / pw(55u));
  uint bk = (b / pw(56u));
  uint hv = (h / pw(57u));
  uint bv = (b / pw(58u));
  float scale = pf(59u);
  float cap = pf(60u);
  uint qb = off4(10u, 0u, iq, h, b);
  uint kb = off4(19u, 0u, 0u, hk, bk);
  uint kn = pw(24u);
  uint vb = off4(28u, 0u, 0u, hv, bv);
  uint vn = pw(33u);
  bool valid = (iq < pw(11u));
  for (uint i = 0u; i < 16u; ++i) {
    uint c = (l + (i * 8u));
    float x = 0.000000;
    if (valid) {
      x = s0[(qb + c)];
    }
    Qs[r][c] = x;
  }
  float m = asfloat(4286578688u);
  float ls = 0.000000;
  float acc0 = 0.000000;
  float acc1 = 0.000000;
  float acc2 = 0.000000;
  float acc3 = 0.000000;
  float acc4 = 0.000000;
  float acc5 = 0.000000;
  float acc6 = 0.000000;
  float acc7 = 0.000000;
  float acc8 = 0.000000;
  float acc9 = 0.000000;
  float acc10 = 0.000000;
  float acc11 = 0.000000;
  float acc12 = 0.000000;
  float acc13 = 0.000000;
  float acc14 = 0.000000;
  float acc15 = 0.000000;
  uint nt = ((lk + 31u) / 32u);
  for (uint tt = 0u; tt < nt; ++tt) {
    uint k0 = (tt * 32u);
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0u; i < 32u; ++i) {
      uint f = (t + (i * 128u));
      uint j = (f / 128u);
      uint c = (f % 128u);
      uint key = (k0 + j);
      float kx = 0.000000;
      float vx = 0.000000;
      if ((key < lk)) {
        kx = s1[((kb + (key * kn)) + c)];
        vx = s2[((vb + (key * vn)) + c)];
      }
      Ks[j][c] = kx;
      Vs[j][c] = vx;
    }
    GroupMemoryBarrierWithGroupSync();
    float d0 = 0.000000;
    float d1 = 0.000000;
    float d2 = 0.000000;
    float d3 = 0.000000;
    for (uint c = 0u; c < 128u; ++c) {
      float qc = Qs[r][c];
      d0 = (d0 + (qc * Ks[l][c]));
      d1 = (d1 + (qc * Ks[(l + 8u)][c]));
      d2 = (d2 + (qc * Ks[(l + 16u)][c]));
      d3 = (d3 + (qc * Ks[(l + 24u)][c]));
    }
    d0 = (d0 * scale);
    if ((cap != 0.000000)) {
      d0 = (cap * tanh(d0));
    }
    if (((k0 + l) >= lk)) {
      d0 = asfloat(4286578688u);
    }
    Ss[r][l] = d0;
    d1 = (d1 * scale);
    if ((cap != 0.000000)) {
      d1 = (cap * tanh(d1));
    }
    if (((k0 + (l + 8u)) >= lk)) {
      d1 = asfloat(4286578688u);
    }
    Ss[r][(l + 8u)] = d1;
    d2 = (d2 * scale);
    if ((cap != 0.000000)) {
      d2 = (cap * tanh(d2));
    }
    if (((k0 + (l + 16u)) >= lk)) {
      d2 = asfloat(4286578688u);
    }
    Ss[r][(l + 16u)] = d2;
    d3 = (d3 * scale);
    if ((cap != 0.000000)) {
      d3 = (cap * tanh(d3));
    }
    if (((k0 + (l + 24u)) >= lk)) {
      d3 = asfloat(4286578688u);
    }
    Ss[r][(l + 24u)] = d3;
    GroupMemoryBarrierWithGroupSync();
    float mt = m;
    for (uint j = 0u; j < 32u; ++j) {
      mt = max(mt, Ss[r][j]);
    }
    float corr = exp((m - mt));
    acc0 = (acc0 * corr);
    acc1 = (acc1 * corr);
    acc2 = (acc2 * corr);
    acc3 = (acc3 * corr);
    acc4 = (acc4 * corr);
    acc5 = (acc5 * corr);
    acc6 = (acc6 * corr);
    acc7 = (acc7 * corr);
    acc8 = (acc8 * corr);
    acc9 = (acc9 * corr);
    acc10 = (acc10 * corr);
    acc11 = (acc11 * corr);
    acc12 = (acc12 * corr);
    acc13 = (acc13 * corr);
    acc14 = (acc14 * corr);
    acc15 = (acc15 * corr);
    float ps = 0.000000;
    for (uint j = 0u; j < 32u; ++j) {
      float p = exp((Ss[r][j] - mt));
      ps = (ps + p);
      acc0 = (acc0 + (p * Vs[j][l]));
      acc1 = (acc1 + (p * Vs[j][(l + 8u)]));
      acc2 = (acc2 + (p * Vs[j][(l + 16u)]));
      acc3 = (acc3 + (p * Vs[j][(l + 24u)]));
      acc4 = (acc4 + (p * Vs[j][(l + 32u)]));
      acc5 = (acc5 + (p * Vs[j][(l + 40u)]));
      acc6 = (acc6 + (p * Vs[j][(l + 48u)]));
      acc7 = (acc7 + (p * Vs[j][(l + 56u)]));
      acc8 = (acc8 + (p * Vs[j][(l + 64u)]));
      acc9 = (acc9 + (p * Vs[j][(l + 72u)]));
      acc10 = (acc10 + (p * Vs[j][(l + 80u)]));
      acc11 = (acc11 + (p * Vs[j][(l + 88u)]));
      acc12 = (acc12 + (p * Vs[j][(l + 96u)]));
      acc13 = (acc13 + (p * Vs[j][(l + 104u)]));
      acc14 = (acc14 + (p * Vs[j][(l + 112u)]));
      acc15 = (acc15 + (p * Vs[j][(l + 120u)]));
    }
    ls = ((ls * corr) + ps);
    m = mt;
  }
  if (valid) {
    float inv = ((ls == 0.000000) ? 0.000000 : (1.000000 / ls));
    dst[off4(1u, l, h, iq, b)] = (acc0 * inv);
    dst[off4(1u, (l + 8u), h, iq, b)] = (acc1 * inv);
    dst[off4(1u, (l + 16u), h, iq, b)] = (acc2 * inv);
    dst[off4(1u, (l + 24u), h, iq, b)] = (acc3 * inv);
    dst[off4(1u, (l + 32u), h, iq, b)] = (acc4 * inv);
    dst[off4(1u, (l + 40u), h, iq, b)] = (acc5 * inv);
    dst[off4(1u, (l + 48u), h, iq, b)] = (acc6 * inv);
    dst[off4(1u, (l + 56u), h, iq, b)] = (acc7 * inv);
    dst[off4(1u, (l + 64u), h, iq, b)] = (acc8 * inv);
    dst[off4(1u, (l + 72u), h, iq, b)] = (acc9 * inv);
    dst[off4(1u, (l + 80u), h, iq, b)] = (acc10 * inv);
    dst[off4(1u, (l + 88u), h, iq, b)] = (acc11 * inv);
    dst[off4(1u, (l + 96u), h, iq, b)] = (acc12 * inv);
    dst[off4(1u, (l + 104u), h, iq, b)] = (acc13 * inv);
    dst[off4(1u, (l + 112u), h, iq, b)] = (acc14 * inv);
    dst[off4(1u, (l + 120u), h, iq, b)] = (acc15 * inv);
  }
}"

/-- The f32 serial sibling, D = 64. -/
def expectedSerialF32D64 : String :=
"struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 64u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint nq = pw(11u);
  uint iq = (e % nq);
  uint rr = (e / nq);
  uint h = (rr % pw(12u));
  uint b = (rr / pw(12u));
  uint lk = pw(20u);
  uint hk = (h / pw(55u));
  uint bk = (b / pw(56u));
  uint hv = (h / pw(57u));
  uint bv = (b / pw(58u));
  float scale = pf(59u);
  float cap = pf(60u);
  uint qb = off4(10u, 0u, iq, h, b);
  uint kb = off4(19u, 0u, 0u, hk, bk);
  uint kn = pw(24u);
  uint vb = off4(28u, 0u, 0u, hv, bv);
  uint vn = pw(33u);
  float qv[64];
  float acc[64];
  float sc[32];
  for (uint c = 0u; c < 64u; ++c) {
    qv[c] = s0[(qb + c)];
    acc[c] = 0.000000;
  }
  float m = asfloat(4286578688u);
  float ls = 0.000000;
  uint nt = ((lk + 31u) / 32u);
  for (uint tt = 0u; tt < nt; ++tt) {
    uint k0 = (tt * 32u);
    for (uint j = 0u; j < 32u; ++j) {
      uint key = (k0 + j);
      float s = asfloat(4286578688u);
      if ((key < lk)) {
        s = 0.000000;
        for (uint c = 0u; c < 64u; ++c) {
          s = (s + (qv[c] * s1[((kb + (key * kn)) + c)]));
        }
        s = (s * scale);
        if ((cap != 0.000000)) {
          s = (cap * tanh(s));
        }
      }
      sc[j] = s;
    }
    float mt = m;
    for (uint j = 0u; j < 32u; ++j) {
      mt = max(mt, sc[j]);
    }
    float corr = exp((m - mt));
    for (uint c = 0u; c < 64u; ++c) {
      acc[c] = (acc[c] * corr);
    }
    float ps = 0.000000;
    for (uint j = 0u; j < 32u; ++j) {
      float p = exp((sc[j] - mt));
      ps = (ps + p);
      uint key = (k0 + j);
      if ((key < lk)) {
        for (uint c = 0u; c < 64u; ++c) {
          acc[c] = (acc[c] + (p * s2[((vb + (key * vn)) + c)]));
        }
      }
    }
    ls = ((ls * corr) + ps);
    m = mt;
  }
  float inv = ((ls == 0.000000) ? 0.000000 : (1.000000 / ls));
  for (uint c = 0u; c < 64u; ++c) {
    dst[off4(1u, c, h, iq, b)] = (acc[c] * inv);
  }
}"

example : LeanSlang.emitFunction (fnLdF16 "s1") =
"float ld_f16_s1(uint e) {
  uint w = s1[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  return f16tof32(h);
}" := by native_decide

/-- The f16 kernel's text from the f32 kernel's. -/
def toF16 (t : String) : String :=
  let off := "(i3 * pw((w + 7u))));\n}"
  let ld := "\n\n" ++ LeanSlang.emitFunction (fnLdF16 "s1") ++ "\n\n" ++ LeanSlang.emitFunction (fnLdF16 "s2")
  ((((((t.replace "RWStructuredBuffer<float> s1;" "RWStructuredBuffer<uint> s1;").replace
    "RWStructuredBuffer<float> s2;" "RWStructuredBuffer<uint> s2;").replace off (off ++ ld)).replace
    "s1[((kb + (key * kn)) + c)]" "ld_f16_s1(((kb + (key * kn)) + c))").replace
    "s2[((vb + (key * vn)) + c)]" "ld_f16_s2(((vb + (key * vn)) + c))").replace
    "s0[(qb + c)]" "f16tof32(f32tof16(s0[(qb + c)]))")

example : LeanSlang.emit (tiled .f32 64) = expectedTiledF32D64 := by native_decide
example : LeanSlang.emit (tiled .f32 128) = expectedTiledF32D128 := by native_decide
example : LeanSlang.emit (serial .f32 64) = expectedSerialF32D64 := by native_decide
example : LeanSlang.emit (serial .f32 128) =
    (expectedSerialF32D64.replace "c < 64u" "c < 128u").replace "[64]" "[128]" := by native_decide
example : LeanSlang.emit (tiled .f16 64) = toF16 expectedTiledF32D64 := by native_decide
example : LeanSlang.emit (tiled .f16 128) = toF16 expectedTiledF32D128 := by native_decide
example : LeanSlang.emit (serial .f16 64) = toF16 expectedSerialF32D64 := by native_decide
example : LeanSlang.emit (serial .f16 128) =
    toF16 ((expectedSerialF32D64.replace "c < 64u" "c < 128u").replace "[64]" "[128]") := by native_decide
example : (tiled .f32 128).entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.FlashAttn
