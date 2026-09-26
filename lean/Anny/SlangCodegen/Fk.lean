import Anny.SlangCodegen.Common

/-!
# `Anny.SlangCodegen.Fk` — forward kinematics from 6D joint rotations

The posed skeleton and the skinning transforms `bone_j = world_j ·
bind_j⁻¹` that `anny_lbs` blends, from:

- `rb`   (J·9)  each joint's bind-world rotation `Rb_j` (row-major), from
               `pheno_skeleton_fit`; held constant (a stop-gradient: the
               fit's Newton–Schulz / Kabsch pass is not differentiated);
- `jpos` (J·3)  each joint's bind-world position (the RBF regression of
               the shaped rest mesh, `anny_csr_gemv3`);
- `rot6` (J·6)  each joint's rotation in its own bind frame, as its 6D
               truncation (`Common.rot6`; the identity is `1 0 0 0 1 0`);
- `trans` (3)   the root's world translation.

With `P_j`, `q_j` the posed world rotation and position,

    A_j = Rb_pᵀ Rb_j R_j,        c_j = Rb_pᵀ (jpos_j − jpos_p),
    P_j = P_p A_j,               q_j = P_p c_j + q_p,
    bone_j = [P_j Rb_jᵀ | q_j − P_j Rb_jᵀ jpos_j].

The root (joint 0) takes the same formula with a virtual parent `P = I`,
`q = 0`, `Rb = I`, `jpos = −trans`, so `P_0 = Rb_0 R_0` and `q_0 =
jpos_0 + trans`: one straight-line body for every joint. At the identity
rotations and zero translation every bone is the identity.

`parents[j] < j` for every j > 0 (topological order; the host checks it).
One thread walks the tree, as `pheno_skeleton_fit` does: joint j needs its
parent's world transform, and J is ~100.

Bindings (set 0):

  0  ConstantBuffer<AnnyFkParams> { uint J; }
  1  StructuredBuffer<uint>    parents  (J)
  2  StructuredBuffer<float>   rb       (J·9)
  3  StructuredBuffer<float>   jpos     (J·3)
  4  StructuredBuffer<float>   rot6     (J·6)
  5  StructuredBuffer<float>   trans    (3)
  6  RWStructuredBuffer<float> world    (J·12)  [P_j | q_j]
  7  RWStructuredBuffer<float> bone     (J·12)
-/

namespace Anny.SlangCodegen.Fk

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common

private def idn (i j : Nat) : E := if i == j then fl 1.0 else fl 0.0

/-- The parent frame of joint `j` and the local transform `(A, c)`:
    declares `root`, `pi`, `Rbp`, `jp`, `Pp`, `qp`, `Rbj`, `jj`, the
    rotation `R` (with `rot6`'s intermediates), `G = Rbpᵀ Rbj`, `A = G R`
    and `c = Rbpᵀ (jj − jp)`. Shared with `FkBackward`. -/
def parentFrame (j : E) : List St :=
  [ let_ bT "root" (eq j (u 0))
  , let_ uT "pi" (sel (v "root") (u 0) (at_ "parents" j)) ] ++
  mat "Rbp" (fun a b => sel (v "root") (idn a b) (at_ "rb" (v "pi" * u 9 + u (3 * a + b)))) ++
  vec "jp" (fun k => sel (v "root") (-(at_ "trans" (u k))) (at_ "jpos" (v "pi" * u 3 + u k))) ++
  mat "Pp" (fun a b => sel (v "root") (idn a b) (at_ "world" (v "pi" * u 12 + u (4 * a + b)))) ++
  vec "qp" (fun k => sel (v "root") (fl 0.0) (at_ "world" (v "pi" * u 12 + u (4 * k + 3)))) ++
  mat "Rbj" (fun a b => at_ "rb" (j * u 9 + u (3 * a + b))) ++
  vec "jj" (fun k => at_ "jpos" (j * u 3 + u k)) ++
  rot6 "R" "rot6" (j * u 6) ++
  matTMul "G" "Rbp" "Rbj" ++
  matMul "A" "G" "R" ++
  vec "e" (fun k => s "jj" k - s "jp" k) ++
  matTVec "c" "Rbp" "e"

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyFkParams", fields := [fld "J" uT] } ]
  , globals :=
      [ paramsCB "AnnyFkParams", roU "parents" 1, roF "rb" 2, roF "jpos" 3, roF "rot6" 4,
        roF "trans" 5, rwF "world" 6, rwF "bone" 7 ]
  , functions :=
      [ { attrs := [.shaderCompute, .numthreads 1 1 1], name := "main", params := [dtid]
        , body :=
            [ if_ (ne (.member (v "tid") "x") (u 0)) [ ret ]
            , for_ "j" (u 0) (p "J")
                (parentFrame (v "j") ++
                 matMul "P" "Pp" "A" ++
                 matVec "Pc" "Pp" "c" ++
                 vec "q" (fun k => s "Pc" k + s "qp" k) ++
                 store12 "world" (v "j" * u 12) "P" "q" ++
                 matMulT "B" "P" "Rbj" ++
                 matVec "Bj" "B" "jj" ++
                 vec "b" (fun k => s "q" k - s "Bj" k) ++
                 store12 "bone" (v "j" * u 12) "B" "b") ] } ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyFkParams {
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyFkParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> parents;
[[vk::binding(2, 0)]]
StructuredBuffer<float> rb;
[[vk::binding(3, 0)]]
StructuredBuffer<float> jpos;
[[vk::binding(4, 0)]]
StructuredBuffer<float> rot6;
[[vk::binding(5, 0)]]
StructuredBuffer<float> trans;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> world;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> bone;

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  for (uint j = 0u; j < params.J; ++j) {
    bool root = (j == 0u);
    uint pi = (root ? 0u : parents[j]);
    float Rbp0 = (root ? 1.000000 : rb[((pi * 9u) + 0u)]);
    float Rbp1 = (root ? 0.000000 : rb[((pi * 9u) + 1u)]);
    float Rbp2 = (root ? 0.000000 : rb[((pi * 9u) + 2u)]);
    float Rbp3 = (root ? 0.000000 : rb[((pi * 9u) + 3u)]);
    float Rbp4 = (root ? 1.000000 : rb[((pi * 9u) + 4u)]);
    float Rbp5 = (root ? 0.000000 : rb[((pi * 9u) + 5u)]);
    float Rbp6 = (root ? 0.000000 : rb[((pi * 9u) + 6u)]);
    float Rbp7 = (root ? 0.000000 : rb[((pi * 9u) + 7u)]);
    float Rbp8 = (root ? 1.000000 : rb[((pi * 9u) + 8u)]);
    float jp0 = (root ? (-trans[0u]) : jpos[((pi * 3u) + 0u)]);
    float jp1 = (root ? (-trans[1u]) : jpos[((pi * 3u) + 1u)]);
    float jp2 = (root ? (-trans[2u]) : jpos[((pi * 3u) + 2u)]);
    float Pp0 = (root ? 1.000000 : world[((pi * 12u) + 0u)]);
    float Pp1 = (root ? 0.000000 : world[((pi * 12u) + 1u)]);
    float Pp2 = (root ? 0.000000 : world[((pi * 12u) + 2u)]);
    float Pp3 = (root ? 0.000000 : world[((pi * 12u) + 4u)]);
    float Pp4 = (root ? 1.000000 : world[((pi * 12u) + 5u)]);
    float Pp5 = (root ? 0.000000 : world[((pi * 12u) + 6u)]);
    float Pp6 = (root ? 0.000000 : world[((pi * 12u) + 8u)]);
    float Pp7 = (root ? 0.000000 : world[((pi * 12u) + 9u)]);
    float Pp8 = (root ? 1.000000 : world[((pi * 12u) + 10u)]);
    float qp0 = (root ? 0.000000 : world[((pi * 12u) + 3u)]);
    float qp1 = (root ? 0.000000 : world[((pi * 12u) + 7u)]);
    float qp2 = (root ? 0.000000 : world[((pi * 12u) + 11u)]);
    float Rbj0 = rb[((j * 9u) + 0u)];
    float Rbj1 = rb[((j * 9u) + 1u)];
    float Rbj2 = rb[((j * 9u) + 2u)];
    float Rbj3 = rb[((j * 9u) + 3u)];
    float Rbj4 = rb[((j * 9u) + 4u)];
    float Rbj5 = rb[((j * 9u) + 5u)];
    float Rbj6 = rb[((j * 9u) + 6u)];
    float Rbj7 = rb[((j * 9u) + 7u)];
    float Rbj8 = rb[((j * 9u) + 8u)];
    float jj0 = jpos[((j * 3u) + 0u)];
    float jj1 = jpos[((j * 3u) + 1u)];
    float jj2 = jpos[((j * 3u) + 2u)];
    float R_a10 = rot6[((j * 6u) + 0u)];
    float R_a11 = rot6[((j * 6u) + 1u)];
    float R_a12 = rot6[((j * 6u) + 2u)];
    float R_a20 = rot6[((j * 6u) + 3u)];
    float R_a21 = rot6[((j * 6u) + 4u)];
    float R_a22 = rot6[((j * 6u) + 5u)];
    float R_n1 = max(sqrt((((R_a10 * R_a10) + (R_a11 * R_a11)) + (R_a12 * R_a12))), 1.0e-20f);
    float R_r10 = (R_a10 / R_n1);
    float R_r11 = (R_a11 / R_n1);
    float R_r12 = (R_a12 / R_n1);
    float R_d = (((R_r10 * R_a20) + (R_r11 * R_a21)) + (R_r12 * R_a22));
    float R_u0 = (R_a20 - (R_d * R_r10));
    float R_u1 = (R_a21 - (R_d * R_r11));
    float R_u2 = (R_a22 - (R_d * R_r12));
    float R_nu = max(sqrt((((R_u0 * R_u0) + (R_u1 * R_u1)) + (R_u2 * R_u2))), 1.0e-20f);
    float R_r20 = (R_u0 / R_nu);
    float R_r21 = (R_u1 / R_nu);
    float R_r22 = (R_u2 / R_nu);
    float R_r30 = ((R_r11 * R_r22) - (R_r12 * R_r21));
    float R_r31 = ((R_r12 * R_r20) - (R_r10 * R_r22));
    float R_r32 = ((R_r10 * R_r21) - (R_r11 * R_r20));
    float R0 = R_r10;
    float R1 = R_r11;
    float R2 = R_r12;
    float R3 = R_r20;
    float R4 = R_r21;
    float R5 = R_r22;
    float R6 = R_r30;
    float R7 = R_r31;
    float R8 = R_r32;
    float G0 = (((Rbp0 * Rbj0) + (Rbp3 * Rbj3)) + (Rbp6 * Rbj6));
    float G1 = (((Rbp0 * Rbj1) + (Rbp3 * Rbj4)) + (Rbp6 * Rbj7));
    float G2 = (((Rbp0 * Rbj2) + (Rbp3 * Rbj5)) + (Rbp6 * Rbj8));
    float G3 = (((Rbp1 * Rbj0) + (Rbp4 * Rbj3)) + (Rbp7 * Rbj6));
    float G4 = (((Rbp1 * Rbj1) + (Rbp4 * Rbj4)) + (Rbp7 * Rbj7));
    float G5 = (((Rbp1 * Rbj2) + (Rbp4 * Rbj5)) + (Rbp7 * Rbj8));
    float G6 = (((Rbp2 * Rbj0) + (Rbp5 * Rbj3)) + (Rbp8 * Rbj6));
    float G7 = (((Rbp2 * Rbj1) + (Rbp5 * Rbj4)) + (Rbp8 * Rbj7));
    float G8 = (((Rbp2 * Rbj2) + (Rbp5 * Rbj5)) + (Rbp8 * Rbj8));
    float A0 = (((G0 * R0) + (G1 * R3)) + (G2 * R6));
    float A1 = (((G0 * R1) + (G1 * R4)) + (G2 * R7));
    float A2 = (((G0 * R2) + (G1 * R5)) + (G2 * R8));
    float A3 = (((G3 * R0) + (G4 * R3)) + (G5 * R6));
    float A4 = (((G3 * R1) + (G4 * R4)) + (G5 * R7));
    float A5 = (((G3 * R2) + (G4 * R5)) + (G5 * R8));
    float A6 = (((G6 * R0) + (G7 * R3)) + (G8 * R6));
    float A7 = (((G6 * R1) + (G7 * R4)) + (G8 * R7));
    float A8 = (((G6 * R2) + (G7 * R5)) + (G8 * R8));
    float e0 = (jj0 - jp0);
    float e1 = (jj1 - jp1);
    float e2 = (jj2 - jp2);
    float c0 = (((Rbp0 * e0) + (Rbp3 * e1)) + (Rbp6 * e2));
    float c1 = (((Rbp1 * e0) + (Rbp4 * e1)) + (Rbp7 * e2));
    float c2 = (((Rbp2 * e0) + (Rbp5 * e1)) + (Rbp8 * e2));
    float P0 = (((Pp0 * A0) + (Pp1 * A3)) + (Pp2 * A6));
    float P1 = (((Pp0 * A1) + (Pp1 * A4)) + (Pp2 * A7));
    float P2 = (((Pp0 * A2) + (Pp1 * A5)) + (Pp2 * A8));
    float P3 = (((Pp3 * A0) + (Pp4 * A3)) + (Pp5 * A6));
    float P4 = (((Pp3 * A1) + (Pp4 * A4)) + (Pp5 * A7));
    float P5 = (((Pp3 * A2) + (Pp4 * A5)) + (Pp5 * A8));
    float P6 = (((Pp6 * A0) + (Pp7 * A3)) + (Pp8 * A6));
    float P7 = (((Pp6 * A1) + (Pp7 * A4)) + (Pp8 * A7));
    float P8 = (((Pp6 * A2) + (Pp7 * A5)) + (Pp8 * A8));
    float Pc0 = (((Pp0 * c0) + (Pp1 * c1)) + (Pp2 * c2));
    float Pc1 = (((Pp3 * c0) + (Pp4 * c1)) + (Pp5 * c2));
    float Pc2 = (((Pp6 * c0) + (Pp7 * c1)) + (Pp8 * c2));
    float q0 = (Pc0 + qp0);
    float q1 = (Pc1 + qp1);
    float q2 = (Pc2 + qp2);
    world[((j * 12u) + 0u)] = P0;
    world[((j * 12u) + 1u)] = P1;
    world[((j * 12u) + 2u)] = P2;
    world[((j * 12u) + 4u)] = P3;
    world[((j * 12u) + 5u)] = P4;
    world[((j * 12u) + 6u)] = P5;
    world[((j * 12u) + 8u)] = P6;
    world[((j * 12u) + 9u)] = P7;
    world[((j * 12u) + 10u)] = P8;
    world[((j * 12u) + 3u)] = q0;
    world[((j * 12u) + 7u)] = q1;
    world[((j * 12u) + 11u)] = q2;
    float B0 = (((P0 * Rbj0) + (P1 * Rbj1)) + (P2 * Rbj2));
    float B1 = (((P0 * Rbj3) + (P1 * Rbj4)) + (P2 * Rbj5));
    float B2 = (((P0 * Rbj6) + (P1 * Rbj7)) + (P2 * Rbj8));
    float B3 = (((P3 * Rbj0) + (P4 * Rbj1)) + (P5 * Rbj2));
    float B4 = (((P3 * Rbj3) + (P4 * Rbj4)) + (P5 * Rbj5));
    float B5 = (((P3 * Rbj6) + (P4 * Rbj7)) + (P5 * Rbj8));
    float B6 = (((P6 * Rbj0) + (P7 * Rbj1)) + (P8 * Rbj2));
    float B7 = (((P6 * Rbj3) + (P7 * Rbj4)) + (P8 * Rbj5));
    float B8 = (((P6 * Rbj6) + (P7 * Rbj7)) + (P8 * Rbj8));
    float Bj0 = (((B0 * jj0) + (B1 * jj1)) + (B2 * jj2));
    float Bj1 = (((B3 * jj0) + (B4 * jj1)) + (B5 * jj2));
    float Bj2 = (((B6 * jj0) + (B7 * jj1)) + (B8 * jj2));
    float b0 = (q0 - Bj0);
    float b1 = (q1 - Bj1);
    float b2 = (q2 - Bj2);
    bone[((j * 12u) + 0u)] = B0;
    bone[((j * 12u) + 1u)] = B1;
    bone[((j * 12u) + 2u)] = B2;
    bone[((j * 12u) + 4u)] = B3;
    bone[((j * 12u) + 5u)] = B4;
    bone[((j * 12u) + 6u)] = B5;
    bone[((j * 12u) + 8u)] = B6;
    bone[((j * 12u) + 9u)] = B7;
    bone[((j * 12u) + 10u)] = B8;
    bone[((j * 12u) + 3u)] = b0;
    bone[((j * 12u) + 7u)] = b1;
    bone[((j * 12u) + 11u)] = b2;
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.Fk
