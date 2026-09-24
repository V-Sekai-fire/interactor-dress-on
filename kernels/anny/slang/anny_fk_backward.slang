struct AnnyFkParams {
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
StructuredBuffer<float> world;
[[vk::binding(7, 0)]]
StructuredBuffer<float> dbone;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> dworld;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> drot6;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float> dtrans;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float> djpos;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  for (uint k = 0u; k < (params.J * 12u); ++k) {
    dworld[k] = 0.000000;
  }
  for (uint k = 0u; k < (params.J * 6u); ++k) {
    drot6[k] = 0.000000;
  }
  for (uint k = 0u; k < (params.J * 3u); ++k) {
    djpos[k] = 0.000000;
  }
  for (uint k = 0u; k < 3u; ++k) {
    dtrans[k] = 0.000000;
  }
  for (uint jr = 0u; jr < params.J; ++jr) {
    uint j = ((params.J - 1u) - jr);
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
    float P0 = world[((j * 12u) + 0u)];
    float P1 = world[((j * 12u) + 1u)];
    float P2 = world[((j * 12u) + 2u)];
    float P3 = world[((j * 12u) + 4u)];
    float P4 = world[((j * 12u) + 5u)];
    float P5 = world[((j * 12u) + 6u)];
    float P6 = world[((j * 12u) + 8u)];
    float P7 = world[((j * 12u) + 9u)];
    float P8 = world[((j * 12u) + 10u)];
    float B0 = (((P0 * Rbj0) + (P1 * Rbj1)) + (P2 * Rbj2));
    float B1 = (((P0 * Rbj3) + (P1 * Rbj4)) + (P2 * Rbj5));
    float B2 = (((P0 * Rbj6) + (P1 * Rbj7)) + (P2 * Rbj8));
    float B3 = (((P3 * Rbj0) + (P4 * Rbj1)) + (P5 * Rbj2));
    float B4 = (((P3 * Rbj3) + (P4 * Rbj4)) + (P5 * Rbj5));
    float B5 = (((P3 * Rbj6) + (P4 * Rbj7)) + (P5 * Rbj8));
    float B6 = (((P6 * Rbj0) + (P7 * Rbj1)) + (P8 * Rbj2));
    float B7 = (((P6 * Rbj3) + (P7 * Rbj4)) + (P8 * Rbj5));
    float B8 = (((P6 * Rbj6) + (P7 * Rbj7)) + (P8 * Rbj8));
    float dB0 = dbone[((j * 12u) + 0u)];
    float dB1 = dbone[((j * 12u) + 1u)];
    float dB2 = dbone[((j * 12u) + 2u)];
    float dB3 = dbone[((j * 12u) + 4u)];
    float dB4 = dbone[((j * 12u) + 5u)];
    float dB5 = dbone[((j * 12u) + 6u)];
    float dB6 = dbone[((j * 12u) + 8u)];
    float dB7 = dbone[((j * 12u) + 9u)];
    float dB8 = dbone[((j * 12u) + 10u)];
    float db0 = dbone[((j * 12u) + 3u)];
    float db1 = dbone[((j * 12u) + 7u)];
    float db2 = dbone[((j * 12u) + 11u)];
    float dq00 = dworld[((j * 12u) + 3u)];
    float dq01 = dworld[((j * 12u) + 7u)];
    float dq02 = dworld[((j * 12u) + 11u)];
    float dq0 = (dq00 + db0);
    float dq1 = (dq01 + db1);
    float dq2 = (dq02 + db2);
    float dBt0 = (dB0 - (db0 * jj0));
    float dBt1 = (dB1 - (db0 * jj1));
    float dBt2 = (dB2 - (db0 * jj2));
    float dBt3 = (dB3 - (db1 * jj0));
    float dBt4 = (dB4 - (db1 * jj1));
    float dBt5 = (dB5 - (db1 * jj2));
    float dBt6 = (dB6 - (db2 * jj0));
    float dBt7 = (dB7 - (db2 * jj1));
    float dBt8 = (dB8 - (db2 * jj2));
    float Btdb0 = (((B0 * db0) + (B3 * db1)) + (B6 * db2));
    float Btdb1 = (((B1 * db0) + (B4 * db1)) + (B7 * db2));
    float Btdb2 = (((B2 * db0) + (B5 * db1)) + (B8 * db2));
    float dP00 = dworld[((j * 12u) + 0u)];
    float dP01 = dworld[((j * 12u) + 1u)];
    float dP02 = dworld[((j * 12u) + 2u)];
    float dP03 = dworld[((j * 12u) + 4u)];
    float dP04 = dworld[((j * 12u) + 5u)];
    float dP05 = dworld[((j * 12u) + 6u)];
    float dP06 = dworld[((j * 12u) + 8u)];
    float dP07 = dworld[((j * 12u) + 9u)];
    float dP08 = dworld[((j * 12u) + 10u)];
    float dBR0 = (((dBt0 * Rbj0) + (dBt1 * Rbj3)) + (dBt2 * Rbj6));
    float dBR1 = (((dBt0 * Rbj1) + (dBt1 * Rbj4)) + (dBt2 * Rbj7));
    float dBR2 = (((dBt0 * Rbj2) + (dBt1 * Rbj5)) + (dBt2 * Rbj8));
    float dBR3 = (((dBt3 * Rbj0) + (dBt4 * Rbj3)) + (dBt5 * Rbj6));
    float dBR4 = (((dBt3 * Rbj1) + (dBt4 * Rbj4)) + (dBt5 * Rbj7));
    float dBR5 = (((dBt3 * Rbj2) + (dBt4 * Rbj5)) + (dBt5 * Rbj8));
    float dBR6 = (((dBt6 * Rbj0) + (dBt7 * Rbj3)) + (dBt8 * Rbj6));
    float dBR7 = (((dBt6 * Rbj1) + (dBt7 * Rbj4)) + (dBt8 * Rbj7));
    float dBR8 = (((dBt6 * Rbj2) + (dBt7 * Rbj5)) + (dBt8 * Rbj8));
    float dP0 = (dP00 + dBR0);
    float dP1 = (dP01 + dBR1);
    float dP2 = (dP02 + dBR2);
    float dP3 = (dP03 + dBR3);
    float dP4 = (dP04 + dBR4);
    float dP5 = (dP05 + dBR5);
    float dP6 = (dP06 + dBR6);
    float dP7 = (dP07 + dBR7);
    float dP8 = (dP08 + dBR8);
    float dA0 = (((Pp0 * dP0) + (Pp3 * dP3)) + (Pp6 * dP6));
    float dA1 = (((Pp0 * dP1) + (Pp3 * dP4)) + (Pp6 * dP7));
    float dA2 = (((Pp0 * dP2) + (Pp3 * dP5)) + (Pp6 * dP8));
    float dA3 = (((Pp1 * dP0) + (Pp4 * dP3)) + (Pp7 * dP6));
    float dA4 = (((Pp1 * dP1) + (Pp4 * dP4)) + (Pp7 * dP7));
    float dA5 = (((Pp1 * dP2) + (Pp4 * dP5)) + (Pp7 * dP8));
    float dA6 = (((Pp2 * dP0) + (Pp5 * dP3)) + (Pp8 * dP6));
    float dA7 = (((Pp2 * dP1) + (Pp5 * dP4)) + (Pp8 * dP7));
    float dA8 = (((Pp2 * dP2) + (Pp5 * dP5)) + (Pp8 * dP8));
    float dc0 = (((Pp0 * dq0) + (Pp3 * dq1)) + (Pp6 * dq2));
    float dc1 = (((Pp1 * dq0) + (Pp4 * dq1)) + (Pp7 * dq2));
    float dc2 = (((Pp2 * dq0) + (Pp5 * dq1)) + (Pp8 * dq2));
    float dR0 = (((G0 * dA0) + (G3 * dA3)) + (G6 * dA6));
    float dR1 = (((G0 * dA1) + (G3 * dA4)) + (G6 * dA7));
    float dR2 = (((G0 * dA2) + (G3 * dA5)) + (G6 * dA8));
    float dR3 = (((G1 * dA0) + (G4 * dA3)) + (G7 * dA6));
    float dR4 = (((G1 * dA1) + (G4 * dA4)) + (G7 * dA7));
    float dR5 = (((G1 * dA2) + (G4 * dA5)) + (G7 * dA8));
    float dR6 = (((G2 * dA0) + (G5 * dA3)) + (G8 * dA6));
    float dR7 = (((G2 * dA1) + (G5 * dA4)) + (G8 * dA7));
    float dR8 = (((G2 * dA2) + (G5 * dA5)) + (G8 * dA8));
    float R_g10 = dR0;
    float R_g11 = dR1;
    float R_g12 = dR2;
    float R_g20 = dR3;
    float R_g21 = dR4;
    float R_g22 = dR5;
    float R_g30 = dR6;
    float R_g31 = dR7;
    float R_g32 = dR8;
    float R_c10 = ((R_r21 * R_g32) - (R_r22 * R_g31));
    float R_c11 = ((R_r22 * R_g30) - (R_r20 * R_g32));
    float R_c12 = ((R_r20 * R_g31) - (R_r21 * R_g30));
    float R_c20 = ((R_g31 * R_r12) - (R_g32 * R_r11));
    float R_c21 = ((R_g32 * R_r10) - (R_g30 * R_r12));
    float R_c22 = ((R_g30 * R_r11) - (R_g31 * R_r10));
    float R_h10 = (R_g10 + R_c10);
    float R_h11 = (R_g11 + R_c11);
    float R_h12 = (R_g12 + R_c12);
    float R_h20 = (R_g20 + R_c20);
    float R_h21 = (R_g21 + R_c21);
    float R_h22 = (R_g22 + R_c22);
    float R_p2 = (((R_r20 * R_h20) + (R_r21 * R_h21)) + (R_r22 * R_h22));
    float R_du0 = ((R_h20 - (R_p2 * R_r20)) / R_nu);
    float R_du1 = ((R_h21 - (R_p2 * R_r21)) / R_nu);
    float R_du2 = ((R_h22 - (R_p2 * R_r22)) / R_nu);
    float R_q = (((R_r10 * R_du0) + (R_r11 * R_du1)) + (R_r12 * R_du2));
    float R_da20 = (R_du0 - (R_q * R_r10));
    float R_da21 = (R_du1 - (R_q * R_r11));
    float R_da22 = (R_du2 - (R_q * R_r12));
    float R_h1b0 = (R_h10 - ((R_d * R_du0) + (R_q * R_a20)));
    float R_h1b1 = (R_h11 - ((R_d * R_du1) + (R_q * R_a21)));
    float R_h1b2 = (R_h12 - ((R_d * R_du2) + (R_q * R_a22)));
    float R_p1 = (((R_r10 * R_h1b0) + (R_r11 * R_h1b1)) + (R_r12 * R_h1b2));
    float R_da10 = ((R_h1b0 - (R_p1 * R_r10)) / R_n1);
    float R_da11 = ((R_h1b1 - (R_p1 * R_r11)) / R_n1);
    float R_da12 = ((R_h1b2 - (R_p1 * R_r12)) / R_n1);
    drot6[((j * 6u) + 0u)] = R_da10;
    drot6[((j * 6u) + 1u)] = R_da11;
    drot6[((j * 6u) + 2u)] = R_da12;
    drot6[((j * 6u) + 3u)] = R_da20;
    drot6[((j * 6u) + 4u)] = R_da21;
    drot6[((j * 6u) + 5u)] = R_da22;
    float uc0 = (((Rbp0 * dc0) + (Rbp1 * dc1)) + (Rbp2 * dc2));
    float uc1 = (((Rbp3 * dc0) + (Rbp4 * dc1)) + (Rbp5 * dc2));
    float uc2 = (((Rbp6 * dc0) + (Rbp7 * dc1)) + (Rbp8 * dc2));
    djpos[((j * 3u) + 0u)] = ((djpos[((j * 3u) + 0u)] - Btdb0) + uc0);
    djpos[((j * 3u) + 1u)] = ((djpos[((j * 3u) + 1u)] - Btdb1) + uc1);
    djpos[((j * 3u) + 2u)] = ((djpos[((j * 3u) + 2u)] - Btdb2) + uc2);
    if (root) {
      dtrans[0u] = uc0;
      dtrans[1u] = uc1;
      dtrans[2u] = uc2;
    } else {
      djpos[((pi * 3u) + 0u)] = (djpos[((pi * 3u) + 0u)] - uc0);
      djpos[((pi * 3u) + 1u)] = (djpos[((pi * 3u) + 1u)] - uc1);
      djpos[((pi * 3u) + 2u)] = (djpos[((pi * 3u) + 2u)] - uc2);
      float dPA0 = (((dP0 * A0) + (dP1 * A1)) + (dP2 * A2));
      float dPA1 = (((dP0 * A3) + (dP1 * A4)) + (dP2 * A5));
      float dPA2 = (((dP0 * A6) + (dP1 * A7)) + (dP2 * A8));
      float dPA3 = (((dP3 * A0) + (dP4 * A1)) + (dP5 * A2));
      float dPA4 = (((dP3 * A3) + (dP4 * A4)) + (dP5 * A5));
      float dPA5 = (((dP3 * A6) + (dP4 * A7)) + (dP5 * A8));
      float dPA6 = (((dP6 * A0) + (dP7 * A1)) + (dP8 * A2));
      float dPA7 = (((dP6 * A3) + (dP7 * A4)) + (dP8 * A5));
      float dPA8 = (((dP6 * A6) + (dP7 * A7)) + (dP8 * A8));
      dworld[((pi * 12u) + 0u)] = ((dworld[((pi * 12u) + 0u)] + dPA0) + (dq0 * c0));
      dworld[((pi * 12u) + 1u)] = ((dworld[((pi * 12u) + 1u)] + dPA1) + (dq0 * c1));
      dworld[((pi * 12u) + 2u)] = ((dworld[((pi * 12u) + 2u)] + dPA2) + (dq0 * c2));
      dworld[((pi * 12u) + 4u)] = ((dworld[((pi * 12u) + 4u)] + dPA3) + (dq1 * c0));
      dworld[((pi * 12u) + 5u)] = ((dworld[((pi * 12u) + 5u)] + dPA4) + (dq1 * c1));
      dworld[((pi * 12u) + 6u)] = ((dworld[((pi * 12u) + 6u)] + dPA5) + (dq1 * c2));
      dworld[((pi * 12u) + 8u)] = ((dworld[((pi * 12u) + 8u)] + dPA6) + (dq2 * c0));
      dworld[((pi * 12u) + 9u)] = ((dworld[((pi * 12u) + 9u)] + dPA7) + (dq2 * c1));
      dworld[((pi * 12u) + 10u)] = ((dworld[((pi * 12u) + 10u)] + dPA8) + (dq2 * c2));
      dworld[((pi * 12u) + 3u)] = (dworld[((pi * 12u) + 3u)] + dq0);
      dworld[((pi * 12u) + 7u)] = (dworld[((pi * 12u) + 7u)] + dq1);
      dworld[((pi * 12u) + 11u)] = (dworld[((pi * 12u) + 11u)] + dq2);
    }
  }
}
