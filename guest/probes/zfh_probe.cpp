// Gate 0F probe 15, control arm: the one TU of probes.elf compiled with Zfh
// (CMakeLists.txt adds -march=..._zfh to this file only), so the half
// arithmetic below becomes flh/fmadd.h/fsh instead of soft conversions. If
// libriscv does not implement Zfh the vmcall traps with an illegal
// instruction; if it does, the value comes back. Either is recorded.
#include "zfh_probe.h"

static volatile _Float16 g_a = _Float16(1.5f);
static volatile _Float16 g_b = _Float16(2.25f);
static volatile _Float16 g_c;

float zfh_probe_run() {
	const _Float16 a = g_a;
	const _Float16 b = g_b;
	g_c = a * b + a; // 1.5 * 2.25 + 1.5 = 4.875, exact in binary16
	return float(g_c);
}
