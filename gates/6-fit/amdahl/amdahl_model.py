"""Amdahl model for fit.elf (loop skirt) -- every input cited in the README.

Parts: asm, bp, csu, ccd, sdf, lin, rest.
Per phase: guest seconds per Newton T, Newton count N, fractions f_i.
Offloaded part i costs  r_i*T_i + k_i*(t_rt + g)  per Newton, where r_i is the
CPU residue (share of the part that stays), k_i round trips per Newton, t_rt the
pump round trip (rule 4: >= one pump tick), g the GPU exec + transfer allowance.
"""

def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}

# --- measured profiles ------------------------------------------------------
# loop skirt phase 0, native samply (runs/loop-p0-direct.buckets.md), default cfg
P0 = norm(dict(asm=18.3, bp=43.2, ccd=20.9, csu=2.3, sdf=0.0, lin=11.5, rest=3.9))
# loop skirt phase 1, guest instret timers (cut-6c guest-psd-prof.log, psd cfg)
P1 = norm(dict(asm=1.19, bp=1.12, ccd=0.251, csu=0.445, sdf=0.209, lin=0.368, rest=0.165))
# loop skirt phase 0, native polysolve timers, psd cfg (C:/b/budget/solo-psd)
P0psd = norm(dict(asm=2.48, bp=3.91, ccd=1.24, csu=0.139, sdf=0.0, lin=0.942, rest=0.309))
# per-component guest/native ratios, phase 1 psd (guest-psd-prof vs solo-psd)
RATIO = dict(asm=10.5, bp=13.6, ccd=15.6, csu=9.0, sdf=10.0, lin=10.1, rest=11.3)

def weighted(p):
    return norm({k: v * RATIO[k] for k, v in p.items()})

# guest phase totals (s) and Newton counts
DEFAULT = [(1013.0, 144), (380.0, 109)]      # Gate 8 flat
PSD = [(267.6, 50), (345.1, 100)]            # cut-6c guest-psd-prof

TRIALS = [1.03, 1.99]  # CSU builds per Newton: loop p0 counters; p1 map

def run(cfg, profiles, moved, t_rt, cpu_fix=None, s_ccd=10.0, g=0.002):
    total = 0.0
    for (secs, n), prof, trials in zip(cfg, profiles, TRIALS):
        T = secs / n
        part = {k: v * T for k, v in prof.items()}
        # CPU code fixes (estimates): remove shares of parts
        if cpu_fix:
            contact = 0.032 * part["asm"]            # stays on CPU (fp64)
            part["asm"] = part["asm"] * (1 - cpu_fix["asm"])
            part["bp"] *= (1 - cpu_fix["bp"])
            part["lin"] *= (1 - cpu_fix["lin"])
        else:
            contact = 0.032 * part["asm"]
        t = 0.0
        for k, v in part.items():
            if k not in moved:
                t += v
                continue
            if k == "asm":
                t += contact + 0.01 * v + (t_rt + g)
            elif k == "bp":
                t += 0.01 * v + (t_rt + g)
            elif k == "csu":
                t += 0.05 * v + trials * (t_rt + g)
            elif k == "sdf":
                t += 0.0                      # filled at fit_begin, overlaps phase 0
            elif k == "ccd":
                t += v / s_ccd + g            # fused into the broad phase's list
        total += t * n
    return total

def zero_cost(cfg, profiles, moved):
    total = 0.0
    for (secs, n), prof in zip(cfg, profiles):
        T = secs / n
        total += sum(v * T for k, v in prof.items() if k not in moved) * n
    return total

base = sum(s for s, _ in DEFAULT)
psd_base = sum(s for s, _ in PSD)
FAST, HZ60 = 0.0006, 1 / 60

for label, prof0 in [("unweighted", P0), ("ratio-weighted", weighted(P0))]:
    profs = [prof0, P1]
    print(f"== default config, P0 {label}")
    blend = {k: (prof0[k] * DEFAULT[0][0] + P1[k] * DEFAULT[1][0]) / base for k in P0}
    print("  blend %:", {k: round(100 * v, 1) for k, v in blend.items()})
    for name, moved in [("bp", {"bp"}), ("+asm", {"bp", "asm"}),
                        ("+csu+sdf", {"bp", "asm", "csu", "sdf"}),
                        ("+ccd", {"bp", "asm", "csu", "sdf", "ccd"})]:
        zc = base / zero_cost(DEFAULT, profs, moved)
        f = base / run(DEFAULT, profs, moved, FAST)
        h = base / run(DEFAULT, profs, moved, HZ60)
        print(f"  {name:10s} zero-cost {zc:5.2f}x  fast {f:5.2f}x  60Hz {h:5.2f}x  "
              f"-> {base / h:6.0f}-{base / f:6.0f} s")

print("== psd config (P0psd native, P1 guest)")
profs = [P0psd, P1]
blend = {k: (P0psd[k] * PSD[0][0] + P1[k] * PSD[1][0]) / psd_base for k in P0}
print("  blend %:", {k: round(100 * v, 1) for k, v in blend.items()})
fix = dict(asm=0.5, bp=0.6, lin=0.1)
cpu = run(PSD, profs, set(), FAST, cpu_fix=fix)
print(f"  psd measured {base / psd_base:.2f}x ({psd_base:.0f} s)")
print(f"  +cpu fixes {psd_base / cpu:.2f}x on psd, {base / cpu:.2f}x vs loop ({cpu:.0f} s)")
for name, moved in [("bp+asm", {"bp", "asm"}),
                    ("+csu+sdf", {"bp", "asm", "csu", "sdf"}),
                    ("+ccd", {"bp", "asm", "csu", "sdf", "ccd"})]:
    for rt_name, rt in [("fast", FAST), ("60Hz", HZ60)]:
        t = run(PSD, profs, moved, rt, cpu_fix=fix)
        print(f"  {name:9s} {rt_name:5s} {psd_base / t:5.2f}x on psd, {base / t:5.2f}x vs loop ({t:4.0f} s)")
# floor: all parallel moved at zero cost after cpu fixes
tot = 0.0
for (secs, n), prof in zip(PSD, profs):
    T = secs / n
    tot += (prof["lin"] * T * 0.9 + prof["rest"] * T + 0.032 * prof["asm"] * T) * n
print(f"  floor (lin*0.9 + rest + contact) {psd_base / tot:.2f}x on psd, {base / tot:.2f}x vs loop ({tot:.0f} s)")
# GPU-only on psd without cpu fixes
for name, moved in [("bp", {"bp"}), ("bp+asm", {"bp", "asm"}), ("+csu+sdf", {"bp", "asm", "csu", "sdf"}),
                    ("+ccd", {"bp", "asm", "csu", "sdf", "ccd"})]:
    zc = psd_base / zero_cost(PSD, profs, moved)
    h = psd_base / run(PSD, profs, moved, HZ60)
    f = psd_base / run(PSD, profs, moved, FAST)
    print(f"  nofix {name:9s} zero {zc:5.2f}x fast {f:5.2f}x 60Hz {h:5.2f}x on psd")
# sensitivity: s_ccd
for s in (3, 10, 30):
    t = run(DEFAULT, [P0, P1], {"bp", "asm", "csu", "sdf", "ccd"}, HZ60, s_ccd=s)
    print(f"  default all-moved 60Hz s_ccd={s}: {base / t:.2f}x")
# CPU-fixes only on the default config
t = run(DEFAULT, [P0, P1], set(), FAST, cpu_fix=fix)
print(f"  default + cpu fixes only: {base / t:.2f}x ({t:.0f} s)")
