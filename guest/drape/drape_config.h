// drape_config -- the drape's configuration, printed into every log.
//
// A mirror of cloth-dynamics' AvbdConfig (Simulation.cpp:76-156, e361584):
// the same knobs, the same defaults, so a run here and the native run are
// comparable line by line (the native [avbd-config] line is in
// gates/5-drape/native/README.md). Upstream reads them from AVBD_* env vars
// once; here they are set by drape_config(key, value) and dumped by every job
// and every forward/backward result, so each log says what it ran with.
//
// What upstream has and this port refuses (set() returns false):
//  - relax < 1 (host-side under-relaxation: one readback per iteration);
//  - bwdIft (the IFT adjoint); gpuSelf = 0 (the CPU spatial hash).
// Everything else is honoured.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

struct DrapeConfig {
	// Solver (AvbdConfig).
	int iters = 16;            // AVBD_ITERS
	float damp = 1.0f;         // AVBD_DAMP: velocity in the predictor
	float relax = 1.0f;        // AVBD_RELAX: only 1 is supported
	bool colors = true;        // AVBD_NO_COLORS=1 -> block Jacobi (one colour)
	bool membrane = true;      // AVBD_NO_MEMBRANE
	bool bending = true;       // AVBD_NO_BENDING
	bool rawStiffness = false; // AVBD_RAW_STIFFNESS
	bool al = false;           // AVBD_AL: dual updates after every iteration
	bool alGammaSet = false;   // AVBD_AL_GAMMA
	float alGamma = 1.0f;
	// Contact / friction (driver code after the solve, as upstream).
	bool contact = true;       // AVBD_NO_CONTACT
	bool frictionPred = true;  // AVBD_NO_FRICTION_PRED
	bool selfCollision = true; // AVBD_NO_SELF_COLLISION (and selfcollisionEnabled)
	int selfPasses = 2;        // AVBD_SELF_PASSES
	bool gpuSelf = true;       // the solver's scan kernel; 0 is refused
	int selfK = 16;            // neighbours per vertex (uploadSelfCollisionRadii(.., 16u))
	// Adjoint.
	int bwdTruncateK = 20;     // AVBD_BWD_TRUNCATE_K (0 = full chain)
	bool bwdIft = false;       // refused
	bool gradientClipping = true;           // Simulation.h:364
	double gradientClippingThreshold = 16.0; // Simulation.h:365, times nV
	// native mode's dL/dmu: upstream runBackwardTask adds the carried dL/dmu
	// twice per step (Simulation.cpp:4745-4749 and 4778-4783), so the carry
	// doubles each step. 1 reproduces that (parity); 0 adds it once.
	bool nativeMuDoubleCarry = true;

	// Scene physics (SceneConfiguration / FabricConfiguration).
	double h = 1.0 / 180.0;    // rotatingSphereScene.timeStep
	double gravity[3] = { 0.0, -9.8, 0.0 }; // Simulation.h:389
	double wind[3] = { 0.0, 0.0, 0.0 };     // constant external force per vertex (NO_WIND default)
	double density = 0.3;      // sphereFabric
	double kTri = 150.0;       // k_stiff_stretching
	double kBend = 0.00001;    // k_stiff_bending
	double kAttach = 10000.0;  // AttachmentSpring::k_stiff
	double mu = 0.3;           // the sphere demo's ground truth (OptimizationTaskSetup.cpp:228)

	// Set one knob by name; false for an unknown key or a refused value.
	bool set(const std::string &k, double v) {
		const bool b = v != 0.0;
		if (k == "iters") { if (v < 1) return false; iters = int(v); }
		else if (k == "damp") damp = float(v);
		else if (k == "relax") { if (v != 1.0) return false; relax = 1.0f; }
		else if (k == "colors") colors = b;
		else if (k == "membrane") membrane = b;
		else if (k == "bending") bending = b;
		else if (k == "rawStiffness") rawStiffness = b;
		else if (k == "al") al = b;
		else if (k == "alGamma") { alGammaSet = true; alGamma = float(v); }
		else if (k == "contact") contact = b;
		else if (k == "frictionPred") frictionPred = b;
		else if (k == "selfCollision") selfCollision = b;
		else if (k == "selfPasses") { if (v < 1) return false; selfPasses = int(v); }
		else if (k == "gpuSelf") { if (!b) return false; }
		else if (k == "selfK") { if (v < 1) return false; selfK = int(v); }
		else if (k == "bwdTruncateK") { if (v < 0) return false; bwdTruncateK = int(v); }
		else if (k == "bwdIft") { if (b) return false; }
		else if (k == "gradientClipping") gradientClipping = b;
		else if (k == "gradientClippingThreshold") gradientClippingThreshold = v;
		else if (k == "nativeMuDoubleCarry") nativeMuDoubleCarry = b;
		else if (k == "h") { if (!(v > 0)) return false; h = v; }
		else if (k == "gravityX") gravity[0] = v;
		else if (k == "gravityY") gravity[1] = v;
		else if (k == "gravityZ") gravity[2] = v;
		else if (k == "windX") wind[0] = v;
		else if (k == "windY") wind[1] = v;
		else if (k == "windZ") wind[2] = v;
		else if (k == "density") { if (!(v > 0)) return false; density = v; }
		else if (k == "kTri") kTri = v;
		else if (k == "kBend") kBend = v;
		else if (k == "kAttach") kAttach = v;
		else if (k == "mu") mu = v;
		else return false;
		return true;
	}

	// The upstream [avbd-config] line, then the scene's own knobs.
	std::string dump() const {
		char b[768];
		std::snprintf(b, sizeof b,
				"[drape-config] solver=AVBD iters=%d damp=%g relax=%g colors=%d | membrane=%d bending=%d "
				"rawStiffness=%d | al=%d gamma=%s | contact=%d frictionPred=%d selfColl=%d passes=%d gpuSelf=%d K=%d | "
				"bwd=AVBD truncateK=%d ift=%d clip=%d(%g*nV) muDoubleCarry=%d | h=%.9g g=(%g,%g,%g) wind=(%g,%g,%g) "
				"density=%g kTri=%g kBend=%g kAttach=%g mu=%.6f",
				iters, damp, relax, int(colors), int(membrane), int(bending), int(rawStiffness), int(al),
				alGammaSet ? "set" : "default", int(contact), int(frictionPred), int(selfCollision), selfPasses,
				int(gpuSelf), selfK, bwdTruncateK, int(bwdIft), int(gradientClipping), gradientClippingThreshold,
				int(nativeMuDoubleCarry), h, gravity[0], gravity[1], gravity[2], wind[0], wind[1], wind[2], density,
				kTri, kBend, kAttach, mu);
		return b;
	}
};
