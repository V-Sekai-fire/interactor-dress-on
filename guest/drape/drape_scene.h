// drape_scene -- the drape's cloth and obstacles, built the way DiffCloth
// builds them.
//
// Ports of cloth-dynamics-standalone (e361584) Simulation.cpp:
//  - getInitParticlePos (2383) and createClothMeshFromConfig (3196): the
//    grid, rotated by the scene orientation (rotatePointsAccordingToConfig,
//    Simulation.h:711; axisToRotation, UtilityFunctions.h:62) and centred;
//    triangles created as (c, b, a), the order the native OBJ faces carry;
//  - createClothMeshFromModel (2759-2830), with keepOriginalScalePoint: the
//    host mesh as given, triangles in their own order;
//  - createBendingConstraints (2676) with TriangleBending's cotangent weights
//    (TriangleBending.cpp:186-225: float weights, n from them);
//  - Triangle's inv_deltaUV and area_rest (Triangle.cpp:498-511);
//  - updateAreaMatrix / updateMassMatrix (m = density * area / 3 per corner);
//  - updateCollisionRadii (3015: min connected edge / 2 - 0.01);
//  - the AVBD stiffness uploads: membrane area * k (Triangle.h:181-195),
//    bending k * 3 / (A0 + A1) (TriangleBending.h:61-79), attachments k
//    (AttachmentSpring.h:96-103);
//  - initScene's PLANE_AND_SPHERE (2426, 2487-2497): the sphere of the
//    rotating-sphere demo (the plane is positioned there but never added to
//    the primitive list, so it is not here either).
// Particle positions and rest positions are float (Particle::pos_rest is a
// Vec3f); every derived quantity is computed in double from them, as upstream.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

// The native sphere demo's initial mu, exactly. `tool_cloth_dynamics -demo
// sphere -seed 1` draws it in OptimizeHelper::getRandomParam: srand(1); the
// first rand() (41) is only logged as the "seed"; parameterFromRandSeed then
// calls VecXd::setRandom(), whose one coefficient is Eigen's
// -1 + 2 rand()/RAND_MAX with the second rand() (18467 of 32767, the UCRT
// LCG), mapped to lb + (x/2 + 1/2)(ub - lb) on [0.01, 0.95]. The logs print
// it as 0.539770; the difference (1.96e-7) matters: the sphere demo is
// chaotic after its self-collision onset, and a 7e-9 change of mu moves the
// loss at 350 steps by 9% (gates/5-drape/README.md).
constexpr double kNativeSphereMu0 = 0.5397701956236457;

#include <cstdint>
#include <string>
#include <vector>

#include "drape_config.h"
#include "primitives.h"

struct DrapeScene {
	std::string name;
	uint32_t nV = 0;
	std::vector<float> rest;     // pos_rest, 3 nV
	std::vector<float> x0, v0;   // pos_init, velocity_init, 3 nV
	std::vector<double> vertArea; // area distributed to each vertex
	std::vector<uint32_t> tri;   // 3 per triangle (p0, p1, p2 of Triangle)
	std::vector<float> triInvUV; // 4 per triangle, row-major
	std::vector<double> triArea; // area_rest
	std::vector<uint32_t> bendIdx; // 4 per bending
	std::vector<float> bendW, bendN;
	std::vector<double> bendAreaSum; // A0 + A1
	std::vector<uint32_t> attachVert;
	std::vector<float> attachFixed;  // 3 per attachment
	std::vector<double> radii;
	std::vector<Primitive> prims;

	// From the material (applyMaterial).
	std::vector<double> massD;
	std::vector<float> mass, triK, bendK, attachK;

	uint32_t nTri() const { return uint32_t(triArea.size()); }
	uint32_t nBend() const { return uint32_t(bendN.size()); }
	uint32_t nAttach() const { return uint32_t(attachVert.size()); }

	// Masses and stiffnesses from cfg.density, kTri, kBend, kAttach (and
	// rawStiffness): what the solver uploads.
	void applyMaterial(const DrapeConfig &cfg);
	std::string describe() const;
};

enum class GridOrientation { Front, Down };

// createClothMeshFromConfig for a gridNumX x gridNumY grid of clothDimX x
// clothDimY, rotated by `o` and centred (keepOriginalScalePoint false).
// `restMin`/`restMax` receive restShapeMinDim/MaxDim.
void scene_grid(DrapeScene &s, int gridNumX, int gridNumY, double clothDimX, double clothDimY, GridOrientation o,
		double restMin[3], double restMax[3]);

// The rotating-sphere demo (OptimizationTaskConfigurations rotatingSphereScene,
// sphereFabric): 25 x 25, 4.5 x 4.5, orientation DOWN, the sphere of radius 2
// with friction cfg.mu. Also applies the material from cfg.
void scene_sphere_demo(DrapeScene &s, const DrapeConfig &cfg);

// A host mesh: positions (3 nV), triangles (3 per), pinned vertex ids (each
// an attachment at its rest position). False with `err` on bad input.
bool scene_mesh(DrapeScene &s, const std::vector<float> &pos, const std::vector<int32_t> &tris,
		const std::vector<int32_t> &pins, std::string &err);

// Finish a scene whose rest/x0/tri are set: triangles' material data, the
// per-vertex area, bendings, radii (particleTriangleMap = triangles in order).
void scene_finish(DrapeScene &s);
