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
// chaotic after its self-collision onset: a 4.4e-9 change of mu moves the
// loss at 350 steps by 6.5% and dL/dmu by 7.8% (gates/5-drape/README.md).
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
	// Fit attachments (Cut 6d, the drape's fit mode): the LAST nFit
	// attachments (the pins come first) pull their vertex toward the mesh
	// collider's nearest surface point (moved fitGap out along the normal;
	// the loop uses 0, the collider's skin is the clearance) at stiffness
	// fitK times the vertex's lumped area (applyMaterial) in place of
	// kAttach; the sim refreshes their targets every fitRefresh steps while
	// its fit mode is on (drape_sim.h). The same attachment kernels serve
	// both kinds.
	uint32_t nFit = 0;
	double fitK = 0.0, fitGap = 0.0;
	int fitRefresh = 1;
	// With fitSimilarity the sim also re-fits the rest shape every refresh:
	// the best similarity (Umeyama: rotation, uniform scale, translation) of
	// the fit's source garment onto the current vertices becomes the rest,
	// so the membrane pulls toward a uniformly scaled skirt and the fit pull
	// sets the scale at the hips, cloth-fit's SimilarityForm with one global
	// transform in place of one per element.
	bool fitSimilarity = false;
	// The anchor loop (the waist): each of its vertices gets a SECOND
	// attachment after the fit set, at anchorK, whose target is refreshed
	// every refresh to the vertex's own position plus the loop's centroid
	// error (its source centroid minus its current centroid): a uniform
	// force that holds the loop's centre and nothing else, so the ring still
	// shrinks onto the body under its own soft fit pull. That is cloth-fit's
	// curve_center_target (weight 1), which holds a boundary curve's centre
	// on its bone while the curve shrinks (PolyFEM's waist: centre y 0.954 =
	// the pelvis joint, radius 0.19 -> 0.135 m). Why not simpler: on the
	// flaring hips every nearest-point target lies below its vertex and an
	// unheld skirt ratchets down 3 cm (Gate 6d passes 4-5); a shift at the fit
	// stiffness does not hold it (pass 5); pinning the ring's vertices onto
	// the surface at kAttach places it but folds it, the nearest-point map
	// of a circle onto the waist's cross-section not being injective (pass
	// 6: self-intersections at the ring in every rung, the tube control
	// included). The similarity rest update pivots on the loop (source
	// centroid -> current centroid). Empty: no hold (8 cm low, pass 3).
	std::vector<uint32_t> fitAnchor;
	uint32_t nAnchorAtt = 0; // the anchor attachments, the LAST nAnchorAtt
	double anchorK = 100.0;
	// The rest update's cadence (steps; a multiple of fitRefresh: the targets
	// are cheap BVH queries, the rest update rebuilds the scene) and the
	// settle: steps run after the fit with the fit pull off (the anchor pins
	// and the collider on, the targets frozen) so the membrane and the
	// contact projection leave nothing through the skin before the check.
	int fitRestEvery = 4;
	int fitSettle = 0;

	// From the material (applyMaterial).
	std::vector<double> massD;
	std::vector<float> mass, triK, bendK, attachK;

	uint32_t nTri() const { return uint32_t(triArea.size()); }
	uint32_t nBend() const { return uint32_t(bendN.size()); }
	uint32_t nAttach() const { return uint32_t(attachVert.size()); }
	uint32_t nPin() const { return nAttach() - nFit - nAnchorAtt; }

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

// The fit set of a scene_mesh scene (Cut 6d): one fit attachment per listed
// vertex (its rest position as the first target) after the pins, replacing
// any earlier fit set; k, gap and refresh as DrapeScene documents them. A
// new scene_mesh drops it. False with `err` on a bad index or refresh < 1.
bool scene_fit_set(DrapeScene &s, const std::vector<int32_t> &verts, double k, double gap, int refresh, bool similarity,
		const std::vector<int32_t> &anchor, int restEvery, int settle, double anchorK, std::string &err);

// Finish a scene whose rest/x0/tri are set: triangles' material data, the
// per-vertex area, bendings, radii (particleTriangleMap = triangles in order).
void scene_finish(DrapeScene &s);
