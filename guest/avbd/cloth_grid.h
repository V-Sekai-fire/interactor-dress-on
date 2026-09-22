// SPDX-License-Identifier: Apache-2.0 OR MIT
#ifndef CLOTH_GRID_H
#define CLOTH_GRID_H

#include <cstdint>
#include <vector>

struct ClothMesh {
	uint32_t nx = 0, ny = 0;
	float width = 0.0f, height = 0.0f;

	std::vector<float> positions;
	std::vector<float> mass;

	std::vector<uint32_t> triIdx;
	std::vector<float> triInvUV;

	std::vector<uint32_t> bendIdx;
	std::vector<float> bendWeight;
	std::vector<float> bendNTarget;

	std::vector<uint32_t> attachVert;
	std::vector<float> attachFixed;

	std::vector<uint32_t> faces;

	uint32_t nVerts() const { return static_cast<uint32_t>(positions.size() / 3); }
	uint32_t nTri() const { return static_cast<uint32_t>(triInvUV.size() / 4); }
	uint32_t nBend() const { return static_cast<uint32_t>(bendNTarget.size()); }
	uint32_t nAttach() const { return static_cast<uint32_t>(attachVert.size()); }
};

ClothMesh build_cloth_mesh(uint32_t nx, uint32_t ny, float width, float height,
		float panelTop, float clothMass);

ClothMesh combine_meshes(const std::vector<ClothMesh> &parts);

std::vector<ClothMesh> partition_islands(const ClothMesh &mesh);

struct PanelTile {
	ClothMesh mesh;
	uint32_t nOwned = 0;
	std::vector<uint32_t> globalId;
};

std::vector<PanelTile> split_connected_panel(const ClothMesh &mesh, uint32_t nTiles);

#endif
