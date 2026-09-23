// godot-lite: Mesh / ArrayMesh as surface-array holders. add_surface_from_arrays
// keeps a copy of the arrays (Godot hands them to the RenderingServer);
// surface_get_arrays returns a fresh copy, as Godot's does. No format
// validation, blend shapes, LODs or AABB.
#pragma once

#include "core/io/resource.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

#include <vector>

namespace gdl {

class Mesh : public Resource {
	GDCLASS(Mesh, Resource);

public:
	enum ArrayType {
		ARRAY_VERTEX = 0,
		ARRAY_NORMAL = 1,
		ARRAY_TANGENT = 2,
		ARRAY_COLOR = 3,
		ARRAY_TEX_UV = 4,
		ARRAY_TEX_UV2 = 5,
		ARRAY_CUSTOM0 = 6,
		ARRAY_CUSTOM1 = 7,
		ARRAY_CUSTOM2 = 8,
		ARRAY_CUSTOM3 = 9,
		ARRAY_BONES = 10,
		ARRAY_WEIGHTS = 11,
		ARRAY_INDEX = 12,
		ARRAY_MAX = 13
	};

	enum PrimitiveType {
		PRIMITIVE_POINTS,
		PRIMITIVE_LINES,
		PRIMITIVE_LINE_STRIP,
		PRIMITIVE_TRIANGLES,
		PRIMITIVE_TRIANGLE_STRIP,
		PRIMITIVE_MAX,
	};

	virtual int get_surface_count() const { return 0; }
	virtual Array surface_get_arrays(int p_surface) const { return Array(); }

	Mesh() = default;
};

class ArrayMesh : public Mesh {
	GDCLASS(ArrayMesh, Mesh);

	struct Surface {
		PrimitiveType primitive = PRIMITIVE_TRIANGLES;
		Array arrays;
	};
	std::vector<Surface> surfaces;

public:
	void add_surface_from_arrays(PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes = TypedArray<Array>(), const Dictionary &p_lods = Dictionary(), uint64_t p_flags = 0);
	int get_surface_count() const override { return int(surfaces.size()); }
	Array surface_get_arrays(int p_surface) const override;

	ArrayMesh() = default;
};

} // namespace gdl
