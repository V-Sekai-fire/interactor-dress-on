#include "scene/resources/mesh.h"

namespace gdl {

void ArrayMesh::add_surface_from_arrays(PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &, const Dictionary &, uint64_t) {
	ERR_FAIL_COND_MSG(p_arrays.size() != ARRAY_MAX, "Arrays must have ARRAY_MAX (" + itos(ARRAY_MAX) + ") elements.");
	ERR_FAIL_COND_MSG(p_arrays[ARRAY_VERTEX].get_type() == Variant::NIL, "ARRAY_VERTEX is required.");
	Surface s;
	s.primitive = p_primitive;
	s.arrays = p_arrays.duplicate(); // the caller's Array stays theirs
	surfaces.push_back(s);
	emit_changed();
}

Array ArrayMesh::surface_get_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, int(surfaces.size()), Array());
	return surfaces[size_t(p_surface)].arrays.duplicate();
}

} // namespace gdl
