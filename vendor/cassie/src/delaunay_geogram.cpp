/**************************************************************************/
/*  delaunay_geogram.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "delaunay_geogram.h"

// Backed by Geogram's GEO::Delaunay "BDEL2d" in double. The Godot module
// used Godot's Delaunay2D (R128 fixed point, in float Vector2) here, which
// the guest does not carry. Geogram's predicates are exact on doubles
// (filtered, with exact fallback), so the old "No solution!" at 1 mm scale is
// addressed by conditioning instead: the input is mapped affinely into the
// unit box [0,1]^2 (one uniform scale, so the triangulation is unchanged)
// before the triangulation sees it.

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/numeric.h>
#include <geogram/delaunay/delaunay.h>

#include <cmath>
#include <cstddef>
#include <mutex>
#include <vector>

namespace {

void ensure_geogram_initialized() {
	static std::once_flag init_flag;
	std::call_once(init_flag, []() {
		GEO::initialize(GEO::GEOGRAM_INSTALL_NONE);
		GEO::Logger::instance()->set_quiet(true);
	});
}

} // namespace

namespace cassie {

bool delaunay_triangulate_2d_raw(const double *p_xy_coords, int n_points,
		int **out_face_indices, int *out_face_count) {
	*out_face_indices = nullptr;
	*out_face_count = 0;
	if (p_xy_coords == nullptr || n_points < 3) {
		return false;
	}

	double lo[2] = { p_xy_coords[0], p_xy_coords[1] };
	double hi[2] = { lo[0], lo[1] };
	for (int i = 0; i < n_points; ++i) {
		for (int a = 0; a < 2; ++a) {
			const double v = p_xy_coords[i * 2 + a];
			if (!std::isfinite(v)) {
				return false;
			}
			if (v < lo[a]) {
				lo[a] = v;
			}
			if (v > hi[a]) {
				hi[a] = v;
			}
		}
	}
	const double span = std::fmax(hi[0] - lo[0], hi[1] - lo[1]);
	if (!(span > 0.0)) {
		return false;
	}
	const double inv = 1.0 / span;
	std::vector<double> unit(std::size_t(n_points) * 2u);
	for (int i = 0; i < n_points; ++i) {
		unit[i * 2 + 0] = (p_xy_coords[i * 2 + 0] - lo[0]) * inv;
		unit[i * 2 + 1] = (p_xy_coords[i * 2 + 1] - lo[1]) * inv;
	}

	ensure_geogram_initialized();
	// Deterministic call to call, as DelaunayFaces does for BDEL.
	GEO::Numeric::random_reset();

	GEO::Delaunay_var delaunay = GEO::Delaunay::create(2, "BDEL2d");
	if (delaunay.get() == nullptr) {
		return false;
	}
	delaunay->set_vertices(GEO::index_t(n_points), unit.data());

	const GEO::index_t nf = delaunay->nb_cells();
	if (nf == 0) {
		return false; // colinear input: Delaunay2d leaves no cells
	}
	int *indices = new int[std::size_t(nf) * 3u];
	for (GEO::index_t c = 0; c < nf; ++c) {
		for (GEO::index_t k = 0; k < 3; ++k) {
			indices[c * 3 + k] = int(delaunay->cell_vertex(c, k));
		}
	}
	*out_face_indices = indices;
	*out_face_count = int(nf);
	return true;
}

} // namespace cassie
