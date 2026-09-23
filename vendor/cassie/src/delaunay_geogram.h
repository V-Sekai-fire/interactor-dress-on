/**************************************************************************/
/*  delaunay_geogram.h                                                    */
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

#pragma once

// Geogram-backed 2D Delaunay for mwt::DMWT's coplanar branch
// (DMWT.cpp:548), the one caller in the curvenet subset. The Godot-typed
// delaunay_triangulate_2d / delaunay_tetrahedralize_3d overloads are gone:
// nothing in the subset calls them, and 3D Delaunay goes through
// cassie::DelaunayFaces (GEO::Delaunay "BDEL") already. This header has no
// Godot or Eigen dependency, so mwt compiles against it without the
// godot_lite prelude.

namespace cassie {

// 2D Delaunay triangulation of n_points points. `p_xy_coords` has length
// 2*n_points; each consecutive pair is (x, y).
//
// Returns the flat face index list (`*out_face_indices`, length
// 3 * *out_face_count) — caller owns and must `delete[]` it.
// Returns false on degenerate input (fewer than 3 points, non-finite
// coordinates, all points coincident or colinear).
bool delaunay_triangulate_2d_raw(const double *p_xy_coords, int n_points,
		int **out_face_indices, int *out_face_count);

} // namespace cassie
