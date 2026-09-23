// gdl: extracted from V-Sekai-fire/entities-godot@c165a519d2 by tools/vendor/godot_core_subset.py; edits listed in CITATION.cff. Do not edit.
/**************************************************************************/
/*  curve.h                                                               */
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

#include "core/io/resource.h"
#include "core/templates/rb_map.h"
namespace gdl {

class Curve3D : public Resource {
	GDCLASS(Curve3D, Resource);

	struct Point {
		Vector3 in;
		Vector3 out;
		Vector3 position;
		real_t tilt = 0.0;
	};

	LocalVector<Point> points;
#ifdef TOOLS_ENABLED
	// For Path3DGizmo.
	mutable Vector<size_t> points_in_cache;
#endif

	bool closed = false;

	mutable bool baked_cache_dirty = false;
	mutable PackedVector3Array baked_point_cache;
	mutable Vector<real_t> baked_tilt_cache;
	mutable PackedVector3Array baked_up_vector_cache;
	mutable PackedVector3Array baked_forward_vector_cache;
	mutable Vector<real_t> baked_dist_cache;
	mutable real_t baked_max_ofs = 0.0;

	void mark_dirty();

	static Vector3 _calculate_tangent(const Vector3 &p_begin, const Vector3 &p_control_1, const Vector3 &p_control_2, const Vector3 &p_end, const real_t p_t);
	void _bake() const;

	struct Interval {
		int idx;
		real_t frac;
	};
	Interval _find_interval(real_t p_offset) const;
	Vector3 _sample_baked(Interval p_interval, bool p_cubic) const;
	real_t _sample_baked_tilt(Interval p_interval) const;
	Basis _sample_posture(Interval p_interval, bool p_apply_tilt = false) const;
	Basis _compose_posture(int p_index) const;

	real_t bake_interval = 0.2;
	bool up_vector_enabled = true;

	void _bake_segment3d(RBMap<real_t, Vector3> &r_bake, real_t p_begin, real_t p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, real_t p_tol) const;
	void _bake_segment3d_even_length(RBMap<real_t, Vector3> &r_bake, real_t p_begin, real_t p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, real_t p_length) const;
	Dictionary _get_data() const;
	void _set_data(const Dictionary &p_data);

	void _add_point(const Vector3 &p_position, const Vector3 &p_in = Vector3(), const Vector3 &p_out = Vector3(), int p_atpos = -1);
	void _remove_point(int p_index);

	Vector<RBMap<real_t, Vector3>> _tessellate_even_length(int p_max_stages = 5, real_t p_length = 0.2) const;

public:
#ifdef TOOLS_ENABLED
	// For Path3DGizmo.
	Basis get_point_baked_posture(int p_index, bool p_apply_tilt = false) const;
#endif

	int get_point_count() const;
	void set_point_count(int p_count);
	void add_point(const Vector3 &p_position, const Vector3 &p_in = Vector3(), const Vector3 &p_out = Vector3(), int p_atpos = -1);
	void set_point_position(int p_index, const Vector3 &p_position);
	Vector3 get_point_position(int p_index) const;
	void set_point_tilt(int p_index, real_t p_tilt);
	real_t get_point_tilt(int p_index) const;
	void set_point_in(int p_index, const Vector3 &p_in);
	Vector3 get_point_in(int p_index) const;
	void set_point_out(int p_index, const Vector3 &p_out);
	Vector3 get_point_out(int p_index) const;
	void remove_point(int p_index);
	void clear_points();

	Vector3 sample(int p_index, real_t p_offset) const;
	Vector3 samplef(real_t p_findex) const;

	void set_closed(bool p_closed);
	bool is_closed() const;
	void set_bake_interval(real_t p_tolerance);
	real_t get_bake_interval() const;
	void set_up_vector_enabled(bool p_enable);
	bool is_up_vector_enabled() const;

	real_t get_baked_length() const;
	Vector3 sample_baked(real_t p_offset, bool p_cubic = false) const;
	Transform3D sample_baked_with_rotation(real_t p_offset, bool p_cubic = false, bool p_apply_tilt = false) const;
	real_t sample_baked_tilt(real_t p_offset) const;
	Vector3 sample_baked_up_vector(real_t p_offset, bool p_apply_tilt = false) const;
	PackedVector3Array get_baked_points() const; // Useful for going through.
	Vector<real_t> get_baked_tilts() const; //useful for going through
	PackedVector3Array get_baked_up_vectors() const;
	Vector<real_t> get_baked_dist_cache() const;
	Vector3 get_closest_point(const Vector3 &p_to_point) const;
	real_t get_closest_offset(const Vector3 &p_to_point) const;
	PackedVector3Array get_points() const;

	PackedVector3Array tessellate(int p_max_stages = 5, real_t p_tolerance = 4) const; // Useful for display.
	PackedVector3Array tessellate_even_length(int p_max_stages = 5, real_t p_length = 0.2) const; // Useful for baking.

	Curve3D();
};

} // namespace gdl
