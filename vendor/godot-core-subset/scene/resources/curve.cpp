// gdl: extracted from V-Sekai-fire/entities-godot@c165a519d2 by tools/vendor/godot_core_subset.py; edits listed in CITATION.cff. Do not edit.
/**************************************************************************/
/*  curve.cpp                                                             */
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

#include "curve.h"

#include "core/math/math_funcs.h"
#include "core/object/class_db.h"

#include <cfloat> // FLT_EPSILON
namespace gdl {

int Curve3D::get_point_count() const {
	return points.size();
}

void Curve3D::set_point_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	int old_size = points.size();
	if (old_size == p_count) {
		return;
	}

	if (old_size > p_count) {
		points.resize(p_count);
		mark_dirty();
	} else {
		for (int i = p_count - old_size; i > 0; i--) {
			_add_point(Vector3());
		}
	}
	notify_property_list_changed();
}

void Curve3D::_add_point(const Vector3 &p_position, const Vector3 &p_in, const Vector3 &p_out, int p_atpos) {
	Point n;
	n.position = p_position;
	n.in = p_in;
	n.out = p_out;
	if ((uint32_t)p_atpos < points.size()) {
		points.insert(p_atpos, n);
	} else {
		points.push_back(n);
	}

	mark_dirty();
}

void Curve3D::add_point(const Vector3 &p_position, const Vector3 &p_in, const Vector3 &p_out, int p_atpos) {
	_add_point(p_position, p_in, p_out, p_atpos);
	notify_property_list_changed();
}

void Curve3D::set_point_position(int p_index, const Vector3 &p_position) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_index, points.size());

	points[p_index].position = p_position;
	mark_dirty();
}

Vector3 Curve3D::get_point_position(int p_index) const {
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_index, points.size(), Vector3());
	return points[p_index].position;
}

void Curve3D::set_point_tilt(int p_index, real_t p_tilt) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_index, points.size());

	points[p_index].tilt = p_tilt;
	mark_dirty();
}

real_t Curve3D::get_point_tilt(int p_index) const {
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_index, points.size(), 0);
	return points[p_index].tilt;
}

void Curve3D::set_point_in(int p_index, const Vector3 &p_in) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_index, points.size());

	points[p_index].in = p_in;
	mark_dirty();
}

Vector3 Curve3D::get_point_in(int p_index) const {
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_index, points.size(), Vector3());
	return points[p_index].in;
}

void Curve3D::set_point_out(int p_index, const Vector3 &p_out) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_index, points.size());

	points[p_index].out = p_out;
	mark_dirty();
}

Vector3 Curve3D::get_point_out(int p_index) const {
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_index, points.size(), Vector3());
	return points[p_index].out;
}

void Curve3D::_remove_point(int p_index) {
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_index, points.size());
	points.remove_at(p_index);
	mark_dirty();
}

void Curve3D::remove_point(int p_index) {
	_remove_point(p_index);
	if (closed && points.size() < 2) {
		set_closed(false);
	}
	notify_property_list_changed();
}

void Curve3D::clear_points() {
	if (!points.is_empty()) {
		points.clear();
		mark_dirty();
		notify_property_list_changed();
	}
}

Vector3 Curve3D::sample(int p_index, real_t p_offset) const {
	int pc = points.size();
	ERR_FAIL_COND_V(pc == 0, Vector3());

	if (p_index >= pc - 1) {
		if (!closed) {
			return points[pc - 1].position;
		} else {
			p_index = pc - 1;
		}
	} else if (p_index < 0) {
		return points[0].position;
	}

	Vector3 p0 = points[p_index].position;
	Vector3 p1 = p0 + points[p_index].out;
	Vector3 p3, p2;
	if (!closed || p_index < pc - 1) {
		p3 = points[p_index + 1].position;
		p2 = p3 + points[p_index + 1].in;
	} else {
		p3 = points[0].position;
		p2 = p3 + points[0].in;
	}

	return p0.bezier_interpolate(p1, p2, p3, p_offset);
}

Vector3 Curve3D::samplef(real_t p_findex) const {
	if (p_findex < 0) {
		p_findex = 0;
	} else if (p_findex >= points.size()) {
		p_findex = points.size();
	}

	return sample((int)p_findex, Math::fmod(p_findex, (real_t)1.0));
}

void Curve3D::mark_dirty() {
	baked_cache_dirty = true;
	emit_changed();
}

void Curve3D::_bake_segment3d(RBMap<real_t, Vector3> &r_bake, real_t p_begin, real_t p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, real_t p_tol) const {
	real_t mp = p_begin + (p_end - p_begin) * 0.5;
	Vector3 beg = p_a.bezier_interpolate(p_a + p_out, p_b + p_in, p_b, p_begin);
	Vector3 mid = p_a.bezier_interpolate(p_a + p_out, p_b + p_in, p_b, mp);
	Vector3 end = p_a.bezier_interpolate(p_a + p_out, p_b + p_in, p_b, p_end);

	Vector3 na = (mid - beg).normalized();
	Vector3 nb = (end - mid).normalized();
	real_t dp = na.dot(nb);

	if (dp < Math::cos(Math::deg_to_rad(p_tol))) {
		r_bake[mp] = mid;
	}
	if (p_depth < p_max_depth) {
		_bake_segment3d(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
		_bake_segment3d(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
	}
}

void Curve3D::_bake_segment3d_even_length(RBMap<real_t, Vector3> &r_bake, real_t p_begin, real_t p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, real_t p_length) const {
	Vector3 beg = p_a.bezier_interpolate(p_a + p_out, p_b + p_in, p_b, p_begin);
	Vector3 end = p_a.bezier_interpolate(p_a + p_out, p_b + p_in, p_b, p_end);

	real_t length = beg.distance_to(end);

	if (length > p_length && p_depth < p_max_depth) {
		real_t mp = (p_begin + p_end) * 0.5;
		Vector3 mid = p_a.bezier_interpolate(p_a + p_out, p_b + p_in, p_b, mp);
		r_bake[mp] = mid;

		_bake_segment3d_even_length(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_length);
		_bake_segment3d_even_length(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_length);
	}
}

Vector3 Curve3D::_calculate_tangent(const Vector3 &p_begin, const Vector3 &p_control_1, const Vector3 &p_control_2, const Vector3 &p_end, const real_t p_t) {
	// Handle corner cases.
	if (Math::is_zero_approx(p_t - 0.0f)) {
		if (p_control_1.is_equal_approx(p_begin)) {
			if (p_control_1.is_equal_approx(p_control_2)) {
				return (p_end - p_begin).normalized();
			} else {
				return (p_control_2 - p_begin).normalized();
			}
		}
	} else if (Math::is_zero_approx(p_t - 1.0f)) {
		if (p_control_2.is_equal_approx(p_end)) {
			if (p_control_2.is_equal_approx(p_control_1)) {
				return (p_end - p_begin).normalized();
			} else {
				return (p_end - p_control_1).normalized();
			}
		}
	}

	if (p_control_1.is_equal_approx(p_end) && p_control_2.is_equal_approx(p_begin)) {
		return (p_end - p_begin).normalized();
	}

	return p_begin.bezier_derivative(p_control_1, p_control_2, p_end, p_t).normalized();
}

void Curve3D::_bake() const {
	if (!baked_cache_dirty) {
		return;
	}

	baked_max_ofs = 0;
	baked_cache_dirty = false;

	if (points.is_empty()) {
#ifdef TOOLS_ENABLED
		points_in_cache.clear();
#endif
		baked_point_cache.clear();
		baked_tilt_cache.clear();
		baked_dist_cache.clear();

		baked_forward_vector_cache.clear();
		baked_up_vector_cache.clear();
		return;
	}

	if (points.size() == 1) {
#ifdef TOOLS_ENABLED
		points_in_cache.resize(1);
		points_in_cache.set(0, 0);
#endif

		baked_point_cache.resize(1);
		baked_point_cache.set(0, points[0].position);
		baked_tilt_cache.resize(1);
		baked_tilt_cache.set(0, points[0].tilt);
		baked_dist_cache.resize(1);
		baked_dist_cache.set(0, 0.0);
		baked_forward_vector_cache.resize(1);
		baked_forward_vector_cache.set(0, Vector3(0.0, 0.0, 1.0));

		if (up_vector_enabled) {
			baked_up_vector_cache.resize(1);
			baked_up_vector_cache.set(0, Vector3(0.0, 1.0, 0.0));
		} else {
			baked_up_vector_cache.clear();
		}

		return;
	}

	// Step 1: Tessellate curve to (almost) even length segments.
	{
		Vector<RBMap<real_t, Vector3>> midpoints = _tessellate_even_length(10, bake_interval);

		const int num_intervals = closed ? points.size() : points.size() - 1;

#ifdef TOOLS_ENABLED
		points_in_cache.resize(closed ? (points.size() + 1) : points.size());
		points_in_cache.set(0, 0);
#endif

		// Point Count: Begins at 1 to account for the last point.
		int pc = 1;
		for (int i = 0; i < num_intervals; i++) {
			pc++;
			pc += midpoints[i].size();
#ifdef TOOLS_ENABLED
			points_in_cache.set(i + 1, pc - 1);
#endif
		}

		baked_point_cache.resize(pc);
		baked_tilt_cache.resize(pc);
		baked_dist_cache.resize(pc);
		baked_forward_vector_cache.resize(pc);

		Vector3 *bpw = baked_point_cache.ptrw();
		real_t *btw = baked_tilt_cache.ptrw();
		Vector3 *bfw = baked_forward_vector_cache.ptrw();

		// Collect positions and sample tilts and tangents for each baked points.
		bpw[0] = points[0].position;
		bfw[0] = _calculate_tangent(points[0].position, points[0].position + points[0].out, points[1].position + points[1].in, points[1].position, 0.0);
		btw[0] = points[0].tilt;
		int pidx = 0;

		for (int i = 0; i < num_intervals; i++) {
			for (const KeyValue<real_t, Vector3> &E : midpoints[i]) {
				pidx++;
				bpw[pidx] = E.value;
				if (!closed || i < num_intervals - 1) {
					bfw[pidx] = _calculate_tangent(points[i].position, points[i].position + points[i].out, points[i + 1].position + points[i + 1].in, points[i + 1].position, E.key);
					btw[pidx] = Math::lerp(points[i].tilt, points[i + 1].tilt, E.key);
				} else {
					bfw[pidx] = _calculate_tangent(points[i].position, points[i].position + points[i].out, points[0].position + points[0].in, points[0].position, E.key);
					btw[pidx] = Math::lerp(points[i].tilt, points[0].tilt, E.key);
				}
			}

			pidx++;
			if (!closed || i < num_intervals - 1) {
				bpw[pidx] = points[i + 1].position;
				bfw[pidx] = _calculate_tangent(points[i].position, points[i].position + points[i].out, points[i + 1].position + points[i + 1].in, points[i + 1].position, 1.0);
				btw[pidx] = points[i + 1].tilt;
			} else {
				bpw[pidx] = points[0].position;
				bfw[pidx] = _calculate_tangent(points[i].position, points[i].position + points[i].out, points[0].position + points[0].in, points[0].position, 1.0);
				btw[pidx] = points[0].tilt;
			}
		}

		// Recalculate the baked distances.
		real_t *bdw = baked_dist_cache.ptrw();
		bdw[0] = 0.0;
		for (int i = 0; i < pc - 1; i++) {
			bdw[i + 1] = bdw[i] + bpw[i].distance_to(bpw[i + 1]);
		}
		baked_max_ofs = bdw[pc - 1];
	}

	if (!up_vector_enabled) {
		baked_up_vector_cache.resize(0);
		return;
	}

	// Step 2: Calculate the up vectors and the whole local reference frame.
	//
	// See Dougan, Carl. "The parallel transport frame." Game Programming Gems 2 (2001): 215-219.
	// for an example discussing about why not the Frenet frame.
	{
		int point_count = baked_point_cache.size();

		baked_up_vector_cache.resize(point_count);
		Vector3 *up_write = baked_up_vector_cache.ptrw();

		const Vector3 *forward_ptr = baked_forward_vector_cache.ptr();
		const Vector3 *points_ptr = baked_point_cache.ptr();

		Basis frame; // X-right, Y-up, -Z-forward.
		Basis frame_prev;

		// Set the initial frame based on Y-up rule.
		{
			Vector3 forward = forward_ptr[0];

			if (std::abs(forward.dot(Vector3(0, 1, 0))) > 1.0 - UNIT_EPSILON) {
				frame_prev = Basis::looking_at(forward, Vector3(1, 0, 0));
			} else {
				frame_prev = Basis::looking_at(forward, Vector3(0, 1, 0));
			}

			up_write[0] = frame_prev.get_column(1);
		}

		// Calculate the Parallel Transport Frame.
		for (int idx = 1; idx < point_count; idx++) {
			Vector3 forward = forward_ptr[idx];

			Basis rotate;
			rotate.rotate_to_align(-frame_prev.get_column(2), forward);
			frame = rotate * frame_prev;
			frame.orthonormalize(); // Guard against float error accumulation.

			up_write[idx] = frame.get_column(1);
			frame_prev = frame;
		}

		bool is_loop = true;
		// Loop smoothing only applies when the curve is a loop, which means two ends meet, and share forward directions.
		{
			if (!points_ptr[0].is_equal_approx(points_ptr[point_count - 1])) {
				is_loop = false;
			}

			real_t dot = forward_ptr[0].dot(forward_ptr[point_count - 1]);
			if (dot < 1.0 - UNIT_EPSILON) { // Alignment should not be too tight, or it doesn't work for coarse bake interval.
				is_loop = false;
			}
		}

		// Twist up vectors, so that they align at two ends of the curve.
		if (is_loop) {
			const Vector3 up_start = up_write[0];
			const Vector3 up_end = up_write[point_count - 1];

			real_t sign = SIGN(up_end.cross(up_start).dot(forward_ptr[0]));
			real_t full_angle = Quaternion(up_end, up_start).get_angle();

			if (std::abs(full_angle) < CMP_EPSILON) {
				return;
			} else {
				const real_t *dists = baked_dist_cache.ptr();
				for (int idx = 1; idx < point_count; idx++) {
					const real_t frac = dists[idx] / baked_max_ofs;
					const real_t angle = Math::lerp((real_t)0.0, full_angle, frac);
					Basis twist(forward_ptr[idx] * sign, angle);

					up_write[idx] = twist.xform(up_write[idx]);
				}
			}
		}
	}
}

real_t Curve3D::get_baked_length() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_max_ofs;
}

Curve3D::Interval Curve3D::_find_interval(real_t p_offset) const {
	Interval interval = {
		-1,
		0.0
	};
	ERR_FAIL_COND_V_MSG(baked_cache_dirty, interval, "Backed cache is dirty");

	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc < 2, interval, "Less than two points in cache");

	int start = 0;
	int end = pc;
	int idx = (end + start) / 2;
	// Binary search to find baked points.
	while (start < idx) {
		real_t offset = baked_dist_cache[idx];
		if (p_offset <= offset) {
			end = idx;
		} else {
			start = idx;
		}
		idx = (end + start) / 2;
	}

	real_t offset_begin = baked_dist_cache[idx];
	real_t offset_end = baked_dist_cache[idx + 1];

	real_t idx_interval = offset_end - offset_begin;
	ERR_FAIL_COND_V_MSG(p_offset < offset_begin || p_offset > offset_end, interval, "Offset out of range.");

	interval.idx = idx;
	if (idx_interval < FLT_EPSILON) {
		interval.frac = 0.5; // For a very short interval, 0.5 is a reasonable choice.
		ERR_FAIL_V_MSG(interval, "Zero length interval.");
	}

	interval.frac = (p_offset - offset_begin) / idx_interval;
	return interval;
}

Vector3 Curve3D::_sample_baked(Interval p_interval, bool p_cubic) const {
	// Assuming p_interval is valid.
	ERR_FAIL_INDEX_V_MSG(p_interval.idx, baked_point_cache.size(), Vector3(), "Invalid interval");

	int idx = p_interval.idx;
	real_t frac = p_interval.frac;

	const Vector3 *r = baked_point_cache.ptr();
	int pc = baked_point_cache.size();

	if (p_cubic) {
		Vector3 pre = idx > 0 ? r[idx - 1] : r[idx];
		Vector3 post = (idx < (pc - 2)) ? r[idx + 2] : r[idx + 1];
		return r[idx].cubic_interpolate(r[idx + 1], pre, post, frac);
	} else {
		return r[idx].lerp(r[idx + 1], frac);
	}
}

real_t Curve3D::_sample_baked_tilt(Interval p_interval) const {
	// Assuming that p_interval is valid.
	ERR_FAIL_INDEX_V_MSG(p_interval.idx, baked_tilt_cache.size(), 0.0, "Invalid interval");

	int idx = p_interval.idx;
	real_t frac = p_interval.frac;

	const real_t *r = baked_tilt_cache.ptr();

	return Math::lerp(r[idx], r[idx + 1], frac);
}

// Internal method for getting posture at a baked point. Assuming caller
// make all safety checks.
Basis Curve3D::_compose_posture(int p_index) const {
	Vector3 forward = baked_forward_vector_cache[p_index];

	Vector3 up;
	if (up_vector_enabled) {
		up = baked_up_vector_cache[p_index];
	} else {
		up = Vector3(0.0, 1.0, 0.0);
	}

	const Basis frame = Basis::looking_at(forward, up);
	return frame;
}

Basis Curve3D::_sample_posture(Interval p_interval, bool p_apply_tilt) const {
	// Assuming that p_interval is valid.
	ERR_FAIL_INDEX_V_MSG(p_interval.idx, baked_point_cache.size(), Basis(), "Invalid interval");
	if (up_vector_enabled) {
		ERR_FAIL_INDEX_V_MSG(p_interval.idx, baked_up_vector_cache.size(), Basis(), "Invalid interval");
	}

	int idx = p_interval.idx;
	real_t frac = p_interval.frac;

	// Get frames at both ends of the interval, then interpolate.
	const Basis frame_begin = _compose_posture(idx);
	const Basis frame_end = _compose_posture(idx + 1);
	const Basis frame = frame_begin.slerp(frame_end, frac).orthonormalized();

	if (!p_apply_tilt) {
		return frame;
	}

	// Applying tilt.
	const real_t tilt = _sample_baked_tilt(p_interval);
	Vector3 tangent = -frame.get_column(2);

	const Basis twist(tangent, tilt);
	return twist * frame;
}

#ifdef TOOLS_ENABLED
// Get posture at a control point. Needed for Gizmo implementation.
Basis Curve3D::get_point_baked_posture(int p_index, bool p_apply_tilt) const {
	if (baked_cache_dirty) {
		_bake();
	}

	// Assuming that p_idx is valid.
	ERR_FAIL_INDEX_V_MSG(p_index, points_in_cache.size(), Basis(), "Invalid control point index");

	int baked_idx = points_in_cache[p_index];
	Basis frame = _compose_posture(baked_idx);

	if (!p_apply_tilt) {
		return frame;
	}

	// Applying tilt.
	const real_t tilt = points[p_index].tilt;
	Vector3 tangent = -frame.get_column(2);
	const Basis twist(tangent, tilt);

	return twist * frame;
}
#endif

Vector3 Curve3D::sample_baked(real_t p_offset, bool p_cubic) const {
	// Make sure that p_offset is finite.
	ERR_FAIL_COND_V_MSG(!Math::is_finite(p_offset), Vector3(), "Offset is non-finite");

	if (baked_cache_dirty) {
		_bake();
	}

	// Validate: Curve may not have baked points.
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, Vector3(), "No points in Curve3D.");

	if (pc == 1) {
		return baked_point_cache[0];
	}

	p_offset = CLAMP(p_offset, 0.0, get_baked_length()); // PathFollower implement wrapping logic.

	Curve3D::Interval interval = _find_interval(p_offset);
	return _sample_baked(interval, p_cubic);
}

Transform3D Curve3D::sample_baked_with_rotation(real_t p_offset, bool p_cubic, bool p_apply_tilt) const {
	// Make sure that p_offset is finite.
	ERR_FAIL_COND_V_MSG(!Math::is_finite(p_offset), Transform3D(), "Offset is non-finite");

	if (baked_cache_dirty) {
		_bake();
	}

	// Validate: Curve may not have baked points.
	const int point_count = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(point_count == 0, Transform3D(), "No points in Curve3D.");

	if (point_count == 1) {
		Transform3D t;
		t.origin = baked_point_cache.get(0);
		ERR_FAIL_V_MSG(t, "Only 1 point in Curve3D.");
	}

	p_offset = CLAMP(p_offset, 0.0, get_baked_length()); // PathFollower implement wrapping logic.

	// 0. Find interval for all sampling steps.
	Curve3D::Interval interval = _find_interval(p_offset);

	// 1. Sample position.
	Vector3 pos = _sample_baked(interval, p_cubic);

	// 2. Sample rotation frame.
	Basis frame = _sample_posture(interval, p_apply_tilt);

	return Transform3D(frame, pos);
}

real_t Curve3D::sample_baked_tilt(real_t p_offset) const {
	// Make sure that p_offset is finite.
	ERR_FAIL_COND_V_MSG(!Math::is_finite(p_offset), 0, "Offset is non-finite");

	if (baked_cache_dirty) {
		_bake();
	}

	// Validate: Curve may not have baked tilts.
	int pc = baked_tilt_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, 0, "No tilts in Curve3D.");

	if (pc == 1) {
		return baked_tilt_cache.get(0);
	}

	p_offset = CLAMP(p_offset, 0.0, get_baked_length()); // PathFollower implement wrapping logic.

	Curve3D::Interval interval = _find_interval(p_offset);
	return _sample_baked_tilt(interval);
}

Vector3 Curve3D::sample_baked_up_vector(real_t p_offset, bool p_apply_tilt) const {
	// Make sure that p_offset is finite.
	ERR_FAIL_COND_V_MSG(!Math::is_finite(p_offset), Vector3(0, 1, 0), "Offset is non-finite");

	if (baked_cache_dirty) {
		_bake();
	}

	// Validate: Curve may not have baked up vectors.
	ERR_FAIL_COND_V_MSG(!up_vector_enabled, Vector3(0, 1, 0), "No up vectors in Curve3D.");

	int count = baked_up_vector_cache.size();
	if (count == 1) {
		return baked_up_vector_cache.get(0);
	}

	p_offset = CLAMP(p_offset, 0.0, get_baked_length()); // PathFollower implement wrapping logic.

	Curve3D::Interval interval = _find_interval(p_offset);
	return _sample_posture(interval, p_apply_tilt).get_column(1);
}

PackedVector3Array Curve3D::get_baked_points() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_point_cache;
}

Vector<real_t> Curve3D::get_baked_tilts() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_tilt_cache;
}

PackedVector3Array Curve3D::get_baked_up_vectors() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_up_vector_cache;
}

Vector<real_t> Curve3D::get_baked_dist_cache() const {
	if (baked_cache_dirty) {
		_bake();
	}

	return baked_dist_cache;
}

Vector3 Curve3D::get_closest_point(const Vector3 &p_to_point) const {
	// Brute force method.

	if (baked_cache_dirty) {
		_bake();
	}

	// Validate: Curve may not have baked points.
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, Vector3(), "No points in Curve3D.");

	if (pc == 1) {
		return baked_point_cache.get(0);
	}

	const Vector3 *r = baked_point_cache.ptr();

	Vector3 nearest;
	real_t nearest_dist = -1.0f;

	for (int i = 0; i < pc - 1; i++) {
		const real_t interval = baked_dist_cache[i + 1] - baked_dist_cache[i];
		Vector3 origin = r[i];
		Vector3 direction = (r[i + 1] - origin) / interval;

		real_t d = CLAMP((p_to_point - origin).dot(direction), 0.0f, interval);
		Vector3 proj = origin + direction * d;

		real_t dist = proj.distance_squared_to(p_to_point);

		if (nearest_dist < 0.0f || dist < nearest_dist) {
			nearest = proj;
			nearest_dist = dist;
		}
	}

	return nearest;
}

PackedVector3Array Curve3D::get_points() const {
	return _get_data()["points"];
}

real_t Curve3D::get_closest_offset(const Vector3 &p_to_point) const {
	// Brute force method.

	if (baked_cache_dirty) {
		_bake();
	}

	// Validate: Curve may not have baked points.
	int pc = baked_point_cache.size();
	ERR_FAIL_COND_V_MSG(pc == 0, 0.0f, "No points in Curve3D.");

	if (pc == 1) {
		return 0.0f;
	}

	const Vector3 *r = baked_point_cache.ptr();

	real_t nearest = 0.0f;
	real_t nearest_dist = -1.0f;
	real_t offset;

	for (int i = 0; i < pc - 1; i++) {
		offset = baked_dist_cache[i];

		const real_t interval = baked_dist_cache[i + 1] - baked_dist_cache[i];
		Vector3 origin = r[i];
		Vector3 direction = (r[i + 1] - origin) / interval;

		real_t d = CLAMP((p_to_point - origin).dot(direction), 0.0f, interval);
		Vector3 proj = origin + direction * d;

		real_t dist = proj.distance_squared_to(p_to_point);

		if (nearest_dist < 0.0f || dist < nearest_dist) {
			nearest = offset + d;
			nearest_dist = dist;
		}
	}

	return nearest;
}

void Curve3D::set_closed(bool p_closed) {
	if (closed == p_closed) {
		return;
	}

	closed = p_closed;
	mark_dirty();
	notify_property_list_changed();
}

bool Curve3D::is_closed() const {
	return closed;
}

void Curve3D::set_bake_interval(real_t p_tolerance) {
	bake_interval = p_tolerance;
	mark_dirty();
}

real_t Curve3D::get_bake_interval() const {
	return bake_interval;
}

void Curve3D::set_up_vector_enabled(bool p_enable) {
	up_vector_enabled = p_enable;
	mark_dirty();
}

bool Curve3D::is_up_vector_enabled() const {
	return up_vector_enabled;
}

Dictionary Curve3D::_get_data() const {
	Dictionary dc;

	PackedVector3Array d;
	d.resize(points.size() * 3);
	Vector3 *w = d.ptrw();
	Vector<real_t> t;
	t.resize(points.size());
	real_t *wt = t.ptrw();

	for (uint32_t i = 0; i < points.size(); i++) {
		w[i * 3 + 0] = points[i].in;
		w[i * 3 + 1] = points[i].out;
		w[i * 3 + 2] = points[i].position;
		wt[i] = points[i].tilt;
	}

	dc["points"] = d;
	dc["tilts"] = t;

	return dc;
}

void Curve3D::_set_data(const Dictionary &p_data) {
	ERR_FAIL_COND(!p_data.has("points"));
	ERR_FAIL_COND(!p_data.has("tilts"));

	PackedVector3Array rp = p_data["points"];
	int pc = rp.size();
	ERR_FAIL_COND(pc % 3 != 0);
	int old_size = points.size();
	int new_size = pc / 3;
	if (old_size != new_size) {
		points.resize(new_size);
	}
	const Vector3 *r = rp.ptr();
	Vector<real_t> rtl = p_data["tilts"];
	const real_t *rt = rtl.ptr();

	for (uint32_t i = 0; i < points.size(); i++) {
		points[i].in = r[i * 3 + 0];
		points[i].out = r[i * 3 + 1];
		points[i].position = r[i * 3 + 2];
		points[i].tilt = rt[i];
	}

	mark_dirty();
	if (old_size != new_size) {
		notify_property_list_changed();
	}
}

PackedVector3Array Curve3D::tessellate(int p_max_stages, real_t p_tolerance) const {
	PackedVector3Array tess;

	if (points.is_empty()) {
		return tess;
	}
	Vector<RBMap<real_t, Vector3>> midpoints;

	const int num_intervals = closed ? points.size() : points.size() - 1;
	midpoints.resize(num_intervals);

	// Point Count: Begins at 1 to account for the last point.
	int pc = 1;
	for (int i = 0; i < num_intervals; i++) {
		if (!closed || i < num_intervals - 1) {
			_bake_segment3d(midpoints.write[i], 0, 1, points[i].position, points[i].out, points[i + 1].position, points[i + 1].in, 0, p_max_stages, p_tolerance);
		} else {
			_bake_segment3d(midpoints.write[i], 0, 1, points[i].position, points[i].out, points[0].position, points[0].in, 0, p_max_stages, p_tolerance);
		}
		pc++;
		pc += midpoints[i].size();
	}

	tess.resize(pc);
	Vector3 *bpw = tess.ptrw();
	bpw[0] = points[0].position;
	int pidx = 0;

	for (int i = 0; i < num_intervals; i++) {
		for (const KeyValue<real_t, Vector3> &E : midpoints[i]) {
			pidx++;
			bpw[pidx] = E.value;
		}

		pidx++;
		if (!closed || i < num_intervals - 1) {
			bpw[pidx] = points[i + 1].position;
		} else {
			bpw[pidx] = points[0].position;
		}
	}

	return tess;
}

Vector<RBMap<real_t, Vector3>> Curve3D::_tessellate_even_length(int p_max_stages, real_t p_length) const {
	Vector<RBMap<real_t, Vector3>> midpoints;
	ERR_FAIL_COND_V_MSG(points.size() < 2, midpoints, "Curve must have at least 2 control point");

	const int num_intervals = closed ? points.size() : points.size() - 1;
	midpoints.resize(num_intervals);

	for (int i = 0; i < num_intervals; i++) {
		if (!closed || i < num_intervals - 1) {
			_bake_segment3d_even_length(midpoints.write[i], 0, 1, points[i].position, points[i].out, points[i + 1].position, points[i + 1].in, 0, p_max_stages, p_length);
		} else {
			_bake_segment3d_even_length(midpoints.write[i], 0, 1, points[i].position, points[i].out, points[0].position, points[0].in, 0, p_max_stages, p_length);
		}
	}
	return midpoints;
}

PackedVector3Array Curve3D::tessellate_even_length(int p_max_stages, real_t p_length) const {
	PackedVector3Array tess;

	Vector<RBMap<real_t, Vector3>> midpoints = _tessellate_even_length(p_max_stages, p_length);
	if (midpoints.is_empty()) {
		return tess;
	}

	const int num_intervals = closed ? points.size() : points.size() - 1;
	// Point Count: Begins at 1 to account for the last point.
	int pc = 1;
	for (int i = 0; i < num_intervals; i++) {
		pc++;
		pc += midpoints[i].size();
	}

	tess.resize(pc);
	Vector3 *bpw = tess.ptrw();
	bpw[0] = points[0].position;
	int pidx = 0;

	for (int i = 0; i < num_intervals; i++) {
		for (const KeyValue<real_t, Vector3> &E : midpoints[i]) {
			pidx++;
			bpw[pidx] = E.value;
		}

		pidx++;
		if (!closed || i < num_intervals - 1) {
			bpw[pidx] = points[i + 1].position;
		} else {
			bpw[pidx] = points[0].position;
		}
	}

	return tess;
}

Curve3D::Curve3D() {
}

} // namespace gdl
