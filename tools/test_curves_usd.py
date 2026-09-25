"""Properties and controls for tools/curves_usd.py: pytest tools/test_curves_usd.py"""
import os
import sys

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curves_usd as cu  # noqa: E402

f32 = st.floats(-10.0, 10.0, width=32, allow_nan=False)
point = st.tuples(f32, f32, f32)
stroke = st.lists(point, min_size=2, max_size=12).map(lambda p: np.array(p, dtype=np.float32))
sketch = st.lists(stroke, min_size=1, max_size=8)
FAST = settings(max_examples=60, deadline=None)


def same(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


@FAST
@given(sketch)
def test_curves_text_round_trip_is_exact(strokes):
    assert same(cu.parse_curves(cu.format_curves(strokes)), strokes)


@FAST
@given(sketch, st.sets(st.integers(0, 7)))
def test_usda_round_trip_keeps_points_names_and_marks(strokes, marks):
    marks = {m for m in marks if m < len(strokes)}
    got, names, boundary, _ = cu.read_usda(cu.write_usda(strokes, boundary=marks))
    assert same(got, strokes)
    assert names == ["stroke_%03d" % i for i in range(len(strokes))]
    assert set(boundary) == marks


@FAST
@given(sketch)
def test_curves_to_usda_to_curves_is_exact_with_the_flip(strokes):
    body = cu.to_body(strokes)
    back = cu.from_body(cu.read_usda(cu.write_usda(body))[0])
    assert same(back, strokes)


@FAST
@given(sketch)
def test_the_flip_is_a_reflection(strokes):
    body = cu.to_body(strokes)
    assert all(np.array_equal(b[:, 2], -s[:, 2]) and np.array_equal(b[:, :2], s[:, :2]) for b, s in zip(body, strokes))


@pytest.mark.parametrize("text", [
    "v 3\n0 0 0\n1 1 1\n",             # fewer points than the count
    "v 2\n0 0 0\n1 1 1\n2 2 2\n",      # more
    "v 1\n0 0 0\n",                     # a one-point stroke
    "v x\n0 0 0\n",                     # a bad count
    "0 0 0\n",                          # a point outside any stroke
    "v 2\n0 0\n1 1 1\n",                # two coordinates
])
def test_malformed_curves_are_rejected(text):
    with pytest.raises(cu.CurvesError):
        cu.parse_curves(text)


def test_counts_that_disagree_with_points_are_rejected():
    text = cu.write_usda([np.zeros((3, 3), np.float32)]).replace("int[] curveVertexCounts = [3]", "int[] curveVertexCounts = [4]")
    with pytest.raises(cu.CurvesError):
        cu.read_usda(text)


def test_garbage_is_not_a_layer():
    with pytest.raises(cu.CurvesError):
        cu.read_usda("#usda 1.0\n( this is not usda")


def test_an_unmarked_stroke_authors_no_boundary():
    assert "primvars:boundary" not in cu.write_usda([np.zeros((2, 3), np.float32)])
