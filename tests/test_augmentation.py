"""Regression tests for issue #1: rotate_landmarks' max_size annotation was
`list(int, int, int)` -- a call to `list()` with 3 positional args, which
raises `TypeError: list expected at most 1 argument, got 3` when Python
evaluates annotations eagerly at function-definition time (every Python
version before 3.14's deferred-annotations default). The module still
imported fine in this environment's Python 3.14, which is exactly why the
bug went unnoticed here."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

import augmentation

SRC_FILE = Path(augmentation.__file__)


def test_module_source_has_no_annotation_that_calls_list_as_a_function():
    """`list(int, int, int)` (a call) is the exact shape of the original bug;
    `list[int, ...]` (a subscript) is syntactically valid but still not what
    was intended for a 3-element size, so neither should remain anywhere in
    the module -- catches this bug recurring on a sibling parameter too."""
    tree = ast.parse(SRC_FILE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and len(node.args) > 1
        ):
            pytest.fail(f"found list(...) called with multiple positional args at line {node.lineno}")


def test_module_imports_and_defines_rotate_landmarks():
    # Reload to prove definition-time evaluation of the (now-fixed)
    # annotation doesn't raise, not just that a cached import succeeded.
    reloaded = importlib.reload(augmentation)
    assert callable(reloaded.rotate_landmarks)
    sig = inspect.signature(reloaded.rotate_landmarks)
    assert "max_size" in sig.parameters


def test_rotate_landmarks_rotates_a_small_synthetic_landmark_array():
    landmarks = np.array(
        [[10.0, 10.0, 10.0], [20.0, 10.0, 10.0], [10.0, 20.0, 10.0]]
    )
    max_size = [32, 32, 32]

    rotated = augmentation.rotate_landmarks(landmarks, angle=90.0, rot_axis=2, max_size=max_size)

    assert rotated.shape == landmarks.shape
    # A 90-degree rotation about the center actually moves at least one point.
    assert not np.allclose(rotated, landmarks)


def test_resize_nd_array_accepts_a_three_element_size():
    image = np.zeros((8, 8, 8), dtype="uint8")
    resized = augmentation.resize_nd_array(image, [4, 4, 4])
    assert resized.shape == (4, 4, 4)
