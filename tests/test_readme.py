"""Phase 1 README-hygiene checks for issue #2: the README's "Augmentation
functions" table must name every augmentation function that actually exists
in src/augmentation.py, and only those -- and a couple of its more specific
behavioral claims must hold against the real module, not just against
prose."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import augmentation

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

# Every top-level, non-underscore-prefixed function defined in the module.
MODULE_FUNCTIONS = sorted(
    name
    for name, obj in vars(augmentation).items()
    if inspect.isfunction(obj)
    and obj.__module__ == augmentation.__name__
    and not name.startswith("_")
)


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _readme_documented_function_names() -> set[str]:
    """Function names referenced as `name(...)` inside inline code spans,
    e.g. `` `resize_nd_array(image, new_size)` ``."""
    text = _readme_text()
    return set(re.findall(r"`([a-z_][a-z0-9_]*)\(", text))


def test_module_has_at_least_the_functions_the_issue_calls_out():
    assert {"resize_nd_array", "rotate_nd_array", "rotate_landmarks"} <= set(MODULE_FUNCTIONS)


def test_readme_documents_every_module_function_and_no_others():
    documented = _readme_documented_function_names()
    assert documented == set(MODULE_FUNCTIONS), (
        f"README documents {documented - set(MODULE_FUNCTIONS)} that don't exist, "
        f"and is missing {set(MODULE_FUNCTIONS) - documented}"
    )


def test_every_documented_function_is_callable_in_the_module():
    for name in _readme_documented_function_names():
        assert hasattr(augmentation, name), f"{name} is documented but missing from module"
        assert callable(getattr(augmentation, name)), f"{name} is documented but not callable"


def test_readme_notes_volumetric_and_landmark_data():
    text = _readme_text().lower()
    assert "3d" in text
    assert "landmark" in text
    assert "medical" in text or "volumetric" in text


def test_readme_claim_resize_changes_shape_matches_real_behavior():
    # Cross-checks the table's "resizes ... to an arbitrary target shape" claim.
    image = np.zeros((8, 8, 8), dtype="uint8")
    resized = augmentation.resize_nd_array(image, [4, 6, 10])
    assert resized.shape == (4, 6, 10)


def test_readme_claim_rotate_landmarks_uses_single_angle_not_a_list():
    # Cross-checks the table's "rotates ... by a single angle" claim: unlike
    # rotate_nd_array, rotate_landmarks must accept a scalar angle directly.
    landmarks = np.array([[10.0, 10.0, 10.0], [20.0, 5.0, 10.0]])
    rotated = augmentation.rotate_landmarks(landmarks, angle=45.0, rot_axis=1, max_size=[32, 32, 32])
    assert rotated.shape == landmarks.shape


def test_readme_claim_padd_3d_landmarks_returns_none():
    # Cross-checks the table's explicit claim that this function currently
    # has no return statement.
    result = augmentation.padd_3d_landmarks([[1, 2, 3]], x_padd=1, y_padd=2, z_padd=3)
    assert result is None


def test_readme_claim_pad_3d_is_symmetric_when_x_and_y_padding_match():
    # Cross-checks the table's caveat: symmetric padding requires x_padd == y_padd.
    image = np.ones((4, 4, 4))
    padded = augmentation.pad_3d(image, x_padd=2, y_padd=2, z_padd=1)
    assert padded.shape == (8, 8, 6)
    assert np.array_equal(padded[2:6, 2:6, 1:5], image)


def test_readme_claim_random_padding_currently_always_raises(monkeypatch):
    # Cross-checks the table's claim that random_padding has no `landmarks`
    # parameter yet references one in its body, so it always raises
    # UnboundLocalError. Padding amounts are pinned equal so the unrelated
    # pad_3d x/y caveat can't mask this and cause a different error instead.
    monkeypatch.setattr(np.random, "randint", lambda low, high: 3)
    image = np.ones((10, 10, 10), dtype="uint8")
    with pytest.raises(UnboundLocalError):
        augmentation.random_padding(image)


def test_readme_claim_random_padding_with_landmark_defaults_to_input_shape(monkeypatch):
    # Cross-checks the table's claim that this is the working, landmarks-
    # taking equivalent. Padding amounts are pinned equal so the unrelated
    # pad_3d x/y caveat (also documented and tested separately) can't make
    # this test flaky.
    monkeypatch.setattr(np.random, "randint", lambda low, high: 3)
    image = np.ones((6, 6, 6), dtype="uint8")
    landmarks = [[1, 1, 1], [2, 2, 2]]
    new_image, new_landmarks = augmentation.random_padding_with_landmark(image, landmarks)
    assert new_image.shape == image.shape
    assert new_landmarks.shape == (2, 3)


def test_readme_claim_random_padding_with_landmark_fails_when_x_y_padding_differ(monkeypatch):
    # Cross-checks the "fails roughly half the time" claim: mismatched
    # sampled x/y padding amounts hit pad_3d's documented shape caveat.
    calls = iter([7, 2, 1])  # paddy=7, paddx=2, paddz=1 -> x_padd != y_padd
    monkeypatch.setattr(np.random, "randint", lambda low, high: next(calls))
    image = np.ones((6, 6, 6), dtype="uint8")
    landmarks = [[1, 1, 1], [2, 2, 2]]
    with pytest.raises(ValueError):
        augmentation.random_padding_with_landmark(image, landmarks)
