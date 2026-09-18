# 3d_augmentation

Data-augmentation helpers for 3D volumetric data (e.g. medical-imaging stacks
such as CT/MRI scans) and the 3D landmark coordinates that go with them. Each
function is a standalone numpy/scipy transform meant to be composed into a
training-time augmentation pipeline for volume- or landmark-based models.

All functions live in `src/augmentation.py`.

## Augmentation functions

| Function | What it does |
| --- | --- |
| `resize_nd_array(image, new_size)` | Resizes a 3D volume to an arbitrary target shape via `scipy.ndimage.zoom` (spline-interpolated resampling, not a crop/pad). |
| `rotate_nd_array(image, rot_angle, rot_axis)` | Picks one axis and one angle at random from the given candidate lists and rotates the volume in-plane, slice by slice, about that axis. |
| `rotate_landmarks(landmarks, angle, rot_axis, max_size)` | Rotates 3D landmark coordinates by a single angle about one axis (0/1/2), using half of `max_size` as the rotation center. |
| `rotate_3d_with_landmarks(image, landmarks, rot_angle, rot_axis)` | Runs the same random rotation as `rotate_nd_array` on a volume and applies the matching rotation to its landmarks via `rotate_landmarks`, so both stay aligned. |
| `pad_3d(images_3d, x_padd, y_padd, z_padd)` | Zero-pads a 3D volume by embedding it in a larger zero-filled array. Only produces symmetric padding when `x_padd == y_padd` -- see the function's docstring for the axis caveat. |
| `padd_3d_landmarks(landmarks, x_padd, y_padd, z_padd)` | Computes landmarks shifted by fixed padding offsets. Currently has no `return` statement, so it always yields `None` (documented as-is; use `random_padding`/`random_padding_with_landmark` for a version that returns usable coordinates). |
| `random_padding(image, ...)` | Intended to apply random per-axis zero-padding, resize to a fixed `(128, 128, 128)` shape, and rescale landmark coordinates to match -- but its signature takes no `landmarks` argument while its body still reads one, so it currently always raises `UnboundLocalError`. Documented as-is; not fixed here. |
| `random_padding_with_landmark(image, landmarks, output_size=[-1, -1, -1])` | The working, landmarks-taking equivalent of `random_padding`: pads by a random `[0, 25)` amount per axis and resizes to `output_size` (defaults to the input's own shape). Inherits `pad_3d`'s caveat, so it only succeeds reliably when the sampled x/y padding amounts happen to match -- it fails roughly half the time with the default range. |

See each function's docstring in `src/augmentation.py` for exact parameter
shapes, return shapes, and the rotation-axis/padding-axis conventions used.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest tests/
```
