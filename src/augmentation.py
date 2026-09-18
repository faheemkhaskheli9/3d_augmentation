import numpy as np
from scipy.ndimage import rotate, zoom
from scipy.spatial.transform import Rotation


def resize_nd_array(image, new_size: list[int]):
    """Resize a 3D array (e.g. a volumetric image/scan stack) to an arbitrary
    target shape.

    A per-axis zoom factor is computed as ``new_size / image.shape`` and
    passed to ``scipy.ndimage.zoom``, which resamples the array with spline
    interpolation -- voxel values are interpolated, not just cropped/padded.

    Args:
        image: 3D numpy array (volume/image stack) to resize.
        new_size: target shape, e.g. ``[depth, height, width]``.

    Returns:
        numpy array with shape equal to ``new_size``.
    """
    real_resize_factor = np.array(new_size) / np.array(list(image.shape))
    image = zoom(image, real_resize_factor)
    return image

def rotate_nd_array(image, rot_angle: float, rot_axis: list[int]):
    """Randomly rotate a 3D volume in-plane about one of a set of candidate
    axes.

    One axis is drawn uniformly at random from ``rot_axis`` and one angle is
    drawn uniformly at random from ``rot_angle`` (both are candidate lists,
    not single values, despite the parameter names). The volume is
    axis-swapped so the chosen axis becomes axis 0, every 2D slice along that
    axis is rotated in-plane by ``-angle`` degrees with
    ``scipy.ndimage.rotate(reshape=False)`` (so each slice keeps its
    original shape), and the axes are swapped back.

    Args:
        image: 3D numpy array (volume/image stack) to rotate.
        rot_angle: sequence of candidate rotation angles in degrees; one
            entry is selected at random via ``numpy.random.choice``.
        rot_axis: sequence of candidate axis indices (0, 1 or 2); one entry
            is selected at random and used as the axis each 2D slice is
            taken perpendicular to.

    Returns:
        Rotated volume as a ``uint8`` numpy array, same shape as ``image``.
    """
    rot_axis = np.random.choice(rot_axis)
    angle = np.random.choice(rot_angle)
    output = []
    image = np.swapaxes(image, axis1=0, axis2=rot_axis)
    for slice in image:
        output.append(rotate(slice, -angle, reshape=False))
    output = np.array(output, dtype="uint8")
    output = np.swapaxes(output, axis1=0, axis2=rot_axis)    
    return output

def rotate_landmarks(landmarks, angle: float, rot_axis: int, max_size: list[int]):
    """Rotate 3D landmark coordinates about the volume's center point.

    The center used for every landmark is ``max_size // 2`` (integer
    division). Each landmark is translated so that center is the origin,
    rotated by ``angle`` degrees about the axis selected by ``rot_axis``
    using ``scipy.spatial.transform.Rotation.from_rotvec``, then translated
    back.

    ``rot_axis`` selects the rotation-axis vector:

    - ``0`` -> rotates about the x-axis, in the negative direction
      (``[-1, 0, 0]``)
    - ``1`` -> rotates about the y-axis (``[0, 1, 0]``)
    - ``2`` -> rotates about the z-axis (``[0, 0, 1]``)

    (axis ``0`` is inverted relative to ``1``/``2`` so it matches the
    ``-angle`` convention ``rotate_nd_array``/``rotate_3d_with_landmarks``
    use on the image side.)

    Args:
        landmarks: array-like of shape ``(N, 3)`` with landmark
            ``[x, y, z]`` coordinates.
        angle: rotation angle in degrees -- a single value, unlike
            ``rotate_nd_array``'s ``rot_angle`` candidate list.
        rot_axis: integer ``0``, ``1`` or ``2`` selecting the rotation axis
            (see above).
        max_size: the ``[x, y, z]`` volume shape whose half is used as the
            rotation center.

    Returns:
        numpy array of shape ``(N, 3)`` with the rotated landmark
        coordinates.
    """
    new_landmakrs = []
    for ld in landmarks:
        rotation_radians = np.radians(angle)
        if rot_axis == 0:
            rotation_axis = np.array([-1, 0, 0])
        if rot_axis == 1:
            rotation_axis = np.array([0, 1, 0])
        if rot_axis == 2:
            rotation_axis = np.array([0, 0, 1])

        rotation_vector = rotation_radians * rotation_axis
        rotation = Rotation.from_rotvec(rotation_vector)
        ld -= np.array(max_size)//2
        rotated_vec = rotation.apply(ld)
        rotated_vec += np.array(max_size)//2
        new_landmakrs.append(rotated_vec)

    return np.array(new_landmakrs)

def rotate_3d_with_landmarks(image, landmarks, rot_angle: float, rot_axis: list[int]):
    """Rotate a 3D volume and its landmarks together, keeping both aligned.

    Used for landmark-detection training, where the ground-truth landmark
    coordinates must be transformed the same way as the image. One axis is
    drawn at random from ``rot_axis`` and one angle at random from
    ``rot_angle`` (the same random-draw logic as ``rotate_nd_array``, but
    duplicated inline here rather than calling it), and the image is rotated
    exactly as ``rotate_nd_array`` does. The landmarks are then rotated with
    ``rotate_landmarks`` using that same angle/axis; the ``max_size`` passed
    to it is the shape of the *axis-swapped* image (after the chosen
    rotation axis has been moved to position 0), not the original image
    shape.

    Args:
        image: 3D numpy array (volume/image stack) to rotate.
        landmarks: array-like of shape ``(N, 3)`` with landmark
            ``[x, y, z]`` coordinates, aligned to ``image``.
        rot_angle: sequence of candidate rotation angles in degrees.
        rot_axis: sequence of candidate axis indices (0, 1 or 2).

    Returns:
        Tuple ``(rotated_image, rotated_landmarks)``: ``rotated_image`` is a
        ``uint8`` numpy array the same shape as ``image``; ``rotated_landmarks``
        is a numpy array of shape ``(N, 3)``.
    """
    rot_axis = np.random.choice(rot_axis)
    angle = np.random.choice(rot_angle)
    output = []
    image = np.swapaxes(image, axis1=0, axis2=rot_axis)
    for slice in image:
        output.append(rotate(slice, -angle, reshape=False))
    output = np.array(output, dtype="uint8")
    output = np.swapaxes(output, axis1=0, axis2=rot_axis)

    new_landmakrs = rotate_landmarks(landmarks, angle, rot_axis, image.shape)
    
    return output, new_landmakrs

def pad_3d(images_3d, x_padd=0, y_padd=0, z_padd=0):
    """Pad a 3D array with zero-valued voxels by embedding it inside a
    larger zero array.

    Given ``images_3d.shape == (height, width, depth)``, allocates a zero
    array of shape ``(height + 2*x_padd, width + 2*y_padd, depth + 2*z_padd)``
    and copies the input into it.

    Caveat: the axis an offset is *allocated* on and the axis it is *placed*
    on are swapped between ``x_padd``/``y_padd`` -- the height axis (axis 0)
    is grown by ``2*x_padd`` but the input is placed at offset ``y_padd`` on
    that axis, and the width axis (axis 1) is grown by ``2*y_padd`` but
    placed at offset ``x_padd``. This only yields symmetric zero-padding
    when ``x_padd == y_padd``; passing different values can raise a
    shape-mismatch error or produce asymmetric padding. ``z_padd`` (depth
    axis) is not affected by this and behaves as expected.

    Args:
        images_3d: 3D numpy array of shape ``(height, width, depth)`` to
            pad.
        x_padd: padding added (on both sides) to the height axis'
            allocated size -- see caveat above about the placement offset.
        y_padd: padding added (on both sides) to the width axis' allocated
            size -- see caveat above.
        z_padd: padding added on both sides of the depth axis.

    Returns:
        New zero-padded numpy array of shape
        ``(height+2*x_padd, width+2*y_padd, depth+2*z_padd)`` (``float64``,
        from ``np.zeros``) containing the original data placed per the
        offsets described above.
    """
    height, width, depth = images_3d.shape

    xi = x_padd
    xf = xi + width
    yi = y_padd
    yf = yi + height
    zi = z_padd
    zf = zi + depth

    new_image = np.zeros((height+(x_padd*2), width+(y_padd*2), depth+(z_padd*2)))
    new_image[yi:yf, xi:xf, zi:zf] = images_3d
    return new_image

def padd_3d_landmarks(landmarks, x_padd, y_padd, z_padd):
    """Compute landmark coordinates shifted by fixed padding offsets.

    Note: this function builds the shifted coordinate array but has no
    ``return`` statement, so calling it always yields ``None`` -- this is
    existing, as-is behavior, documented rather than fixed here. Use
    ``random_padding`` or ``random_padding_with_landmark`` for
    padding-plus-resize on landmarks that does return usable coordinates.

    Args:
        landmarks: array-like reshapeable to ``(N, 3)`` landmark
            coordinates.
        x_padd: offset added to each landmark's first coordinate.
        y_padd: offset added to each landmark's second coordinate.
        z_padd: offset added to each landmark's third coordinate.

    Returns:
        ``None`` (see note above).
    """
    new_landmarks = []
    landmarks = np.array(landmarks).reshape(-1, 3)
    for ld in landmarks:
        new_landmarks.append([(ld[0]+x_padd),
                              (ld[1]+y_padd),
                              (ld[2]+z_padd)
                              ])

    new_landmarks = np.array(new_landmarks)

def random_padding(image,
                   min_x_padding=0,
                   max_x_padding=25,
                   min_y_padding=0,
                   max_y_padding=25,
                   min_z_padding=0,
                   max_z_padding=25):
    """Intended to apply random zero-padding then resize a 3D volume to a
    fixed ``(128, 128, 128)`` shape, adjusting landmark coordinates to
    match -- but currently broken; see the bug note below.

    Padding amounts are drawn independently per axis via
    ``numpy.random.randint`` and forwarded to ``pad_3d`` -- see that
    function's docstring for exactly how the x/y amounts map onto array
    axes (they are cross-wired there, and this function forwards the
    sampled values as-is without correcting for that, so it can also raise
    the shape-mismatch error described there whenever the two sampled
    values differ).

    Bug: despite the intent above, this function's signature has **no**
    ``landmarks`` parameter, yet its body reads and reassigns a local named
    ``landmarks`` while building the return value. Because Python treats
    any name assigned inside a function as local, referencing it before
    that assignment raises ``UnboundLocalError`` -- so calling this
    function currently always raises, once/if it gets past the ``pad_3d``
    call above. Documented as-is rather than fixed here; use
    ``random_padding_with_landmark`` for the working, landmarks-taking
    equivalent (itself still subject to the ``pad_3d`` caveat above).

    Args:
        image: 3D numpy array (volume/image stack) to pad and resize.
        min_x_padding: lower bound (inclusive) for one axis' padding amount.
            Defaults to 0.
        max_x_padding: upper bound (exclusive) for that axis' padding
            amount. Defaults to 25.
        min_y_padding: lower bound (inclusive) for a second axis' padding
            amount. Defaults to 0.
        max_y_padding: upper bound (exclusive) for that axis' padding
            amount. Defaults to 25.
        min_z_padding: lower bound (inclusive) for the depth axis' padding
            amount. Defaults to 0.
        max_z_padding: upper bound (exclusive) for the depth axis' padding
            amount. Defaults to 25.

    Returns:
        Never returns normally today (see bug note above). If the bug were
        fixed, the intent is ``(new_image, new_landmarks)``: a
        ``(128, 128, 128)`` volume and an ``(N, 3)`` array of rescaled
        landmark coordinates.
    """
    paddy = np.random.randint(min_x_padding, max_x_padding)
    paddx = np.random.randint(min_y_padding, max_y_padding)
    paddz = np.random.randint(min_z_padding, max_z_padding)

    new_image = pad_3d(image, paddx, paddy, paddz)

    zoom_h, zoom_w, zoom_d = np.array([128, 128, 128]) / new_image.shape
    new_image = resize_nd_array(new_image, [128, 128, 128])

    new_landmarks = []
    landmarks = np.array(landmarks).reshape(-1, 3)
    for ld in landmarks:
        new_landmarks.append([(ld[0]+paddy)*zoom_h,
                              (ld[1]+paddx)*zoom_w,
                              (ld[2]+paddz)*zoom_d
                              ])

    new_landmarks = np.array(new_landmarks)

    return new_image, new_landmarks

def random_padding_with_landmark(image, landmarks, output_size: list[int] = [-1, -1, -1]):
    """Apply random zero-padding (0-25 voxels per axis) then resize a 3D
    volume to ``output_size``, adjusting landmark coordinates to match.

    Behaves like ``random_padding`` was intended to (and, unlike
    ``random_padding``, actually has a working ``landmarks`` parameter and
    reliably returns), except: the padding amount on every axis is
    independently sampled from a fixed ``[0, 25)`` range (no configurable
    min/max), and the output shape defaults to the input image's own shape
    rather than a fixed ``(128, 128, 128)`` -- pass an explicit 3-element
    ``output_size`` to resize to something else.

    Caveat inherited from ``pad_3d``: because the x- and y-axis padding
    amounts are sampled independently, this call raises a shape-mismatch
    error unless the two happen to be equal -- see ``pad_3d``'s docstring.
    In practice this makes this function fail roughly half the time with
    the default ``[0, 25)`` sampling range.

    Args:
        image: 3D numpy array (volume/image stack) to pad and resize.
        landmarks: array-like reshapeable to ``(N, 3)`` landmark
            coordinates, aligned to ``image``.
        output_size: target ``[x, y, z]`` shape after padding and resizing;
            the sentinel ``[-1, -1, -1]`` (default) means "use
            ``image.shape``".

    Returns:
        Tuple ``(new_image, new_landmarks)``: ``new_image`` is a numpy
        array of shape ``output_size``; ``new_landmarks`` is a numpy array
        of shape ``(N, 3)`` with coordinates shifted and rescaled to match.
    """
    paddy = np.random.randint(0, 25)
    paddx = np.random.randint(0, 25)
    paddz = np.random.randint(0, 25)

    if output_size == [-1, -1, -1]:
        output_size = image.shape

    new_image = pad_3d(image, paddx, paddy, paddz)

    zoom_h, zoom_w, zoom_d = np.array(output_size) / new_image.shape
    new_image = resize_nd_array(new_image, output_size)

    new_landmarks = []
    landmarks = np.array(landmarks).reshape(-1, 3)
    for ld in landmarks:
        new_landmarks.append([(ld[0]+paddy)*zoom_h,
                              (ld[1]+paddx)*zoom_w,
                              (ld[2]+paddz)*zoom_d
                              ])

    new_landmarks = np.array(new_landmarks)

    return new_image, new_landmarks
