import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate


def resize_nd_array(image, new_size: list[int, int, int]):
    "Resize 3d array of images stack"
    real_resize_factor = np.array(new_size) / np.array(list(image.shape))
    image = zoom(image, real_resize_factor)
    return image

def rotate_nd_array(image, rot_angle: float, rot_axis: list[int]):
    "Rotate nd Image stack"
    rot_axis = np.random.choice(rot_axis)
    angle = np.random.choice(rot_angle)
    output = []
    image = np.swapaxes(image, axis1=0, axis2=rot_axis)
    for slice in image:
        output.append(rotate(slice, -angle, reshape=False))
    output = np.array(output, dtype="uint8")
    output = np.swapaxes(output, axis1=0, axis2=rot_axis)    
    return output

def rotate_landmarks(landmarks, angle: float, rot_axis: int, max_size:list(int, int, int)):
    """Rotate landmarks around center.
    half of max size will be assigned as center.

    Args:
        landmarks (_type_): _description_
        angle (float): _description_
        rot_axis (int): _description_
        max_size (list): _description_

    Returns:
        _type_: _description_
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
    """Rotate 3d Images stack, with its landmarks.
    used for landmarksdetection

    Args:
        image (_type_): _description_
        landmarks (_type_): _description_
        rot_angle (float): _description_
        rot_axis (list[int]): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        images_3d (_type_): _description_
        x_padd (int, optional): _description_. Defaults to 0.
        y_padd (int, optional): _description_. Defaults to 0.
        z_padd (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
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
    """Random Padd 3d Images Stack

    Args:
        image (_type_): _description_
        min_x_padding (int, optional): _description_. Defaults to 0.
        max_x_padding (int, optional): _description_. Defaults to 25.
        min_y_padding (int, optional): _description_. Defaults to 0.
        max_y_padding (int, optional): _description_. Defaults to 25.
        min_z_padding (int, optional): _description_. Defaults to 0.
        max_z_padding (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_
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

def random_padding_with_landmark(image, landmarks, output_size: list[int, int, int]=[-1, -1, -1]):
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
