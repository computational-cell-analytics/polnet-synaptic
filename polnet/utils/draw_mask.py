"""
Utility functions for drawing segmentation masks from filament coordinates.

Adapted from TARDIS (tardis_em/cnn/data_processing/draw_mask.py)
Original authors: Robert Kiewisz, Tristan Bepler (MIT License 2021-2025)
"""

from math import pow, sqrt
from typing import Tuple, Iterable, Union

import numpy as np


def draw_instances(
    mask_size: Union[list, tuple],
    coordinate: np.ndarray,
    pixel_size: float,
    circle_size=70,
    label=True,
    dtype=None,
) -> np.ndarray:
    """
    Draws labeled or binary 3D masks based on the input coordinates, mask size, and additional parameters.

    If label generation is enabled, unique segment labels are created to distinguish between different segments.
    Spheres centered around specific points are drawn in the masks, with sizes determined by the input circle size
    and pixel size. Input parameters related to the shape, size, and type of the mask, as well as the labeling behavior,
    are fully customizable. This function is suitable for constructing semantic or instance masks.

    :param mask_size: The dimensions of the mask to be created.
    :type mask_size: list | tuple
    :param coordinate: An array of coordinates specifying the locations to draw the mask,
        shape [Label x X x Y x Z].
    :type coordinate: np.ndarray
    :param pixel_size: Size of a pixel in the mask, used to scale the mask appropriately.
    :type pixel_size: float
    :param circle_size: Diameter of the sphere to be drawn at each coordinate point. Defaults to 250.
    :type circle_size: int, optional
    :param label: Flag indicating whether a labeled mask or a binary mask should be created. Defaults to True.
    :type label: bool, optional
    :param dtype: Data type for the output mask. If not provided, defaults to np.uint16 or np.uint8.
    :type dtype: str | None, optional

    :return: A 3D mask generated based on the input parameters.
    :rtype: np.ndarray
    """
    if label:
        if coordinate.ndim != 2 or coordinate.shape[1] not in [3, 4]:
            raise ValueError(
                "Coordinates are of not correct shape, expected: "
                f"shape [Label x X x Y x Z] but {coordinate.shape} given!"
            )

    if label:
        label_mask = np.zeros(mask_size, dtype=np.uint16 if dtype is None else dtype)
    else:
        label_mask = np.zeros(mask_size, dtype=np.uint8 if dtype is None else dtype)

    if pixel_size == 0:
        pixel_size = 1

    r = (circle_size // 2) // pixel_size

    # Number of segments in coordinates
    if label:
        segments = np.unique(coordinate[:, 0])

        for i in segments:
            # Pick coordinates for each segment
            points = coordinate[np.where(coordinate[:, 0] == i)[0]][:, 1:]

            label = interpolation(points)
            all_cz, all_cy, all_cx = [], [], []

            for j in range(len(label)):
                c = label[j, :]  # Point center
                cz, cy, cx = draw_mask(r=r, c=c, label_mask=label_mask)
                all_cz.append(cz)
                all_cy.append(cy)
                all_cx.append(cx)

            all_cz, all_cy, all_cx = (
                np.concatenate(all_cz),
                np.concatenate(all_cy),
                np.concatenate(all_cx),
            )
            label_mask[all_cz, all_cy, all_cx] = i + 1

        return label_mask
    else:
        all_cz, all_cy, all_cx = [], [], []

        for c in coordinate:
            cz, cy, cx = draw_mask(r=r, c=c, label_mask=label_mask)
            all_cz.append(cz)
            all_cy.append(cy)
            all_cx.append(cx)

        all_cz, all_cy, all_cx = (
            np.concatenate(all_cz),
            np.concatenate(all_cy),
            np.concatenate(all_cx),
        )
        label_mask[all_cz, all_cy, all_cx] = 1

        return np.where(label_mask == 1, 1, 0).astype(np.uint8)


def draw_mask(
    r: int, c: np.ndarray, label_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draws a sphere mask on the given label array.

    :param r: The radius of the sphere in voxels.
    :param c: A numpy array containing the center coordinates [x, y, z].
    :param label_mask: A numpy array representing the label mask (3D).
    :return: A tuple of (cx, cy, cz) index arrays for the sphere voxels.
    """
    if label_mask.ndim != 3:
        raise ValueError(f"Unsupported dimensions {label_mask.ndim}, expected 3.")

    x = int(c[0])
    y = int(c[1])
    z = int(c[2])

    # input in polnet convention [x, y, z]
    cx, cy, cz = draw_sphere(r=r, c=(x, y, z), shape=label_mask.shape)
    return cx, cy, cz

def draw_sphere(
    r: int, c: tuple, shape: tuple
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the coordinates of a 3D spherical structure within a volume of specified shape.
    This function simulates a sphere using a 3D binary array, applies trimming to the sphere,
    and shifts its position within a given coordinate frame. The output consists of the
    adjusted coordinates of the sphere within the specified 3D volume.

    :param r: Radius of the sphere
    :type r: int
    :param c: Center coordinates of the sphere in the 3D volume as (z, y, x)
    :type c: tuple
    :param shape: Shape of the 3D volume to contain the sphere
    :type shape: tuple

    :return: Coordinates of the sphere within the 3D volume along z, y, and x axes
    :rtype: tuple of numpy.ndarray
    """
    r_dim = round(r * 2)
    sphere_frame = np.zeros((r_dim, r_dim, r_dim), dtype=np.int8)
    trim = round(r_dim / 6)
    c_frame = round(r)

    z, y, x = sphere_frame.shape
    for z_dim in range(z):
        for y_dim in range(y):
            for x_dim in range(x):
                dist_zyx_to_c = sqrt(
                    pow(abs(z_dim - c_frame), 2)
                    + pow(abs(y_dim - c_frame), 2)
                    + pow(abs(x_dim - c_frame), 2)
                )
                if dist_zyx_to_c > c_frame:
                    sphere_frame[z_dim, y_dim, x_dim] = False
                else:
                    sphere_frame[z_dim, y_dim, x_dim] = True
    sphere_frame[:trim, :] = False  # Trim bottom of the sphere
    sphere_frame[-trim:, :] = False  # Trim top of the sphere
    z, y, x = np.where(sphere_frame)

    c = ((c[0] - c_frame), (c[1] - c_frame), (c[2] - c_frame))
    z, y, x = z + c[0], y + c[1], x + c[2]

    # Remove pixel out of frame
    zyx = np.array((z, y, x)).T
    del_id = []
    for id_, i in enumerate(zyx):
        if i[0] >= shape[0] or i[1] >= shape[1] or i[2] >= shape[2]:
            del_id.append(id_)
        if i[0] < 0:  # Remove negative value
            del_id.append(id_)

    zyx = np.delete(zyx, del_id, 0)

    return zyx[:, 0], zyx[:, 1], zyx[:, 2]

def interpolation(points: np.ndarray) -> np.ndarray:
    """
    Performs 3D interpolation for XYZ coordinates using a given interpolation
    generator over sequential point pairs, and appends the last point to the
    output in integer format. The function creates a continuous series of
    interpolated points connecting all input coordinates.

    :param points: An array of shape (N, 3) representing a sequence of 3D points.

    :return: An array of shape (M, 3) containing the interpolated 3D coordinates,
             where M >= N due to newly generated intermediate points.
    """
    new_coord = []
    for i in range(0, len(points) - 1):
        """3D interpolation for XYZ dimension"""
        new_coord.append(list(interpolate_generator(points[i : i + 2, :])))

    # Append last point
    new_coord.append(list(np.round(points[-1, :]).astype(np.int32)))

    return np.vstack(new_coord)

def interpolate_generator(points: np.ndarray) -> Iterable:
    """
    Generates interpolated points between two given 2D or 3D points. The function
    determines the interpolation path in a pixel-grid-like manner between the start
    and end points, based on Bresenham-like algorithms. Interpolation supports
    only 2D and 3D coordinate systems, and exactly two points must be provided for
    interpolation.

    :param points: An array of shape (2, 2) for 2D points or (2, 3) for 3D points.
                   Defines the two points between which interpolation is performed.
    :return: An iterable generator that yields tuples representing (x, y) in 2D
             space and (x, y, z) in 3D space.
    """
    if points.shape not in [(2, 3), (2, 2)]:
        raise ValueError(
            "Interpolation supports only 2D/3D for 2 points at a time; "
            f"but {points.shape} was given!"
        )

    points = np.round(points).astype(np.int32)
    if points.shape == (2, 2):
        dim_ = 2
    else:
        dim_ = 3

    # Collect first and last point in array for XYZ
    x0, x1 = points[0, 0], points[1, 0]
    y0, y1 = points[0, 1], points[1, 1]
    if dim_ == 2:
        z0, z1 = 0, 0
    else:
        z0, z1 = points[0, 2], points[1, 2]

    # Delta between first and last point to interpolate
    delta_x, delta_y, delta_z = x1 - x0, y1 - y0, z1 - z0

    # Calculate axis to iterate throw
    max_delta = np.where(
        (abs(delta_x), abs(delta_y), abs(delta_z))
        == np.max((abs(delta_x), abs(delta_y), abs(delta_z)))
    )[0][0]
    if delta_x == 0 and delta_y == 0 and delta_z == 0:
        max_delta = 3

    # Calculate scaling direction + or - or None
    dx_sign, dy_sign, dz_sign = np.sign(delta_x), np.sign(delta_y), np.sign(delta_z)

    # Calculating scaling threshold
    delta_err_x = (
        0.0
        if delta_x == 0
        else (
            abs(delta_x / delta_y)
            if delta_y != 0
            else abs(delta_x / delta_z) if delta_z != 0 else 0.0
        )
    )
    delta_err_y = (
        0.0
        if delta_y == 0
        else (
            abs(delta_y / delta_x)
            if delta_x != 0
            else abs(delta_y / delta_z) if delta_z != 0 else 0.0
        )
    )

    if dim_ != 2:
        delta_err_z = (
            0.0
            if delta_z == 0
            else (
                np.minimum(abs(delta_z / delta_x), abs(delta_z / delta_y))
                if delta_x != 0 and delta_y != 0
                else (
                    abs(delta_z / delta_y)
                    if delta_x == 0 and delta_y != 0
                    else abs(delta_z / delta_x) if delta_x != 0 else 0.0
                )
            )
        )

    # Zero out threshold
    error_x, error_y, error_z = 0, 0, 0
    x, y, z = x0, y0, z0

    if max_delta == 0:  # Scale XYZ by iterating throw X axis
        for x in range(x0, x1, dx_sign):
            if dim_ != 2:
                yield x, y, z
            else:
                yield x, y

            # Iteratively add and scale Y axis
            error_y = error_y + delta_err_y
            while error_y >= 0.5:
                y += dy_sign
                error_y -= 1

            if dim_ != 2:
                # Iteratively add and scale Z axis
                error_z = error_z + delta_err_z
                while error_z >= 0.5:
                    z += dz_sign
                    error_z -= 1
    if max_delta == 1:  # Scale XYZ by iterating throw Y axis
        for y in range(y0, y1, dy_sign):
            if dim_ != 2:
                yield x, y, z
            else:
                yield x, y

            # Iteratively add and scale X axis
            error_x = error_x + delta_err_x
            while error_x >= 0.5:
                x += dx_sign
                error_x -= 1

            if dim_ != 2:
                # Iteratively add and scale Z axis
                error_z = error_z + delta_err_z
                while error_z >= 0.5:
                    z += dz_sign
                    error_z -= 1
    if max_delta == 2:  # Scale XYZ by iterating throw Z axis
        for z in range(z0, z1, dz_sign):
            if dim_ != 2:
                yield x, y, z
            else:
                yield x, y
    if max_delta == 3:  # Nothing to do
        if dim_ != 2:
            yield x, y, z
        else:
            yield x, y