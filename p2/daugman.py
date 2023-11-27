import cv2
import numpy as np
import itertools
import math


def daugman(gray_img, center, start_r, end_r, step):
    intensities = []
    mask = np.zeros_like(gray_img)

    radii = list(range(start_r, end_r, step))
    for r in radii:
        cv2.circle(mask, center, r, 255, 1)
        diff = gray_img & mask
        intensities.append(np.add.reduce(diff[diff > 0]) / (2 * math.pi * r))
        mask.fill(0)

    intensities_np = np.array(intensities, dtype=np.float32)
    del intensities

    intensities_np = intensities_np[:-1] - intensities_np[1:]
    intensities_np = abs(cv2.GaussianBlur(intensities_np, (1, 5), 0))
    idx = np.argmax(intensities_np)

    return intensities_np[idx], radii[idx]


def find_iris(gray, *, daugman_start, daugman_end, daugman_step, points_step):
    h, w = gray.shape
    if h != w:
        print('Your image is not a square!')

    single_axis_range = range(int(h / 3), h - int(h / 3), points_step)
    all_points = itertools.product(single_axis_range, single_axis_range)

    intensity_values = []
    coords = []

    for point in all_points:
        val, r = daugman(gray, point, daugman_start, daugman_end, daugman_step)
        intensity_values.append(val)
        coords.append((point, r))

    best_idx = intensity_values.index(max(intensity_values))
    return coords[best_idx]