import numpy as np
import cv2 as cv


def daugman(img, center, start_r, end_r, step=1):
    intensities = []
    mask = np.zeros_like(img)

    radii = list(range(start_r, end_r, step))
    for r in radii:
        cv.circle(mask, center, r, (255, 255, 255), 1)
        diff = img & mask
        intensities.append(np.add.reduce(diff[diff > 0]) / (2 * np.pi * r))
        mask.fill(0)

    intensities_np = np.array(intensities, dtype=np.float32)
    del intensities

    intensities_np = intensities_np[:-1] - intensities_np[1:]
    intensities_np = abs(cv.GaussianBlur(intensities_np, (1, 5), 0))
    idx = np.argmax(intensities_np)

    return intensities_np[idx], radii[idx]


def find_circle(image, centers, min_rad, max_rad, step=1):
    intensity_values = []
    coords = []

    for point in centers:
        val, r = daugman(image, point, min_rad, max_rad, step)
        intensity_values.append(val)
        coords.append((point, r))

    best_idx = intensity_values.index(max(intensity_values))
    return coords[best_idx]
