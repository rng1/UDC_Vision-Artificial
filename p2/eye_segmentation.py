from os import listdir

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def daugman(gray_img, center, start_r, end_r, step=1):
    intensities = []
    mask = np.zeros_like(gray_img)

    radii = list(range(start_r, end_r, step))
    for r in radii:
        cv.circle(mask, center, r, (255, 255, 255), 1)
        diff = gray_img & mask
        intensities.append(np.add.reduce(diff[diff > 0]) / (2 * math.pi * r))
        mask.fill(0)

    intensities_np = np.array(intensities, dtype=np.float32)
    del intensities

    intensities_np = intensities_np[:-1] - intensities_np[1:]
    intensities_np = abs(cv.GaussianBlur(intensities_np, (1, 5), 0))
    idx = np.argmax(intensities_np)

    return intensities_np[idx], radii[idx]


def find_iris(image, centers, min_rad, max_rad, step=1):
    intensity_values = []
    coords = []

    for point in centers:
        val, r = daugman(image, point, min_rad, max_rad, step)
        intensity_values.append(val)
        coords.append((point, r))

    best_idx = intensity_values.index(max(intensity_values))
    return coords[best_idx]


def calculate_centers(image, use_local_min=False):
    possible_centers = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j] < 0.1 * 255
                    and (not use_local_min
                         or image[i, j] == np.min(image[i - 1:i + 2, j - 1:j + 2]))):
                possible_centers.append((j, i))

    return possible_centers


def get_segment_circles(image, iris_out, iris_in):
    out = image.copy()
    cv.circle(out, iris_out[0], iris_out[1], (255, 0, 255), 1)
    cv.circle(out, iris_in[0], iris_in[1], (255, 0, 255), 1)

    return out


def get_segment_mask(image, seg_out, seg_in):
    mask_out = np.zeros_like(image)
    mask_in = np.zeros_like(image)

    cv.circle(mask_in, seg_in[0], seg_in[1], (255, 255, 255), thickness=cv.FILLED)
    cv.circle(mask_out, seg_out[0], seg_out[1], (255, 255, 255), thickness=cv.FILLED)
    cv.subtract(mask_out, mask_in, mask_out)

    return mask_out, mask_in


def plot_all(axes, images, titles):
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image, cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main(image, title, use_local_min=False):
    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
        else:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    blur = cv.GaussianBlur(gray, (3, 3), 1.5)
    morph_open = cv.morphologyEx(blur, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    centers = calculate_centers(morph_open, use_local_min)

    iris_out = find_iris(morph_open, centers, min_rad=30, max_rad=80, step=1)
    iris_in = find_iris(morph_open, centers, min_rad=1, max_rad=30, step=1)

    circled_image = get_segment_circles(image, iris_out, iris_in)
    mask_iris, mask_pupil = get_segment_mask(gray, iris_out, iris_in)

    highlighted_image = np.copy(image)
    highlighted_image[np.where(mask_pupil == 255)] = [0, 255, 0]
    highlighted_image[np.where(mask_iris == 255)] = [255, 0, 0]

    segmented_image = cv.addWeighted(image, 0.9, highlighted_image, 0.1, 0)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    plot_all(axes, [segmented_image, circled_image], [title, title+" circles"])


start_time = time.time()
folder = "test_images/1/"
for im in listdir(folder):
    main(cv.imread(folder + im), im.strip(".bmp"), use_local_min=False)
print("Elapsed time: %s s" % (time.time() - start_time))
