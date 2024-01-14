import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


def get_possible_centers(image):
    possible_centers = []

    i_min, i_max = image.shape[0] // 3, 2 * image.shape[0] // 3
    j_min, j_max = image.shape[1] // 3, 2 * image.shape[1] // 3

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if image[i, j] < 0.15 * 255:
                possible_centers.append((j, i))

    return possible_centers


def get_segment_circles(image, iris, pupil):
    out = image.copy()
    cv.circle(out, iris[0], iris[1], (255, 0, 255), 1)
    cv.circle(out, pupil[0], pupil[1], (255, 0, 255), 1)

    return out


def get_segment_mask(image, iris, pupil):
    mask_iris = np.zeros_like(image)
    mask_pupil = np.zeros_like(image)

    cv.circle(mask_iris, iris[0], iris[1], (255, 255, 255), thickness=cv.FILLED)
    cv.circle(mask_pupil, pupil[0], pupil[1], (255, 255, 255), thickness=cv.FILLED)
    mask_iris = np.logical_xor(mask_iris, mask_pupil)

    return mask_iris.astype(bool), mask_pupil.astype(bool)


def plot_all(images, titles):
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def cleaner(image, pct):
    histogram = cv.calcHist([image], [0], None, [256], [1, 256])
    max_value = np.argmax(histogram)
    values = list(range(math.floor(max_value - max_value*pct), math.floor(max_value + max_value*pct)))
    mask = np.isin(image, values)
    return np.where(mask, image, 0)


def center_in_bounding_box(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[0])
    return image[y:y + h, x:x + w]
