import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


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


def preprocessing(image):
    """
    TODO: realizar preprocesado para la imagen de entrada, incluyendo:
        - meter aquí dentro los bucles para sacar los centros posibles
        - suavizar aquí la imagen con gauss
        - eliminar los destellos de luz que me vayan a joder los círculos después
        - encontrar una forma de mejorar la selección de min_rad y max_rad
            (eso no va necesariamente aquí pero para tenerlo en cuenta)

    gl!
    """
    return image


img = cv.imread("test_images/1/left/aeval3.bmp")
gauss_img = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_RGB2GRAY), (5, 5), 1.5)

possible_centers = []
for i in range(gauss_img.shape[0]):
    for j in range(gauss_img.shape[1]):
        if (gauss_img[i, j] < 0.1 * 255
                and gauss_img[i, j] == np.min(gauss_img[i - 1:i + 2, j - 1:j + 2])):
            possible_centers.append((j, i))

iris_center_out, iris_rad_out = find_iris(gauss_img, possible_centers, min_rad=30, max_rad=80, step=1)
iris_center_in, iris_rad_in = find_iris(gauss_img, possible_centers, min_rad=1, max_rad=30, step=1)
out = img.copy()
cv.circle(out, iris_center_out, iris_rad_out, (0, 255, 0), 1)
cv.circle(out, iris_center_in, iris_rad_in, (0, 255, 0), 1)
plt.imshow(out, cmap="gray")
plt.show()
