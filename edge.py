import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import filters


def gradient_image(in_image, operator):
    match operator.lower():
        case "roberts":
            gx = np.array([[-1, 0], [0, 1]])
            gy = np.array([[0, -1], [1, 0]])
        case "prewitt":
            gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        case "sobel":
            gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        case "central_diff":
            # TODO: comprobar si esto va bien porque lo dudo sinceramente
            gx = np.array([[-1, 0, 1]])
            gy = gx.T
        case _:
            raise ValueError(f"`operator` must be a valid option, got `{operator}`")

    return filters.filter_image(in_image, gx), filters.filter_image(in_image, gy)


def LoG(in_image, sigma):
    gauss_image = filters.gaussian_filter(in_image, sigma)
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    return filters.filter_image(gauss_image, laplace_kernel)


def non_max_suppression(magnitude, theta):
    theta[theta < 0] += np.pi
    suppressed = np.zeros_like(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            t = theta[i, j]

            if 0 <= t < np.pi / 8 or 7 * np.pi / 8 <= t <= np.pi:
                adj_1 = magnitude[i, j - 1]
                adj_2 = magnitude[i, j + 1]
            elif np.pi / 8 <= t < 3 * np.pi / 8:
                adj_1 = magnitude[i - 1, j + 1]
                adj_2 = magnitude[i + 1, j - 1]
            elif 3 * np.pi / 8 <= t < 5 * np.pi / 8:
                adj_1 = magnitude[i - 1, j]
                adj_2 = magnitude[i + 1, j]
            elif 5 * np.pi / 8 <= t < 7 * np.pi / 8:
                adj_1 = magnitude[i + 1, j + 1]
                adj_2 = magnitude[i - 1, j - 1]
            else:
                continue

            mask = (magnitude[i, j] >= adj_1) & (magnitude[i, j] >= adj_2)
            suppressed[i, j] = magnitude[i, j] * mask

    return suppressed


def hysteresis_thresholding(in_image, tlow, thigh):
    tlow = np.clip(tlow, a_min=None, a_max=thigh)
    mask_low = in_image > tlow
    mask_high = in_image > thigh

    labels_low, num_labels = ndimage.label(mask_low)

    sums = ndimage.sum_labels(mask_high, labels_low, np.arange(num_labels + 1))

    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]

    return thresholded


def edge_canny(in_image, sigma, tlow, thigh):
    gauss_image = filters.gaussian_filter(in_image, sigma)
    gx, gy = gradient_image(gauss_image, "sobel")
    magnitude = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    suppressed = non_max_suppression(magnitude, theta)
    out_image = hysteresis_thresholding(suppressed, tlow, thigh)

    return out_image


def canny(in_image):
    plt.imshow(edge_canny(in_image, 2, 0.1, 0.2), cmap=plt.cm.gray)
    plt.show()


def plot_output(in_image):
    # Parámetros
    operator = "sobel"
    sigma_LoG = 3

    out_image_gradient = gradient_image(in_image, operator)
    out_image_LoG = LoG(in_image, sigma_LoG)
    out_image_canny = edge_canny(in_image, 1.5, 0.35, 0.5)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = [f"Gx ({operator.capitalize()})", f"Gy ({operator.capitalize()})", f"LoG (σ={str(sigma_LoG)})", "Canny"]
    imgs = [out_image_gradient[0], out_image_gradient[1], out_image_LoG, out_image_canny]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')

    plt.tight_layout()
    plt.show()

    plot_all_operators(in_image)


# TODO: borrar esto antes de entregar
def plot_all_operators(in_image):
    roberts = gradient_image(in_image, "roberts")
    prewitt = gradient_image(in_image, "prewitt")
    sobel = gradient_image(in_image, "sobel")
    central_diff = gradient_image(in_image, "central_diff")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ["Gx Roberts", "Gx Prewitt", "Gx Sobel", "Gx CentralDiff",
              "Gy Roberts", "Gy Prewitt", "Gy Sobel", "Gy CentralDiff"]
    imgs = [roberts[0], prewitt[0], sobel[0], central_diff[0],
            roberts[1], prewitt[1], sobel[1], central_diff[1]]

    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')

    plt.tight_layout()
    plt.show()
