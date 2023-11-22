import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from helper import norm, plot_group, plot_image
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
            gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        case "central_diff":
            gx = np.array([[-1, 0, 1]])
            gy = gx.T
        case _:
            raise ValueError(f"`operator` must be a valid option, got \"{operator}\"")

    return filters.filter_image(in_image, gx), filters.filter_image(in_image, gy)


def LoG(in_image, sigma):
    gauss_image = filters.gaussian_filter(in_image, sigma)
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    return filters.filter_image(gauss_image, laplace_kernel)


def find_adjacent(theta):
    angle_ranges = {
        (45, 135): ((-1, 0), (1, 0)),
        (135, 225): ((0, -1), (0, 1)),
        (225, 315): ((-1, 0), (1, 0))
    }

    for angle_range, (offset_1, offset_2) in angle_ranges.items():
        if angle_range[0] <= theta < angle_range[1]:
            return offset_1, offset_2

    return (0, -1), (0, 1)


def non_max_suppression(magnitude, orientation):
    suppressed = np.zeros_like(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            theta = orientation[i, j]

            offset_1, offset_2 = find_adjacent(theta)

            adj_1 = magnitude[i + offset_1[0], j + offset_1[1]]
            adj_2 = magnitude[i + offset_2[0], j + offset_2[1]]

            mask = (magnitude[i, j] >= adj_1) & (magnitude[i, j] >= adj_2)
            suppressed[i, j] = magnitude[i, j] * mask

    return suppressed


def hysteresis_thresholding(in_image, tlow, thigh):
    tlow = np.clip(tlow, a_min=None, a_max=thigh)
    mask_tlow = in_image > tlow
    mask_thigh = in_image > thigh

    labels_tlow, num_labels = ndimage.label(mask_tlow)

    sums = ndimage.sum_labels(mask_thigh, labels_tlow, np.arange(num_labels + 1))

    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_tlow]

    return thresholded


def edge_canny(in_image, sigma, tlow, thigh, steps):
    gauss_image = filters.gaussian_filter(in_image, sigma)
    gx, gy = gradient_image(gauss_image, "sobel")
    magnitude = np.hypot(gx, gy)
    orientation = np.degrees(np.arctan2(gy, gx)) + 180
    suppressed = non_max_suppression(magnitude, orientation)
    out_image = hysteresis_thresholding(suppressed, tlow, thigh)

    if steps:
        step_images = [gauss_image, magnitude, orientation, suppressed, out_image]
        step_titles = [f"Gauss (σ={str(sigma)}", "Magnitud", "Orientación,", "Supresión no máxima", "Histéresis"]
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8, 5), sharex=True, sharey=True)
        plot_group(axes, step_images, step_titles)
    else:
        return out_image


def plot_output(in_image, mode="all", operator="sobel", sigma_LoG=2, sigma_canny=1.5, tlow=0.3, thigh=0.5):
    in_image = norm(in_image)

    out_image_gradient = gradient_image(in_image, operator)
    out_image_LoG = LoG(in_image, sigma_LoG)

    out_image_canny = edge_canny(in_image, sigma_canny, tlow, thigh, steps=(mode == "canny_steps"))

    titles = [f"Gx ({operator.capitalize()})", f"Gy ({operator.capitalize()})", f"LoG (σ={str(sigma_LoG)})",
              f"Canny (σ={str(sigma_canny)}, tlow={tlow}, thigh={thigh})"]
    images = [out_image_gradient[0], out_image_gradient[1], out_image_LoG, out_image_canny]

    match mode:
        case "all":
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
            plot_group(axes, images, titles)
        case "split":
            for image, title in zip(images, titles):
                plot_image(image, title)
        case "all_gradients":
            gradient_images = [gradient_image(in_image, "roberts")[0], gradient_image(in_image, "roberts")[1],
                               gradient_image(in_image, "sobel")[0], gradient_image(in_image, "sobel")[1],
                               gradient_image(in_image, "prewitt")[0], gradient_image(in_image, "prewitt")[1],
                               gradient_image(in_image, "central_diff")[0], gradient_image(in_image, "central_diff")[1]]
            gradient_titles = ["Gx (Roberts)", "Gy (Roberts)",
                               "Gx (Sobel)", "Gx (Sobel)",
                               "Gy (Prewitt)", "Gy (Prewitt)",
                               "Gy (Central diff)", "Gy (Central diff)"]
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(4, 8), sharex=True, sharey=True)
            plot_group(axes, gradient_images, gradient_titles)
        case "gradient":
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
            plot_group(axes, out_image_gradient, [titles[0], titles[1]])
        case "log":
            plot_image(images[2], titles[2])
        case "canny":
            plot_image(images[3], titles[3])
