import numpy as np
import matplotlib.pyplot as plt


def dilate_erode(in_image, se, center=None, operation=""):
    se = np.asarray(se)
    (se_r, se_c) = se.shape

    in_image = np.asarray(in_image)
    (im_r, im_c) = in_image.shape

    if center is None:
        center = [int(np.floor(se_r / 2)), int(np.floor(se_c / 2))]

    out_image = np.zeros((im_r, im_c), dtype=int)

    for i in range(im_r):
        for j in range(im_c):
            i_min = max(i - center[0], 0)
            i_max = min(i + se_r - center[0], im_r)
            j_min = max(j - center[1], 0)
            j_max = min(j + se_c - center[1], im_c)

            window = in_image[i_min:i_max, j_min:j_max]

            if operation == "dilate":
                result = np.max(window)
            elif operation == "erode":
                result = np.min(window)
            else:
                raise ValueError(f"Operation must be 'erode' or 'dilate', got {operation}")

            out_image[i, j] = result

    return out_image


def hit_or_miss(in_image, obj_se, bg_se, center=None):
    obj_se = np.asarray(obj_se)
    bg_se = np.asarray(bg_se)
    in_image = np.asarray(in_image)

    (im_r, im_c) = in_image.shape
    (obj_se_r, obj_se_c) = obj_se.shape

    if obj_se.shape != bg_se.shape:
        raise ValueError("Error: elementos estructurantes incoherentes")

    out_image = np.zeros((im_r, im_c), dtype=int)

    for i in range(im_r):
        for j in range(im_c):
            i_min = max(i - center[0], 0)
            i_max = min(i + obj_se_r - center[0], im_r)
            j_min = max(j - center[1], 0)
            j_max = min(j + obj_se_c - center[1], im_c)

            window = in_image[i_min:i_max, j_min:j_max]

            obj_match = np.logical_and(obj_se == 1, window == 1)
            bg_match = np.logical_and(bg_se == 0, window == 0)

            if np.all(obj_match) and np.all(bg_match):
                out_image[i, j] = 1

    return out_image


def noise(in_image, color=""):
    if color == "white":
        value = 255
    elif color == "black":
        value = 0
    else:
        raise ValueError(f"Color must be 'black' or 'white', got {color}")

    out_image = in_image.copy()
    (r, c) = in_image.shape

    for i in range(r):
        for j in range(c):
            if np.random.rand() > 0.99:
                out_image[i, j] = value

    return out_image


def plot_closing(in_image, se):
    noise_im = noise(in_image, color="black")
    opening_im = dilate_erode(dilate_erode(in_image, se, operation="dilate"), se, operation="erode")

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ["Noise (black)", "Closing"]
    imgs = [noise_im, opening_im]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis("off")

    plt.tight_layout()
    plt.show()


def plot_opening(in_image, se):
    noise_im = noise(in_image, color="white")
    opening_im = dilate_erode(dilate_erode(in_image, se, operation="erode"), se, operation="dilate")

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ["Noise (white)", "Opening"]
    imgs = [noise_im, opening_im]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis("off")

    plt.tight_layout()
    plt.show()

def plot_dilate_erode(in_image, se):
    dilate_im = dilate_erode(in_image, se, operation="dilate")
    erode_im = dilate_erode(in_image, se, operation="erode")

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ["Original", "Dilation", "Erosion"]
    imgs = [in_image, dilate_im, erode_im]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis("off")

    plt.tight_layout()
    plt.show()


def plot_output(in_image):
    se = np.zeros((3, 3), dtype=int)

    plot_dilate_erode(in_image, se)
    plot_opening(in_image, se)
    plot_closing(in_image, se)
