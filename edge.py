import numpy as np
import matplotlib.pyplot as plt
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
        case _:
            raise ValueError(f"`operator` must be a valid option, got `{operator}`")

    return filters.filter_image(in_image, gx), filters.filter_image(in_image, gy)


def LoG(in_image, sigma):
    gauss_image = filters.gaussian_filter(in_image, sigma)
    laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    return filters.filter_image(gauss_image, laplace_kernel)


def plot_output(in_image):
    # Parámetros
    operator = "roberts"
    sigma = 10

    out_image_gradient = gradient_image(in_image, operator)
    out_image_LoG = LoG(in_image, sigma)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ["Original", f"Gx ({operator.capitalize()})", f"Gy ({operator.capitalize()})", f"LoG (σ={str(sigma)})"]
    imgs = [in_image, out_image_gradient[0], out_image_gradient[1], out_image_LoG]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')

    plt.tight_layout()
    plt.show()
