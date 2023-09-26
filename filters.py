import numpy as np
import matplotlib.pyplot as plt


def gauss_kernel_1d(sigma):
    """Calcula un kernel Gaussiano unidimensional con `sigma` dado

    :param sigma: int
        Parámetro σ de entrada.

    :return kernel
        El vector 1xN con el kernel de salida, teniendo en cuenta que:
            - El centro x=0 de la Gaussiana está en la posición ⌊N/2⌋ + 1.
            - N se calcula a partir de sigma como N = 2⌈3σ⌉ + 1
    """
    if sigma < 1:
        raise ValueError(f"`sigma` must be greater than 0, got {sigma}")

    n = 2 * round(3 * sigma) + 1
    radius = np.floor(n / 2)
    x = np.arange(-radius, radius + 1)
    kernel = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))

    return kernel


def gaussian_filter(in_image, sigma):
    """Realiza un suavizado Gaussiano bidimensional usando un filtro NxN de parámetro `sigma`,
    donde N se calcula a partir de sigma como N = 2⌈3σ⌉ + 1.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param sigma: int
        Parámetro σ de entrada.

    :return: out_image
        Matriz MxN con la imagen de salida después de la aplicación del filtro.
    """
    in_image = np.asarray(in_image)
    in_image = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))
    row, col = in_image.shape

    kernel = np.asarray(gauss_kernel_1d(sigma))
    print(kernel)

    it1_image = np.zeros((row - 1, col - 1))
    out_image = np.zeros((row - 1, col - 1))

    for i in range(row - 1):
        for j in range(col - 1):
            window = in_image[i, j:j + len(kernel)]
            it1_image[i, j] = window.sum() / kernel.sum()

    it1_image = it1_image.transpose()

    for i in range(row - 1):
        for j in range(col - 1):
            window = it1_image[i, j:j + len(kernel)]
            out_image[i, j] = window.sum() / kernel.sum()

    return out_image.transpose()


def median_filter(in_image, filter_size):
    """Implementa el filtro de medianas bidimensional.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param filter_size: int
        Valor entero N indicando que el tamaño de la ventana es de NxN. La posición central de la ventana es
        (⌊N/2⌋ + 1, ⌊N/2⌋ + 1).

    :return: out_image
        Matriz MxN con la imagen de salida después de la aplicación del filtro.
    """
    if filter_size < 1:
        raise ValueError(f"`filter_size` must be greater than 0, got {filter_size}")

    in_image = np.asarray(in_image)
    in_image = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))

    row, col = in_image.shape
    out_image = np.zeros((row - 1, col - 1))

    for i in range(row - 1):
        for j in range(col - 1):
            window = in_image[i:i + filter_size, j:j + filter_size]
            out_image[i, j] = np.median(window)

    return out_image


def get_output_and_show_filtered_images(in_image):
    out_image_gaussian = gaussian_filter(in_image, 1)
    out_image_median = median_filter(in_image, filter_size=9)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ['Original', 'Median filter', "Gaussian filter"]
    imgs = [in_image, out_image_median, out_image_gaussian]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')

    plt.tight_layout()
    plt.show()
