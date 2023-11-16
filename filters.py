import matplotlib.pyplot as plt
import numpy as np
import skimage


def filter_image(in_image, kernel):
    """Aplica un filtro de convolución a una imagen de entrada utilizando el kernel dado.

    :param in_image: array
        Matriz MxN con la imagen de entrada.
    :param kernel: array
        Kernel de convolución.
    :return: out_image
        Matriz MxN con la imagen de salida después de la aplicación del filtro.
    """
    if in_image.ndim > 2:
        in_image = skimage.color.rgb2gray(in_image)

    in_image = np.asarray(in_image)
    out_image = np.zeros_like(in_image)

    image_row, image_col = in_image.shape
    kernel_row, kernel_col = kernel.shape
    pad_row = (kernel_row - 1) // 2
    pad_col = (kernel_col - 1) // 2

    padded_image = np.pad(in_image, ((pad_row, pad_row), (pad_col, pad_col)), mode='reflect')

    for i in range(image_row):
        for j in range(image_col):
            out_image[i, j] = np.sum(padded_image[i:(i + kernel_row), j:(j + kernel_col)] * kernel)

    return out_image


def gauss_kernel_1d(sigma):
    """Calcula un kernel Gaussiano unidimensional con `sigma` dado

    :param sigma: int
        Parámetro σ de entrada.
    :return kernel
        El vector 1xN con el kernel de salida, teniendo en cuenta que:
            - El centro x=0 de la Gaussiana está en la posición ⌊N/2⌋ + 1.
            - N se calcula a partir de sigma como N = 2⌈3σ⌉ + 1
    """
    if sigma < 0:
        raise ValueError(f"`sigma` must be greater than 0, got {sigma}")

    n = 2 * round(3 * sigma) + 1
    radius = np.floor(n / 2)
    x = np.arange(-radius, radius + 1)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))

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
    kernel1d = gauss_kernel_1d(sigma)[np.newaxis]
    return filter_image(filter_image(in_image, kernel1d), kernel1d.T)


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
    if in_image.ndim > 2:
        in_image = skimage.color.rgb2gray(in_image)

    in_image = np.asarray(in_image)
    out_image = np.zeros_like(in_image)

    row, col = in_image.shape
    pad = (filter_size - 1) // 2

    padded_image = np.pad(in_image, pad, mode='constant')

    for i in range(row):
        for j in range(col):
            window = padded_image[i:i + filter_size, j:j + filter_size]
            out_image[i, j] = np.median(window)

    return out_image


def plot_output(in_image):
    out_image_gaussian = gaussian_filter(in_image, 9)
    out_image_median = median_filter(in_image, filter_size=9)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ["Original", "Median filter", "Gaussian filter"]
    imgs = [in_image, out_image_median, out_image_gaussian]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')

    plt.tight_layout()
    plt.show()
