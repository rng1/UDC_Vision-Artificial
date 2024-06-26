import numpy as np
import matplotlib.pyplot as plt
import skimage

from helper import plot_img_and_hist, norm


def adjust_intensity(in_image, in_range=None, out_range=None):
    """Implementa un algoritmo de alteración del rango dinámico de la imagen.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param in_range: vector, opcional
        Vector 1x2 con el rango de niveles de intensidad [imin, imax] de entrada. Si el vector está vacío (por defecto),
        el mínimo y el máximo de la imagen de entrada se usan como imin e imax.

    :param out_range: vector, opcional
        Vector 1x2 con el rango de niveles de intensidad [omin, omax] de salida. El valor por defecto es [0, 1]

    :return: out_image
        Matriz MxN con la imagen de salida después de la alteración de su rango dinámico.
    """
    if out_range is None:
        out_range = [0, 1]
    elif not isinstance(out_range, list) or len(out_range) != 2:
        raise ValueError(f"`out_range` debe ser un vector 1x2, got {out_range}.")

    if in_range is None:
        in_range = [np.min(in_image), np.max(in_image)]
    elif not isinstance(in_range, list) or len(in_range) != 2:
        raise ValueError(f"`in_range` debe ser un vector 1x2, got {in_range}.")

    """
    Se normaliza la imagen en base al valor mínimo de entrada.
    Se calcula la escala de la transformación.
    Se multiplica la imagen normalizada por la escala para ajustar el rango de intensidades de la imagen al
        rango introducido.
    Se suma el mínimo rango de salida a la imagen para desplazarla.
    """
    out_image = out_range[0] + (
                ((out_range[1] - out_range[0]) * (in_image - in_range[0])) / (in_range[1] - in_range[0]))

    return out_image


def equalize_histogram(in_image, n_bins=256):
    """Implementa un algoritmo de ecualización de histograma.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param n_bins: int, opcional
        Número de bins utilizados en el procesamiento. Se asume que el intervalo de entrada [0 1] se divide en n_bins
        intervalos iguales para hacer el procesamiento, y que la imagen de salida vuelve a quedar en el intervalo [0 1].

    :return: out_image
        Matriz MxN con la imagen de salida después de la ecualización de su histograma.
    """
    cdf = 0
    out_image = np.copy(in_image)
    d = {}

    """
    Se recorren los valores únicos de intensidad y se cuentan las ocurrencias, actualizando la función
    de distribución acumulativa. Mediante la ecuación de ecualización de histograma se calcula un nuevo valor
    de intensidad para ese punto, que se almacena en el diccionario para mapearlo después en la imagen.
    """
    for i in np.unique(in_image.ravel()):
        occurrence = np.count_nonzero(np.sort(in_image.ravel()) == i)
        cdf += occurrence
        h = round((cdf - 1) / (in_image.size - 1) * (n_bins - 1))
        d[i] = h

    for k, v in d.items():
        out_image[in_image == k] = v

    norm_out_image = (out_image - np.min(out_image)) / (np.max(out_image) - np.min(out_image))

    return norm_out_image


def plot_output(in_image):
    in_image = norm(in_image)
    out_image_intensity = adjust_intensity(in_image, out_range=[0.3, 0.7])
    out_image_equalized = equalize_histogram(in_image,n_bins=10)

    # Mostrar resultados
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=object)

    axes[0, 0] = fig.add_subplot(2, 3, 1)
    for i in range(1, 3):
        axes[0, i] = fig.add_subplot(2, 3, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 3):
        axes[1, i] = fig.add_subplot(2, 3, 4 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(in_image, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))
    ax_cdf.set_yticks([])

    ax_img, _, ax_cdf = plot_img_and_hist(out_image_intensity, axes[:, 1])
    ax_img.set_title('Intensity adjustment')
    ax_cdf.set_yticks([])

    ax_img, _, ax_cdf = plot_img_and_hist(out_image_equalized, axes[:, 2])
    ax_img.set_title('Histogram equalization')
    ax_cdf.set_yticks([])

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()
