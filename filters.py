import numpy as np
import matplotlib.pyplot as plt

MEDIAN_FILTER_SIZE = 3


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

    indexer = filter_size // 2
    out_image = np.zeros_like(in_image)
    for i in range(len(in_image)):
        for j in range(len(in_image[0])):
            values = []

            for z in range(filter_size):
                for k in range(filter_size):
                    row_idx = i + z - indexer
                    col_idx = j + k - indexer

                    if 0 <= row_idx < len(in_image) and 0 <= col_idx < len(in_image[0]):
                        values.append(in_image[row_idx][col_idx])

            values.sort()
            out_image[i][j] = values[len(values) // 2]
    return out_image


def get_output_and_show_filtered_images(in_image):
    out_image_median = median_filter(in_image, MEDIAN_FILTER_SIZE)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    titles = ['Original', 'Median filter']
    imgs = [in_image, out_image_median]
    for n in range(0, len(imgs)):
        ax[n].imshow(imgs[n], cmap=plt.cm.gray)
        ax[n].set_title(titles[n])
        ax[n].axis('off')

    plt.tight_layout()
    plt.show()
