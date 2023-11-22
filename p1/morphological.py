import numpy as np
import matplotlib.pyplot as plt
from helper import norm, plot_group


def window_calc(i, j, center, se_shape, img_shape, in_image):
    """Función auxiliar.
    Calcula la ventana de la imagen de entrada en la que se realizará la operación oportuna.

    :param i: int
        Columna correspondiente a la iteración.

    :param j: int
        Fila correspondiente a la iteración del bucle.

    :param center: vector
        Vector 1x2 con las coordenadas del centro de SE.

    :param se_shape: vector
        Vector 1x2 con el número de filas y columnas de SE.

    :param img_shape: vector
        Vector 1x2 con el número de filas y columnas de la imagen de entrada.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :return:
        Matriz PxQ con la ventana reducida de la imagen.
    """
    i_min = max(i - center[0], 0)
    i_max = min(i + se_shape[0] - center[0], img_shape[0])
    j_min = max(j - center[1], 0)
    j_max = min(j + se_shape[1] - center[1], img_shape[1])
    return in_image[i_min:i_max, j_min:j_max]


def noise(in_image, color=""):
    """Función auxiliar.
    Introduce ruido aleatorio blanco o negro para comprobar el funcionamiento de los operadores de apertura y cierre.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param color: {'black', 'white'}
        black:
            Color del ruido, correspondiente a la operación de cierre.
        white:
            Color del ruido, correspondiente a la operación de apertura.

    :return: out_image
        Matriz MxN con la imagen de salida después de la aplicación del ruido.
    """
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


def dilate_erode(in_image, se, operation="", center=None):
    """Implementa los operadores morfológicos de erosión y dilatación, según un parámetro.
    La apertura es una dilatación de una erosión, y el cierre viceversa.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param se: array
        Matriz PxQ de ceros y unos definiendo el elemento estructurante.

    :param operation: {'dilate', 'erode'}
        dilate:
            Calcula el máximo de la ventana que recorre la matriz de entrada.
        erode:
            Calcula el mínimo de la ventana que recorre la matriz de entrada.

    :param center: vector, opcional
        Vector 1x2 con las coordenadas del centro de SE. Se asume que [0 0] es la esquina superior izquierda.
        Por defecto, el centro se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).

    :return: out_image
        Matriz MxN con la imagen de salida después de la aplicación del operador morfológico.
    """
    in_image = np.asarray(in_image)
    se = np.asarray(se)

    img_shape = in_image.shape
    se_shape = se.shape

    if center is None:
        center = [int(np.floor(se_shape[0] / 2)), int(np.floor(se_shape[1] / 2))]

    out_image = np.zeros_like(in_image)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            window = window_calc(i, j, center, se_shape, img_shape, in_image)

            if operation == "dilate":
                result = np.max(window)
            elif operation == "erode":
                result = np.min(window)
            else:
                raise ValueError(f"Operation must be 'erode' or 'dilate', got {operation}")

            out_image[i, j] = result

    return out_image


def hit_or_miss(in_image, obj_se, bg_se, center=None):
    """Implementa la transformada hit-or-miss de una imagen, dados dos elementos estructurantes de objeto y fondo.

    :param in_image: array
        Matriz MxN con la imagen de entrada.

    :param obj_se: array
        Matriz PxQ de ceros y unos definiendo el elemento estructurante del objeto.

    :param bg_se: array
        Matriz PxQ de ceros y unos definiendo el elemento estructurante del fondo.

    :param center: vector, opcional
        Vector 1x2 con las coordenadas del centro de SE. Se asume que [0 0] es la esquina superior izquierda.
        Por defecto, el centro se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).

    :return: out_image
        Matriz MxN con la imagen de salida después de la aplicación de la transformada.
    """
    obj_se = np.asarray(obj_se)
    bg_se = np.asarray(bg_se)
    in_image = np.asarray(in_image)
    in_image = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))

    if obj_se.shape != bg_se.shape:
        raise ValueError("Error: elementos estructurantes incoherentes")

    kernel = np.zeros_like(obj_se)
    kernel[obj_se == 1] = 1
    kernel[bg_se == 1] = -1

    img_shape = in_image.shape
    k_shape = kernel.shape

    if center is None:
        center = [int(np.floor(k_shape[0] / 2)), int(np.floor(k_shape[1] / 2))]

    out_image = np.zeros_like(in_image)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            window = window_calc(i, j, center, k_shape, img_shape, in_image)

            if window.shape == kernel.shape:
                cond = np.logical_or(kernel == 0, (kernel == 1) & (window == 1) | (kernel == -1) & (window == 0))
                if np.all(cond):
                    out_image[i, j] = 1

    return out_image


def plot_closing(in_image, se):
    noise_im = noise(in_image, color="black")
    closing_im = dilate_erode(dilate_erode(in_image, se, operation="dilate"), se, operation="erode")

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
    titles = ["Noise (black)", "Closing"]
    images = [noise_im, closing_im]
    plot_group(axes, images, titles)


def plot_opening(in_image, se):
    noise_im = noise(in_image, color="white")
    opening_im = dilate_erode(dilate_erode(in_image, se, operation="erode"), se, operation="dilate")

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
    titles = ["Noise (white)", "Opening"]
    images = [noise_im, opening_im]
    plot_group(axes, images, titles)


def plot_dilate_erode(in_image, se):
    dilate_im = dilate_erode(in_image, se, operation="dilate")
    erode_im = dilate_erode(in_image, se, operation="erode")

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 5), sharex=True, sharey=True)
    titles = ["Original", "Dilation", "Erosion"]
    images = [in_image, dilate_im, erode_im]
    plot_group(axes, images, titles)


def plot_hit_or_miss(in_image):
    obj_se = np.array((
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]), dtype="int")
    bg_se = np.array((
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]), dtype="int")
    hit_or_miss_im = hit_or_miss(in_image, obj_se, bg_se)

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                             sharex=True, sharey=True)
    titles = ["Original", "Hit-or-miss"]
    images = [in_image, hit_or_miss_im]
    plot_group(axes, images, titles)


def plot_output(in_image, mode="all"):
    in_image = norm(in_image)

    hit_or_miss_image = np.array((
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0]))
    se = np.zeros((3, 3), dtype=int)

    if mode == "dil_er" or mode == "all":
        plot_dilate_erode(in_image, se)
    if mode == "op" or mode == "all":
        plot_opening(in_image, se)
    if mode == "cl" or mode == "all":
        plot_closing(in_image, se)
    if mode == "h_m" or mode == "all":
        plot_hit_or_miss(hit_or_miss_image)
