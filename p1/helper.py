import matplotlib.pyplot as plt
import skimage
import numpy as np


def norm(image):
    if image.ndim > 2:
        if image.shape[2] == 4:
            image = skimage.color.rgba2rgb(image)
        image = skimage.color.rgb2gray(image)
    else:
        return image/255
    return image


def plot_group(axes, images, titles):
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_image(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_img_and_hist(image, axes, bins=256):
    image = skimage.util.img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Mostrar imagen
    ax_img.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax_img.set_axis_off()

    # Mostrar histograma
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Mostrar distribución
    img_cdf, bins = skimage.exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf
