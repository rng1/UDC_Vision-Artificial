import numpy as np


def equalize_intensity(in_image, bins=256):
    cdf = 0
    out_image = np.copy(in_image)
    d = {}

    for i in np.unique(in_image.ravel()):
        occurrence = np.count_nonzero(np.sort(in_image.ravel()) == i)
        cdf += occurrence
        h = round((cdf - 1) / (in_image.size - 1) * (bins - 1))
        d[i] = h

    for k, v in d.items():
        out_image[in_image == k] = v

    return out_image
