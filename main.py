import skimage

import histograms
import morphological

in_image = skimage.data.page()
jota = skimage.io.imread("img/jota.png")

histograms.plot_output(in_image)
morphological.plot_output(jota)
