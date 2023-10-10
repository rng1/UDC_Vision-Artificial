import skimage

import histograms
import morphological

in_image = skimage.data.page()
jota = skimage.io.imread("C:\\Users\\rnara\\Desktop\\jota.png") # TODO: cambiar la ruta

histograms.plot_output(in_image)
morphological.plot_output(jota)
