import skimage
import histograms
import morphological
import filters

jota = skimage.io.imread("img/jota.png")

histograms.plot_output(skimage.data.page())
filters.plot_output(skimage.data.coins())
morphological.plot_output(jota)