import skimage
import histograms
import morphological
import filters
import edge

#histograms.plot_output(skimage.data.page())
#filters.plot_output(skimage.data.coins())
#morphological.plot_output(skimage.io.imread("img/jota.png"))
edge.plot_output(skimage.io.imread("img/lena.jpg"))
