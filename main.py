import skimage

import histograms

in_image = skimage.data.page()

histograms.get_output_and_plot_histograms(in_image)
