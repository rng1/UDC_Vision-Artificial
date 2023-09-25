import skimage
import histograms
import filters

in_image = skimage.data.moon()[::10, ::10]

histograms.get_output_and_plot_histograms(in_image)
filters.get_output_and_show_filtered_images(in_image)
