import skimage
import histograms
import filters

in_image = skimage.data.coins()

filters.get_output_and_show_filtered_images(in_image)
