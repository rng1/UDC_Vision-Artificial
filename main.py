import numpy as np
import matplotlib.pyplot as plt
import skimage
import plotter
import histograms
import test_images

in_image = skimage.data.moon()

out_image = histograms.equalize_intensity(in_image)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 2), dtype=object)
axes[0, 0] = plt.subplot(2, 2, 1)
axes[0, 1] = plt.subplot(2, 2, 2, sharex=axes[0, 0], sharey=axes[0, 0])
axes[1, 0] = plt.subplot(2, 2, 3)
axes[1, 1] = plt.subplot(2, 2, 4)

ax_img, ax_hist, _ = plotter.plot_img_and_hist(in_image, axes[:, 0])
ax_img.set_title('Low contrast image')
ax_hist.set_ylabel('Number of pixels')

ax_img, ax_hist, ax_cdf = plotter.plot_img_and_hist(out_image, axes[:, 1])
ax_img.set_title('Histogram equalization')
ax_cdf.set_ylabel('Fraction of total intensity')

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
