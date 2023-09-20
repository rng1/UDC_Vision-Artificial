import numpy as np
import matplotlib.pyplot as plt
import skimage

import plotter
import histograms
import test_images

in_image = skimage.data.text()

p2, p98 = np.percentile(in_image, (2, 98))
out_image_intensity = histograms.adjust_intensity(in_image, in_range=[20, 120])
out_image_equalized = histograms.equalize_intensity(in_image)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=object)

axes[0, 0] = fig.add_subplot(2, 3, 1)
for i in range(1, 3):
    axes[0, i] = fig.add_subplot(2, 3, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 3):
    axes[1, i] = fig.add_subplot(2, 3, 4 + i)

ax_img, ax_hist, ax_cdf = plotter.plot_img_and_hist(in_image, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))
ax_cdf.set_yticks([])

ax_img, _, ax_cdf = plotter.plot_img_and_hist(out_image_intensity, axes[:, 1])
ax_img.set_title('Intensity adjustment')
ax_cdf.set_yticks([])

ax_img, _, ax_cdf = plotter.plot_img_and_hist(out_image_equalized, axes[:, 2])
ax_img.set_title('Histogram equalization')
ax_cdf.set_yticks([])

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
