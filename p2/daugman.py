import numpy as np
import cv2 as cv


def daugman_method(image, center, min_rad, max_rad, step=1):
    """ Compute the pixel intensities for circles with radii in the range from ´min_rad´ to ´max_rad´,
    incrementing by ´step´, centered at the given ´center´. Then identify the circle that is immediately before
    the largest decrease in intensity.

    :param image: the input image
    :param center: the center of the potential segmentation circle
    :param min_rad: starting point of the possible radii range
    :param max_rad: ending point of the possible radii range
    :param step: step at which the range increments in value, the lower the step, the more precise the search.
    :return: intensity value and radius of the best iteration
    """
    intensities = []
    mask = np.zeros_like(image)

    radii_range = list(range(min_rad, max_rad, step))

    # Loop over each value in the range of radii, drawing the circle, calculating its average intensity and finding
    # the index of maximum difference of intensities where the element is segmented
    for radius in radii_range:
        cv.circle(mask, center, radius, (255, 255, 255), 1)
        circle_intensity = image & mask
        avg_intensity = np.sum(circle_intensity.flatten()) / (2 * np.pi * radius)
        intensities.append(avg_intensity)
        mask.fill(0)  # Reset the mask for the next iteration

    # Calculate the differences between intensities to find the maximum drop between neighbours
    # A Gaussian blur is applied on all collected deltas to avoid false-positives.
    intensities_diff = np.array(intensities)[:-1] - np.array(intensities)[1:]
    intensities_diff = abs(cv.GaussianBlur(intensities_diff, (1, 5), 0))
    max_diff_index = np.argmax(intensities_diff)

    return intensities_diff[max_diff_index], radii_range[max_diff_index]


def find_optimal_circle(image, centers, min_rad, max_rad, step=1):
    """ Find the optimal circle in an image using the Daugman method. Iterate over a list of potential circle centers
    and for each center, apply the Daugman method to find the circle with the highest intensity.

    :param image: the input image
    :param centers: a list of potential centers for the segmentation circle
    :param min_rad: minimum radius of the circle
    :param max_rad: maximum radius of the circle
    :param step: step size for the radius. Default is 1.
    :return: a tuple containing the center point and radius of the optimal circle.
    """
    intensity_values = []
    coordinates = []

    # Apply the Daugman method for every possible center
    for center in centers:
        intensity, radius = daugman_method(image, center, min_rad, max_rad, step)
        intensity_values.append(intensity)
        coordinates.append((center, radius))

    best_index = intensity_values.index(max(intensity_values))
    return coordinates[best_index]
