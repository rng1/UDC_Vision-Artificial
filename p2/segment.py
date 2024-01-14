import time
from os import listdir

import cv2 as cv
import numpy as np

import daugman
import helper

start_time_total = time.time()

folder = "C:\\Users\\rnara\\PycharmProjects\\UDC_Vision-Artificial\\p2\\test_images\\final/"
for img in listdir(folder):
    start_time_img = time.time()

    # Ranges for the radii of the iris and pupil
    lpupil_rad = 10
    upupil_rad = 30
    liris_rad = 35
    uiris_rad = 80

    it = 0
    max_it = 10  # Max iterations for the polish function

    max_distance = 10

    eye_image = cv.imread(folder + img)
    if eye_image.ndim == 3:
        eye_image = cv.cvtColor(eye_image, cv.COLOR_RGB2GRAY)

    gauss_image = cv.GaussianBlur(eye_image, (3, 3), 2)
    proc_image = cv.morphologyEx(gauss_image, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    possible_centers = helper.get_possible_centers(proc_image)
    iris_boundary = daugman.find_circle(proc_image, possible_centers, min_rad=liris_rad, max_rad=uiris_rad, step=1)
    pupil_boundary = daugman.find_circle(proc_image, possible_centers, min_rad=lpupil_rad, max_rad=upupil_rad, step=1)

    # If the distance between vectors is greater than allowed, the center is positioned incorrectly.
    # Delete that center from the list and try again until it is or the maximum number of iterations is reached.
    while helper.distance(pupil_boundary[0], iris_boundary[0]) > max_distance:
        it += 1
        possible_centers.remove(iris_boundary[0])
        iris_boundary = daugman.find_circle(proc_image, possible_centers, min_rad=liris_rad, max_rad=uiris_rad, step=1)
        if it == max_it:
            print("Max number of iterations reached")
            break

    circled_image = helper.get_segment_circles(eye_image, iris_boundary, pupil_boundary)
    mask_iris, mask_pupil = helper.get_segment_mask(eye_image, iris_boundary, pupil_boundary)

    # Segment iris and pupil
    iris_segment = eye_image.copy()
    iris_segment[~mask_iris] = 0
    pupil_segment = eye_image.copy()
    pupil_segment[~mask_pupil] = 0

    # Center the image around its bounding box
    iris_segment = helper.center_in_bounding_box(iris_segment)
    pupil_segment = helper.center_in_bounding_box(pupil_segment)

    # Clean up all non-valuable information
    iris_segment = helper.cleaner(iris_segment, 0.5)
    pupil_segment = helper.cleaner(pupil_segment, 0.9)

    print(f"Image \"{img}\" processed in: {(time.time() - start_time_img)} s")

    helper.plot_all([eye_image, circled_image, iris_segment, pupil_segment],
                    [img, "Circled image", "Segmented iris", "Segmented pupil"])

print(f"Total elapsed time: {(time.time() - start_time_total)} s")
