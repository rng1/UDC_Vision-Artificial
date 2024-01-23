import time
from os import listdir

from daugman import find_optimal_circle
from helper import *

st = time.time()

start_time_total = time.time()

folder = "p2/img/"
for img in listdir(folder):
    start_time_img = time.time()

    # Ranges for the radii of the iris and pupil
    lpupil_rad = 10
    upupil_rad = 30
    liris_rad = 35
    uiris_rad = 80

    pupil_pct_acc = 0.97
    iris_pct_acc = 0.5

    it = 0
    max_it = 1  # Max iterations for the polish function

    max_distance = 5

    # Image preprocessing
    eye_image = cv.imread(folder + img)
    if eye_image.ndim == 3:
        eye_image = cv.cvtColor(eye_image, cv.COLOR_RGB2GRAY)

    gauss_image = cv.GaussianBlur(eye_image, (3, 3), 2)
    proc_image = cv.morphologyEx(gauss_image, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Daugman method
    pupil_centers = get_potential_centers(proc_image)
    pupil_boundary = find_optimal_circle(proc_image, pupil_centers, min_rad=lpupil_rad, max_rad=upupil_rad, step=1)
    iris_centers = get_adjacent_points(proc_image, pupil_boundary[0], max_distance)
    iris_boundary = find_optimal_circle(proc_image, iris_centers, min_rad=liris_rad, max_rad=uiris_rad, step=1)

    circled_image = get_segment_circles(eye_image, iris_boundary, pupil_boundary)
    mask_iris, mask_pupil = get_segment_mask(eye_image, iris_boundary, pupil_boundary)

    # Segment iris and pupil
    iris_segment = eye_image.copy()
    iris_segment[~mask_iris] = 0
    pupil_segment = eye_image.copy()
    pupil_segment[~mask_pupil] = 0

    # Center the image around its bounding box
    iris_segment = center_in_bounding_box(iris_segment)
    pupil_segment = center_in_bounding_box(pupil_segment)

    # Clean up all non-valuable information
    iris_segment = cleaner(iris_segment, iris_pct_acc)
    pupil_segment = cleaner(pupil_segment, pupil_pct_acc)

    # Invert the background of the images for plotting
    iris_segment_plot = np.where(iris_segment == 0, 255, iris_segment)
    pupil_segment_plot = np.where(pupil_segment == 0, 255, pupil_segment)

    print(f"Image \"{img}\" processed in: {(time.time() - start_time_img)} s")

    plot_all([eye_image, circled_image, iris_segment_plot, pupil_segment_plot],
             [img, "Circled image", "Segmented iris", "Segmented pupil"])

print(f"Total elapsed time: {time.time() - st} s")
