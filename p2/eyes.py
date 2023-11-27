import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


folder = "test_images/1/left"

for images in os.listdir(folder):
    if not images.endswith(".db"):
        img_path = os.path.join(folder, images)

        in_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        blur_img = cv.GaussianBlur(in_img, (5, 5), 1.5)

        thr_img = np.where(blur_img < 0.1, 1, 0)

        possible_centers = []

        for i in range(blur_img.shape[0]):
            for j in range(blur_img.shape[1]):
                if blur_img[i, j] < 0.1*255:
                    possible_centers.append((j, i))

        cimg = cv.cvtColor(blur_img, cv.COLOR_GRAY2BGR)
        circles = cv.HoughCircles(blur_img, cv.HOUGH_GRADIENT, 1, 10,
                                  param1=50, param2=40, minRadius=40, maxRadius=70)

        circles = np.uint16(np.around(circles))
        detected_centers = []

        for circle in circles[0, :]:
            x, y, r = circle
            for point in possible_centers:
                px, py = point
                # Check if the circle center is close to the candidate point
                if (x, y) == (px, py):
                    detected_centers.append((x, y, r))
                    break

        for i in detected_centers:
            # draw the outer circle
            cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        plt.imshow(cimg, cmap="gray")
        plt.title(img_path)
        plt.show()
