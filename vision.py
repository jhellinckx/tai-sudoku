import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

import utils

RGB_RED = (255, 0, 0)
RGB_GREEN = (0, 255, 0)
RGB_BLUE = (0, 0, 255)

CORNER_RADIUS = 18
CORNER_COLOR = RGB_GREEN
CONTOUR_THICKNESS = 8
CONTOUR_COLOR = RGB_RED

RES_DIR = 'res'
GRIDS_DIR = os.path.join(RES_DIR, 'grids')

def get_img_path(img_filename):
    return os.path.join(GRIDS_DIR, img_filename)

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_sudoku(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_blurred, img_threshold, img_dilate = pre_process(img)
    #utils.plot_images([img, img_blurred, img_threshold, img_dilate], ['Original', 'Blur', 'Threshold', 'Dilate'])
    corners, img_contours, img_grid_contour, img_corners = get_corners(img_dilate)
    #utils.plot_images([img_dilate, img_contours], ['Preprocessed', 'Contours'], cols=1)
    utils.plot_images([img_grid_contour, img_corners], ['Largest contour', 'Corners'], cols=1)
    
def pre_process(image):
    # Blurring/smoothing to reduce noise
    img_blurred = cv2.GaussianBlur(image, (17, 17), 0)

    # Gray scale image, each pixel in the range 0-255
    # Too noisy to perform advanced operations on it
    # Reduce that noise by transforming the image to pure black and white by applying adaptive thresholding
    # Here, for each pixel, it calculates a threshold value over a 11x11 window (i.e. neighbourhood)
    # The threshold is computed by taking the mean of the pixels values in the window and substracting 2
    # It then changes the current pixel value to 255 if said value is greater that the threshold, else it sets it to 0
    # In normal (non-adaptative) thresholding, the threshold is computed by taking the mean of the entire image which
    # leads to problems when the lightning is uneven in the image or when some shadows are present
    # http://aishack.in/tutorials/thresholding/
    # http://aishack.in/tutorials/sudoku-grabber-opencv-detection/
    img_threshold = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Use dilation to increase the thickness of the "features" e.g. grid lines
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]).astype('uint8')
    img_dilate = cv2.dilate(img_threshold, kernel)
    return img_blurred, img_threshold, img_dilate

def get_corners(image):
    # Find all the contours in the image (i.e. connected components)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # We assume the sudoku grid is the contour with the largest area, so fetch the first one
    grid_contour = contours[0]

    # We want to draw the contours in RGB, but our image is encoded as grayscale
    # So, re-encode it as RGB
    image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    img_contours = cv2.drawContours(image.copy(), contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
    img_grid_contour = cv2.drawContours(image.copy(), [grid_contour], -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

    # We now need to find the 4 corners of the sudoku grid
    # The findContours function represent its contours by a succession of (x, y) points
    # So we can use some heuristics to find the 4 corners of the grid
    # Note that the origin is located at the top left of the image
    # e.g. bottom_right point has the largest (x + y) value
    points = [(point[0][0], point[0][1]) for point in grid_contour]
    diff = lambda elems: elems[0] - sum(elems[1:])
    operations = [(np.argmin, sum), (np.argmin, diff), (np.argmax, diff), (np.argmax, sum)]
    corners = []
    for retrieve, transform in operations:
        corner_idx = retrieve(list(map(transform, points)))
        corners.append(points[corner_idx])

    img_corners = image.copy()
    for corner in corners:
        img_corners = cv2.circle(img_corners, corner, CORNER_RADIUS, CORNER_COLOR, cv2.FILLED)

    return corners, img_contours, img_grid_contour, img_corners

    


if __name__ == '__main__':
    get_sudoku(get_img_path('sudoku7.jpg'))