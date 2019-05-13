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
CONTOUR_THICKNESS = 6
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
    #utils.plot_images([img_grid_contour, img_corners], ['Largest contour', 'Corners'], cols=1)
    img_raw_grid, img_warp_grid, m = get_grid(img_dilate, *corners)
    utils.plot_images([img_raw_grid, img_warp_grid], ['Raw corners rectangle', 'With perspective transform'], cols=1)
    
def pre_process(image):
    # Blurring/smoothing to reduce noise
    img_blurred = cv2.GaussianBlur(image, (21, 21), 0)

    # Gray scale image, each pixel in the range 0-255
    # Too noisy to perform advanced operations on it
    # Reduce that noise by transforming the image to pure black and white by applying adaptive thresholding
    # Here, for each pixel, it calculates a threshold value over a 15x15 window (i.e. neighbourhood)
    # The threshold is computed by taking the mean of the pixels values in the window and substracting 2
    # It then changes the current pixel value to 255 if said value is greater that the threshold, else it sets it to 0
    # In normal (non-adaptative) thresholding, the threshold is computed by taking the mean of the entire image which
    # leads to problems when the lightning is uneven in the image or when some shadows are present
    # http://aishack.in/tutorials/thresholding/
    # http://aishack.in/tutorials/sudoku-grabber-opencv-detection/
    img_threshold = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)

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
    operations = [(np.argmin, sum), (np.argmax, diff), (np.argmax, sum), (np.argmin, diff)]
    corners = []
    for retrieve, transform in operations:
        corner_idx = retrieve(list(map(transform, points)))
        corners.append(points[corner_idx])

    # Generate an image with the corners of the grid drawn on it
    img_corners = image.copy()
    for corner in corners:
        img_corners = cv2.circle(img_corners, corner, CORNER_RADIUS, CORNER_COLOR, cv2.FILLED)

    return corners, img_contours, img_grid_contour, img_corners

    
def get_grid(image, top_left, top_right, bottom_right, bottom_left):
    # We still need to take care of perspective and rotation
    # Given that we already know the 4 corners of the grid, we can apply perspective tranformation
    # which will also crop the grid for us
    # Let us first generate a grid by cropping the image with the raw corners, without any changes
    img_raw_grid = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # Get the area of the source grid, as defined by the extracted corners
    src_area = np.array([top_left, top_right, bottom_right, bottom_left]).astype('float32')

    # Find the longest edge of the grid rectangle, which will then be used to construct the destination area 
    # of the perspective transformation
    top_left, top_right, bottom_right, bottom_left = list(map(np.array, [top_left, top_right, bottom_right, bottom_left]))
    norm = np.linalg.norm
    edge_max = max([norm(top_left - top_right), norm(top_right - bottom_right), 
                norm(bottom_left - bottom_right), norm(top_left - bottom_left)])
    # The new area is a square with side of length edge_max
    # Note that the ordering of the points must correspond in src_area and dst_area
    dst_area = np.array([[0, 0], [edge_max, 0], [edge_max, edge_max], [0, edge_max]]).astype('float32')

    # Get the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_area, dst_area)

    # Apply the transformation on the image to get the warped grid
    img_warp_grid = cv2.warpPerspective(image, transform_matrix, (int(edge_max), int(edge_max)))
    return img_raw_grid, img_warp_grid, transform_matrix


if __name__ == '__main__':
    sudoku_easy = 'sudoku7.jpg'
    sudoku_rotation_right = 'sudoku2.jpg'
    sudoku_rotation_left = 'sudoku4.jpg'
    get_sudoku(get_img_path(sudoku_easy))