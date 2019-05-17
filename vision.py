import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import *

RGB_RED = (255, 0, 0)
RGB_GREEN = (0, 255, 0)
RGB_BLUE = (0, 0, 255)

CORNER_RADIUS = 18
CORNER_COLOR = RGB_GREEN

CENTER_RADIUS = 8
CENTER_COLOR = RGB_GREEN

CONTOUR_THICKNESS = 6
CONTOUR_COLOR = RGB_RED

DIGIT_BB_COLOR = RGB_GREEN
DIGIT_BB_THICKNESS = 2

SOL_DIGIT_FONT = cv2.FONT_HERSHEY_SIMPLEX
SOL_DIGIT_COLOR = (0, 200, 66)
SOL_DIGIT_THICKNESS = 2
SOL_DIGIT_FONT_SCALE = 1 # Base font size multiplier
HERSHEY_SIMPLEX_SCALE = lambda cell_height: SOL_DIGIT_FONT_SCALE * cell_height/50

SUDOKU_DIM = 9
SUDOKU_MIN_DIGITS_NUM = 17

BLUR_KERNEL_SIZE = (9, 9)
ADAPTIVE_THRESH_WINDOW_SIZE = 11
ADAPTIVE_THRESH_SUB_MEAN = 2

DIGIT_CENTER_PAD_RATIO = 0.2
CELL_BORDER_CROP_RATIO = 0.15
DIGIT_CC_AREA_MIN_RATIO = 0.1
SUDOKU_GRID_AREA_MIN_RATIO = 0.05

IMAGE_FIXED_WIDTH = 1000

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_sudoku_digits(img, is_file=False):
    if is_file:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    img_original = resize_ar(img, IMAGE_FIXED_WIDTH)
    img_original = img_original.copy()
    #start = time.time()
    img = cv2.cvtColor(img_original.copy(), cv2.COLOR_RGB2GRAY)
    img_blurred, img_threshold, img_dilate = pre_process(img)
    #plot_images([img, img_blurred, img_threshold, img_dilate], ['Original', 'Blur', 'Threshold', 'Dilate'])
    grid_found, corners, img_contours, img_grid_contour, img_corners, img_vision, bb_grid = get_corners(img_original, img_dilate)
    if not grid_found:
        return (False, img_original, None, None, None, None, img_threshold, img_contours)
    #print(time.time() - start)
    #plot_images([img_dilate, img_contours], ['Preprocessed', 'Contours'], cols=1)
    # use sudoku9.jpg
    # bottom_left = corners[-1]
    # img_corners = cv2.line(img_corners, (0, 0), (1000, 1000), RGB_GREEN, 8)
    # img_corners = cv2.line(img_corners, bottom_left, (bottom_left[0] + 1500, bottom_left[1] - 1500), RGB_GREEN, 8)
    #plot_images([img_grid_contour, img_corners], ['Largest contour', 'Corners'], cols=1)
    img_raw_grid, img_warp_grid, m = get_grid_roi(img_threshold, *corners)
    #plot_images([img_threshold, img_warp_grid], ['Original', 'With perspective transform'], cols=1)
    valid_grid, digits, cells, centers, img_centers, bb_cells, img_vision = get_digits_rois(img_warp_grid, img_vision, m)
    if not valid_grid:
        return (False, img_original, None, None, None, img_vision, img_threshold, img_contours)
    return (True, img_original, digits, bb_grid, (img_warp_grid, m, centers), img_vision, img_threshold, img_contours)
    #plot_images([img_warp_grid, img_centers], ['Warped grid', 'Centers of cells'], cols=2, figsize=(15, 15), fontsize=15)
    #plot_images(cells, cols=9)
    # img_solution = write_solution_digits(img_original, img_warp_grid, m, centers, digits)
    #plot_images([img_original, img_solution], figsize=(10, 10), cols=2)
    # return digits
    # return (True, img_original, img_solution)

def warp_draw_image(img_bg, img_fg, m, threshold=100):
    bg_width, bg_height = img_bg.shape[1], img_bg.shape[0]
    img_fg_warp = cv2.warpPerspective(img_fg, m, (bg_width, bg_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    img_fg_warp_gray = cv2.cvtColor(img_fg_warp, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img_fg_warp_gray, threshold, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg_warp = cv2.bitwise_and(img_fg_warp, img_fg_warp, mask=mask)
    img_bg = cv2.bitwise_and(img_bg, img_bg, mask=mask_inv)
    img_drawn = cv2.add(img_bg, img_fg_warp)
    return img_drawn

def write_solution_digits(image_original, digits, image_warp, m, cell_centers):
    image_original = image_original.copy()
    image_warp = cv2.cvtColor(image_warp.copy(), cv2.COLOR_GRAY2RGB)
    warp_height = image_warp.shape[0]
    warp_width = image_warp.shape[1]
    original_height = image_original.shape[0]
    original_width = image_original.shape[1]
    image_solution_digits = np.full((warp_height, warp_width, 3), 0).astype('uint8')
    font_scale = HERSHEY_SIMPLEX_SCALE(warp_height / SUDOKU_DIM)
    for (cell_center_x, cell_center_y), digit in zip(cell_centers, digits):
        if digit is not None:
            text_width, text_height = cv2.getTextSize(digit, SOL_DIGIT_FONT, font_scale, SOL_DIGIT_THICKNESS)[0]
            text_center_x = cell_center_x - int(text_width / 2)
            text_center_y = cell_center_y + int(text_height / 2)
            cv2.putText(image_solution_digits, digit, (text_center_x, text_center_y), SOL_DIGIT_FONT, font_scale, SOL_DIGIT_COLOR, SOL_DIGIT_THICKNESS, cv2.LINE_AA)
    
    return warp_draw_image(image_original, image_solution_digits, m, 100)
    
def pre_process(image):
    # Blurring/smoothing to reduce noise
    img_blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, 0)

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
    img_threshold = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_WINDOW_SIZE, 
                                          ADAPTIVE_THRESH_SUB_MEAN)

    # Use dilation to increase the thickness of the "features" e.g. grid lines
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]).astype('uint8')
    img_dilate = cv2.dilate(img_threshold, kernel)
    return img_blurred, img_threshold, img_dilate

def get_corners(img_original, image):
    # Find all the contours in the image (i.e. connected components)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if not contours:
        return (False, None, None, None, None, None, None)
    # We assume the sudoku grid is the contour with the largest area, so fetch the first one
    grid_contour = contours[0]

    # We want to draw the bounding box and contours in RGB, but our image is encoded as grayscale
    # So, re-encode it as RGB
    image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    x, y, w, h = cv2.boundingRect(grid_contour)
    bb_grid = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), RGB_GREEN, CONTOUR_THICKNESS)
    #plot_images([image, bb_grid], figsize=(10, 10), cols=2)

    grid_found = cv2.contourArea(grid_contour) > (image.shape[0] * image.shape[1] * SUDOKU_GRID_AREA_MIN_RATIO)
    img_contours = cv2.drawContours(image.copy(), contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
    img_grid_contour = cv2.drawContours(img_original.copy(), [grid_contour], -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

    img_vision = img_grid_contour.copy()
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
        img_vision = cv2.circle(img_vision, corner, CORNER_RADIUS, CORNER_COLOR, cv2.FILLED)

    return grid_found, corners, img_contours, img_grid_contour, img_corners, img_vision, (x, y, w, h)

    
def get_grid_roi(image, top_left, top_right, bottom_right, bottom_left):
    # We still need to take care of perspective and rotation
    # Since we know the 4 corners of the grid, we can easily apply perspective transformation
    # which will also crop the grid for us and handle rotations
    
    # Let us first generate a grid by cropping the image with the raw corners, without any changes
    img_raw_grid = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # Let's now start with the perspective transformation
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

def get_digits_rois(image_grid, img_vision, m):
    # Get the length of the grid square
    side_length = image_grid.shape[0]
    # Sudoku grids are 9x9 matrices, so start off by finding the center of each cell
    # by partitioning the grid into 81 cells of the same size
    cell_length = side_length / SUDOKU_DIM
    pad = int(cell_length / 2)
    centers = [(int(cell_length * j + pad), int(cell_length * i + pad)) for i in range(SUDOKU_DIM) for j in range(SUDOKU_DIM)]

    # Generate an image that shows the 81 centers of the cells
    img_centers = cv2.cvtColor(image_grid.copy(), cv2.COLOR_GRAY2RGB)
    for center in centers:
        img_centers = cv2.circle(img_centers, center, CENTER_RADIUS, CENTER_COLOR, cv2.FILLED)
    #show_image(img_centers)
    
    # Now that we have the approximate center of each cell, crop the grid for each cell w.r.t. its center 
    # Also crop the cells to get rid of the noisy borders
    cell_pad = int(pad - cell_length * CELL_BORDER_CROP_RATIO)
    cells = [image_grid[center_y - cell_pad:center_y + cell_pad, center_x - cell_pad:center_x + cell_pad] for center_x, center_y in centers]
    #plot_images(cells, None, cols=9, figsize=(4, 4))

    # We still need to determine wether or not each cell contains a digit
    # As for the grid contour detection, find the biggest connected component, assume it is the digit in the cell
    digits = [None for i in range(len(cells))]
    bb_cells = cells
    bb_digits = [None for i in range(len(cells))]
    for i, cell in enumerate(cells):
        cell_area = cell.shape[0]**2
        contours, h = cv2.findContours(cell, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if not contours:
            continue
        digit_contour = contours[0]
        # If the area of the CC is not greater than a fixed proportion of the cell, assume the cell is empty
        if cell_area * DIGIT_CC_AREA_MIN_RATIO > cv2.contourArea(digit_contour):
            continue
        # Create an image that is centered around the digit
        # First find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(digit_contour)
        bb_digits[i] = (x, y, w, h)
        # Create an image of the cell with the digit bounding box
        bb_cell = cv2.cvtColor(cell.copy(), cv2.COLOR_GRAY2RGB)
        bb_cells[i] = cv2.rectangle(bb_cell, (x, y), (x + w, y + h), DIGIT_BB_COLOR, DIGIT_BB_THICKNESS)
        # Crop the cell with the bounding box
        cell = cell[y:y + h, x:x + w]
        # Add some area around the digit such that it creates a well-centered image
        if h > w:
            # Wrap the image around some zeros vectors to create a vertically centered image
            pad_length = int(h * DIGIT_CENTER_PAD_RATIO)
            vertical_pad = np.zeros((pad_length, w))
            cell = np.concatenate((vertical_pad, cell, vertical_pad), axis=0)
            # Do the same horizontally
            horizontal_pad = np.zeros((cell.shape[0], int((cell.shape[0] - w) / 2)))
            cell = np.concatenate((horizontal_pad, cell, horizontal_pad), axis=1)
        else:
            # Inverse the operations
            pad_length = int(w * DIGIT_CENTER_PAD_RATIO)
            horizontal_pad = np.zeros((h, pad_length))
            cell = np.concatenate((horizontal_pad, cell, horizontal_pad), axis=1)
            vertical_pad = np.zeros((int((cell.shape[1] - h) / 2), cell.shape[1]))
            cell = np.concatenate((vertical_pad, cell, vertical_pad), axis=0)
        digits[i] = cell
    num_recognized_digits = len(list(filter(lambda x: x is not None, digits)))
    valid_grid = num_recognized_digits >= SUDOKU_MIN_DIGITS_NUM
    # Add the digits bounding boxes in the vision image
    # Create the image containing only the bounding boxes
    img_vision_bbs = np.full((side_length, side_length, 3), 0).astype('uint8')
    for bb_digit, (cx, cy) in zip(bb_digits, centers):
        if bb_digit is not None:
            (x, y, w, h) = bb_digit
            x, y = x + cx - pad + int(cell_pad / 2), y + cy - pad + int(cell_pad / 2)
            img_vision_bbs = cv2.rectangle(img_vision_bbs, (x, y), (x + w, y + h), DIGIT_BB_COLOR, DIGIT_BB_THICKNESS)
    img_vision = warp_draw_image(img_vision, img_vision_bbs, m, 100)
    #plot_images([d for d in digits if d is not None], cols=3, figsize=(6, 6))
    #plot_images(cells, cols=9, figsize=(4, 4))
    return valid_grid, digits, cells, centers, img_centers, bb_cells, img_vision

if __name__ == '__main__':
    get_sudoku_digits(sudoku_perspective, True)










