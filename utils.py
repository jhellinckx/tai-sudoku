import os
from math import ceil

import matplotlib.image as mpli
import matplotlib.pyplot as plt
import numpy as np
import cv2

RES_DIR = 'res'
GRIDS_DIR = os.path.join(RES_DIR, 'grids')
GRID_IMG_EXT = '.jpg'
GRID_SOL_EXT = '.sud'

def get_grid_path(img_filename):
    return os.path.join(GRIDS_DIR, img_filename)

def get_sudoku(sudoku_name):
    return get_grid_path(sudoku_name + GRID_IMG_EXT), get_grid_path(sudoku_name + GRID_SOL_EXT)

def flatten(ls):
    return [x for l in ls for x in l]

def get_solution(sudoku_sol_path):
    with open(sudoku_sol_path, 'r') as f:
        return list(map(int, flatten(filter(lambda x: x != '', map(lambda x: x.split(' '), map(str.strip, f.readlines()))))))

sudoku_color = get_grid_path('sudoku21.jpg')
sudoku_easy = get_grid_path('sudoku7.jpg')
sudoku_easy2 = get_grid_path('sudoku22.jpg')
sudoku_easy3 = get_grid_path('sudoku23.jpg')
sudoku_right = get_grid_path('sudoku2.jpg')
sudoku_left = get_grid_path('sudoku4.jpg')
sudoku_hard = get_grid_path('sudoku19.jpg')
sudoku_small = get_grid_path('sudoku36.jpg')
sudoku_perspective = get_grid_path('sudoku10.jpg')

def resize_ar(image, width):
    img_width, img_height = image.shape[1], image.shape[0]
    if img_width == width:
        return image
    scale = width / img_width
    new_height = int(img_height * scale)
    new_width = int(img_width * scale)
    upscale = new_width > img_width
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR if upscale else cv2.INTER_AREA)

def plot_images(images, titles=None, cols=2, figsize=(10, 10), color='gray', fontsize=10, hspace=0.2, wspace=0.2):
    if titles is None:
        titles = [None for i in range(len(images))]
    rows = ceil(len(images) / cols)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.imshow(img, color)
        plt.xticks([])
        plt.yticks([])
    plt.show()