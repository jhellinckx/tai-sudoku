import os

import numpy as np

import ocr
import vision
from utils import *

def grid_dataset_accuracy():
    sudoku_names = [sudoku_name for sudoku_name in os.listdir(GRIDS_DIR) if not sudoku_name.startswith('.')]
    sudoku_names = set(map(lambda x: x.split('.')[0], sudoku_names))
    # Percentage of digits cells effectively recognized as digits
    y_digits_num = 0
    y_digits_pred = 0
    
    y_empty_num = 0
    y_empty_pred = 0

    y_digits_ocr_num = 0
    y_digits_ocr_pred = 0

    ocr_errors = []
    ocr_errors_y = []
    ocr_errors_y_pred = []
    ocr_errors_sudoku = []
    for i, sudoku_name in enumerate(sudoku_names):
        img_path, y_path = get_sudoku(sudoku_name)
        y = np.array(get_solution(y_path))
        y_pred, digit_imgs = get_sudoku_grid(img_path)
        for j, (y_digit, y_digit_pred) in enumerate(zip(y, y_pred)):
            if y_digit == 0:
                y_empty_num += 1
                if y_digit_pred == 0:
                    y_empty_pred += 1
            elif y_digit != 0:
                y_digits_num += 1
                if y_digit_pred != 0:
                    y_digits_ocr_num += 1
                    y_digits_pred += 1
                    if y_digit_pred == y_digit:
                        y_digits_ocr_pred += 1
                    else:
                        ocr_errors_sudoku.append(sudoku_name[6:])
                        ocr_errors_y.append(y_digit)
                        ocr_errors_y_pred.append(y_digit_pred)
                        ocr_errors.append(digit_imgs[j])
        print(f'Done {i + 1}/{len(sudoku_names)}')
    ocr_errors_titles = []
    for y, y_pred, sudoku in zip(ocr_errors_y, ocr_errors_y_pred, ocr_errors_sudoku):
        ocr_errors_titles.append(f'{sudoku}: {y} -> {y_pred}')
    plot_images(ocr_errors, ocr_errors_titles, cols=10, figsize=(10, 10), fontsize=8, hspace=0.4)
    empty_cell_accuracy = y_empty_pred / y_empty_num
    digit_cell_accuracy = y_digits_pred / y_digits_num
    ocr_accuracy = y_digits_ocr_pred / y_digits_ocr_num
    print(f'Empty cell accuracy : {empty_cell_accuracy}')
    print(f'Digit cell accuracy : {digit_cell_accuracy}')
    print(f'OCR accuracy : {ocr_accuracy}')


def get_sudoku_grid(img_path):
    digit_imgs = vision.get_digits(img_path)
    return np.array(ocr.predict_digits(digit_imgs)), digit_imgs

if __name__ == '__main__':
    grid_dataset_accuracy()
    
