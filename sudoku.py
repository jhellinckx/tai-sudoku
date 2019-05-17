import os
import time
import argparse

import numpy as np
import cv2

import ocr
import vision
import solver
from utils import *

def grid_dataset_accuracy():
    sudoku_names = [sudoku_name for sudoku_name in os.listdir(GRIDS_DIR) if not sudoku_name.startswith('.')]
    sudoku_names = set(map(lambda x: x.split('.')[0], sudoku_names))
    # Percentage of digits cells effectively recognized as digits
    y_digits_num = 0
    y_digits_pred = 0
    # Percentage of empty cells effectively recognized as empty cells
    y_empty_num = 0
    y_empty_pred = 0
    # Percentage of correctly recognized digits by the NN 
    # when the cell was correctly recognized as a digit cell
    y_digits_ocr_num = 0
    y_digits_ocr_pred = 0
    ocr_errors = []
    ocr_errors_y = []
    ocr_errors_y_pred = []
    ocr_errors_sudoku = []
    for i, sudoku_name in enumerate(sudoku_names):
        img_path, y_path = get_sudoku(sudoku_name)
        y = np.array(get_solution(y_path))
        y_pred, digit_imgs = get_sudoku_grid(img_path, ocr.conv_model)
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
        print(f'Done {i + 1}/{len(sudoku_names)}', end='\r')
    print()
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

def get_sudoku_grid(img_path, model_filename):
    digit_imgs = vision.get_sudoku_digits(img_path)
    return np.array(ocr.predict_digits(digit_imgs, model_filename)), digit_imgs

def sudoku(picture, is_file=False, prev_solution_digits=None):
    start = time.time()
    valid_grid, img_original, digit_imgs, bb_grid, vision_state, img_vision, img_threshold, img_contours = vision.get_sudoku_digits(picture, is_file)
    #print(f'Detect digits time: {time.time() - start}')
    if not valid_grid:
        return (False, None, img_original, None, None, None, img_threshold, img_contours)
    # digits = ocr.predict_digits(digit_imgs, ocr.DEFAULT_MODEL)
    if prev_solution_digits is None:
        digits = ocr.predict_digits(digit_imgs)
        #print(f'OCR time: {time.time() - start}')
        digits = list(map(str, digits))
        solved, solution_digits = solver.solve_sudoku(''.join(digits))
        #print(f'Solve time: {time.time() - start}')
        if not solved:
            #print('Not solved')
            return (False, None, img_original, None, None, img_vision, img_threshold, img_contours)
        solution_digits = list(map(lambda z: z[0] if z[0] != z[1] else None, zip(solution_digits, digits)))
        print('=== SOLVED ===')
    else:
        solution_digits = prev_solution_digits
    img_solution = vision.write_solution_digits(img_original, solution_digits, *vision_state)
    #print(f'Write digits time: {time.time() - start}')
    return (True, solution_digits, img_original, img_solution, bb_grid, img_vision, img_threshold, img_contours)

def sudoku_picture(picture_path):
    solved, _, img_original, img_solution, _, _, _, _ = sudoku(picture_path, True)
    if not solved:
        print('Could not detect/solve a sudoku in the given picture.')
        return
    plot_images([img_original, img_solution], figsize=(10, 10), cols=2)

def sudoku_video(show_vision, show_preprocess, show_contours):
    cap = cv2.VideoCapture(0)
    t = time.time()
    i = 0
    tracker = None
    bb_grid = None
    solution_digits = None
    tracking = False
    while(True):
        ret, frame = cap.read()
        frame = resize_ar(frame, vision.IMAGE_FIXED_WIDTH)
        if tracking:
            success, box = tracker.update(frame)
            #print(success)
            if not success:
                bb_grid = None
                solution_digits = None
                tracking = False
        
        solved, solution_digits, img_original, img_solution, bb_grid, img_vision, img_threshold, img_contours = sudoku(frame, False, solution_digits)

        if not tracking and bb_grid is not None:
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bb_grid)
            tracking = True
        cv2.imshow('sudoku', img_solution if solved else img_original)
        if show_vision:
            cv2.imshow('vision', img_vision if img_vision is not None else img_original)
        if show_preprocess:
            cv2.imshow('preprocessing', img_threshold if img_threshold is not None else img_original)
        if show_contours:
            cv2.imshow('contours', img_contours if img_contours is not None else img_original)
        i += 1
        if (time.time() - t) >= 1:
            t = time.time()
            #print(f'fps: {i}')
            i = 0
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Solve sudoku')
    parser.add_argument('-v', '--video', action='store_true')
    parser.add_argument('-p', '--picture', type=str)
    parser.add_argument('-cv', '--vision', action='store_true')
    parser.add_argument('-pre', '--preprocessing', action='store_true')
    parser.add_argument('-c', '--contours', action='store_true')
    args = parser.parse_args()
    if args.video:
        sudoku_video(args.vision, args.preprocessing, args.contours)
    elif args.picture:
        sudoku_picture(args.picture)


if __name__ == '__main__':
    main()
    #grid_dataset_accuracy()
    
