import os
import time

import numpy as np

import ocr
import vision
import solver
import argparse
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


def main():
    parser = argparse.ArgumentParser(description='Solve sudoku')
    parser.add_argument('-v', '--video', type=str)
    parser.add_argument('-p', '--picture', type=str)
    args = parser.parse_args()
    if args.video:
        print(args.video) # TODO
    elif args.picture:
        valid_grid, img_original, digit_imgs, vision_state = vision.get_sudoku_digits_file(args.picture)
        if not valid_grid:
            print('Could not detect a sudoku grid in the input picture.')
            return
        digits = ocr.predict_digits(digit_imgs, ocr.DEFAULT_MODEL)
        digits = list(map(str, digits))
        solved, solution_digits = solver.solve_sudoku(''.join(digits))
        if not solved:
            print('Could not solve the sudoku.')
        solution_digits = list(map(lambda z: z[0] if z[0] != z[1] else None, zip(solution_digits, digits)))
        img_solution = vision.write_solution_digits(img_original, solution_digits, *vision_state)
        plot_images([img_original, img_solution], figsize=(10, 10), cols=2)


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    # t = time.time()
    # i = 0
    # while(True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()

    #     # Our operations on the frame come here
    #     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     found, img, img_solution = vision.get_sudoku_digits(frame)

    #     # Display the resulting frame
    #     if found:
    #         cv2.imshow('frame',img_solution)
    #     else:
    #         cv2.imshow('frame', img)
    #     i += 1
    #     if (time.time() - t) >= 1:
    #         t = time.time()
    #         print(f'fps: {i}')
    #         i = 0
            
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
    # grid_dataset_accuracy()
    main()
    
