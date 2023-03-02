#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sudoku Solver from image.
@author: Christoph M. Jankowsky
"""

from time import perf_counter

import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor

from solver import solve_sudoku
from model import Net

def preprocess_image(image):
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    return img

def transorm_image(image):
    contours, hierarchy = cv2.findContours(prc_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctr_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)

    epsilon = 0.1*cv2.arcLength(contours[-1], True)
    box = cv2.approxPolyDP(contours[-1], epsilon, True)
    box_image = cv2.drawContours(image.copy(), box, -1, (0, 0, 255), 10)

    input_pts = np.float32(box)
    output_pts = np.float32([[0, 0], [0, 1000 - 1], [1000 - 1, 1000 - 1], [1000 - 1, 0]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    transformed_image = cv2.warpPerspective(image.copy(),M,(1000, 1000),flags=cv2.INTER_LINEAR)

    return transformed_image

def extract_cells(image):
    edge_h = np.shape(image)[0]
    edge_w = np.shape(image)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(trn_image)[1] // 9

    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = trn_image[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])
    cells = np.array(tempgrid)

    return cells

def show_image(image, title="window"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_solution(board):
    # Create a black image
    img = np.zeros((450,450,3), np.uint8)

    # Write some Text

    font                    = cv2.FONT_HERSHEY_SIMPLEX
    at                      = (10,30)
    fontScale               = 1
    fontColor               = (255,255,255)
    thickness               = 1
    lineType                = 2

    for row in range(9):
        cv2.putText(img, np.array2string(board[row, :]), 
            (at[0], at[1]+50*row), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    #Display the image
    cv2.imshow("img",img)
    cv2.waitKey(0)

def print_sudoku(board):
    for n_row, row in enumerate(board):
        if n_row%3 == 0:
            print("\n")
        print(f"{row[0:3]}   {row[3:6]}   {row[6:9]}")

if __name__ == "__main__":
    image_path = "./test_images/sample_sudoku.jpeg"
    image = cv2.imread(image_path)
    prc_image = preprocess_image(image)
    trn_image = transorm_image(prc_image)
    cells = extract_cells(trn_image)
    cells = np.array([cell[15:-15, 15:-15] for cell in cells])
    cells = np.array([cv2.resize(cell, (28, 28), interpolation=cv2.INTER_LINEAR) for cell in cells])
    grid = np.concatenate(cells, axis=1)

    model = Net()
    model.load_state_dict(torch.load("./model_results/model.pth"))
    model.eval()

    digits = [i for i in range(10)]
    board = np.zeros((9, 9))
    with torch.no_grad():
        for n, cell in enumerate(cells):
            if cell.mean() > 20:
                data = torch.from_numpy(np.asarray(cell, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
                output = model(data)
                # get the index of the max log-probability
                prediction = output.max(1, keepdim=True)[1]
                # show_image(cell, f"{digits[prediction[0,0]]}")
                board[n//9, n%9] = prediction
    
    print(f"Solving Sudoku:")
    print_sudoku(board)
    start = perf_counter()
    if solve_sudoku(board):
        stop = perf_counter()
        print(f"\nSolved Sudoku in {stop-start:.2f} seconds.\n")
        print(f"Solution:")
        print_sudoku(board)
        # show_solution(board)
    else:
        print("Could not solve board.")
    

