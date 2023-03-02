#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of several Sudoku solving algorithms.
@author: Christoph M. Jankowsky
"""

import numpy as np

def solve_sudoku(board):
    """
    solves a given sudoku using the brute-force method of backtracking

    :param board: sudoko board as a 2D numpy array
    :return: returns True if board is solved, otherwise False
    """ 
    
    def find_empty(board):
        """
        finds an empty cell in a given sudoku board

        :param board: sudoko board as a 2D numpy array
        :return: returns row and column index of the empty cell
        """ 
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    return row, col

    def is_valid(board, number, row, col):
        """
        checks if a given number at a given place in a given sudoku board
        is valid, i. e. if it does not break any sudoku rules

        :param board: sudoko board as a 2D numpy array
        :param number: number whose validity needs to be checked
        :param row: row index of the cell in question
        :param col: col index of the cell in question
        :return: returns True if number is valid at given cell
        """ 

        # check if number already in row
        if number in board[row, :]:
            return False

        # check if number already in column
        if number in board[:, col]:
            return False

        # check if number already in block
        sub_row = 3 * (row//3)
        sub_col = 3 * (col//3)
        block = board[sub_row:sub_row+3, sub_col:sub_col+3]
        if number in block:
            return False
        
        return True

    if 0 not in board:
        return True

    row, col = find_empty(board)

    for number in range(1, 10):
        if is_valid(board, number, row, col):
            board[row, col] = number
            if solve_sudoku(board):
                return True
            board[row, col] = 0
    return False

if __name__ == "__main__":
    board = np.array([
        [0, 0, 4,   0, 2, 0,    0, 8, 0],
        [0, 0, 0,   1, 0, 0,    0, 0, 0],
        [9, 1, 2,   5, 7, 0,    0, 4, 0],

        [0, 0, 7,   6, 0, 2,    4, 0, 0],
        [0, 4, 5,   7, 0, 0,    0, 1, 0],
        [2, 0, 0,   0, 0, 0,    5, 0, 0],

        [0, 0, 3,   2, 9, 0,    0, 7, 0],
        [0, 0, 0,   0, 0, 3,    9, 0, 4],
        [0, 0, 9,   0, 0, 0,    8, 3, 0]
    ])

    if solve_sudoku(board):
        print(board)
    else:
        print("Could not solve board.")

