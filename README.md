# SudokuSensei

A simple Sudoku solver.
Currently using a brute-force backtracking algorithm.

## To Do
  - implement real world sudoku scanner using OpenCV
  - implement different solving algorithms
  
## Example
  
  ```
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
  ```
  Returns
  ```
  [[3 7 4 9 2 6 1 8 5]
   [8 5 6 1 3 4 7 2 9]
   [9 1 2 5 7 8 3 4 6]
   [1 3 7 6 5 2 4 9 8]
   [6 4 5 7 8 9 2 1 3]
   [2 9 8 3 4 1 5 6 7]
   [4 8 3 2 9 5 6 7 1]
   [7 2 1 8 6 3 9 5 4]
   [5 6 9 4 1 7 8 3 2]]
  ```
