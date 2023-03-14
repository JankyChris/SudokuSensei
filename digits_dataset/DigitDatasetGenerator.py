#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Printed digit dataset generator.
@author: Christoph M. Jankowsky
"""

import numpy as np
from numpy import random
import os

from tqdm import tqdm
import cv2

fonts   = [cv2.FONT_HERSHEY_SIMPLEX,
           cv2.FONT_HERSHEY_DUPLEX,
           cv2.FONT_HERSHEY_COMPLEX,
           cv2.FONT_HERSHEY_TRIPLEX]

digits  = ["1", "2", "3", "4", "5", "6", "7", "8", "9", " "]
color   = (255,255,255)

def print_digit(image, digit, dirt=False, border=False):
    at = (random.randint(6, 9), random.randint(20, 24))
    scale = 0.1 * random.randint(6, 10)
    font = random.choice(fonts)
    thickness = random.randint(1, 2)
    line_type = random.randint(1, 2)

    cv2.putText(image, 
                digit, 
                at, 
                font, 
                scale,
                color,
                thickness,
                line_type)
    
    if dirt:
        print_digit(image, ".")
    
    if border:
        pass


def test():
    for digit in digits:

        if random.randint(10) < 1:
            dirt = True
        else: 
            dirt = False

        if random.randint(10) < 1:
            border = True
        else:
            border = False

        image = np.zeros((28,28,3), np.uint8)

        print_digit(image, digit, dirt, border)

        image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imshow(f"{digit}",image)
        cv2.waitKey(0)

def generate_assets(digit, n_images, not_zero):
    for n in tqdm(range(n_images)):
            dirt =  (random.randint(10) < 2)
            border = (random.randint(10) < 1)

            image = np.zeros((28,28,3), np.uint8)
            print_digit(image, digit, dirt, border)

            if not_zero:
                file_name = f"{digit}/{digit}_{n:04d}.jpg"
            else:
                file_name = f"0/0_{n:04d}.jpg"

            image = cv2.GaussianBlur(image, (3, 3), 0)
            cv2.imwrite(file_name, image)




if __name__ == "__main__":
    n_images = 1000
    os.chdir("./digits_dataset/assets")

    for digit in digits[:-1]:
        image_path = f"./digits_dataset/assets/{digit}/"
        # os.chdir(image_path)
        print(f"Generating {n_images} training samples for digit: {digit}.")
        generate_assets(digit, n_images, not_zero=True)
    
    for digit in digits[-1]:
        image_path = f"./digits_dataset/assets/0"
        print(f"Generating {n_images} training samples for empty cells.")
        generate_assets(digit, n_images, not_zero=False)
        

