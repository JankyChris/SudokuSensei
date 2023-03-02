#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Printed digit dataset generator.
@author: Christoph M. Jankowsky
"""

import numpy as np
import cv2

fonts = [cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX]
digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
fontColor               = (255,255,255)
thickness               = 1
lineType                = 2

for digit in digits[:-1]:
    for font in fonts:
        img = np.zeros((28,28,3), np.uint8)
        at = (np.random.randint(0, 14), np.random.randint(14, 28))
        fontScale = 0.1 * np.random.randint(5, 11)

        cv2.putText(img, 
                    digit, 
                    at, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        cv2.imshow("img",img)
        cv2.waitKey(0)

for digit in digits[-1]:
    for font in fonts:
        img = np.zeros((28,28,3), np.uint8)
        at = (np.random.randint(0, 28), np.random.randint(0, 28))
        fontScale = 0.1 * np.random.randint(0, 10)

        cv2.putText(img, 
                    digit, 
                    at, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        cv2.imshow("img",img)
        cv2.waitKey(0)
