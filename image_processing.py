
# a couple of methods for processing the images
import cv2
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

def filter_(img):
    # Otsu's thresholding
    ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.float32)/9
    th3 = cv2.filter2D(img, -1, kernel)
    return th

def zhangsuen(input_):
    #pseudocode: http://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
    # black = 1, white = 0
    img = input_.copy()
    shape = img.shape
    step = 1;

    # optimization step
    check_pixels = set()
    for row in range(1, shape[0] - 1):
        for col in range(1, shape[1] - 1):
            check_pixels.add((row, col))

    while len(check_pixels) > 0:
        selected_pixels = set()
        if step == 1:
            new_pixels = set()

        check_pixels = list(check_pixels)
        for pixel in check_pixels:
            row, col = pixel

            # pixels all have 8 neighbors
            if row < 1 or col < 1 or row > shape[0] - 2 or col > shape[1] - 2:
                continue 

            # if condition 0 not met
            if img[row, col] == 0:
                continue

            # draws box, goes in a circle around box
            tmp = img[row-1:row+2,col-1:col+2]

            p2 = tmp[0,1]
            p3 = tmp[0,2]
            p4 = tmp[1,2]
            p5 = tmp[2,2]
            p6 = tmp[2,1]
            p7 = tmp[2,0]
            p8 = tmp[1,0]
            p9 = tmp[0,0]

            blackNum = sum([1 if b!=0 else 0 for b in [p2, p3, p4, p5, p6, p7, p8, p9]])

            if blackNum < 2 or blackNum > 6:
                continue

            trans = [p2,p3,p4,p5,p6,p7,p8,p9]
            trans2 = [p3,p4,p5,p6,p7,p8,p9,p2]
            numTrans = sum([1 if t[0] == 0 and t[1] != 0 else 0 for t in zip(trans, trans2)])

            # if condition 2 not met
            if numTrans != 1:
                continue

            if (step == 1 and (p4== 0 or p6==0 or (p2==0 and p8==0)))\
                or (step == 0 and (p2 == 0 or p8 == 0 or (p4==0 and p6==0))):
                selected_pixels.add((row, col))
                

        for pixel in selected_pixels:
            row, col = pixel
            img[row, col] = 0
            new_pixels.add((row-1, col))
            new_pixels.add((row-1, col+1))
            new_pixels.add((row, col+1))
            new_pixels.add((row+1, col+1))
            new_pixels.add((row+1, col))
            new_pixels.add((row+1, col-1))
            new_pixels.add((row, col-1))
            new_pixels.add((row-1, col-1))

        step = (step + 1) % 2

        if step == 1:
            check_pixels = new_pixels

    return img