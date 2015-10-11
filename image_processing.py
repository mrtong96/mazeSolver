
# a couple of methods for processing the images
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

        for pixel in check_pixels:
            row, col = pixel

            if row < 1 or row > shape[0] - 2:
                continue
            if col < 1 or col > shape[1] - 2:
                continue

            # if condition 0 not met
            if img[row, col] == 0:
                continue

            p2 = img[row-1, col]
            p3 = img[row-1, col+1]
            p4 = img[row, col+1]
            p5 = img[row+1, col+1]
            p6 = img[row+1, col]
            p7 = img[row+1, col-1]
            p8 = img[row, col-1]
            p9 = img[row-1, col-1]

            #import pdb; pdb.set_trace()
            blackPix = map(lambda x: x != 0, [p2, p3, p4, p5, p6, p7, p8, p9])
            blackNum = sum([1 if b else 0 for b in blackPix])
            if blackNum < 2 or blackNum> 6:
                continue

            trans = [p2,p3,p4,p5,p6,p7,p8,p9]
            trans2 = [p3,p4,p5,p6,p7,p8,p9,p2]
            trans = map(lambda x: x[0] ==0 and x[1] != 0, zip(trans, trans2))
            numTrans = sum([1 if t else 0 for t in trans])
            
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