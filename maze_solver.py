
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from pathing import ImageSolver
from image_processing import filter_, zhangsuen

def main():
    names = ['noisy_images/noise{}.png'.format(i) for i in range(5)]

    for name in names:
        print 'processing image: {}'.format(name)
        t0 = time.time()

        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        smallest_dim = min(img.shape)
        scale = 720.0 / smallest_dim
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        img_copy = img.copy()

        print 'resizing time: {}'.format(time.time() - t0)

        # gets the area of interest
        filtered = filter_(img)
        # fix I found on stack overflow:
        # http://stackoverflow.com/questions/20743850/python-opencv-error-finding-contours
        ret,thresh = cv2.threshold(filtered.copy(),127,255,cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_per = map(lambda x: (x, cv2.arcLength(x, True)), contours)
        cont_per.sort(key=lambda x: -x[1])
        longest_2 = zip(*cont_per[0:2])[0]
        rects = map(lambda x: cv2.boundingRect(x), longest_2)

        # finds rectangle bounding maze
        DELTA = 5
        min_x0, min_y0, max_x0, max_y0 =\
            rects[0][0], rects[0][1], rects[0][0] + rects[0][2], rects[0][1] + rects[0][3]
        min_x1, min_y1, max_x1, max_y1 =\
            rects[1][0], rects[1][1], rects[1][0] + rects[1][2], rects[1][1] + rects[1][3]
        min_x = min(min_x0, min_x1) - DELTA
        min_y = min(min_y0, min_y1) - DELTA
        max_x = max(max_x0, max_x1) + DELTA
        max_y = max(max_y0, max_y1) + DELTA

        filtered_copy = filtered.copy()
        cv2.rectangle(filtered_copy,(min_x,min_y),(max_x,max_y),0,1)

        cropped_img = filtered[min_y: max_y, min_x: max_x]

        print 'find contours and crop: {}'.format(time.time() - t0)

        if cropped_img.size == 0:
            print 'error: no maze found'
            continue

        # now process the cropped_img
        z = zhangsuen(cropped_img.copy())

        print 'zs alg: {}'.format(time.time() - t0)

        # draws bounding boxes around rectangles
        filtered_copy = filtered.copy()
        cv2.rectangle(filtered_copy,(min_x,min_y),(max_x,max_y),0,1)

        i = ImageSolver(z)
        y = cv2.cvtColor(z, cv2.COLOR_GRAY2RGB)
        y = i.highlightPOIs(y)

        print 'initialize solver: {}'.format(time.time() - t0)

        a = i.solveMaze()
        if a:
            for pixel in a:
                y[pixel] = [0,255,0]

            pixels = map(lambda x: (min_x + x[1], min_y + x[0]), a)
            for pixel in pixels:
                cv2.circle(img, pixel, 5, [0, 255, 0])

        print 'solve maze: {}'.format(time.time() - t0)

        # display stuff
        plt.subplot(2, 2, 1)
        plt.title('original image')
        plt.imshow(img_copy, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.title('filtered image')
        plt.imshow(filtered_copy, cmap='gray')
        plt.subplot(2, 2, 3)
        plt.title('processed image')
        plt.imshow(z, cmap='gray')
        plt.subplot(2, 2, 0)
        plt.title('final result')
        plt.imshow(img, cmap='gray')

        # show time for sanity checks
        print 'total time: {}'.format(time.time() - t0)
        plt.show()

if __name__ == '__main__':
    main()



