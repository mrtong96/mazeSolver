
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy import weave
import sys


names = ['0', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['noisy_images/noise{}.jpg'.format(i) for i in range(6)]
'''
names = ['0.png', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['images/maze' + name for name in names]
'''
def filter_(img):
	# Otsu's thresholding
	ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return th

def zhangsuen(img):
	#pseudocode: http://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
	shape = img.shape
	step = 1;
	changed = True
	while(changed):
		changed = False
		for row in range(1, shape[0] - 1):
			for col in range(1, shape[1] - 1):
				isBlack = (img[row,col] == 0)
				numTransitions = (img[row - 1, col - 1] == 255 or img[row - 1, col] == 255) + \
					(img[row - 1, col] == 255 or img[row - 1, col + 1] == 255) + \
					(img[row - 1, col + 1] == 255 or img[row, col + 1] == 255) + \
					(img[row, col + 1] == 255 or img[row + 1, col + 1] == 255) + \
					(img[row + 1, col + 1] == 255 or img[row + 1, col] == 255) + \
					(img[row + 1, col] == 255 or img[row + 1, col - 1] == 255) + \
					(img[row + 1, col - 1] == 255 or img[row, col - 1] == 255) + \
					(img[row, col - 1] == 255 or img[row - 1, col - 1] == 255)
				numBlackNeighbors = (img[row - 1, col - 1] == 0) + (img[row - 1, col] == 0) + (img[row - 1, col + 1] == 0) + (img[row, col - 1] == 0) + (img[row, col + 1] == 0) + (img[row + 1, col - 1] == 0) + (img[row + 1, col] == 0) + (img[row + 1, col + 1] == 0)
				if(step == 1):
					fuckThis = (img[row - 1, col] == 255 or img[row, col + 1] == 255 or img[row + 1, col] == 255)
					fuckThis2 = (img[row, col - 1] == 255 or img[row, col + 1] == 255 or img[row + 1, col] == 255)
					step = 2
				elif(step == 2):
					fuckThis = (img[row - 1, col] == 255 or img[row, col + 1] == 255 or img[row, col - 1] == 255)
					fuckThis2 = (img[row, col - 1] == 255 or img[row - 1, col] == 255 or img[row + 1, col] == 255)
					step = 1
				if(isBlack and numBlackNeighbors >= 2 and numBlackNeighbors <= 6 and numTransitions == 1 and fuckThis and fuckThis2):
					img[row,col] == 255
					changed = True
	return img

for name in names:
	img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	img_copy = img.copy()

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

	cropped_img = filtered[min_y: max_y, min_x: max_x]

	# draws bounding boxes around rectangles
	filtered_copy = filtered.copy()
	cv2.rectangle(filtered_copy,(min_x,min_y),(max_x,max_y),0,1)

	#filtered_copy = filtered.copy()

	zhang_suen = zhangsuen(cropped_img.copy())


	plt.subplot(2, 3, 0)
	plt.title('original')
	plt.imshow(img, cmap='gray')
	plt.subplot(2, 3, 1)
	plt.title('filtered')
	plt.imshow(filtered_copy, cmap='gray')
	plt.subplot(2, 3, 2)
	plt.title('with rectangle')
	plt.imshow(filtered, cmap='gray')
	plt.subplot(2, 3, 3)
	plt.title('cropped_img')
	plt.imshow(cropped_img, cmap='gray')
	plt.subplot(2, 3, 4)
	plt.title('zhang suen')
	plt.imshow(zhang_suen, cmap='gray')
	plt.show()
	print(cropped_img)

