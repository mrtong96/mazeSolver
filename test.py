
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy


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

	# draws bounding boxes around rectangles
	filtered_copy = filtered.copy()
	for rect in rects:
		x,y,w,h = rect
		print x,y,w+x,h+y

		cv2.rectangle(filtered_copy,(x,y),(x+w,y+h),0,1)

	#filtered_copy = filtered.copy()

	plt.subplot(2, 2, 0)
	plt.title('original')
	plt.imshow(img, cmap='gray')
	plt.subplot(2, 2, 1)
	plt.title('filtered')
	plt.imshow(filtered_copy, cmap='gray')
	plt.subplot(2, 2, 2)
	plt.title('with rectangle')
	plt.imshow(filtered, cmap='gray')

	plt.show()

