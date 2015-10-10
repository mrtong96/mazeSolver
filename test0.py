
# testing filtering stuff

import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

names = ['0', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['noisy_images/noise{}.jpg'.format(i) for i in range(6)]


for name in names:
	img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	img_copy = img.copy()

	# global thresholding
	ret1, th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

	# Otsu's thresholding
	ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# add median filter (3x3)
	kernel = np.ones((3,3), np.float32)/9
	th3 = cv2.filter2D(img, -1, kernel)

	# gaussian then otsu's
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# otsu's then gaussian
	ret5,th5 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	th5 = cv2.GaussianBlur(img, (5,5), 0)

	plt.subplot(2,1,0)
	plt.title('otsu')
	plt.imshow(th2, cmap='gray')
	plt.subplot(2,1,1)
	plt.title('gaussian+otsu')
	plt.imshow(th4, cmap='gray')

	'''
	plt.subplot(3,2,0)
	plt.title('origininal')
	plt.imshow(img, cmap='gray')
	plt.subplot(3,2,1)
	plt.title('global')
	plt.imshow(th1, cmap='gray')
	plt.subplot(3,2,2)
	plt.title('otsu')
	plt.imshow(th2, cmap='gray')
	plt.subplot(3,2,3)
	plt.title('otsu+median')
	plt.imshow(th3, cmap='gray')
	plt.subplot(3,2,4)
	plt.title('gaussian+otsu')
	plt.imshow(th4, cmap='gray')
	plt.subplot(3,2,5)
	plt.title('otsu+gaussian')
	plt.imshow(th5, cmap='gray')
	'''

	plt.show()

